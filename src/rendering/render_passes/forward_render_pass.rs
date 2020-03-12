use std::{iter, mem, ptr};
use std::borrow::{Borrow, BorrowMut};
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt::Debug;
use std::hash::Hasher;
use std::io::{Cursor, Error};
use std::iter::once;
use std::mem::ManuallyDrop;
use std::ops::{Deref, Range};
use std::process::exit;
use std::rc::Rc;
use std::sync::{Arc, Mutex, Weak};

use cgmath::{Matrix, Matrix4, SquareMatrix};
use gfx_hal::{Backend, command, format, format::Format, image, IndexType, memory, pass, pass::Attachment, pso};
use gfx_hal::buffer::IndexBufferView;
use gfx_hal::command::{ClearColor, ClearDepthStencil, ClearValue, CommandBuffer, ImageBlit, ImageCopy, SubpassContents};
use gfx_hal::device::Device;
use gfx_hal::format::ChannelType;
use gfx_hal::image::{Extent, Filter, Layout, Level, Offset, SubresourceLayers, SubresourceRange};
use gfx_hal::image::Layout::{TransferDstOptimal, TransferSrcOptimal};
use gfx_hal::image::Usage;
use gfx_hal::image::ViewError::Layer;
use gfx_hal::memory::Barrier;
use gfx_hal::pass::{Subpass, SubpassDependency, SubpassRef};
use gfx_hal::pool::CommandPool;
use gfx_hal::pso::{Comparison, DepthTest, DescriptorPool, DescriptorPoolCreateFlags, DescriptorRangeDesc, DescriptorSetLayoutBinding, DescriptorType, FrontFace, ShaderStageFlags, VertexInputRate};
use gfx_hal::queue::{CommandQueue, Submission};
use gfx_hal::window::{Extent2D, Surface, SwapImageIndex};
use winit::event::WindowEvent::CursorMoved;

use crate::rendering::{CameraData, ForwardPipeline, GPUMesh, InstanceData, MeshID, Pipeline, RenderPass, ResolvePipeline, ResourceManager, ShaderCode, Uniform, Vertex};
use crate::rendering::framebuffer::Framebuffer;
use crate::rendering::renderer::Renderer;
use glium::draw_parameters::sync;

//use crate::rendering::pipelines::{ResolvePipeline, ForwardPipeline, Pipeline};

/* Constants */
const ENTRY_NAME: &str = "main";

// Uniform
const CAMERA_UNIFORM_BINDING: u32 = 0;
const MODEL_MATRIX_UNIFORM_BINDING: u32 = 1;

pub struct ForwardRenderPass<B: Backend> {
    device: Arc<B::Device>,

    resource_manager: Arc<Mutex<ResourceManager<B>>>,

    pub render_pass: ManuallyDrop<B::RenderPass>,

    forward_pipeline: ForwardPipeline<B>,
    resolve_pipeline: ResolvePipeline<B>,

    extent: Extent2D,
    viewport: pso::Viewport,

    pub desc_pool: ManuallyDrop<B::DescriptorPool>,
    pub desc_set: Vec<B::DescriptorSet>,
    pub render_set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    framebuffer: Framebuffer<B, B::Device>,

    pub cmd_buffers: Vec<B::CommandBuffer>,

    // uniforms
    camera_uniform: Uniform<B>,

    instances: HashMap<MeshID, Vec<Matrix4<f32>>>,
}

impl<B: Backend> ForwardRenderPass<B> {
    pub fn new(renderer: &mut Renderer<B>, resource_manager: &Arc<Mutex<ResourceManager<B>>>) -> Self {
        let device = &renderer.device;
        let adapter = &renderer.adapter;

        // Setup renderpass and pipelines
        let render_set_layout = ManuallyDrop::new(
            unsafe {
                device.create_descriptor_set_layout(
                    &[
                        DescriptorSetLayoutBinding {
                            binding: 0,
                            ty: DescriptorType::UniformBuffer,
                            count: 1,
                            stage_flags: ShaderStageFlags::all(),
                            immutable_samplers: false,
                        },
                    ],
                    &[],
                )
            }.expect("Can't create descriptor set layout"),
        );

        // Descriptors
        let mut desc_pool = ManuallyDrop::new(
            unsafe {
                device.create_descriptor_pool(
                    renderer.frames_in_flight, // sets
                    &[
                        DescriptorRangeDesc {
                            ty: DescriptorType::UniformBuffer,
                            count: renderer.frames_in_flight,
                        }
                    ],
                    DescriptorPoolCreateFlags::empty(),
                )
            }
                .expect("Can't create descriptor pool"),
        );

        let mut desc_set: Vec<B::DescriptorSet> = Vec::new();
        unsafe {
            for _ in 0..renderer.frames_in_flight {
                desc_set.push(desc_pool.allocate_set(&render_set_layout).unwrap());
            }
        };

        let caps = renderer.surface.capabilities(&adapter.physical_device);
        let formats = &renderer.surface.supported_formats(&adapter.physical_device);
        println!("formats: {:?}", formats);
        let format = formats.clone().map_or(Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });
        let depth_stencil_format = formats.clone().map_or(Format::D24UnormS8Uint, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Sfloat)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let render_pass = {
            let attachment = pass::Attachment {
                format: Some(Format::Rgba32Sfloat), //Some(format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::TransferSrcOptimal,
            };

            let depth_attachment = pass::Attachment {
                format: Some(Format::D24UnormS8Uint), //Some(format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::DepthStencilAttachmentOptimal,
            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: Some(&(1, Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            // Subpass dependencies enable early fragment test mode
            // source: https://rust-tutorials.github.io/learn-gfx-hal/09_depth_buffer.html
            let in_dependency = SubpassDependency {
                passes: SubpassRef::External..SubpassRef::Pass(0),
                stages: pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT
                    .. pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT | pso::PipelineStage::EARLY_FRAGMENT_TESTS,
                accesses: image::Access::empty()
                    ..(image::Access::COLOR_ATTACHMENT_READ
                    | image::Access::COLOR_ATTACHMENT_WRITE
                    | image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE),
                flags: memory::Dependencies::empty()
            };
            let out_dependency = SubpassDependency {
                passes: SubpassRef::Pass(0)..SubpassRef::External,
                stages: pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT | pso::PipelineStage::EARLY_FRAGMENT_TESTS
                    .. pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: (image::Access::COLOR_ATTACHMENT_READ
                    | image::Access::COLOR_ATTACHMENT_WRITE
                    | image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE)..image::Access::empty(),
                flags: memory::Dependencies::empty()
            };

            ManuallyDrop::new(
                unsafe { device.create_render_pass(&[attachment, depth_attachment], &[subpass], &[in_dependency, out_dependency]) }
                    .expect("Can't create render pass"),
            )
        };

        // uniforms
        let camera_uniform = Uniform::new(&renderer,
                                          &[CameraData::default()],
                                          CAMERA_UNIFORM_BINDING,
                                          &desc_set);

        let mut cmd_buffers = Vec::with_capacity(renderer.frames_in_flight);
        for i in 0..renderer.frames_in_flight {
            unsafe {
                cmd_buffers.push(renderer.cmd_pools[i].allocate_one(command::Level::Primary));
            }
        }

        let forward_pipeline = ForwardPipeline::new(device, render_pass.deref(), render_set_layout.deref());
        let resolve_pipeline = ResolvePipeline::new(device, render_pass.deref(), render_set_layout.deref());

        let mut framebuffer = Framebuffer::new(device,
                                               adapter,
                                               &renderer.queue_group,
                                               &render_pass,
                                               renderer.dimensions,
                                               Usage::COLOR_ATTACHMENT | Usage::TRANSFER_SRC | Usage::TRANSFER_DST,
                                               Format::Rgba32Sfloat,
                                               renderer.frames_in_flight).unwrap();

        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: renderer.dimensions.width as _,
                h: renderer.dimensions.height as _,
            },
            depth: 0.0..1.0,
        };

        ForwardRenderPass {
            device: device.clone(),

            resource_manager: resource_manager.clone(),

            render_pass,

            forward_pipeline,
            resolve_pipeline,

            extent: renderer.dimensions,
            viewport,

            desc_pool,
            desc_set,
            render_set_layout,
            framebuffer,

            cmd_buffers,

            camera_uniform,

            instances: HashMap::new(),
        }
    }

    pub fn recreate_pipeline(&mut self) {
        self.resolve_pipeline = ResolvePipeline::new(&self.device, self.render_pass.deref(), self.render_set_layout.deref());
        self.forward_pipeline = ForwardPipeline::new(&self.device, self.render_pass.deref(), self.render_set_layout.deref());
    }

    pub fn update_camera(&mut self, camera_data: CameraData, frame_idx: usize) {
        self.camera_uniform.buffers[frame_idx].update_data(0, &[camera_data]);
    }

    pub fn reset_instances(&mut self) {
        for (_, transforms) in self.instances.iter_mut() {
            transforms.clear();
        }
    }

    pub fn add_instance(&mut self, mesh_id: MeshID, transform: Matrix4<f32>) {
        match self.instances.get_mut(&mesh_id) {
            Some(transforms) => {
                transforms.push(transform);
            }
            None => {
                self.instances.insert(mesh_id, vec![transform]);
            }
        };
    }

}

impl<B: Backend> Drop for ForwardRenderPass<B> {
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
        unsafe {
            // TODO: When ManuallyDrop::take (soon to be renamed to ManuallyDrop::read) is stabilized we should use that instead.
            self.device
                .destroy_descriptor_pool(ManuallyDrop::into_inner(ptr::read(&self.desc_pool)));
            self.device
                .destroy_descriptor_set_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.render_set_layout,
                )));

            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.render_pass)));
        }
    }
}

impl<B: Backend> RenderPass<B> for ForwardRenderPass<B> {
    fn sync(&mut self, frame_idx: usize) {
        let (fe,
            fi,
            framebuffer,
            pool,
            command_buffers,
            semaphore) = self.framebuffer.get_frame_data(frame_idx);

        unsafe {
            self.device
                .wait_for_fence(fe, !0)
                .expect("Failed to wait for fence");
            self.device
                .reset_fence(fe)
                .expect("Failed to reset fence");
            pool.reset(false);
        }
    }

    fn submit(&mut self, frame_idx: usize, queue: &mut B::CommandQueue, wait_semaphores: Vec<&B::Semaphore>)
              -> &B::Semaphore {
        let (fe,
            fi,
            framebuffer,
            pool,
            command_buffers,
            semaphore) = self.framebuffer.get_frame_data(frame_idx);

        unsafe {
            let wait_sem = wait_semaphores.iter().map(|sem| {
                (*sem, pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT)
            }).collect::<Vec<(&B::Semaphore, pso::PipelineStage)>>();

            let submission = Submission {
                command_buffers: command_buffers.iter(),
                wait_semaphores: wait_sem,
                signal_semaphores: vec![&*semaphore],
            };
            queue.submit(
                submission,
                Some(fe),
            );
        }

        semaphore
    }

    fn get_render_pass(&self) -> &ManuallyDrop<B::RenderPass> {
        &self.render_pass
    }

    fn get_descriptor_set(&self, frame_index: usize) -> &B::DescriptorSet {
        &self.desc_set[frame_index]
    }

    fn blit_to_surface(&mut self, queue: &mut B::CommandQueue, surface_image: &B::Image, frame_idx: usize, acquire_semaphore: &B::Semaphore) -> &B::Semaphore {
        self.sync(frame_idx);

        let (fe,
            fi,
            framebuffer,
            pool,
            command_buffers,
            semaphore) = self.framebuffer.get_frame_data(frame_idx);

        unsafe {
            // blitting
            let mut cmd_buffer = match command_buffers.pop() {
                Some(cmd_buffer) => cmd_buffer,
                None => pool.allocate_one(command::Level::Primary),
            };

            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            let mut target_image_barrier = Barrier::Image {
                states: (image::Access::TRANSFER_READ, Layout::Undefined)
                    ..(image::Access::TRANSFER_WRITE, Layout::TransferDstOptimal),
                target: surface_image,
                families: None,
                range: SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };

            cmd_buffer.pipeline_barrier(
                pso::PipelineStage::TRANSFER..pso::PipelineStage::TRANSFER,
                gfx_hal::memory::Dependencies::empty(),
                &[target_image_barrier],
            );

            let mut image_barrier = Barrier::Image {
                states: (image::Access::MEMORY_WRITE, Layout::TransferSrcOptimal)
                    ..(image::Access::MEMORY_READ, Layout::TransferSrcOptimal),
                target: &*fi.image,
                families: None,
                range: SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };

            cmd_buffer.pipeline_barrier(
                pso::PipelineStage::TOP_OF_PIPE..pso::PipelineStage::TRANSFER,
                gfx_hal::memory::Dependencies::all(),
                &[image_barrier],
            );

//            cmd_buffer.copy_image(&fi.image,
//                                  Layout::TransferSrcOptimal,
//                                  surface_image,
//                                  Layout::TransferDstOptimal,
//                                  iter::once(ImageCopy {
//                                      extent: Extent {
//                                          width: self.extent.width,
//                                          height: self.extent.height,
//                                          depth: 1,
//                                      },
//                                      src_subresource: SubresourceLayers {
//                                          aspects: format::Aspects::COLOR,
//                                          level: 0,
//                                          layers: 0..1,
//                                      },
//                                      src_offset: Offset { x: 0, y: 0, z: 0 },
//                                      dst_subresource: SubresourceLayers {
//                                          aspects: format::Aspects::COLOR,
//                                          level: 0,
//                                          layers: 0..1,
//                                      },
//                                      dst_offset: Offset { x: 0, y: 0, z: 0 },
//                                  }));

            cmd_buffer.blit_image(&fi.image,
                                  Layout::TransferSrcOptimal,
                                  surface_image,
                                  Layout::TransferDstOptimal,
                                  Filter::Linear,
                                  iter::once(ImageBlit {
                                      src_subresource: SubresourceLayers {
                                          aspects: format::Aspects::COLOR,
                                          level: 0,
                                          layers: 0..1,
                                      },
                                      src_bounds: Offset { x: 0, y: 0, z: 0 }
                                          ..Offset { x: self.extent.width as i32, y: self.extent.height as i32, z: 1 },
                                      dst_subresource: SubresourceLayers {
                                          aspects: format::Aspects::COLOR,
                                          level: 0,
                                          layers: 0..1,
                                      },
                                      dst_bounds: Offset { x: 0, y: 0, z: 0 }
                                          ..Offset { x: self.extent.width as i32, y: self.extent.height as i32, z: 1 },
                                  }));

            let image_barrier = Barrier::Image {
                states: (image::Access::TRANSFER_READ, Layout::TransferSrcOptimal)
                    ..(image::Access::TRANSFER_WRITE, Layout::ColorAttachmentOptimal),
                target: &*fi.image,
                families: None,
                range: SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };

            cmd_buffer.pipeline_barrier(
                pso::PipelineStage::TRANSFER..pso::PipelineStage::TRANSFER,
                gfx_hal::memory::Dependencies::empty(),
                &[image_barrier],
            );

            let target_image_barrier = Barrier::Image {
                states: (image::Access::TRANSFER_WRITE, Layout::TransferDstOptimal)
                    ..(image::Access::MEMORY_READ, Layout::Present),
                target: surface_image,
                families: None,
                range: SubresourceRange {
                    aspects: format::Aspects::COLOR,
                    levels: 0..1,
                    layers: 0..1,
                },
            };

            cmd_buffer.pipeline_barrier(
                pso::PipelineStage::TRANSFER..pso::PipelineStage::TRANSFER,
                gfx_hal::memory::Dependencies::empty(),
                &[target_image_barrier],
            );

            cmd_buffer.finish();

            command_buffers.push(cmd_buffer);
        }

        unsafe {
            let submission = Submission {
                command_buffers: command_buffers.iter(),
                wait_semaphores: vec![(&*semaphore, pso::PipelineStage::TRANSFER),
                                      (acquire_semaphore, pso::PipelineStage::TRANSFER)],
                signal_semaphores: vec![&*semaphore],
            };
            queue.submit(
                submission,
                Some(fe),
            );
        }

        semaphore
    }

    fn record(&mut self, frame_idx: usize) {
        let (fe,
            fi,
            framebuffer,
            pool,
            command_buffers,
            semaphore) = self.framebuffer.get_frame_data(frame_idx);

        unsafe {
            let mut command_buffer = match command_buffers.pop() {
                Some(cmd_buffer) => cmd_buffer,
                None => pool.allocate_one(command::Level::Primary),
            };

            command_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            command_buffer.set_viewports(0, &[self.viewport.clone()]);
            command_buffer.set_scissors(0, &[self.viewport.rect]);
            command_buffer.bind_graphics_pipeline(&self.forward_pipeline.get_pipeline());

            command_buffer.bind_graphics_descriptor_sets(
                &self.forward_pipeline.get_layout(),
                0,
                iter::once(&self.desc_set[frame_idx]),
                &[],
            );

            command_buffer.begin_render_pass(
                &self.render_pass,
                framebuffer,
                self.viewport.rect,
                &[
                    command::ClearValue {
                        color: command::ClearColor {
                            float32: [0.3, 0.3, 0.3, 1.0],
                        },
                    },
                    command::ClearValue {
                        depth_stencil: ClearDepthStencil {
                            depth: 1f32,
                            stencil: 0,
                        },
                    }
                ],
                command::SubpassContents::Inline,
            );

            let resource_manager = self.resource_manager.lock().unwrap();
            for (id, transforms) in self.instances.iter() {
                let mesh = resource_manager.get_mesh(id);
                let vert_buf = &**mesh.vertex_buffer;
                let ind_buf = &**mesh.index_buffer;

                for transform in transforms {
                    let mut data: &[f32; 16] = transform.as_ref();
                    let push_data: [u32; 16] = std::mem::transmute_copy(data);

                    command_buffer.push_graphics_constants(&self.forward_pipeline.get_layout(),
                                                           ShaderStageFlags::VERTEX,
                                                           0,
                                                           &push_data);

                    command_buffer.bind_vertex_buffers(0, iter::once((vert_buf, 0)));

                    let index_buffer_view = IndexBufferView {
                        buffer: ind_buf,
                        offset: 0,
                        index_type: mesh.index_type,
                    };
                    command_buffer.bind_index_buffer(index_buffer_view);
                    command_buffer.draw_indexed(0..mesh.num_indices, 0, 0..1);
                }
            }

            command_buffer.end_render_pass();

            command_buffer.finish();

            command_buffers.push(command_buffer);
        }
    }
}