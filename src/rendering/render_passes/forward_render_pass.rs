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
use gfx_hal::{Backend, command, format, format::Format, image, IndexType, memory, pass, pass::Attachment, pso, query};
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
use gfx_hal::pso::{Comparison, DepthTest, Descriptor, DescriptorPool, DescriptorPoolCreateFlags, DescriptorRangeDesc, DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, FrontFace, ShaderStageFlags, VertexInputRate};
use gfx_hal::query::Query;
use gfx_hal::queue::{CommandQueue, Submission};
use gfx_hal::range::RangeArg;
use gfx_hal::window::{Extent2D, Surface, SwapImageIndex};
use glium::draw_parameters::sync;
use winit::event::WindowEvent::CursorMoved;

use crate::rendering::{CameraData, ForwardPipeline, GPUBuffer, GPUMesh, InstanceData, MeshID, Pipeline, RenderPass, ResourceManager, ShaderCode, Uniform, Vertex};
use crate::rendering::framebuffer::Framebuffer;
use crate::rendering::renderer::Renderer;
use bytes::{Bytes, Buf};
use crate::rendering::utility::resources::BufferID;
use gfx_hal::pso::State::Static;

//use crate::rendering::pipelines::{ResolvePipeline, ForwardPipeline, Pipeline};

/* Constants */
const ENTRY_NAME: &str = "main";

// Uniform
const CAMERA_UNIFORM_BINDING: u32 = 0;
const MODEL_MATRIX_UNIFORM_BINDING: u32 = 1;

// Pipeline options
pub enum PipelineOptions {
    Default,
    Wireframe,
}

pub struct ForwardRenderPass<B: Backend> {
    device: Arc<B::Device>,

    resource_manager: Arc<Mutex<ResourceManager<B>>>,

    pub render_pass: ManuallyDrop<B::RenderPass>,

    frames_in_flight: u32,

    forward_pipeline: ForwardPipeline<B>,
    forward_wireframe_pipeline: ForwardPipeline<B>,

    extent: Extent2D,
    viewport: pso::Viewport,

    pub desc_pool: ManuallyDrop<B::DescriptorPool>,
    pub desc_set: Vec<B::DescriptorSet>,
    pub render_set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    framebuffer: Framebuffer<B, B::Device>,

    timestamp_query_pool: ManuallyDrop<B::QueryPool>,

    pub cmd_buffers: Vec<B::CommandBuffer>,

    // uniforms
    camera_uniform: Uniform<B>,
    //    instance_ssbo: Uniform<B>,
    instance_buffer: GPUBuffer<B>,
    instance_buffer_id: Option<BufferID>,

    meshes: HashMap<MeshID, Vec<Matrix4<f32>>>,
    meshes_wireframe: HashMap<MeshID, Vec<Matrix4<f32>>>,
    instanced_meshes: HashMap<MeshID, Vec<(Range<usize>, Matrix4<f32>)>>,
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
                        DescriptorSetLayoutBinding { // Camera UBO
                            binding: 0,
                            ty: DescriptorType::UniformBuffer,
                            count: 1,
                            stage_flags: ShaderStageFlags::all(),
                            immutable_samplers: false,
                        },
                        DescriptorSetLayoutBinding { // Instance Data SSBO
                            binding: 1,
                            ty: DescriptorType::StorageBuffer,
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
                        },
                        DescriptorRangeDesc {
                            ty: DescriptorType::StorageBuffer,
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
                    ..pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT | pso::PipelineStage::EARLY_FRAGMENT_TESTS,
                accesses: image::Access::empty()
                    ..(image::Access::COLOR_ATTACHMENT_READ
                    | image::Access::COLOR_ATTACHMENT_WRITE
                    | image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE),
                flags: memory::Dependencies::empty(),
            };
            let out_dependency = SubpassDependency {
                passes: SubpassRef::Pass(0)..SubpassRef::External,
                stages: pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT | pso::PipelineStage::EARLY_FRAGMENT_TESTS
                    ..pso::PipelineStage::COLOR_ATTACHMENT_OUTPUT,
                accesses: (image::Access::COLOR_ATTACHMENT_READ
                    | image::Access::COLOR_ATTACHMENT_WRITE
                    | image::Access::DEPTH_STENCIL_ATTACHMENT_READ
                    | image::Access::DEPTH_STENCIL_ATTACHMENT_WRITE)..image::Access::empty(),
                flags: memory::Dependencies::empty(),
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

        // instance/draw buffer
        let instance_buffer = GPUBuffer::new_with_size(device, adapter,
                                                       std::mem::size_of::<InstanceData>() * 1000,
                                                       gfx_hal::buffer::Usage::STORAGE);
        for idx in 0..renderer.frames_in_flight {
            unsafe {
                device.write_descriptor_sets(iter::once(DescriptorSetWrite {
                    set: &desc_set[idx],
                    binding: 1,
                    array_offset: 0,
                    descriptors: iter::once(
                        Descriptor::Buffer(instance_buffer.get_buffer(), None..None)
                    ),
                }));
            }
        }

        let mut cmd_buffers = Vec::with_capacity(renderer.frames_in_flight);
        for i in 0..renderer.frames_in_flight {
            unsafe {
                cmd_buffers.push(renderer.cmd_pools[i].allocate_one(command::Level::Primary));
            }
        }

        let forward_pipeline = ForwardPipeline::new(device, render_pass.deref(), render_set_layout.deref(), pso::PolygonMode::Fill);
        let forward_wireframe_pipeline = ForwardPipeline::new(device, render_pass.deref(), render_set_layout.deref(), pso::PolygonMode::Line(Static(1.0)));
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

        let timestamp_query_pool = unsafe {
            ManuallyDrop::new(device.create_query_pool(
                query::Type::Timestamp,
                renderer.frames_in_flight as u32 * 2) // * 2 for start and end time stamp
                .expect("Failed to create query pool"))
        };

        ForwardRenderPass {
            device: device.clone(),

            resource_manager: resource_manager.clone(),

            render_pass,
            frames_in_flight: renderer.frames_in_flight as u32,

            forward_pipeline,
            forward_wireframe_pipeline,

            extent: renderer.dimensions,
            viewport,

            desc_pool,
            desc_set,
            render_set_layout,
            framebuffer,

            timestamp_query_pool,

            cmd_buffers,

            camera_uniform,
            instance_buffer,
            instance_buffer_id: None,

            meshes: HashMap::new(),
            meshes_wireframe: HashMap::new(),
            instanced_meshes: HashMap::new(),
        }
    }

    pub fn recreate_pipeline(&mut self) {
        self.forward_pipeline = ForwardPipeline::new(&self.device, self.render_pass.deref(), self.render_set_layout.deref(), pso::PolygonMode::Fill);
        self.forward_wireframe_pipeline = ForwardPipeline::new(&self.device, self.render_pass.deref(), self.render_set_layout.deref(), pso::PolygonMode::Line(Static(1.0)));
    }

    pub fn update_camera(&mut self, camera_data: CameraData, frame_idx: usize) {
        self.camera_uniform.buffers[frame_idx].update_data(0, &[camera_data]);
    }

    pub fn reset(&mut self) {
        for (_, transforms) in self.meshes.iter_mut() {
            transforms.clear();
        }
        for (_, transforms) in self.meshes_wireframe.iter_mut() {
            transforms.clear();
        }
        for (_, ranges) in self.instanced_meshes.iter_mut() {
            ranges.clear();
        }
    }

    pub fn add_mesh(&mut self, mesh_id: MeshID, transform: Matrix4<f32>, pipeline: PipelineOptions) {
        let mut mesh_collection = match pipeline {
            PipelineOptions::Default => &mut self.meshes,
            PipelineOptions::Wireframe => &mut self.meshes_wireframe
        };

        match mesh_collection.get_mut(&mesh_id) {
            Some(transforms) => {
                transforms.push(transform);
            }
            None => {
                mesh_collection.insert(mesh_id, vec![transform]);
            }
        };
    }

    /// Adds a given mesh and buffer to the recorded instances for rendering.
    /// The buffer is assumed to contain mat4 data. (model matrices)
    ///
    /// Replaces instance buffer if it was already added.
    ///
    /// TODO: type checking of buffer (probably requires refactoring of buffers)
    pub fn add_instances(&mut self, mesh_id: MeshID, instance_data_range: (Range<usize>, Matrix4<f32>)) {
        match self.instanced_meshes.get_mut(&mesh_id) {
            Some(instances) => {
                instances.push(instance_data_range);
            }
            None => {
                self.instanced_meshes.insert(mesh_id, vec![instance_data_range]);
            }
        };
    }

    pub fn use_instance_buffer(&mut self, buffer: &Arc<Mutex<GPUBuffer<B>>>, frame_idx: usize) {
        let rm_lock = self.resource_manager.lock().unwrap();
        let buf_lock = buffer.lock().unwrap();

        unsafe {
            self.device.write_descriptor_sets(iter::once(DescriptorSetWrite {
                set: &self.desc_set[frame_idx],
                binding: 1,
                array_offset: 0,
                descriptors: iter::once(
                    Descriptor::Buffer(buf_lock.get_buffer(), None..None)
                ),
            }));
        }
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

            self.device
                .destroy_query_pool(ManuallyDrop::into_inner(ptr::read(&self.timestamp_query_pool)));
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

            command_buffer.reset_query_pool(&self.timestamp_query_pool, 0 .. self.frames_in_flight * 2);

            command_buffer.write_timestamp(pso::PipelineStage::TOP_OF_PIPE, Query {
                pool: &*self.timestamp_query_pool,
                id: (frame_idx * 2) as u32,
            });

            command_buffer.set_viewports(0, &[self.viewport.clone()]);
            command_buffer.set_scissors(0, &[self.viewport.rect]);

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


            {
                let rm_lock = self.resource_manager.lock().unwrap();

                if !self.meshes.is_empty() {
                    command_buffer.bind_graphics_pipeline(&self.forward_pipeline.get_pipeline(false));
                    command_buffer.bind_graphics_descriptor_sets(
                        &self.forward_pipeline.get_layout(false),
                        0,
                        iter::once(&self.desc_set[frame_idx]),
                        &[],
                    );
                }

                // generate simple draw calls
                for (id, transforms) in self.meshes.iter() {
                    let mesh = rm_lock.get_mesh(id);
                    let vert_buf = &**mesh.vertex_buffer;
                    let ind_buf = &**mesh.index_buffer;

                    for transform in transforms {
                        let mut data: &[f32; 16] = transform.as_ref();
                        let push_data: [u32; 16] = std::mem::transmute_copy(data);

                        command_buffer.push_graphics_constants(&self.forward_pipeline.get_layout(false),
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

                if !self.meshes_wireframe.is_empty() {
                    command_buffer.bind_graphics_pipeline(&self.forward_wireframe_pipeline.get_pipeline(false));
                    command_buffer.bind_graphics_descriptor_sets(
                        &self.forward_wireframe_pipeline.get_layout(false),
                        0,
                        iter::once(&self.desc_set[frame_idx]),
                        &[],
                    );
                }

                // generate simple draw calls for wire frames
                for (id, transforms) in self.meshes_wireframe.iter() {
                    let mesh = rm_lock.get_mesh(id);
                    let vert_buf = &**mesh.vertex_buffer;
                    let ind_buf = &**mesh.index_buffer;

                    for transform in transforms {
                        let mut data: &[f32; 16] = transform.as_ref();
                        let push_data: [u32; 16] = std::mem::transmute_copy(data);

                        command_buffer.push_graphics_constants(&self.forward_pipeline.get_layout(false),
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
            } // drop lock

            {
                let rm_lock = self.resource_manager.lock().unwrap();

                if !self.instanced_meshes.is_empty() {
                    command_buffer.bind_graphics_pipeline(&self.forward_pipeline.get_pipeline(true));
                    command_buffer.bind_graphics_descriptor_sets(
                        &self.forward_pipeline.get_layout(true),
                        0,
                        iter::once(&self.desc_set[frame_idx]),
                        &[],
                    );
                }

                // generate instanced draw calls
                for (id, instance_ranges) in self.instanced_meshes.iter() {
                    let mesh = rm_lock.get_mesh(id);
                    let vert_buf = &**mesh.vertex_buffer;
                    let ind_buf = &**mesh.index_buffer;

                    for (range, transform) in instance_ranges {
                        command_buffer.bind_vertex_buffers(0, iter::once((vert_buf, 0)));

                        let mut data: &[f32; 16] = transform.as_ref();
                        let push_data: [u32; 16] = std::mem::transmute_copy(data);

                        command_buffer.push_graphics_constants(&self.forward_pipeline.get_layout(true),
                                                               ShaderStageFlags::VERTEX,
                                                               0,
                                                               &push_data);
                        command_buffer.push_graphics_constants(&self.forward_pipeline.get_layout(true),
                                                               ShaderStageFlags::VERTEX,
                                                               64,
                                                               &[range.start as u32]);

                        let index_buffer_view = IndexBufferView {
                            buffer: ind_buf,
                            offset: 0,
                            index_type: mesh.index_type,
                        };
                        command_buffer.bind_index_buffer(index_buffer_view);
                        command_buffer.draw_indexed(0..mesh.num_indices, 0, 0..range.end as u32);
                    }
                }
            } // drop lock

            command_buffer.end_render_pass();

            command_buffer.write_timestamp(pso::PipelineStage::BOTTOM_OF_PIPE, Query {
                pool: &*self.timestamp_query_pool,
                id: (frame_idx * 2  + 1) as u32,
            });

            command_buffer.finish();

            command_buffers.push(command_buffer);
        }
    }

    /// Return the execution time of this pass in nano seconds
    fn execution_time(&mut self, frame_idx: usize) -> u64 {
        let mut data: [u8; 16] = [0u8; 16];

        unsafe {
            self.device.get_query_pool_results(
                &self.timestamp_query_pool,
                (frame_idx * 2) as u32 .. ((frame_idx * 2) + 2) as u32,
                &mut data,
                0,
                gfx_hal::query::ResultFlags::WAIT | gfx_hal::query::ResultFlags::BITS_64).unwrap();

        }

        // TODO: check if VkQueueFamilyProperties::timestampValidBits and timestampPeriod is already handled by gfx_hal
        // It seems like this is not done. Has to be investigated further.
        // https://www.reddit.com/r/vulkan/comments/b6m6wx/how_to_use_timestamps/
        let start = Bytes::copy_from_slice(&data[0..8]).get_u64_le();
        let end = Bytes::copy_from_slice(&data[8..16]).get_u64_le();

        return end - start;
    }
}