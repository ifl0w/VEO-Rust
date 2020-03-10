use std::{iter, mem, ptr};
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::HashMap;
use std::convert::TryInto;
use std::fmt::Debug;
use std::hash::Hasher;
use std::io::{Cursor, Error};
use std::mem::ManuallyDrop;
use std::ops::{Deref, Range};
use std::process::exit;
use std::sync::{Arc, Mutex, Weak};

use cgmath::{Matrix, Matrix4, SquareMatrix};
use gfx_hal::{Backend, command, format, format::Format, image, IndexType, pass, pass::Attachment, pso};
use gfx_hal::buffer::IndexBufferView;
use gfx_hal::command::{ClearDepthStencil, CommandBuffer, ImageBlit};
use gfx_hal::device::Device;
use gfx_hal::format::ChannelType;
use gfx_hal::image::{Extent, Filter, Layout, Level, Offset, SubresourceLayers};
use gfx_hal::image::Layout::{TransferDstOptimal, TransferSrcOptimal};
use gfx_hal::image::Usage;
use gfx_hal::pass::Subpass;
use gfx_hal::pool::CommandPool;
use gfx_hal::pso::{Comparison, DepthTest, DescriptorPool, DescriptorPoolCreateFlags, DescriptorRangeDesc, DescriptorSetLayoutBinding, DescriptorType, FrontFace, ShaderStageFlags, VertexInputRate};
use gfx_hal::queue::{CommandQueue, Submission};
use gfx_hal::window::{Extent2D, Surface, SwapImageIndex};
use winit::event::WindowEvent::CursorMoved;

use crate::rendering::{CameraData, ForwardPipeline, GPUMesh, InstanceData, MeshID, Pipeline, RenderPass, ResolvePipeline, ResourceManager, ShaderCode, Uniform, Vertex};
use crate::rendering::framebuffer::Framebuffer;
use crate::rendering::renderer::Renderer;
use std::iter::once;

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

    pub pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pub pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    pub desc_pool: ManuallyDrop<B::DescriptorPool>,
    pub desc_set: Vec<B::DescriptorSet>,
    pub set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    pub framebuffer: Arc<Mutex<Framebuffer<B, B::Device>>>,

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
        let set_layout = ManuallyDrop::new(
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
                desc_set.push(desc_pool.allocate_set(&set_layout).unwrap());
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
                format: Some(format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: Layout::Undefined..Layout::Present,
            };
//            let depth_attachment = pass::Attachment {
//                format: Some(depth_stencil_format),
//                samples: 1,
//                ops: pass::AttachmentOps::new(
//                    pass::AttachmentLoadOp::Clear,
//                    pass::AttachmentStoreOp::Store,
//                ),
//                stencil_ops: pass::AttachmentOps::DONT_CARE,
//                layouts: Layout::Undefined..Layout::DepthStencilAttachmentOptimal,
//            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, Layout::ColorAttachmentOptimal)],
                depth_stencil: None, //Some(&(0, Layout::DepthStencilAttachmentOptimal)),
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            ManuallyDrop::new(
                unsafe { device.create_render_pass(&[attachment], &[subpass], &[]) }
                    .expect("Can't create render pass"),
            )
        };

        let pipeline_option = Self::create_pipeline(device, &render_pass, &set_layout);
        // TODO refine error handling
        if pipeline_option.is_none() {
            exit(1); //"Pipeline creation failed!");
        }
        let (pipeline, pipeline_layout) = pipeline_option.unwrap();

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

        let forward_pipeline = ForwardPipeline::new(device, render_pass.deref(), set_layout.deref());
        let resolve_pipeline = ResolvePipeline::new(device, render_pass.deref(), set_layout.deref());

        let mut framebuffer = Arc::new(Mutex::new(Framebuffer::new(device,
                                                                   adapter,
                                                                   &renderer.queue_group,
                                                                   &render_pass,
                                                                   renderer.dimensions,
                                                                   Usage::COLOR_ATTACHMENT,
                                                                   format,
                                                                   renderer.frames_in_flight).unwrap()));

        ForwardRenderPass {
            device: device.clone(),

            resource_manager: resource_manager.clone(),

            render_pass,

            forward_pipeline,
            resolve_pipeline,

            extent: renderer.dimensions,

            pipeline,
            pipeline_layout,
            desc_pool,
            desc_set,
            set_layout,
            framebuffer,

            cmd_buffers,

            camera_uniform,

            instances: HashMap::new(),
        }
    }

    fn create_pipeline(device: &Arc<B::Device>,
                       render_pass: &ManuallyDrop<B::RenderPass>,
                       set_layout: &ManuallyDrop<B::DescriptorSetLayout>)
                       -> Option<(ManuallyDrop<B::GraphicsPipeline>, ManuallyDrop<B::PipelineLayout>)> {
        let pipeline_layout = ManuallyDrop::new(
            unsafe {
                device.create_pipeline_layout(
                    iter::once(&**set_layout),
                    &[
                        (ShaderStageFlags::VERTEX, 0..64) // model matrix push constant
                    ],
                )
            }
                .expect("Can't create pipelines layout"),
        );
        let pipeline = {
            let mut shader_code = ShaderCode::new("src/rendering/shaders/forward_pass.vert.glsl");
            let mut compile_result = shader_code.compile(shaderc::ShaderKind::Vertex, ENTRY_NAME.parse().unwrap());
            if compile_result.is_none() {
                println!("Shader could not be compiled.");
                return None;
            }
            let vs_module = {
                let spirv = pso::read_spirv(Cursor::new(compile_result.unwrap().0))
                    .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };

            shader_code = ShaderCode::new("src/rendering/shaders/forward_pass.frag.glsl");
            compile_result = shader_code.compile(shaderc::ShaderKind::Fragment, ENTRY_NAME.parse().unwrap());
            if compile_result.is_none() {
                println!("Shader could not be compiled.");
                return None;
            }
            let fs_module = {
                let spirv = pso::read_spirv(Cursor::new(compile_result.unwrap().0))
                    .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };


            let pipeline = {
                let (vs_entry, fs_entry) = (
                    pso::EntryPoint {
                        entry: ENTRY_NAME,
                        module: &vs_module,
                        specialization: pso::Specialization::default(),
                    },
                    pso::EntryPoint {
                        entry: ENTRY_NAME,
                        module: &fs_module,
                        specialization: pso::Specialization::default(),
                    },
                );

                let shader_entries = pso::GraphicsShaderSet {
                    vertex: vs_entry,
                    hull: None,
                    domain: None,
                    geometry: None,
                    fragment: Some(fs_entry),
                };

                let subpass = Subpass {
                    index: 0,
                    main_pass: &**render_pass,
                };

                let mut pipeline_desc = pso::GraphicsPipelineDesc::new(
                    shader_entries,
                    pso::Primitive::TriangleList,
                    pso::Rasterizer::FILL,
                    &*pipeline_layout,
                    subpass,
                );

                pipeline_desc.rasterizer.cull_face = pso::Face::BACK;
                pipeline_desc.rasterizer.front_face = FrontFace::CounterClockwise;

                pipeline_desc.depth_stencil.depth = Some(DepthTest {
                    fun: Comparison::GreaterEqual,
                    write: true,
                });
                pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });
                pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
                    binding: 0,
                    stride: mem::size_of::<Vertex>() as u32,
                    rate: VertexInputRate::Vertex,
                });
//                pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
//                    binding: 1,
//                    stride: mem::size_of::<InstanceData>() as u32,
//                    rate: VertexInputRate::Instance(1),
//                });

                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: Format::Rgb32Sfloat,
                        offset: 0,
                    },
                });
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: Format::Rgb32Sfloat,
                        offset: 12,
                    },
                });
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 2,
                    binding: 0,
                    element: pso::Element {
                        format: Format::Rgb32Sfloat,
                        offset: 24,
                    },
                });
//                pipeline_desc.attributes.push(pso::AttributeDesc {
//                    location: 3,
//                    binding: 1,
//                    element: pso::Element {
//                        format: Format::Rgb32Sfloat,
//                        offset: 36,
//                    },
//                });

                unsafe { device.create_graphics_pipeline(&pipeline_desc, None) }
            };

            unsafe {
                device.destroy_shader_module(vs_module);
            }
            unsafe {
                device.destroy_shader_module(fs_module);
            }

            ManuallyDrop::new(pipeline.unwrap())
        };

        Some((pipeline, pipeline_layout))
    }

    pub fn recreate_pipeline(&mut self) {
        self.resolve_pipeline = ResolvePipeline::new(&self.device, self.render_pass.deref(), self.set_layout.deref());
        self.forward_pipeline = ForwardPipeline::new(&self.device, self.render_pass.deref(), self.set_layout.deref());

//        let pipeline_option = Self::create_pipeline(&self.device, &self.render_pass, &self.set_layout);
//
//        // TODO refine
//        if pipeline_option.is_none() {
//            return;
//        }
//
//        let (new_pipeline, new_pipeline_layout) = pipeline_option.unwrap();
//
//        unsafe {
//            self.device
//                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.pipeline)));
//            self.device
//                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
//                    &self.pipeline_layout,
//                )));
//        }
//
//        self.pipeline = new_pipeline;
//        self.pipeline_layout = new_pipeline_layout;
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

//    pub fn create_command_buffer(&mut self, frame_idx: usize) -> &B::CommandBuffer {
//
//    }
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
                    &self.set_layout,
                )));

            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.render_pass)));

            self.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.pipeline)));
            self.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.pipeline_layout,
                )));
        }
    }
}

impl<B: Backend> RenderPass<B> for ForwardRenderPass<B> {
    fn get_render_pass(&self) -> &ManuallyDrop<B::RenderPass> {
        &self.render_pass
    }

    fn fill_command_buffer(&self, framebuffer: &mut B::Framebuffer, command_buffer: &mut B::CommandBuffer, frame_idx: usize) {
        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: self.extent.width as _,
                h: self.extent.height as _,
            },
            depth: 0.0..1.0,
        };

        unsafe {
            command_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            command_buffer.set_viewports(0, &[viewport.clone()]);
            command_buffer.set_scissors(0, &[viewport.rect]);
            command_buffer.bind_graphics_pipeline(&self.pipeline);
//            cmd_buffer.bind_vertex_buffers(0, iter::once((&*self.vertex_buffer, 0)));
            command_buffer.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                iter::once(&self.desc_set[frame_idx]),
                &[],
            );

            command_buffer.begin_render_pass(
                &self.render_pass,
                framebuffer,
                viewport.rect,
                &[
                    command::ClearValue {
                        color: command::ClearColor {
                            float32: [0.3, 0.3, 0.3, 1.0],
                        },
                    },
                    command::ClearValue {
                        depth_stencil: ClearDepthStencil {
                            depth: 0f32,
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

                    command_buffer.push_graphics_constants(&self.pipeline_layout,
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

            command_buffer
        };
    }

    fn get_descriptor_set(&self, frame_index: usize) -> &B::DescriptorSet {
        &self.desc_set[frame_index]
    }

    fn blit_to_surface(&mut self, queue: &mut B::CommandQueue, surface_image: &B::ImageView, frame_idx: usize)
                       -> Arc<Mutex<Framebuffer<B, B::Device>>> {
        let mut fb_lock = self.framebuffer.lock().unwrap();
        let (fe, fi, framebuffer, pool, command_buffers, semaphore) = fb_lock.get_frame_data(frame_idx);

        unsafe {
            self.device
                .wait_for_fence(fe, !0)
                .expect("Failed to wait for fence");
            self.device
                .reset_fence(fe)
                .expect("Failed to reset fence");
            pool.reset(false);
        }


        unsafe {
            // blitting
            let mut cmd_buffer = match command_buffers.pop() {
                Some(cmd_buffer) => cmd_buffer,
                None => pool.allocate_one(command::Level::Primary),
            };

//            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            self.fill_command_buffer(framebuffer, &mut cmd_buffer, frame_idx);


//            cmd_buffer.blit_image(&*fi.image,
//                                  TransferSrcOptimal,
//                                  surface_image,
//                                  TransferDstOptimal,
//                                  Filter::Nearest,
//                                  ImageBlit {
//                                      src_subresource: SubresourceLayers {
//                                          aspects: format::Aspects::COLOR,
//                                          level: 0,
//                                          layers: 0..1,
//                                      },
//                                      src_bounds: Range {
//                                          start: Offset { x: 0, y: 0, z: 0 },
//                                          end: Offset { x: fi.extent.width as i32, y: fi.extent.height as i32, z: 0 },
//                                      },
//                                      dst_subresource: SubresourceLayers {
//                                          aspects: format::Aspects::COLOR,
//                                          level: 0,
//                                          layers: 0..1,
//                                      },
//                                      dst_bounds: Range {
//                                          start: Offset { x: 0, y: 0, z: 0 },
//                                          end: Offset { x: fi.extent.width as i32, y: fi.extent.height as i32, z: 0 },
//                                      },
//                                  });

//            cmd_buffer.finish();

            command_buffers.push(cmd_buffer);
        }

//        let submission = Submission {
//            command_buffers: command_buffers.iter(),
//            wait_semaphores: None,
//            signal_semaphores: iter::once(semaphore),
//        };
//        queue.submit(
//            submission,
//            Some(fe),
//        );

        self.framebuffer.clone()
    }

    fn render(&mut self, queue: &mut B::CommandQueue, frame_idx: usize) -> Arc<Mutex<Framebuffer<B, B::Device>>> {
        let mut fb_lock = self.framebuffer.lock().unwrap();

        let (fence,
            image,
            framebuffer,
            pool,
            command_buffers,
            semaphore) = fb_lock.get_frame_data(frame_idx);

        // Wait for the fence of the previous submission of this frame and reset it; ensures we are
        // submitting only up to maximum number of frames_in_flight if we are submitting faster than
        // the gpu can keep up with. This would also guarantee that any resources which need to be
        // updated with a CPU->GPU data copy are not in use by the GPU, so we can perform those updates.
        // In this case there are none to be done, however.
        unsafe {
//            self.device
//                .wait_for_fence(fence, !0)
//                .expect("Failed to wait for fence");
            self.device
                .reset_fence(fence)
                .expect("Failed to reset fence");
            pool.reset(false);
        }

        unsafe {
            let mut cmd_buffer = match command_buffers.pop() {
                Some(cmd_buffer) => cmd_buffer,
                None => pool.allocate_one(command::Level::Primary),
            };

            self.fill_command_buffer(framebuffer, &mut cmd_buffer, frame_idx);

            command_buffers.push(cmd_buffer);

            let submission = Submission {
                command_buffers: command_buffers.iter(),
                wait_semaphores: None,
                signal_semaphores: iter::once(&semaphore),
            };
            queue.submit(
                submission,
                Some(fence),
            );
        }

        self.framebuffer.clone()
    }
}