use std::{iter, mem, ptr};
use std::collections::HashMap;
use std::io::Cursor;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::{Arc, Weak};

use cgmath::{Matrix4, SquareMatrix};
use gfx_hal::{Backend, command, format::Format, IndexType, pass, pass::Attachment, pso};
use gfx_hal::buffer::IndexBufferView;
use gfx_hal::command::CommandBuffer;
use gfx_hal::device::Device;
use gfx_hal::format::ChannelType;
use gfx_hal::image::{Extent, Layout};
use gfx_hal::pass::Subpass;
use gfx_hal::pool::CommandPool;
use gfx_hal::pso::{Comparison, DepthTest, DescriptorPool, DescriptorPoolCreateFlags, DescriptorRangeDesc, DescriptorSetLayoutBinding, DescriptorType, FrontFace, ShaderStageFlags, VertexInputRate};
use gfx_hal::window::{Surface, SwapImageIndex};

use crate::rendering::{CameraData, GPUMesh, MeshID, RenderPass, ResourceManager, Uniform, Vertex, InstanceData};
use crate::rendering::renderer::Renderer;

/* Constants */
const ENTRY_NAME: &str = "main";

// Uniform
const CAMERA_UNIFORM_BINDING: u32 = 0;
const MODEL_MATRIX_UNIFORM_BINDING: u32 = 1;

pub struct ForwardRenderPass<B: Backend> {
    device: Arc<B::Device>,

    pub render_pass: ManuallyDrop<B::RenderPass>,
    pub pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pub pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    pub desc_pool: ManuallyDrop<B::DescriptorPool>,
    pub desc_set: Vec<B::DescriptorSet>,
    pub set_layout: ManuallyDrop<B::DescriptorSetLayout>,

    // uniforms
    camera_uniform: Uniform<B>,

    instances: HashMap<MeshID, Vec<Matrix4<f32>>>,
}

impl<B: Backend> ForwardRenderPass<B> {
    pub fn new(renderer: &Renderer<B>) -> Self {
        let device = &renderer.device;
        let adapter = &renderer.adapter;

        // Setup renderpass and pipeline
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
                            count: 1,
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
        let formats = renderer.surface.supported_formats(&adapter.physical_device);
        println!("formats: {:?}", formats);
        let format = formats.map_or(Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
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

        let (pipeline, pipeline_layout) = Self::create_pipeline(device, &render_pass, &set_layout);

        // uniforms
        let camera_uniform = Uniform::new(&renderer,
                                          &[CameraData::default()],
                                          CAMERA_UNIFORM_BINDING,
                                          &desc_set);

        ForwardRenderPass {
            device: device.clone(),
            render_pass,
            pipeline,
            pipeline_layout,
            desc_pool,
            desc_set,
            set_layout,

            camera_uniform,

            instances: HashMap::new(),
        }
    }

    fn create_pipeline(device: &Arc<B::Device>,
                       render_pass: &ManuallyDrop<B::RenderPass>,
                       set_layout: &ManuallyDrop<B::DescriptorSetLayout>)
                       -> (ManuallyDrop<B::GraphicsPipeline>, ManuallyDrop<B::PipelineLayout>) {
        let pipeline_layout = ManuallyDrop::new(
            unsafe {
                device.create_pipeline_layout(
                    iter::once(&**set_layout),
                    &[],
                )
            }
                .expect("Can't create pipeline layout"),
        );
        let pipeline = {
            let vs_module = {
                let shader = include_glsl_vs!("src/rendering/shaders/test.vert.glsl");
                let spirv = pso::read_spirv(Cursor::new(shader))
                    .unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };
            let fs_module = {
                let shader = include_glsl_fs!("src/rendering/shaders/test.frag.glsl");
                let spirv =
                    pso::read_spirv(Cursor::new(shader))
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

        (pipeline, pipeline_layout)
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

    pub fn create_commandbuffer(&self,
                                renderer: &mut Renderer<B>,
                                resource_manager: &ResourceManager<B>,
                                framebuffer: &B::Framebuffer) -> B::CommandBuffer {
        let frame_idx = renderer.current_swap_chain_image;
        let cmd_buffer = unsafe {
            let mut cmd_buffer = renderer.cmd_pools[frame_idx].allocate_one(command::Level::Primary);

            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            cmd_buffer.set_viewports(0, &[renderer.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[renderer.viewport.rect]);
            cmd_buffer.bind_graphics_pipeline(&self.pipeline);
//            cmd_buffer.bind_vertex_buffers(0, iter::once((&*self.vertex_buffer, 0)));
            cmd_buffer.bind_graphics_descriptor_sets(
                &self.pipeline_layout,
                0,
                iter::once(&self.desc_set[frame_idx]),
                &[],
            );

            cmd_buffer.begin_render_pass(
                &self.render_pass,
                &framebuffer,
                renderer.viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor {
                        float32: [0.3, 0.3, 0.3, 1.0],
                    },
                }],
                command::SubpassContents::Inline,
            );

            let (id, transform) = self.instances.iter().next().unwrap();
            let mesh = resource_manager.get_mesh(id);
            let vert_buf = &**mesh.vertex_buffer;
            let ind_buf = &**mesh.index_buffer;
            cmd_buffer.bind_vertex_buffers(0, iter::once((vert_buf, 0)));

            let index_buffer_view = IndexBufferView {
                buffer: ind_buf,
                offset: 0,
                index_type: mesh.index_type,
            };
            cmd_buffer.bind_index_buffer(index_buffer_view);
            cmd_buffer.draw_indexed(0..mesh.num_indices, 0, 0..1);

//            cmd_buffer.execute_commands(iter::once(&self.mesh_cmd_buffer[frame_idx]));

            cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            cmd_buffer
        };

        cmd_buffer
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

    fn generate_command_buffer(&self, renderer: &mut Renderer<B>,
                               resource_manager: &ResourceManager<B>,
                               framebuffer: &B::Framebuffer) -> B::CommandBuffer {
        self.create_commandbuffer(renderer, resource_manager, framebuffer)
    }

    fn get_descriptor_set(&self, frame_index: usize) -> &<B as Backend>::DescriptorSet {
        &self.desc_set[frame_index]
    }
}