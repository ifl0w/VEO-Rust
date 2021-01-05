
use std::io::Cursor;
use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::{iter, mem, ptr};



use gfx_hal::device::Device;



use gfx_hal::pass::Subpass;

use gfx_hal::pso::{
    Comparison, DepthStencilDesc, DepthTest, FrontFace, PolygonMode,
    ShaderStageFlags, VertexInputRate,
};
use gfx_hal::queue::{CommandQueue};

use gfx_hal::{
    command, format, format::Format, image, pass, pass::Attachment, pso, Backend, IndexType,
};

use crate::rendering::{Pipeline, ShaderCode, Vertex, ENTRY_NAME};

pub struct ForwardPipeline<B: Backend> {
    device: Arc<B::Device>,
    info: (
        ManuallyDrop<B::GraphicsPipeline>,
        ManuallyDrop<B::PipelineLayout>,
    ),
    info_instanced: (
        ManuallyDrop<B::GraphicsPipeline>,
        ManuallyDrop<B::PipelineLayout>,
    ),
}

impl<B: Backend> ForwardPipeline<B> {
    fn create_pipeline(
        device: &Arc<B::Device>,
        render_pass: &B::RenderPass,
        set_layout: &B::DescriptorSetLayout,
        enable_instancing: bool,
        polygon_mode: PolygonMode,
    ) -> Option<(
        ManuallyDrop<B::GraphicsPipeline>,
        ManuallyDrop<B::PipelineLayout>,
    )> {
        let pipeline_layout = ManuallyDrop::new(
            unsafe {
                device.create_pipeline_layout(
                    iter::once(set_layout),
                    &[
                        (ShaderStageFlags::VERTEX, 0..68), // model matrix and offset into instance data
                    ],
                )
            }
            .expect("Can't create pipelines layout"),
        );
        let pipeline = {
            #[cfg(debug_assertions)]
            let mut shader_code = ShaderCode::new("src/rendering/shaders/forward_pass.vert.glsl");
            #[cfg(not(debug_assertions))]
            let mut shader_code = ShaderCode::from_bytes(
                include_bytes!("../../shaders/forward_pass.vert.glsl").to_vec(),
            );

            let mut compile_result =
                shader_code.compile(shaderc::ShaderKind::Vertex, ENTRY_NAME.parse().unwrap());
            if compile_result.is_none() {
                println!("Shader could not be compiled.");
                return None;
            }
            let vs_module = {
                let spirv = pso::read_spirv(Cursor::new(compile_result.unwrap().0)).unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };

            #[cfg(debug_assertions)]
            let mut shader_code = ShaderCode::new("src/rendering/shaders/forward_pass.frag.glsl");
            #[cfg(not(debug_assertions))]
            let mut shader_code = ShaderCode::from_bytes(
                include_bytes!("../../shaders/forward_pass.frag.glsl").to_vec(),
            );

            compile_result =
                shader_code.compile(shaderc::ShaderKind::Fragment, ENTRY_NAME.parse().unwrap());
            if compile_result.is_none() {
                println!("Shader could not be compiled.");
                return None;
            }
            let fs_module = {
                let spirv = pso::read_spirv(Cursor::new(compile_result.unwrap().0)).unwrap();
                unsafe { device.create_shader_module(&spirv) }.unwrap()
            };

            let pipeline = {
                let (vs_entry, fs_entry) = (
                    pso::EntryPoint {
                        entry: ENTRY_NAME,
                        module: &vs_module,
                        specialization: if enable_instancing {
                            gfx_hal::spec_const_list![true]
                        } else {
                            pso::Specialization::default()
                        },
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
                    main_pass: render_pass,
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
                pipeline_desc.rasterizer.polygon_mode = polygon_mode;

                pipeline_desc.depth_stencil = DepthStencilDesc {
                    depth: Some(DepthTest {
                        fun: Comparison::LessEqual,
                        write: true,
                    }),
                    depth_bounds: false,
                    stencil: None,
                };

                pipeline_desc.blender.targets.push(pso::ColorBlendDesc {
                    mask: pso::ColorMask::ALL,
                    blend: Some(pso::BlendState::ALPHA),
                });
                pipeline_desc.vertex_buffers.push(pso::VertexBufferDesc {
                    binding: 0,
                    stride: mem::size_of::<Vertex>() as u32,
                    rate: VertexInputRate::Vertex,
                });

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
}

impl<B: Backend> Pipeline<B> for ForwardPipeline<B> {
    fn new(
        device: &Arc<B::Device>,
        render_pass: &B::RenderPass,
        set_layout: &B::DescriptorSetLayout,
        polygon_mode: PolygonMode,
    ) -> Self {
        ForwardPipeline {
            device: device.clone(),
            info: Self::create_pipeline(device, render_pass, set_layout, false, polygon_mode)
                .unwrap(),
            info_instanced: Self::create_pipeline(
                device,
                render_pass,
                set_layout,
                true,
                polygon_mode,
            )
            .unwrap(),
        }
    }

    fn get_pipeline(&self, instanced: bool) -> &B::GraphicsPipeline {
        if instanced {
            &self.info_instanced.0
        } else {
            &self.info.0
        }
    }

    fn get_layout(&self, instanced: bool) -> &B::PipelineLayout {
        if instanced {
            &self.info_instanced.1
        } else {
            &self.info.1
        }
    }
}

impl<B: Backend> Drop for ForwardPipeline<B> {
    fn drop(&mut self) {
        self.device.wait_idle().unwrap();
        unsafe {
            // TODO: When ManuallyDrop::take (soon to be renamed to ManuallyDrop::read) is stabilized we should use that instead.

            self.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.info.0)));
            self.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(&self.info.1)));
        }
    }
}
