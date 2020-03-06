#![cfg_attr(
not(any(
feature = "vulkan",
feature = "dx11",
feature = "dx12",
feature = "metal",
feature = "gl",
feature = "wgl"
)),
allow(dead_code, unused_extern_crates, unused_imports)
)]

#[cfg(feature = "dx11")]
pub extern crate gfx_backend_dx11 as Backend;
#[cfg(feature = "dx12")]
pub extern crate gfx_backend_dx12 as Backend;
#[cfg(any(feature = "gl", feature = "wgl"))]
pub extern crate gfx_backend_gl as Backend;
#[cfg(feature = "metal")]
pub extern crate gfx_backend_metal as Backend;
#[cfg(feature = "vulkan")]
pub extern crate gfx_backend_vulkan as Backend;

use std::{
    borrow::Borrow,
    io::Cursor,
    iter,
    mem::{self, ManuallyDrop},
    ptr,
};
use std::hash::Hasher;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gfx_hal::{
    buffer,
    command,
    format as f,
    format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle},
    image as i,
    IndexType,
    memory as m,
    pass,
    pass::Subpass,
    pool,
    prelude::*,
    pso,
    pso::{
        PipelineStage,
        ShaderStageFlags,
        VertexInputRate,
    },
    queue::{QueueGroup, Submission},
    window,
};
use gfx_hal::adapter::Adapter;
use gfx_hal::buffer::IndexBufferView;
use gfx_hal::command::{CommandBuffer, CommandBufferInheritanceInfo, SubpassContents};
use gfx_hal::pso::{Descriptor, DescriptorRangeDesc, DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, DepthTest, Comparison, FrontFace};
use gfx_hal::window::Surface;
use glium::buffer::Buffer;
use glium::RawUniformValue::Vec2;
use winit::event::{Event, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::core::{Exit, Filter, MainWindow, Message, System};
use crate::NSE;
use crate::rendering::{Camera, CameraDataUbo, Cube, Mesh, Octree, Plane, Transformation};
use crate::rendering::utility::{GPUBuffer, GPUMesh, ResourceManager, Vertex};

use self::Backend::Instance;
use gfx_hal::pso::Comparison::LessEqual;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn wasm_main() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
    main();
}

#[cfg_attr(rustfmt, rustfmt_skip)]
const DIMS: window::Extent2D = window::Extent2D { width: 1024, height: 768 };

const ENTRY_NAME: &str = "main";


const COLOR_RANGE: i::SubresourceRange = i::SubresourceRange {
    aspects: f::Aspects::COLOR,
    levels: 0..1,
    layers: 0..1,
};

// Constants
const WINDOW_TITLE: &'static str = "NSE";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

pub struct RenderSystem {
    window: Window,

    pub(in crate::rendering) renderer: Renderer<Backend::Backend>,
    pub(in crate::rendering) resource_manager: ResourceManager<Backend::Backend>,

    messages: Vec<Message>,
}

impl RenderSystem {
    #[cfg(any(
    feature = "vulkan",
    feature = "dx11",
    feature = "dx12",
    feature = "metal",
    feature = "gl",
    feature = "wgl"
    ))]
    pub fn new(nse: &NSE) -> Arc<Mutex<Self>> {
        let window = RenderSystem::init_window(&nse.event_loop);

        let instance = Instance::create("render-engine", 1)
            .expect("Failed to create an instance!");

        let surface = unsafe {
            instance.create_surface(&window)
                .expect("Failed to create a surface!")
        };
        let mut adapters = instance.enumerate_adapters();

        for adapter in &adapters {
            println!("{:?}", adapter.info);
        }

        let adapter = Arc::new(adapters.remove(0));

        let mut renderer = Renderer::new(Some(instance), surface, adapter.clone());
        let mut resource_manager = ResourceManager::new(renderer.device.clone(), adapter.clone());

        let messages = vec![Message::new(MainWindow { window_id: window.id() })];

        let rs = RenderSystem {
            window,

            renderer,
            resource_manager,

            messages,
        };

        Arc::new(Mutex::new(rs))
    }


    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_WIDTH, WINDOW_HEIGHT))
            .build(event_loop)
            .expect("Failed to create window.")
    }
}

pub struct Renderer<B: gfx_hal::Backend> {
    pub instance: Option<B::Instance>,
    pub device: Arc<B::Device>,
    pub queue_group: QueueGroup<B>,
    pub desc_pool: ManuallyDrop<B::DescriptorPool>,
    pub surface: ManuallyDrop<B::Surface>,
    pub adapter: Arc<gfx_hal::adapter::Adapter<B>>,
    pub format: gfx_hal::format::Format,
    pub dimensions: window::Extent2D,
    pub viewport: pso::Viewport,
    pub render_pass: ManuallyDrop<B::RenderPass>,
    pub pipeline: ManuallyDrop<B::GraphicsPipeline>,
    pub pipeline_layout: ManuallyDrop<B::PipelineLayout>,
    pub desc_set: Vec<B::DescriptorSet>,
    pub set_layout: ManuallyDrop<B::DescriptorSetLayout>,
    pub submission_complete_semaphores: Vec<B::Semaphore>,
    pub submission_complete_fences: Vec<B::Fence>,
    pub cmd_pools: Vec<B::CommandPool>,
    pub cmd_buffers: Vec<B::CommandBuffer>,
    pub mesh_cmd_buffer: Vec<B::CommandBuffer>,
    pub camera_buffer: Vec<GPUBuffer<B>>,
    pub frames_in_flight: usize,
    pub frame: u64,
}

impl<B> Renderer<B>
    where
        B: gfx_hal::Backend,
{
    fn new(
        instance: Option<B::Instance>,
        mut surface: B::Surface,
        adapter: Arc<gfx_hal::adapter::Adapter<B>>,
    ) -> Renderer<B> {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        // Build a new device and associated command queues
        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family) && family.queue_type().supports_graphics()
            })
            .unwrap();
        let mut gpu = unsafe {
            adapter
                .physical_device
                .open(&[(family, &[1.0])], gfx_hal::Features::empty())
                .unwrap()
        };
        let mut queue_group = gpu.queue_groups.pop().unwrap();
        let device = Arc::new(gpu.device);

        // Define maximum number of frames we want to be able to be "in flight" (being computed
        // simultaneously) at once
        let frames_in_flight = 3;

        let mut command_pool = unsafe {
            device.create_command_pool(queue_group.family, pool::CommandPoolCreateFlags::empty())
        }
            .expect("Can't create command pool");

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
                        }
                    ],
                    &[],
                )
            }
                .expect("Can't create descriptor set layout"),
        );

        // Descriptors
        let mut desc_pool = ManuallyDrop::new(
            unsafe {
                device.create_descriptor_pool(
                    frames_in_flight, // sets
                    &[
                        DescriptorRangeDesc {
                            ty: DescriptorType::UniformBuffer,
                            count: 1,
                        }
                    ],
                    pso::DescriptorPoolCreateFlags::empty(),
                )
            }
                .expect("Can't create descriptor pool"),
        );

        let mut desc_set: Vec<B::DescriptorSet> = Vec::new();
        unsafe {
            for _ in 0 .. frames_in_flight {
                desc_set.push(desc_pool.allocate_set(&set_layout).unwrap());
            }
        };

        // Buffer allocations
        println!("Memory types: {:?}", memory_types);
        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let mut camera_buffer: Vec<GPUBuffer<B>> = Vec::new();
        unsafe {
            for _ in 0 .. frames_in_flight {
                camera_buffer.push(GPUBuffer::new(device.clone(), &[CameraDataUbo::default()], buffer::Usage::UNIFORM, memory_types.clone()))
            }
        };

        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);
        println!("formats: {:?}", formats);
        let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let swap_config = window::SwapchainConfig::from_caps(&caps, format, DIMS);
        println!("{:?}", swap_config);
        let extent = swap_config.extent;
        unsafe {
            surface
                .configure_swapchain(&device, swap_config)
                .expect("Can't configure swapchain");
        };

        let render_pass = {
            let attachment = pass::Attachment {
                format: Some(format),
                samples: 1,
                ops: pass::AttachmentOps::new(
                    pass::AttachmentLoadOp::Clear,
                    pass::AttachmentStoreOp::Store,
                ),
                stencil_ops: pass::AttachmentOps::DONT_CARE,
                layouts: i::Layout::Undefined..i::Layout::Present,
            };

            let subpass = pass::SubpassDesc {
                colors: &[(0, i::Layout::ColorAttachmentOptimal)],
                depth_stencil: None,
                inputs: &[],
                resolves: &[],
                preserves: &[],
            };

            ManuallyDrop::new(
                unsafe { device.create_render_pass(&[attachment], &[subpass], &[]) }
                    .expect("Can't create render pass"),
            )
        };

        // The number of the rest of the resources is based on the frames in flight.
        let mut submission_complete_semaphores = Vec::with_capacity(frames_in_flight);
        let mut submission_complete_fences = Vec::with_capacity(frames_in_flight);
        // Note: We don't really need a different command pool per frame in such a simple demo like this,
        // but in a more 'real' application, it's generally seen as optimal to have one command pool per
        // thread per frame. There is a flag that lets a command pool reset individual command buffers
        // which are created from it, but by default the whole pool (and therefore all buffers in it)
        // must be reset at once. Furthermore, it is often the case that resetting a whole pool is actually
        // faster and more efficient for the hardware than resetting individual command buffers, so it's
        // usually best to just make a command pool for each set of buffers which need to be reset at the
        // same time (each frame).
        let mut cmd_pools = Vec::with_capacity(frames_in_flight);
        let mut cmd_buffers = Vec::with_capacity(frames_in_flight);

        cmd_pools.push(command_pool);
        for _ in 1..frames_in_flight {
            unsafe {
                cmd_pools.push(
                    device
                        .create_command_pool(
                            queue_group.family,
                            pool::CommandPoolCreateFlags::empty(),
                        )
                        .expect("Can't create command pool"),
                );
            }
        }

        for i in 0..frames_in_flight {
            submission_complete_semaphores.push(
                device
                    .create_semaphore()
                    .expect("Could not create semaphore"),
            );
            submission_complete_fences
                .push(device.create_fence(true).expect("Could not create fence"));
            cmd_buffers.push(unsafe { cmd_pools[i].allocate_one(command::Level::Primary) });
        }

        let mut mesh_cmd_buffer = Vec::with_capacity(frames_in_flight);
        for i in 0..frames_in_flight {
            mesh_cmd_buffer.push(unsafe {
                cmd_pools[i].allocate_one(command::Level::Secondary)
            });
        }

        let pipeline_layout = ManuallyDrop::new(
            unsafe {
                device.create_pipeline_layout(
                    iter::once(&*set_layout),
                    &[(pso::ShaderStageFlags::VERTEX, 0..8)],
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
                    main_pass: &*render_pass,
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

                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 0,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rgb32Sfloat,
                        offset: 0,
                    },
                });
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 1,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rgb32Sfloat,
                        offset: 12,
                    },
                });
                pipeline_desc.attributes.push(pso::AttributeDesc {
                    location: 2,
                    binding: 0,
                    element: pso::Element {
                        format: f::Format::Rgb32Sfloat,
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

        // Rendering setup
        let viewport = pso::Viewport {
            rect: pso::Rect {
                x: 0,
                y: 0,
                w: extent.width as _,
                h: extent.height as _,
            },
            depth: 0.0..1.0,
        };

        Renderer {
            instance,
            device,
            queue_group,
            desc_pool,
            surface: ManuallyDrop::new(surface),
            adapter,
            format,
            dimensions: DIMS,
            viewport,
            render_pass,
            pipeline,
            pipeline_layout,
            desc_set,
            set_layout,
            submission_complete_semaphores,
            submission_complete_fences,
            cmd_pools,
            cmd_buffers,
            mesh_cmd_buffer,
            camera_buffer,
            frames_in_flight,
            frame: 0,
        }
    }

    fn recreate_swapchain(&mut self) {
        let caps = self.surface.capabilities(&self.adapter.physical_device);
        let swap_config = window::SwapchainConfig::from_caps(&caps, self.format, self.dimensions);
        println!("{:?}", swap_config);
        let extent = swap_config.extent.to_extent();

        unsafe {
            self.surface
                .configure_swapchain(&self.device, swap_config)
                .expect("Can't create swapchain");
        }

        self.viewport.rect.w = extent.width as _;
        self.viewport.rect.h = extent.height as _;
    }

    fn render(&mut self, mesh: &GPUMesh<B>) {
        let surface_image = unsafe {
            match self.surface.acquire_image(!0) {
                Ok((image, _)) => image,
                Err(_) => {
                    self.recreate_swapchain();
                    return;
                }
            }
        };

        let framebuffer = unsafe {
            self.device
                .create_framebuffer(
                    &self.render_pass,
                    iter::once(surface_image.borrow()),
                    i::Extent {
                        width: self.dimensions.width,
                        height: self.dimensions.height,
                        depth: 1,
                    },
                )
                .unwrap()
        };

        // Compute index into our resource ring buffers based on the frame number
        // and number of frames in flight. Pay close attention to where this index is needed
        // versus when the swapchain image index we got from acquire_image is needed.
        let frame_idx = self.frame as usize % self.frames_in_flight;

        // Wait for the fence of the previous submission of this frame and reset it; ensures we are
        // submitting only up to maximum number of frames_in_flight if we are submitting faster than
        // the gpu can keep up with. This would also guarantee that any resources which need to be
        // updated with a CPU->GPU data copy are not in use by the GPU, so we can perform those updates.
        // In this case there are none to be done, however.
        unsafe {
            let fence = &self.submission_complete_fences[frame_idx];
            self.device
                .wait_for_fence(fence, !0)
                .expect("Failed to wait for fence");
            self.device
                .reset_fence(fence)
                .expect("Failed to reset fence");
            self.cmd_pools[frame_idx].reset(false);
        }

        unsafe {
            self.device.write_descriptor_sets(iter::once(DescriptorSetWrite {
                set: &self.desc_set[frame_idx],
                binding: 0,
                array_offset: 0,
                descriptors: iter::once(
                    Descriptor::Buffer(self.camera_buffer[frame_idx].get_buffer(), None..None)
                ),
            }));
        }

        // Rendering
        let cmd_buffer = &mut self.cmd_buffers[frame_idx];
        unsafe {
            cmd_buffer.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);

            cmd_buffer.set_viewports(0, &[self.viewport.clone()]);
            cmd_buffer.set_scissors(0, &[self.viewport.rect]);
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
                self.viewport.rect,
                &[command::ClearValue {
                    color: command::ClearColor {
                        float32: [0.1, 0.1, 0.1, 1.0],
                    },
                }],
                command::SubpassContents::Inline,
            );

            let vert_buf = &**mesh.vertex_buffer;
            let ind_buf = &**mesh.index_buffer;
            cmd_buffer.bind_vertex_buffers(0, iter::once((vert_buf, 0)));

            let index_buffer_view = IndexBufferView {
                buffer: ind_buf,
                offset: 0,
                index_type: IndexType::U32,
            };
            cmd_buffer.bind_index_buffer(index_buffer_view);
            cmd_buffer.draw_indexed(0..mesh.num_indices, 0, 0..1);

//            cmd_buffer.execute_commands(iter::once(&self.mesh_cmd_buffer[frame_idx]));

//            cmd_buffer.draw(0 .. 6, 0 .. 1);
            cmd_buffer.end_render_pass();
            cmd_buffer.finish();

            let submission = Submission {
                command_buffers: iter::once(&*cmd_buffer),
                wait_semaphores: None,
                signal_semaphores: iter::once(&self.submission_complete_semaphores[frame_idx]),
            };
            self.queue_group.queues[0].submit(
                submission,
                Some(&self.submission_complete_fences[frame_idx]),
            );

            // present frame
            let result = self.queue_group.queues[0].present_surface(
                &mut self.surface,
                surface_image,
                Some(&self.submission_complete_semaphores[frame_idx]),
            );

            self.device.destroy_framebuffer(framebuffer);

            if result.is_err() {
                self.recreate_swapchain();
            }
        }

        // Increment our frame
        self.frame += 1;
    }
}

impl<B> Drop for Renderer<B>
    where
        B: gfx_hal::Backend,
{
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

            for p in self.cmd_pools.drain(..) {
                self.device.destroy_command_pool(p);
            }
            for s in self.submission_complete_semaphores.drain(..) {
                self.device.destroy_semaphore(s);
            }
            for f in self.submission_complete_fences.drain(..) {
                self.device.destroy_fence(f);
            }
            self.device
                .destroy_render_pass(ManuallyDrop::into_inner(ptr::read(&self.render_pass)));
            self.surface.unconfigure_swapchain(&self.device);

            self.device
                .destroy_graphics_pipeline(ManuallyDrop::into_inner(ptr::read(&self.pipeline)));
            self.device
                .destroy_pipeline_layout(ManuallyDrop::into_inner(ptr::read(
                    &self.pipeline_layout,
                )));
            if let Some(instance) = &self.instance {
                let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
                instance.destroy_surface(surface);
            }
        }
        println!("DROPPED!");
    }
}

impl System for RenderSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![
            crate::filter!(Mesh, Transformation),
            crate::filter!(Camera, Transformation),
            crate::filter!(Octree, Mesh, Transformation),
        ]
    }

    fn get_messages(&mut self) -> Vec<Message> {
        let mut ret = vec![];
        if !self.messages.is_empty() {
            ret = self.messages.clone();
            self.messages = vec![];
        }

        ret
    }

    fn handle_input(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, window_id } => {
                if *window_id == self.window.id() {
                    match event {
                        | WindowEvent::CloseRequested => {
                            self.messages = vec![Message::new(Exit {})];
                        }
                        | _ => {}
                    }
                }
            }
            _ => ()
        }
    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, _: Duration) {
        let mutex = filter[1].lock().unwrap();
        if mutex.entities.is_empty() {
            println!("No camera provided.");
            self.window.request_redraw();
            return;
        }
        let camera_entity = mutex.entities[0].lock().unwrap();
        let cam_cam_comp = camera_entity.get_component::<Camera>().ok().unwrap();
        let cam_trans_comp = camera_entity.get_component::<Transformation>().ok().unwrap();
        let camera_ubo = CameraDataUbo::new(cam_cam_comp, cam_trans_comp);

        /*
        let mut instanced_meshes: Vec<(Mesh, Arc<CpuBufferPoolChunk<InstanceData, Arc<StdMemoryPool>>>)> = vec![];

        for mesh_entity in &filter[0].lock().unwrap().entities {
            let mutex = mesh_entity.lock().unwrap();
            let mesh = mutex.get_component::<Mesh>().unwrap();
            let trans = mutex.get_component::<Transformation>().unwrap();

            match self.instance_info.get_mut(&mesh) {
                Some(vec) => {
                    vec.push(InstanceData { model_matrix: trans.get_model_matrix().into() });
                }
                None => {
                    let data = InstanceData { model_matrix: trans.get_model_matrix().into() };
                    self.instance_info.insert(mesh.clone(), vec![data]);
                }
            }
        }

        for octree in &filter[2].lock().unwrap().entities {
            let mutex = octree.lock().unwrap();
            let mesh = mutex.get_component::<Mesh>().unwrap();
            let _trans = mutex.get_component::<Transformation>().unwrap();
            let octree = mutex.get_component::<Octree>().unwrap();

            if octree.instance_data_buffer.is_some() {
                instanced_meshes.push((mesh.clone(), octree.instance_data_buffer.as_ref().unwrap().clone()))
            }
        }

        self.forward_pass_command_buffer(camera_ubo, instanced_meshes);

        self.draw_frame();

        self.surface.window().request_redraw();
        */

        let frame_idx = self.renderer.frame as usize % self.renderer.frames_in_flight;

        let mut mesh = None;
        unsafe {
            let mut m_cmd = &mut self.renderer.mesh_cmd_buffer[frame_idx];

            for mesh_entity in &filter[0].lock().unwrap().entities {
                let mutex = mesh_entity.lock().unwrap();
                mesh = Some(mutex.get_component::<Mesh>().unwrap().clone());
                let _trans = mutex.get_component::<Transformation>().unwrap();
//                let octree = mutex.get_component::<Octree>().unwrap();

//                m_cmd.begin(command::CommandBufferFlags::RENDER_PASS_CONTINUE, CommandBufferInheritanceInfo::default());

//                m_cmd.begin_primary(command::CommandBufferFlags::ONE_TIME_SUBMIT);
//
//                m_cmd.set_viewports(0, &[self.renderer.viewport.clone()]);
//                m_cmd.set_scissors(0, &[self.renderer.viewport.rect]);
//
//                m_cmd.bind_graphics_pipeline(&self.renderer.pipeline);
//
//                m_cmd.bind_graphics_descriptor_sets(
//                    &self.renderer.pipeline_layout,
//                    0,
//                    iter::once(&self.renderer.desc_set),
//                    &[],
//                );
//
//                m_cmd.bind_vertex_buffers(0, iter::once((&**mesh.vertex_buffer, 0)));
//
//                let index_buffer_view = IndexBufferView {
//                    buffer: &**mesh.index_buffer,
//                    offset: 0,
//                    index_type: IndexType::U16,
//                };
//                m_cmd.bind_index_buffer(index_buffer_view);
//                m_cmd.draw_indexed(0..mesh.num_indices, 1, 0..1);

//            if octree.instance_data_buffer.is_some() {
//                instanced_meshes.push((mesh.clone(), octree.instance_data_buffer.as_ref().unwrap().clone()))
//            }
            }

            m_cmd
        };

        self.renderer.camera_buffer[frame_idx].update_data(0, &[camera_ubo]);

        if mesh.is_some() {
            self.renderer.render(self.resource_manager.meshes.get(&mesh.unwrap().id).unwrap());
        }

        self.window.request_redraw();
    }
}
