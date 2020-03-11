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
#[cfg(
not(any(
feature = "vulkan",
feature = "dx12",
feature = "metal",
feature = "gl",
feature = "wgl"
)))]
pub extern crate gfx_backend_empty as Backend;
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
use std::borrow::BorrowMut;
use std::cell::Cell;
use std::hash::Hasher;
use std::iter::{once, Map};
use std::ops::{Deref, RangeInclusive};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cgmath::{Matrix4, SquareMatrix};
use gfx_hal::{buffer, command, format as f, format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle}, image as i, IndexType, Instance, memory as m, pass, pass::Subpass, pool, prelude::*, pso, pso::{
    PipelineStage,
    ShaderStageFlags,
    VertexInputRate,
}, queue::{QueueGroup, Submission}, window};
use gfx_hal::adapter::Adapter;
use gfx_hal::buffer::IndexBufferView;
use gfx_hal::command::{CommandBuffer, CommandBufferInheritanceInfo, SubpassContents};
use gfx_hal::pso::{Comparison, DepthTest, Descriptor, DescriptorRangeDesc, DescriptorSetLayoutBinding, DescriptorSetWrite, DescriptorType, FrontFace};
use gfx_hal::pso::Comparison::LessEqual;
use gfx_hal::window::Surface;
use mopa::Any;
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

use crate::core::{Exit, Filter, MainWindow, Message, System};
use crate::NSE;
use crate::rendering::{Camera, CameraData, Cube, ForwardRenderPass, Mesh, Octree, Plane, RenderPass, Transformation, SwapchainWrapper};
use crate::rendering::utility::{GPUBuffer, GPUMesh, ResourceManager, Uniform, Vertex};
use std::collections::{HashMap, BTreeMap};
use std::thread::sleep;
use std::rc::Rc;
use winit::event::VirtualKeyCode::Mute;

/* Constants */
// Window
const WINDOW_TITLE: &'static str = "NSE";
const WINDOW_DIMENSIONS: window::Extent2D = window::Extent2D { width: 1024, height: 768 };

pub struct RenderSystem {
    window: Window,

    pub(in crate::rendering) renderer: Renderer<Backend::Backend>,
    pub(in crate::rendering) resource_manager: Arc<Mutex<ResourceManager<Backend::Backend>>>,

    // Render passes
    forward_render_pass: Arc<Mutex<ForwardRenderPass<Backend::Backend>>>,

    messages: Vec<Message>,
}

impl RenderSystem {
    #[cfg(any(
    feature = "vulkan",
    feature = "gl",
    ))]
    pub fn new(nse: &NSE) -> Arc<Mutex<Self>> {
        let (window, instance, adapter) = RenderSystem::init(nse);

        let surface = unsafe {
            instance.create_surface(&window)
                .expect("Failed to create a surface!")
        };

        let mut renderer = Renderer::new(Some(instance), surface, adapter.clone());
        let mut resource_manager = ResourceManager::new(&renderer);
        let mut forward_render_pass =
            Arc::new(Mutex::new(ForwardRenderPass::new(&mut renderer, &resource_manager)));

        let messages = vec![Message::new(MainWindow { window_id: window.id() })];

        let rs = RenderSystem {
            window,

            renderer,
            resource_manager,

            forward_render_pass,

            messages,
        };

        Arc::new(Mutex::new(rs))
    }

    fn init(nse: &NSE) ->
    (Window,
     Backend::Instance,
     Arc<Adapter<Backend::Backend>>) {
        let window = RenderSystem::init_window(&nse.event_loop);

        let instance: Backend::Instance = Instance::create("render-engine", ((1) << 22 | (1) << 12))
            .expect("Failed to create an instance!");
        println!("{:?}", instance.extensions);

        let mut adapters = instance.enumerate_adapters();

        for adapter in &adapters {
            println!("{:?}", adapter.info);
        }

        let adapter = Arc::new(adapters.remove(0));

        (window, instance, adapter)
    }


    fn init_window(event_loop: &EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size(winit::dpi::LogicalSize::new(WINDOW_DIMENSIONS.width, WINDOW_DIMENSIONS.height))
            .build(event_loop)
            .expect("Failed to create window.")
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

    fn handle_input(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, window_id } => {
                if *window_id == self.window.id() {
                    match event {
                        | WindowEvent::CloseRequested => {
                            self.messages = vec![Message::new(Exit {})];
                        }
                        WindowEvent::KeyboardInput { input, .. } => {
                            match input {
                                | winit::event::KeyboardInput { virtual_keycode, state, .. } => {
                                    match (virtual_keycode, state) {
                                        (Some(VirtualKeyCode::F5), ElementState::Pressed) => {
                                            println!("Recreating pipelines...");
                                            self.forward_render_pass.lock().unwrap().recreate_pipeline();
                                        }
                                        _ => ()
                                    }
                                }
                            }
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
        let camera_data = CameraData::new(cam_cam_comp, cam_trans_comp);

        let frame_idx = self.renderer.current_swap_chain_image;

        self.forward_render_pass.lock().unwrap().reset_instances();
        for mesh_entity in &filter[0].lock().unwrap().entities {
            let mutex = mesh_entity.lock().unwrap();
            let mesh = mutex.get_component::<Mesh>().unwrap();
            let trans = mutex.get_component::<Transformation>().unwrap();

            self.forward_render_pass.lock().unwrap().add_instance(mesh.id, trans.model_matrix);
        }

        self.forward_render_pass.lock().unwrap().update_camera(camera_data, frame_idx);

        // Currently only a single pass can be rendered since fences and semaphores are in the
        // renderer instead of the render passes
        self.renderer.render(&self.forward_render_pass);

        self.window.request_redraw();
    }

    fn get_messages(&mut self) -> Vec<Message> {
        let mut ret = vec![];
        if !self.messages.is_empty() {
            ret = self.messages.clone();
            self.messages = vec![];
        }

        ret
    }
}

pub struct Renderer<B: gfx_hal::Backend> {
    pub instance: Option<B::Instance>,
    pub device: Arc<B::Device>,
    pub queue_group: QueueGroup<B>,
    pub surface: ManuallyDrop<B::Surface>,
    pub adapter: Arc<gfx_hal::adapter::Adapter<B>>,
    pub format: gfx_hal::format::Format,
    pub dimensions: window::Extent2D,
    pub viewport: pso::Viewport,
    pub submission_complete_semaphores: Vec<B::Semaphore>,
    pub submission_complete_fences: Vec<B::Fence>,
    pub cmd_pools: Vec<B::CommandPool>,
    pub frame_buffers: Vec<B::Framebuffer>,
    pub swapchain: SwapchainWrapper<B, B::Device>,
    pub frames_in_flight: usize,
    pub current_swap_chain_image: usize,
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
        // Build a new device and associated command queues
        let family = adapter
            .queue_families
            .iter()
            .find(|family| {
                surface.supports_queue_family(family)
                    && family.queue_type().supports_graphics()
                    && family.queue_type().supports_compute()
                    && family.queue_type().supports_transfer()
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
        let frames_in_flight: usize = 2;

        let mut caps = surface.capabilities(&adapter.physical_device);
//        caps.image_count = RangeInclusive::new(frames_in_flight as u32, *caps.image_count.end()); // TODO Delete line when framebuffer is implemented correctly

        let formats = surface.supported_formats(&adapter.physical_device);
        println!("formats: {:?}", formats);
        let format = formats.map_or(f::Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let mut swap_config = window::SwapchainConfig::from_caps(&caps, format, WINDOW_DIMENSIONS)
            .with_present_mode(gfx_hal::window::PresentMode::IMMEDIATE);
        println!("{:?}", swap_config);
        let extent = swap_config.extent;
//        unsafe {
//            surface
//                .configure_swapchain(&device, swap_config)
//                .expect("Can't configure swapchain");
//        };

        let swapchain = unsafe {
            SwapchainWrapper::new(&device, &adapter, &mut surface, WINDOW_DIMENSIONS)
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

        for _ in 0..frames_in_flight {
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
        }

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

        let mut frame_buffers = Vec::new();
//        frame_buffers.reserve(frames_in_flight);

        Renderer {
            instance,
            device,
            queue_group,
            surface: ManuallyDrop::new(surface),
            adapter,
            format,
            dimensions: WINDOW_DIMENSIONS,
            viewport,
            submission_complete_semaphores,
            submission_complete_fences,
            cmd_pools,
            frame_buffers,
            swapchain,
            frames_in_flight,
            current_swap_chain_image: 0,
        }
    }

//    fn recreate_swapchain(&mut self) {
//        let caps = self.surface.capabilities(&self.adapter.physical_device);
//        let swap_config = window::SwapchainConfig::from_caps(&caps, self.format, self.dimensions);
//        println!("{:?}", swap_config);
//        let extent = swap_config.extent.to_extent();
//
//        unsafe {
//            self.surface
//                .configure_swapchain(&self.device, swap_config)
//                .expect("Can't create swapchain");
//        }
//
//        self.viewport.rect.w = extent.width as _;
//        self.viewport.rect.h = extent.height as _;
//    }

    fn sync_and_reset(&mut self, frame_idx: usize) {
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
    }

    fn render(&mut self, render_pass: &Arc<Mutex<ForwardRenderPass<B>>>) {

        // Compute index into our resource ring buffers based on the frame number
        // and number of frames in flight. Pay close attention to where this index is needed
        // versus when the swapchain image index we got from acquire_image is needed.
        let frame_idx = self.current_swap_chain_image;

//        let surface_image = unsafe {
//            match self.surface.acquire_image(!0) {
//                Ok((image, _)) => image,
//                Err(_) => {
//                    self.recreate_swapchain();
//                    return;
//                }
//            }
//        };

        let (swap_idx, image, aquire_semaphore) = self.swapchain.acquire_image(frame_idx);


        // TODO move to render_pass and recreate on swapchain recreation
//        let framebuffer = unsafe {
//            self.device
//                .create_framebuffer(
//                    render_pass.get_render_pass(),
//                    iter::once(surface_image.borrow()),
//                    i::Extent {
//                        width: self.dimensions.width,
//                        height: self.dimensions.height,
//                        depth: 1,
//                    },
//                )
//                .unwrap()
//        };

//        self.sync_and_reset(frame_idx);

        // Rendering
//        let mut command_buffers = render_pass.generate_command_buffer(self, resource_manager, &framebuffer);

        let mut queue = &mut self.queue_group.queues[0];
        self.device.wait_idle();

        unsafe {
            let mut lock = render_pass.lock().unwrap();

            lock.sync(frame_idx);
            lock.render(frame_idx);
//            lock.submit(frame_idx, queue, None);

            let (fe,
                fi,
                framebuffer,
                pool,
                command_buffers,
                semaphore) = lock.get_framebuffer().get_frame_data(frame_idx);

            let submission = Submission {
                command_buffers: command_buffers.iter(),
                wait_semaphores: None,
                signal_semaphores: iter::once(&self.submission_complete_semaphores[frame_idx])
            };
            queue.submit(
                submission,
                Some(fe),
            );
        }
        self.device.wait_idle();
        unsafe {
            let mut lock = render_pass.lock().unwrap();
//            let present_semaphore = lock.get_framebuffer().get_frame_semaphore(frame_idx);

            // blitting
            lock.sync(frame_idx);
            lock.blit_to_surface(queue, image, frame_idx);
//            lock.submit(frame_idx, queue, Some(aquire_semaphore));

            let (fe,
                fi,
                framebuffer,
                pool,
                command_buffers,
                semaphore) = lock.get_framebuffer().get_frame_data(frame_idx);

            let w1 = (&aquire_semaphore, pso::PipelineStage::TOP_OF_PIPE);
            let w2 = (&&mut self.submission_complete_semaphores[frame_idx], pso::PipelineStage::TOP_OF_PIPE);

            let mut wait_sem = vec![w1, w2];

            let submission = Submission {
                command_buffers: command_buffers.iter(),
                wait_semaphores: wait_sem,
                signal_semaphores: vec![&semaphore],
            };
            queue.submit(
                submission,
                Some(fe),
            );
        }
        self.device.wait_idle();



//        let mut fb_lock = output.lock().unwrap();
//        let (fence,_,_,_,_,present_semaphore) = fb_lock.get_frame_data(frame_idx);

        // TODO Submit rendering and then generate a second pass to blit to swapchain

        unsafe {
            let mut lock = render_pass.lock().unwrap();
            let (fe,
                fi,
                framebuffer,
                pool,
                command_buffers,
                semaphore) = lock.get_framebuffer().get_frame_data(frame_idx);
//            let submission = Submission {
//                command_buffers: command_buffers.iter(),
//                wait_semaphores: None,
//                signal_semaphores: iter::once(present_semaphore),
//            };
//            self.queue_group.queues[0].submit(
//                submission,
//                Some(&self.submission_complete_fences[frame_idx]),
//            );

//            let fence = &self.submission_complete_fences[frame_idx];
//            self.device
//                .wait_for_fence(fence, !0)
//                .expect("Failed to wait for fence");
//            self.device
//                .reset_fence(fence)
//                .expect("Failed to reset fence");

            // present frame
//            let result = queue.present_surface(
//                &mut self.surface,
//                surface_image,
//                Some(present_semaphore),
//            );

//            self.device.wait_idle();

            self.swapchain.present(queue, swap_idx, Some(vec![semaphore]));

//            if self.frame_buffers.len() > self.current_swap_chain_image {
//
//                let tmp = self.frame_buffers.remove(self.current_swap_chain_image);
//                unsafe {
//                    self.device.destroy_framebuffer(tmp);
//                }
//            }
//            self.frame_buffers.insert(self.current_swap_chain_image, framebuffer);

//            self.device.destroy_framebuffer(framebuffer);

//            if result.is_err() {
//                self.recreate_swapchain();
//            }
        }

        // Increment our frame
        self.current_swap_chain_image = (self.current_swap_chain_image + 1) % self.frames_in_flight;
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

            for p in self.cmd_pools.drain(..) {
                self.device.destroy_command_pool(p);
            }
            for s in self.submission_complete_semaphores.drain(..) {
                self.device.destroy_semaphore(s);
            }
            for f in self.submission_complete_fences.drain(..) {
                self.device.destroy_fence(f);
            }

            self.surface.unconfigure_swapchain(&self.device);

            if let Some(instance) = &self.instance {
                let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
                instance.destroy_surface(surface);
            }
        }
        println!("DROPPED!");
    }
}

