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

#[cfg(not(any(
feature = "vulkan",
feature = "dx12",
feature = "metal",
feature = "gl",
feature = "wgl"
)))]
pub extern crate gfx_backend_empty as Backend;
// #[cfg(any(feature = "gl", feature = "wgl"))]
// pub extern crate gfx_backend_gl as Backend;
#[cfg(feature = "vulkan")]
pub extern crate gfx_backend_vulkan as Backend;

use std::{mem::ManuallyDrop, ptr};
use std::borrow::Borrow;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cgmath::{Matrix4, SquareMatrix};
use gfx_hal::{Instance, pool, prelude::*, pso, queue::QueueGroup, window};
use gfx_hal::adapter::Adapter;
use gfx_hal::device::Device;
use gfx_hal::window::{PresentationSurface, Surface};
use winit::event::{ElementState, Event, VirtualKeyCode, WindowEvent};
use winit::event_loop::EventLoop;
use winit::window::Window;

use crate::core::{Exit, Filter, MainWindow, Message, System};
use crate::NSE;
use crate::rendering::{
    AABB, Camera, CameraData, ForwardRenderPass, Frustum, Mesh, Octree, PipelineOptions,
    RenderPass, SwapchainWrapper, Transformation,
};
use crate::rendering::nse_gui::octree_gui::ProfilingData;
use crate::rendering::utility::ResourceManager;

/* Constants */
// Window
const WINDOW_TITLE: &'static str = "NSE";
const WINDOW_DIMENSIONS: window::Extent2D = window::Extent2D {
    width: 1600,
    height: 900,
};

pub struct RenderSystem {
    window: Window,

    pub(in crate::rendering) renderer: Renderer<Backend::Backend>,
    pub(in crate::rendering) resource_manager: Arc<Mutex<ResourceManager<Backend::Backend>>>,

    // Render passes
    forward_render_pass: Arc<Mutex<ForwardRenderPass<Backend::Backend>>>,

    messages: Vec<Message>,
}

impl RenderSystem {
    // #[cfg(any(feature = "vulkan", feature = "gl",))]
    pub fn new(nse: &NSE) -> Arc<Mutex<Self>> {
        if cfg!(not(any(feature = "vulkan", feature = "gl",))) {
            panic!("Compiled without platform support...")
        }

        let (window, instance, adapter) = RenderSystem::init(nse);

        let surface = unsafe {
            instance
                .create_surface(&window)
                .expect("Failed to create a surface!")
        };

        let mut renderer = Renderer::new(Some(instance), surface, adapter.clone());
        let resource_manager = ResourceManager::new(&renderer);
        let forward_render_pass = Arc::new(Mutex::new(ForwardRenderPass::new(
            &mut renderer,
            &resource_manager,
        )));

        let messages = vec![Message::new(MainWindow {
            window_id: window.id(),
        })];

        let rs = RenderSystem {
            window,

            renderer,
            resource_manager,

            forward_render_pass,

            messages,
        };

        Arc::new(Mutex::new(rs))
    }

    fn init(nse: &NSE) -> (Window, Backend::Instance, Arc<Adapter<Backend::Backend>>) {
        let window = RenderSystem::init_window(&nse.event_loop);

        let instance: Backend::Instance = Instance::create("render-engine", (1) << 22 | (1) << 12)
            .expect("Failed to create an instance!");
        // println!("{:?}", instance.extensions);

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
            .with_inner_size(winit::dpi::PhysicalSize::new(
                WINDOW_DIMENSIONS.width,
                WINDOW_DIMENSIONS.height,
            ))
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
            crate::filter!(AABB),
            crate::filter!(Frustum),
        ]
    }

    fn handle_input(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, window_id } => {
                if *window_id == self.window.id() {
                    match event {
                        WindowEvent::CloseRequested => {
                            self.messages = vec![Message::new(Exit {})];
                        }
                        WindowEvent::KeyboardInput { input, .. } => match input {
                            winit::event::KeyboardInput {
                                virtual_keycode,
                                state,
                                ..
                            } => match (virtual_keycode, state) {
                                (Some(VirtualKeyCode::F5), ElementState::Pressed) => {
                                    println!("Recreating pipelines...");
                                    self.forward_render_pass.lock().unwrap().recreate_pipeline();
                                }
                                _ => (),
                            },
                        },
                        WindowEvent::Resized(size) => {
                            println!("Resizing framebuffers to {:?}", size);

                            let extent = window::Extent2D {
                                width: size.width,
                                height: size.height,
                            };
                            self.renderer.dimensions = extent;

                            self.renderer.swapchain.recreate(self.renderer.surface.deref_mut(), self.renderer.dimensions);
                            self.forward_render_pass.lock().unwrap().recreate_framebuffers(&mut self.renderer);
                        }
                        _ => {}
                    }
                }
            }
            _ => (),
        }
    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, _: Duration) {
        let mutex = filter[1].lock().unwrap();
        if mutex.entities.is_empty() {
            println!("No camera provided.");
            return;
        }
        let camera_entity = mutex.entities[0].lock().unwrap();
        let cam_cam_comp = camera_entity.get_component::<Camera>().ok().unwrap();
        let cam_trans_comp = camera_entity
            .get_component::<Transformation>()
            .ok()
            .unwrap();
        let camera_data = CameraData::new(cam_cam_comp, cam_trans_comp);

        let frame_idx = self.renderer.current_swap_chain_image;

        {
            let mut fwp_lock = self.forward_render_pass.lock().unwrap();

            fwp_lock.reset(); // reset render state of last frame

            // add single meshes
            for mesh_entity in &filter[0].lock().unwrap().entities {
                let mutex = mesh_entity.lock().unwrap();
                let mesh = mutex.get_component::<Mesh>().unwrap();
                let trans = mutex.get_component::<Transformation>().unwrap();

                fwp_lock.add_mesh(mesh.id, trans.model_matrix, PipelineOptions::Default);
            }

            // add octree instances
            for octree_entity in &filter[2].lock().unwrap().entities {
                let mutex = octree_entity.lock().unwrap();
                let mesh = mutex.get_component::<Mesh>().unwrap();
                let octree_trans = mutex.get_component::<Transformation>().unwrap();
                let octree = mutex.get_component::<Octree>().unwrap();

                match octree.get_instance_buffer() {
                    Some(ib) => {
                        // TODO somehow rework instancing. This is not a good solution.
                        fwp_lock.add_instances(
                            mesh.id,
                            (0..octree.info.render_count, octree_trans.get_model_matrix()),
                        );
                        fwp_lock.use_instance_buffer(ib, frame_idx);
                    }
                    None => (),
                }
            }

            for aabb_entity in &filter[3].lock().unwrap().entities {
                let mutex = aabb_entity.lock().unwrap();
                let aabb = mutex.get_component::<AABB>().unwrap().clone();

                let debug_mesh = aabb.debug_mesh;

                //                aabb.update_debug_mesh(&self.resource_manager);
                //                mutex.add_component(aabb);

                match debug_mesh {
                    Some((id, _)) => {
                        fwp_lock.add_mesh(id, Matrix4::identity(), PipelineOptions::Wireframe);
                    }
                    None => {}
                }
            }

            for frustum_entities in &filter[4].lock().unwrap().entities {
                let mutex = frustum_entities.lock().unwrap();
                let frustum = mutex.get_component::<Frustum>().unwrap().clone();

                let debug_mesh = frustum.debug_mesh;

                //                aabb.update_debug_mesh(&self.resource_manager);
                //                mutex.add_component(aabb);

                match debug_mesh {
                    Some((id, _)) => {
                        fwp_lock.add_mesh(id, Matrix4::identity(), PipelineOptions::Wireframe);
                    }
                    None => {}
                }
            }

            fwp_lock.update_camera(camera_data, frame_idx);
        } // drop lock after block

        // Currently only a single pass can be rendered since fences and semaphores are in the
        // renderer instead of the render passes
        let execution_time = self.renderer.render(&self.forward_render_pass);

        // send execution time
        self.messages.push(Message::new(ProfilingData {
            render_time: Some(execution_time),
            ..Default::default()
        }));
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
                .open(
                    &[(family, &[1.0])],
                    gfx_hal::Features::NON_FILL_POLYGON_MODE
                        | gfx_hal::Features::VERTEX_STORES_AND_ATOMICS
                        | gfx_hal::Features::FRAGMENT_STORES_AND_ATOMICS
                        | gfx_hal::Features::SHADER_STORAGE_BUFFER_ARRAY_DYNAMIC_INDEXING
                        | gfx_hal::Features::SHADER_UNIFORM_BUFFER_ARRAY_DYNAMIC_INDEXING,
                )
                .unwrap()
        };
        let queue_group = gpu.queue_groups.pop().unwrap();
        let device = Arc::new(gpu.device);

        // Define maximum number of frames we want to be able to be "in flight" (being computed
        // simultaneously) at once
        let frames_in_flight: usize = 3;

        let swapchain =
            unsafe { SwapchainWrapper::new(&device, &adapter, &mut surface, WINDOW_DIMENSIONS) };

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

        for _ in 0..frames_in_flight {
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
                w: swapchain.extent.width as _,
                h: swapchain.extent.height as _,
            },
            depth: 0.0..1.0,
        };

        let frame_buffers = Vec::new();
        //        frame_buffers.reserve(frames_in_flight);

        Renderer {
            instance,
            device,
            queue_group,
            surface: ManuallyDrop::new(surface),
            adapter,
            format: swapchain.format,
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

    fn sync_and_reset(&mut self, frame_idx: usize) {
        // Wait for the fence of the previous submission of this frame and reset it; ensures we are
        // submitting only up to maximum number of frames_in_flight if we are submitting faster than
        // the gpu can keep up with. This would also guarantee that any resources which need to be
        // updated with a CPU->GPU data copy are not in use by the GPU, so we can perform those updates.
        // In this case there are none to be done, however.
        unsafe {
            let fence = &mut self.submission_complete_fences[frame_idx];
            self.device
                .wait_for_fence(fence, !0)
                .expect("Failed to wait for fence");
            self.device
                .reset_fence(fence)
                .expect("Failed to reset fence");
            self.cmd_pools[frame_idx].reset(false);
        }
    }

    fn render(&mut self, render_pass: &Arc<Mutex<ForwardRenderPass<B>>>) -> u64 {
        // Compute index into our resource ring buffers based on the frame number
        // and number of frames in flight. Pay close attention to where this index is needed
        // versus when the swapchain image index we got from acquire_image is needed.
        let frame_idx = self.current_swap_chain_image;

        let image = unsafe {
            let res = self.surface.acquire_image(!0);

            if res.is_err() {
                return 0;
            }

            let acquired = res.unwrap();

            if acquired.1.is_some() {
                println!("Not optimal surface: {:?}", acquired.1)
            }

            acquired.0
        };

        // Rendering
        let queue = &mut self.queue_group.queues[0];

        let mut render_pass_lock = render_pass.lock().unwrap();

        let present_semaphore = {
            render_pass_lock.sync(frame_idx);
            render_pass_lock.record(frame_idx);
            render_pass_lock.submit(frame_idx, queue, vec![]);

            // blitting
            // render_pass_lock.blit_to_surface(queue, &image, frame_idx, acquire_semaphore)
            render_pass_lock.blit_to_surface(queue, image.borrow(), frame_idx)
        };

        unsafe {
            let res = queue.present(&mut self.surface, image, Some(present_semaphore));

            if res.is_err() {
                println!("Could not present frame!");
            }
        }

        // Increment our frame
        self.current_swap_chain_image = (self.current_swap_chain_image + 1) % self.frames_in_flight;

        // get render time
        let execution_time = render_pass_lock.execution_time(frame_idx);

        return execution_time;
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

            if let Some(instance) = &self.instance {
                let surface = ManuallyDrop::into_inner(ptr::read(&self.surface));
                instance.destroy_surface(surface);
            }
        }
        println!("DROPPED!");
    }
}
