use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool};
use vulkano::buffer::cpu_pool::CpuBufferPoolChunk;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{FixedSizeDescriptorSet, FixedSizeDescriptorSetsPool, PersistentDescriptorSetBuf, UnsafeDescriptorSetLayout};
use vulkano::device::{Device, DeviceExtensions, Features, Queue};
use vulkano::format::{ClearValue, Format};
use vulkano::framebuffer::{
    Framebuffer,
    FramebufferAbstract,
    RenderPassAbstract,
    Subpass,
};
use vulkano::image::{AttachmentImage, ImageUsage, SwapchainImage};
use vulkano::instance::{ApplicationInfo, Instance, InstanceExtensions, layers_list, PhysicalDevice, Version};
use vulkano::instance::debug::{DebugCallback, MessageSeverity, MessageType};
use vulkano::memory::pool::StdMemoryPool;
use vulkano::pipeline::{
    GraphicsPipeline,
    GraphicsPipelineAbstract,
    viewport::Viewport,
};
use vulkano::pipeline::depth_stencil::DepthStencil;
use vulkano::pipeline::vertex::OneVertexOneInstanceDefinition;
use vulkano::swapchain::{acquire_next_image, AcquireError, Capabilities, ColorSpace, CompositeAlpha, PresentMode, SupportedPresentModes, Surface, Swapchain, FullscreenExclusive};
use vulkano::sync::{self, GpuFuture, SharingMode};
use vulkano_win::VkSurfaceBuild;
//use winit::{Event, EventsLoop, Window, WindowBuilder};
use winit::dpi::LogicalSize;

use crate::core::{Filter, System};
use crate::NSE;
use crate::rendering::{Camera, CameraDataUbo, InstanceData, Mesh, Octree, Transformation, Vertex};
use imgui::Context;
use imgui_winit_support::{WinitPlatform, HiDpiMode};
use std::ops::Deref;
use winit::window::{WindowBuilder, Window};
use winit::event::Event;
use winit::event_loop::EventLoop;
use vulkano::descriptor::PipelineLayoutAbstract;

// Constants
const WINDOW_TITLE: &'static str = "NSE";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

const VALIDATION_LAYERS: &[&str] = &[
    "VK_LAYER_LUNARG_standard_validation"
];

/// Required device extensions
fn device_extensions() -> DeviceExtensions {
    DeviceExtensions {
        khr_swapchain: true,
        ..vulkano::device::DeviceExtensions::none()
    }
}

#[cfg(all(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = true;
#[cfg(not(debug_assertions))]
const ENABLE_VALIDATION_LAYERS: bool = false;

struct QueueFamilyIndices {
    graphics_family: i32,
    present_family: i32,
}

impl QueueFamilyIndices {
    fn new() -> Self {
        Self { graphics_family: -1, present_family: -1 }
    }

    fn is_complete(&self) -> bool {
        self.graphics_family >= 0 && self.present_family >= 0
    }
}

type MainDescriptorSet = ((), PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<CameraDataUbo>>>);
type VertexDefinition = OneVertexOneInstanceDefinition::<Vertex, InstanceData>;
//type MainDescriptorSet = (((), PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<CameraDataUbo>>>),
//                          PersistentDescriptorSetBuf<Arc<CpuAccessibleBuffer<InstanceData>>>);

pub struct RenderSystem {
    pub(in crate::rendering) instance: Arc<Instance>,
    #[allow(unused)]
    debug_callback: Option<DebugCallback>,

    pub(in crate::rendering) surface: Arc<Surface<Window>>,

    physical_device_index: usize,
    // can't store PhysicalDevice directly (lifetime issues)
    device: Arc<Device>,

    pub(in crate::rendering) graphics_queue: Arc<Queue>,
    present_queue: Arc<Queue>,

    swap_chain: Arc<Swapchain<Window>>,
    swap_chain_images: Vec<Arc<SwapchainImage<Window>>>,

    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    graphics_pipeline: Arc<GraphicsPipeline<VertexDefinition, Box<dyn PipelineLayoutAbstract + Send + Sync>, Arc<dyn RenderPassAbstract + Send + Sync>>>,

    swap_chain_framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,

    #[allow(dead_code)]
    camera_ubo: Vec<Arc<CpuAccessibleBuffer<CameraDataUbo>>>,
    pub(in crate::rendering) instance_buffer_pool: CpuBufferPool<InstanceData>,
    instance_info: HashMap<Mesh, Vec<InstanceData>>,

    camera_descriptor_sets: Vec<Arc<FixedSizeDescriptorSet<MainDescriptorSet>>>,

    command_buffers: Vec<Arc<AutoCommandBuffer>>,

    depth_format: Format,

    previous_frame_end: Option<Box<dyn GpuFuture>>,
    recreate_swap_chain: bool,

    #[allow(dead_code)]
    start_time: Instant,

}

impl System for RenderSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![
            crate::filter!(Mesh, Transformation),
            crate::filter!(Camera, Transformation),
            crate::filter!(Octree, Mesh, Transformation),
        ]
    }

    fn handle_input(&mut self, _event: &Event<()>) {

    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, _: Duration) {
        let mutex = filter[1].lock().unwrap();
        if mutex.entities.is_empty() {
            println!("No camera provided.");
            return;
        }
        let camera_entity = mutex.entities[0].lock().unwrap();
        let cam_cam_comp = camera_entity.get_component::<Camera>().ok().unwrap();
        let cam_trans_comp = camera_entity.get_component::<Transformation>().ok().unwrap();
        let camera_ubo = CameraDataUbo::new(cam_cam_comp, cam_trans_comp);

        self.instance_info
            .iter_mut()
            .for_each(|(_, val)| val.clear());

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
    }
}

impl RenderSystem {
    pub fn new(nse: &NSE) -> Arc<Mutex<Self>> {
        let instance = Self::create_instance();
        let debug_callback = Self::setup_debug_callback(&instance);
        let surface = Self::create_surface(&instance, &nse.event_loop);

        let physical_device_index = Self::pick_physical_device(&instance, &surface);
        let (device, graphics_queue, present_queue) = Self::create_logical_device(
            &instance, &surface, physical_device_index);

        let (swap_chain, swap_chain_images) = Self::create_swap_chain(&instance, &surface, physical_device_index,
                                                                      &device, &graphics_queue, &present_queue, None);

        let depth_format = Self::find_depth_format();
        let depth_image = Self::create_depth_image(&device, swap_chain.dimensions(), depth_format);

        let render_pass = Self::create_render_pass(&device, swap_chain.format(), depth_format);

        let graphics_pipeline = Self::create_graphics_pipeline(&device, swap_chain.dimensions(), &render_pass);

        let swap_chain_framebuffers = Self::create_framebuffers(&swap_chain_images, &render_pass, &depth_image);

        let start_time = Instant::now();

        let camera_ubo =
            Self::create_uniform_buffers(&device, swap_chain_images.len(), CameraDataUbo { ..Default::default() });

        let instance_buffer_pool: CpuBufferPool<InstanceData> = CpuBufferPool::vertex_buffer(device.clone());

        let descriptor_sets_pool = Self::create_descriptor_pool(&graphics_pipeline);
        let camera_descriptor_sets = Self::create_descriptor_sets(&descriptor_sets_pool, &camera_ubo);

        let previous_frame_end = Some(Self::create_sync_objects(&device));

        let mut imgui = Context::create();
        // configure imgui-rs Context if necessary

        let mut platform = WinitPlatform::init(&mut imgui); // step 1
        platform.attach_window(imgui.io_mut(), surface.window(), HiDpiMode::Default); // step 2

        let rs = RenderSystem {
            instance,
            debug_callback,

            surface,

            physical_device_index,
            device,

            graphics_queue,
            present_queue,

            swap_chain,
            swap_chain_images,

            render_pass,
            graphics_pipeline,

            swap_chain_framebuffers,

            camera_ubo,
            instance_buffer_pool,
            instance_info: HashMap::new(),

            camera_descriptor_sets,

            command_buffers: vec![],

            depth_format,

            previous_frame_end,
            recreate_swap_chain: false,

            start_time,
        };

        Arc::new(Mutex::new(rs))
    }

    fn create_instance() -> Arc<Instance> {
        if ENABLE_VALIDATION_LAYERS && !Self::check_validation_layer_support() {
            println!("Validation layers requested, but not available!")
        }

        let supported_extensions = InstanceExtensions::supported_by_core()
            .expect("failed to retrieve supported extensions");
        println!("Supported extensions: {:?}", supported_extensions);

        let app_info = ApplicationInfo {
            application_name: Some("Hello Triangle".into()),
            application_version: Some(Version { major: 1, minor: 0, patch: 0 }),
            engine_name: Some("No Engine".into()),
            engine_version: Some(Version { major: 1, minor: 0, patch: 0 }),
        };

        let required_extensions = Self::get_required_extensions();

        if ENABLE_VALIDATION_LAYERS && Self::check_validation_layer_support() {
            Instance::new(Some(&app_info), &required_extensions, VALIDATION_LAYERS.iter().cloned())
                .expect("failed to create Vulkan instance")
        } else {
            Instance::new(Some(&app_info), &required_extensions, None)
                .expect("failed to create Vulkan instance")
        }
    }

    fn check_validation_layer_support() -> bool {
        let layers: Vec<_> = layers_list().unwrap().map(|l| l.name().to_owned()).collect();
        VALIDATION_LAYERS.iter()
            .all(|layer_name| layers.contains(&layer_name.to_string()))
    }

    fn get_required_extensions() -> InstanceExtensions {
        let mut extensions = vulkano_win::required_extensions();
        if ENABLE_VALIDATION_LAYERS {
            extensions.ext_debug_utils = true;
        }

        extensions
    }

    fn setup_debug_callback(instance: &Arc<Instance>) -> Option<DebugCallback> {
        if !ENABLE_VALIDATION_LAYERS {
            return None;
        }

        let msg_severity = MessageSeverity {
            error: true,
            warning: true,
            information: true,
            verbose: true,
        };

        let msg_types = MessageType {
            general: true,
            validation: true,
            performance: true,
        };

        DebugCallback::new(&instance, msg_severity, msg_types, |msg| {
            println!("validation layer: {:?}", msg.description);
        }).ok()
    }

    fn pick_physical_device(instance: &Arc<Instance>, surface: &Arc<Surface<Window>>) -> usize {
        PhysicalDevice::enumerate(&instance)
            .position(|device| Self::is_device_suitable(surface, &device))
            .expect("failed to find a suitable GPU!")
    }

    fn is_device_suitable(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> bool {
        let indices = Self::find_queue_families(surface, device);
        let extensions_supported = Self::check_device_extension_support(device);

        let swap_chain_adequate = if extensions_supported {
            let capabilities = surface.capabilities(*device)
                .expect("failed to get surface capabilities");
            !capabilities.supported_formats.is_empty() &&
                capabilities.present_modes.iter().next().is_some()
        } else {
            false
        };

        indices.is_complete() && extensions_supported && swap_chain_adequate
    }

    fn check_device_extension_support(device: &PhysicalDevice) -> bool {
        let available_extensions = DeviceExtensions::supported_by_device(*device);
        let device_extensions = device_extensions();
        available_extensions.intersection(&device_extensions) == device_extensions
    }

    fn choose_swap_surface_format(available_formats: &[(Format, ColorSpace)]) -> (Format, ColorSpace) {
        // NOTE: the 'preferred format' mentioned in the tutorial doesn't seem to be
        // queryable in Vulkano (no VK_FORMAT_UNDEFINED enum)
        *available_formats.iter()
            .find(|(format, color_space)|
                *format == Format::B8G8R8A8Unorm && *color_space == ColorSpace::SrgbNonLinear
            )
            .unwrap_or_else(|| &available_formats[0])
    }

    fn choose_swap_present_mode(available_present_modes: SupportedPresentModes) -> PresentMode {
        if available_present_modes.mailbox {
            PresentMode::Mailbox
        } else if available_present_modes.immediate {
            PresentMode::Immediate
        } else {
            PresentMode::Fifo
        }
    }

    fn choose_swap_extent(capabilities: &Capabilities) -> [u32; 2] {
        if let Some(current_extent) = capabilities.current_extent {
            return current_extent;
        } else {
            let mut actual_extent = [WINDOW_WIDTH, WINDOW_HEIGHT];
            actual_extent[0] = capabilities.min_image_extent[0]
                .max(capabilities.max_image_extent[0].min(actual_extent[0]));
            actual_extent[1] = capabilities.min_image_extent[1]
                .max(capabilities.max_image_extent[1].min(actual_extent[1]));
            actual_extent
        }
    }

    fn create_swap_chain(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
        device: &Arc<Device>,
        graphics_queue: &Arc<Queue>,
        present_queue: &Arc<Queue>,
        old_swapchain: Option<Arc<Swapchain<Window>>>,
    ) -> (Arc<Swapchain<Window>>, Vec<Arc<SwapchainImage<Window>>>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let capabilities = surface.capabilities(physical_device)
            .expect("failed to get surface capabilities");

        let surface_format = Self::choose_swap_surface_format(&capabilities.supported_formats);
        let present_mode = Self::choose_swap_present_mode(capabilities.present_modes);
        let extent = Self::choose_swap_extent(&capabilities);

        let mut image_count = capabilities.min_image_count + 1;
        if capabilities.max_image_count.is_some() && image_count > capabilities.max_image_count.unwrap() {
            image_count = capabilities.max_image_count.unwrap();
        }

        let image_usage = ImageUsage {
            color_attachment: true,
            ..ImageUsage::none()
        };

        let indices = Self::find_queue_families(&surface, &physical_device);

        let sharing: SharingMode = if indices.graphics_family != indices.present_family {
            vec![graphics_queue, present_queue].as_slice().into()
        } else {
            graphics_queue.into()
        };

        let (swap_chain, images) = Swapchain::new(
            device.clone(),
            surface.clone(),
            image_count,
            surface_format.0,
            extent,
            1, // layers
            image_usage,
            sharing,
            capabilities.current_transform,
            CompositeAlpha::Opaque,
            present_mode,
            FullscreenExclusive::Default,
            true, // clipped
            surface_format.1
        ).expect("failed to create swap chain!");

        (swap_chain, images)
    }

    fn find_depth_format() -> Format {
        // this format is guaranteed to be supported by vulkano and as it stands now, I can't figure
        // how to do the queries performed in the original tutorial in vulkano...
        //Format::D16Unorm
        Format::D24Unorm_S8Uint
    }

    fn create_depth_image(device: &Arc<Device>, dimensions: [u32; 2], format: Format) -> Arc<AttachmentImage<Format>> {
        AttachmentImage::with_usage(
            device.clone(),
            dimensions,
            format,
            ImageUsage { depth_stencil_attachment: true, ..ImageUsage::none() },
        ).unwrap()
    }

    fn create_render_pass(device: &Arc<Device>, color_format: Format, depth_format: Format) -> Arc<dyn RenderPassAbstract + Send + Sync> {
        Arc::new(single_pass_renderpass!(device.clone(),
            attachments: {
                color: {
                    load: Clear,
                    store: Store,
                    format: color_format,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: depth_format,
                    samples: 1,
                    initial_layout: ImageLayout::Undefined,
                    final_layout: ImageLayout::DepthStencilAttachmentOptimal,
                }
            },
            pass: {
                color: [color],
                depth_stencil: {depth}
            }
        ).unwrap())
    }

    fn create_graphics_pipeline(
        device: &Arc<Device>,
        swap_chain_extent: [u32; 2],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
    ) -> Arc<GraphicsPipeline<VertexDefinition, Box<dyn PipelineLayoutAbstract + Send + Sync>, Arc<dyn RenderPassAbstract + Send + Sync>>> {
        mod vertex_shader {
            vulkano_shaders::shader! {
               ty: "vertex",
               path: "examples/shaders/test.vert.glsl"
            }
        }

        mod fragment_shader {
            vulkano_shaders::shader! {
                ty: "fragment",
                path: "examples/shaders/test.frag.glsl"
            }
        }

        let vert_shader_module = vertex_shader::Shader::load(device.clone())
            .expect("failed to create vertex shader module!");
        let frag_shader_module = fragment_shader::Shader::load(device.clone())
            .expect("failed to create fragment shader module!");

        let dimensions = [swap_chain_extent[0] as f32, swap_chain_extent[1] as f32];
        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions,
            depth_range: 0.0..1.0,
        };

        Arc::new(GraphicsPipeline::start()
            .vertex_input(VertexDefinition::new())
//            .vertex_input_single_buffer::<Vertex>()
            .vertex_shader(vert_shader_module.main_entry_point(), ())
            .triangle_list()
            .primitive_restart(false)
            .viewports(vec![viewport]) // NOTE: also sets scissor to cover whole viewport
            .fragment_shader(frag_shader_module.main_entry_point(), ())
            .depth_clamp(false)
            // NOTE: there's an outcommented .rasterizer_discard() in Vulkano...
//            .polygon_mode_fill() // = default
            .polygon_mode_line()
            .line_width(1.0) // = default
            .cull_mode_back()
            .front_face_counter_clockwise()
            // NOTE: no depth_bias here, but on pipeline::raster::Rasterization
            .blend_pass_through() // = default
            .depth_stencil(DepthStencil::simple_depth_test())
            .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
            .build(device.clone())
            .unwrap())
    }

    fn create_framebuffers(
        swap_chain_images: &[Arc<SwapchainImage<Window>>],
        render_pass: &Arc<dyn RenderPassAbstract + Send + Sync>,
        depth_image: &Arc<AttachmentImage<Format>>,
    ) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
        swap_chain_images.iter()
            .map(|image| {
                let fba: Arc<dyn FramebufferAbstract + Send + Sync> = Arc::new(Framebuffer::start(render_pass.clone())
                    .add(image.clone()).unwrap()
                    .add(depth_image.clone()).unwrap()
                    .build().unwrap());
                fba
            }
            ).collect::<Vec<_>>()
    }

    fn create_uniform_buffers<T: 'static + Copy>(
        device: &Arc<Device>,
        num_buffers: usize,
        data: T,
    ) -> Vec<Arc<CpuAccessibleBuffer<T>>> {
        let mut buffers = Vec::new();

        for _ in 0..num_buffers {
            let buffer = CpuAccessibleBuffer::from_data(
                device.clone(),
                BufferUsage::uniform_buffer_transfer_destination(),
                false,
                data,
            ).unwrap();

            buffers.push(buffer);
        }

        buffers
    }

    fn create_descriptor_pool(graphics_pipeline: &Arc<GraphicsPipeline<VertexDefinition, Box<dyn PipelineLayoutAbstract + Send + Sync>, Arc<dyn RenderPassAbstract + Send + Sync>>>)
                              -> Mutex<FixedSizeDescriptorSetsPool>
    {
        let layout = graphics_pipeline.layout().descriptor_set_layout(0).unwrap();
        Mutex::new(FixedSizeDescriptorSetsPool::new(layout.clone()))
    }

    fn create_descriptor_sets(
        pool: &Mutex<FixedSizeDescriptorSetsPool>,
        camera_ubo: &[Arc<CpuAccessibleBuffer<CameraDataUbo>>],
    ) -> Vec<Arc<FixedSizeDescriptorSet<MainDescriptorSet>>>
    {
        camera_ubo
            .iter()
            .map(|camera_buff| {
                Arc::new(
                    pool
                        .lock()
                        .unwrap()
                        .next()
                        .add_buffer(camera_buff.clone())
                        .unwrap()
                        .build()
                        .unwrap()
                )
            })
            .collect()
    }

    fn forward_pass_command_buffer(&mut self,
                                   camera_ubo: CameraDataUbo,
                                   instanced_meshes: Vec<(Mesh, Arc<CpuBufferPoolChunk<InstanceData, Arc<StdMemoryPool>>>)>)
    {
        let queue_family = self.graphics_queue.family();

        self.command_buffers = self.swap_chain_framebuffers
            .iter()
            .enumerate()
            .map(|(i, framebuffer)| {
                let mut builder =
                    AutoCommandBufferBuilder::primary_simultaneous_use(
                        self.device.clone(),
                        queue_family)
                        .unwrap()
                        .update_buffer(self.camera_ubo[i].clone(), camera_ubo)
                        .unwrap()
                        .begin_render_pass(
                            framebuffer.clone(),
                            false,
                            vec![[0.0, 0.0, 0.0, 1.0].into(), ClearValue::DepthStencil((1.0, 0))])
                        .unwrap();

                for (mesh, model_matrices) in self.instance_info.iter() {
                    let instance_buffer = self.instance_buffer_pool.chunk(model_matrices.clone()).unwrap();

                    builder = builder.draw_indexed(
                        self.graphics_pipeline.clone(),
                        &DynamicState::none(),
                        vec![mesh.vertex_buffer.clone(), Arc::new(instance_buffer)],
                        mesh.index_buffer.clone(),
                        self.camera_descriptor_sets[i].clone(),
                        (),
                    ).unwrap();
                }

                for (mesh, instance_buffer) in instanced_meshes.iter() {
                    builder = builder.draw_indexed(
                        self.graphics_pipeline.clone(),
                        &DynamicState::none(),
                        vec![mesh.vertex_buffer.clone(), instance_buffer.clone()],
                        mesh.index_buffer.clone(),
                        self.camera_descriptor_sets[i].clone(),
                        (),
                    ).unwrap();
                }

//                for e in entities {
//                    let mut entity = e.lock().unwrap();
//                    let mesh = entity.get_component::<Mesh>().ok().unwrap();
//                    builder = builder.draw_indexed(
//                        self.graphics_pipeline.clone(),
//                        &DynamicState::none(),
//                        vec![mesh.vertex_buffer.clone()],
//                        mesh.index_buffer.clone(),
//                        self.camera_descriptor_sets[i].clone(),
//                        (),
//                    ).unwrap();
//                }

                Arc::new(builder.end_render_pass()
                    .unwrap()
                    .build()
                    .unwrap())
            })
            .collect();
    }

    fn create_command_buffers(&mut self) {
        let queue_family = self.graphics_queue.family();

        self.command_buffers = self.swap_chain_framebuffers
            .iter()
            .enumerate()
            .map(|(i, framebuffer)| {
                Arc::new(AutoCommandBufferBuilder::primary_simultaneous_use(self.device.clone(), queue_family)
                    .unwrap()
                    .update_buffer(self.camera_ubo[i].clone(), CameraDataUbo { ..Default::default() })
                    .unwrap()
                    .begin_render_pass(
                        framebuffer.clone(),
                        false,
                        vec![[0.0, 0.0, 0.0, 1.0].into(), ClearValue::DepthStencil((1.0, 0))])
                    .unwrap()
                    .end_render_pass()
                    .unwrap()
                    .build()
                    .unwrap())
            })
            .collect();
    }

    fn create_sync_objects(device: &Arc<Device>) -> Box<dyn GpuFuture> {
        Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>
    }

    fn find_queue_families(surface: &Arc<Surface<Window>>, device: &PhysicalDevice) -> QueueFamilyIndices {
        let mut indices = QueueFamilyIndices::new();
        // TODO: replace index with id to simplify?
        for (i, queue_family) in device.queue_families().enumerate() {
            if queue_family.supports_graphics() {
                indices.graphics_family = i as i32;
            }

            if surface.is_supported(queue_family).unwrap() {
                indices.present_family = i as i32;
            }

            if indices.is_complete() {
                break;
            }
        }

        indices
    }

    fn create_logical_device(
        instance: &Arc<Instance>,
        surface: &Arc<Surface<Window>>,
        physical_device_index: usize,
    ) -> (Arc<Device>, Arc<Queue>, Arc<Queue>) {
        let physical_device = PhysicalDevice::from_index(&instance, physical_device_index).unwrap();
        let indices = Self::find_queue_families(&surface, &physical_device);

        let families = [indices.graphics_family, indices.present_family];
        use std::iter::FromIterator;
        let unique_queue_families: HashSet<&i32> = HashSet::from_iter(families.iter());

        let queue_priority = 1.0;
        let queue_families = unique_queue_families.iter().map(|i| {
            (physical_device.queue_families().nth(**i as usize).unwrap(), queue_priority)
        });

        // NOTE: the tutorial recommends passing the validation layers as well
        // for legacy reasons (if ENABLE_VALIDATION_LAYERS is true). Vulkano handles that
        // for us internally.

        let minimal_features = Features {
            fill_mode_non_solid: true,
            ..Features::none()
        };

        let (device, mut queues) = Device::new(physical_device,
                                               &minimal_features,
                                               &device_extensions(), queue_families)
            .expect("failed to create logical device!");

        let graphics_queue = queues.next().unwrap();
        let present_queue = queues.next().unwrap_or_else(|| graphics_queue.clone());

        (device, graphics_queue, present_queue)
    }

    fn create_surface(instance: &Arc<Instance>, events_loop: &EventLoop<()>) -> Arc<Surface<Window>> {
        let surface = WindowBuilder::new()
            .with_title(WINDOW_TITLE)
//            .with_dimensions(LogicalSize::new(f64::from(WINDOW_WIDTH), f64::from(WINDOW_HEIGHT)))
            .build_vk_surface(&events_loop, instance.clone())
            .expect("failed to create window surface!");
        surface
    }

    fn draw_frame(&mut self) {
        self.previous_frame_end.as_mut().unwrap().cleanup_finished();

        if self.recreate_swap_chain {
            self.recreate_swap_chain();
            self.recreate_swap_chain = false;
        }

        let (image_index, _, acquire_future) = match acquire_next_image(self.swap_chain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swap_chain = true;
                return;
            }
            Err(err) => panic!("{:?}", err)
        };

        let command_buffer = self.command_buffers[image_index].clone();

        let future = self.previous_frame_end.take().unwrap()
            .join(acquire_future)
            .then_execute(self.graphics_queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.present_queue.clone(), self.swap_chain.clone(), image_index)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                self.previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(vulkano::sync::FlushError::OutOfDate) => {
                self.recreate_swap_chain = true;
                self.previous_frame_end
                    = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("{:?}", e);
                self.previous_frame_end
                    = Some(Box::new(vulkano::sync::now(self.device.clone())) as Box<_>);
            }
        }
    }

    fn recreate_swap_chain(&mut self) {
        let (swap_chain, images) = Self::create_swap_chain(&self.instance, &self.surface, self.physical_device_index,
                                                           &self.device, &self.graphics_queue, &self.present_queue, Some(self.swap_chain.clone()));

        let depth_image = Self::create_depth_image(&self.device, swap_chain.dimensions(), self.depth_format);

        self.swap_chain = swap_chain;
        self.swap_chain_images = images;

        self.render_pass = Self::create_render_pass(&self.device, self.swap_chain.format(), self.depth_format);
        self.graphics_pipeline = Self::create_graphics_pipeline(&self.device, self.swap_chain.dimensions(),
                                                                &self.render_pass);
        self.swap_chain_framebuffers = Self::create_framebuffers(&self.swap_chain_images, &self.render_pass, &depth_image);
        self.create_command_buffers();
    }
}

