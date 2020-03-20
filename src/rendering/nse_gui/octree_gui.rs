use std::collections::VecDeque;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use glium::{Display, Surface};
use glium::glutin;
use glium::glutin::event::WindowEvent;
use glium::glutin::window::WindowBuilder;
use imgui::*;
use imgui::{Context, FontConfig, FontGlyphRanges, FontSource};
//use imgui_gfx_renderer::Shaders;
//use imgui_rs_vulkan_renderer::Renderer;
//use imgui_glium_renderer::glium::Display;
use imgui_glium_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use winit::event::{ElementState, Event, VirtualKeyCode};

use crate::core::{Filter, Message, Payload, System};
use crate::NSE;
use crate::rendering::{Camera, Mesh, Octree, OctreeConfig, RenderSystem, Transformation, OctreeInfo, OctreeOptimizations};
use std::rc::Rc;
use mopa::Any;
use std::cell::RefCell;
use cgmath::{Transform, Vector3, Vector4};

pub struct OctreeGuiSystem {
    imgui: Arc<Mutex<Context>>,
    platform: WinitPlatform,
    renderer: Renderer,
    display: Arc<Mutex<glium::Display>>,

    // octree data
    octree_config: OctreeConfig,
    octree_optimizations: OctreeOptimizations,

    // profiling data
    profiling_data: ProfilingData,

    frame_times: VecDeque<f32>,

    // message passing
    messages: Vec<Message>,
}

impl OctreeGuiSystem {
    pub fn new(nse: &NSE, _render_system: Arc<Mutex<RenderSystem>>) -> Arc<Mutex<Self>> {
        let mut imgui = Context::create();

        // configure imgui-rs Context if necessary
        imgui.set_ini_filename(None);

        let context = glutin::ContextBuilder::new().with_vsync(false);
        let builder = WindowBuilder::new()
            .with_title("Octree - Config")
            .with_decorations(true)
            .with_inner_size(glutin::dpi::LogicalSize::new(420f64, 768f64));
        let display =
            Display::new(builder, context, &nse.event_loop).expect("Failed to initialize display");

        let mut platform = WinitPlatform::init(&mut imgui); // step 1
        platform.attach_window(imgui.io_mut(), &display.gl_window().window(), HiDpiMode::Default); // step 2

        let hidpi_factor = platform.hidpi_factor();
        let font_size = (13.0 * hidpi_factor) as f32;
        imgui.fonts().add_font(&[
            FontSource::DefaultFontData {
                config: Some(FontConfig {
                    size_pixels: font_size,
                    ..FontConfig::default()
                }),
            },
            FontSource::TtfData {
                data: include_bytes!("resources/mplus-1p-regular.ttf"),
                size_pixels: font_size,
                config: Some(FontConfig {
                    rasterizer_multiply: 1.75,
                    glyph_ranges: FontGlyphRanges::japanese(),
                    ..FontConfig::default()
                }),
            },
        ]);

        imgui.io_mut().font_global_scale = (1.0 / hidpi_factor) as f32;

        let renderer = Renderer::init(&mut imgui, &display).expect("Failed to initialize renderer");

        let mut frame_times = VecDeque::new();
        frame_times.resize(500, 0.0);

        Arc::new(Mutex::new(OctreeGuiSystem {
            imgui: Arc::new(Mutex::new(imgui)),
            platform,
            renderer,
            display: Arc::new(Mutex::new(display)),

            octree_config: OctreeConfig::default(),
            octree_optimizations: OctreeOptimizations::default(),

            profiling_data: ProfilingData::default(),
            frame_times,

            messages: vec![],
        }))
    }

    fn display_octree_ui(&mut self, ui: &Ui, config: &OctreeConfig, info: &OctreeInfo) {
        if ui.collapsing_header(im_str!("Settings")).default_open(true).build() {
            ui.text(format!("RAM Allocation: {:.2} MB", info.byte_size as f64 / (1024f64 * 1024f64)));
            ui.text(format!("Max. number of rendered nodes: {}", config.max_rendered_nodes.unwrap_or(0)));
            ui.text(format!("GPU Allocation: {:.2} MB", info.gpu_byte_size as f64 / (1024f64 * 1024f64)));

            ui.separator();

            ui.text(format!("Optimizations"));
            if ui.checkbox(im_str!("Frustum Culling"), &mut self.octree_optimizations.frustum_culling) {
                self.messages.push(Message::new(self.octree_optimizations.clone()));
            }
            if ui.checkbox(im_str!("Depth Culling"), &mut self.octree_optimizations.depth_culling) {
                self.messages.push(Message::new(self.octree_optimizations.clone()));
            }

            if Slider::new(im_str!("Depth Culling Threshold"), RangeInclusive::new(0.00001, 0.001))
                .build(&ui, &mut self.octree_optimizations.depth_threshold) {
                self.messages.push(Message::new(self.octree_optimizations.clone()));
            }

            ui.separator();

            Slider::new(im_str!("Max. Rendered Nodes"), RangeInclusive::new(1e3 as u64, 5e6 as u64))
                .power(100.0)
                .build(&ui, self.octree_config.max_rendered_nodes.as_mut().unwrap());

            Slider::new(im_str!("Octree Depth"), RangeInclusive::new(2, 11))
                .build(&ui, self.octree_config.depth.as_mut().unwrap());

            if ui.button(im_str!("Regenerate Octree"), [0.0, 0.0]) {
                self.messages.push(Message::new(self.octree_config.clone()));
            };
            if ui.button(im_str!("Reset Octree"), [0.0, 0.0]) {
                self.octree_config = OctreeConfig::default();
                self.messages.push(Message::new(self.octree_config.clone()));
            };
        }
    }

    fn display_profiling_ui(&mut self, delta_time: Duration, ui: &Ui) {
        let rendered_nodes = &mut self.profiling_data.rendered_nodes.unwrap_or(0);
        let render_time = self.profiling_data.render_time.unwrap_or(0) as f64 / 1e6 as f64;

        let frame_times = &mut self.frame_times;
        frame_times.pop_front();
        frame_times.push_back(delta_time.as_secs_f32());

        let f_times: Vec<f32> = frame_times.iter().cloned().collect();

        if ui.collapsing_header(im_str!("Profiling")).default_open(true).build() {
            // Plot Frame Times
            ui.plot_lines(im_str!("Frame Times"), &f_times[..])
                .graph_size([0.0, 50.0])
                .overlay_text(&im_str!("{} ms", delta_time.as_millis()))
                .build();

            // print times of seperate systems
            if self.profiling_data.system_times.is_some() {
                for (system_name, system_time) in self.profiling_data.system_times.as_ref().unwrap() {
                    ui.text(im_str!("{}: {}", system_name, system_time.as_millis()));
                }
            }

            ui.separator();

            ui.text(im_str!("Rendered Nodes: {}", rendered_nodes));

            ui.text(im_str!("Render Time Nodes: {:.2} ms", render_time));

            ui.separator();

            let mouse_pos = ui.io().mouse_pos;
            ui.text(format!(
                "Mouse Position: ({:.1},{:.1})",
                mouse_pos[0], mouse_pos[1]
            ));
        }
    }

    fn display_camera_ui(&mut self, ui: &Ui, camera: &Camera, camera_transform: &Transformation) {
        if ui.collapsing_header(im_str!("Camera")).default_open(true).build() {

            let view_dir = camera_transform.get_model_matrix() * (-Vector4::unit_z());

            InputFloat3::new(&ui, im_str!("View Direction"), &mut [
                view_dir.x,
                view_dir.y,
                view_dir.z,
            ]).read_only(true).build();

            InputFloat3::new(&ui, im_str!("Camera Position"), &mut [
                camera_transform.position.x,
                camera_transform.position.y,
                camera_transform.position.z,
            ]).read_only(true).build();
        }
    }
}

impl System for OctreeGuiSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![
            crate::filter!(Octree, Mesh, Transformation),
            crate::filter!(Camera, Transformation)
        ]
    }

    fn handle_input(&mut self, _event: &Event<()>) {
        let platform = &mut self.platform;
        let display = self.display.lock().unwrap();
        let gl_window = display.gl_window();
        let mut imgui = self.imgui.lock().unwrap();

        match _event {
            Event::MainEventsCleared => {
                platform.prepare_frame(imgui.io_mut(), &gl_window.window()) // step 4
                    .expect("Failed to prepare frame");

                gl_window.window().request_redraw();
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                window_id
            } => {
                if *window_id == gl_window.window().id() {
                    println!("Close Octree Config Window");

                    gl_window.window().set_visible(false);
                    return;
                }
            }
            Event::WindowEvent { event, .. } => {
                match event {
                    WindowEvent::KeyboardInput { input, .. } => {
                        match input {
                            | winit::event::KeyboardInput { virtual_keycode, state, .. } => {
                                match (virtual_keycode, state) {
                                    (Some(VirtualKeyCode::F12), ElementState::Pressed) => {
                                        println!("Open Octree Config Window");
                                        gl_window.window().set_visible(true);
                                    }
                                    _ => ()
                                }
                            }
                        }
                    }
                    _ => ()
                }
            }
            _ => ()
        }

        platform.handle_event(imgui.io_mut(), &gl_window.window(), &_event); // step 3
    }

    fn consume_messages(&mut self, messages: &Vec<Message>) {
        for m in messages {
            if m.is_type::<ProfilingData>() {
                let data = m.get_payload::<ProfilingData>().unwrap();
                self.profiling_data.replace(data);
            }
        }
    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, delta_time: Duration) {
        let ctx = self.imgui.clone();
        let display = self.display.clone();
        let mut ctx_lock = ctx.lock().unwrap();
        let mut display_lock = display.lock().unwrap();

        let ui = ctx_lock.frame();
        let gl_window = display_lock.gl_window();

        let octree_entities = &filter[0].lock().unwrap().entities;
        let camera_entities = &filter[1].lock().unwrap().entities;

        let window = Window::new(im_str!("Octree"))
            .collapsible(false)
            .movable(false)
            .position([10.0, 10.0], Condition::FirstUseEver)
            .size([400.0, 0.0], Condition::FirstUseEver);
        let window_token = window.begin(&ui).unwrap();

        for entity in octree_entities {
            let mut entitiy_mutex = entity.lock().unwrap();
            let _octree_transform = entitiy_mutex.get_component::<Transformation>().ok().unwrap();
            let octree = entitiy_mutex.get_component::<Octree>().ok().unwrap();

            self.display_profiling_ui(delta_time, &ui);

            ui.new_line();

            self.display_octree_ui(&ui, &octree.config, &octree.info);

            ui.new_line();
        }

        for entity in camera_entities {
            let mut entitiy_mutex = entity.lock().unwrap();
            let camera_transform = entitiy_mutex.get_component::<Transformation>().ok().unwrap();
            let camera = entitiy_mutex.get_component::<Camera>().ok().unwrap();

            self.display_camera_ui(&ui, camera, camera_transform);
        }

        window_token.end(&ui);

        // construct the UI
        self.platform.prepare_render(&ui, &gl_window.window()); // step 5
        // render the UI with a renderer
        let draw_data = ui.render();
        // application-specific rendering *over the UI*

        let mut target = display_lock.draw();
        target.clear_color_srgb(0.1, 0.1, 0.11, 1.0);

        self.renderer
            .render(&mut target, draw_data)
            .expect("Rendering failed");

        target.finish().expect("Failed to swap buffers");
    }

    fn get_messages(&mut self) -> Vec<Message> {
        let ret = self.messages.clone();

        self.messages.clear();

        ret
    }
}

#[derive(Debug, Clone, Default)]
pub struct ProfilingData {
    pub rendered_nodes: Option<u32>,
    pub instance_data_generation: Option<u64>,
    pub render_time: Option<u64>, // in nano seconds
    pub system_times: Option<Vec<(String, Duration)>>
}

impl ProfilingData {
    pub fn replace(&mut self, other: &Self) {
        Self::replace_option(&mut self.rendered_nodes, &other.rendered_nodes);
        Self::replace_option(&mut self.instance_data_generation, &other.instance_data_generation);
        Self::replace_option(&mut self.render_time, &other.render_time);
        Self::replace_option(&mut self.system_times, &other.system_times);
    }

    fn replace_option<T>(target: &mut Option<T>, source: &Option<T>)
        where T: Clone {
        match source {
            Some(val) => target.replace(val.clone()),
            None => None
        };
    }
}

impl Payload for ProfilingData {}