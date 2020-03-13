use std::collections::VecDeque;
use std::ops::RangeInclusive;
use std::sync::{Arc, Mutex};
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
use crate::rendering::RenderSystem;

pub struct OctreeGuiSystem {
    imgui: Context,
    platform: WinitPlatform,
    renderer: Renderer,
    display: glium::Display,

    // octree data
    octree_depth: i32,

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
            imgui,
            platform,
            renderer,
            display,

            octree_depth: 5,

            profiling_data: ProfilingData::default(),
            frame_times,

            messages: vec![],
        }))
    }
}

impl System for OctreeGuiSystem {
    fn get_filter(&mut self) -> Vec<Filter> { vec![] }

    fn handle_input(&mut self, _event: &Event<()>) {
        let platform = &mut self.platform;
        let gl_window = self.display.gl_window();
        let imgui = &mut self.imgui;

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
                if *window_id == self.display.gl_window().window().id() {
                    println!("Close Octree Config Window");

                    self.display.gl_window().window().set_visible(false);
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
                                        self.display.gl_window().window().set_visible(true);
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

                match data.rendered_nodes {
                    Some(v) => self.profiling_data.rendered_nodes.replace(v),
                    None => None
                };
                match data.render_time {
                    Some(v) => self.profiling_data.render_time.replace(v),
                    None => None
                };
                match data.instance_data_generation {
                    Some(v) => self.profiling_data.instance_data_generation.replace(v),
                    None => None
                };
            }
        }
    }

    fn execute(&mut self, _: &Vec<Arc<Mutex<Filter>>>, delta_time: Duration) {
        let ui = self.imgui.frame();
//        let render_system_lock = self.render_system.lock().unwrap();
//        let window = render_system_lock.surface.window();
        let gl_window = self.display.gl_window();
        // application-specific rendering *under the UI*

        let messages = &mut self.messages;

        let octree_depth = &mut self.octree_depth;

        let rendered_nodes = &mut self.profiling_data.rendered_nodes.unwrap_or(0);
        let render_time = self.profiling_data.render_time.unwrap_or(0) as f64 / 1e6 as f64;

        let frame_times = &mut self.frame_times;
        frame_times.pop_front();
        frame_times.push_back(delta_time.as_secs_f32());

        let f_times: Vec<f32> = frame_times.iter().cloned().collect();

        Window::new(im_str!("Octree"))
            .size([300.0, 0.0], Condition::FirstUseEver)
            .build(&ui, || {
                if ui.collapsing_header(im_str!("Settings")).default_open(true).build() {
                    ui.new_line();

                    Slider::new(im_str!("Octree Depth"), RangeInclusive::new(2, 10))
                        .build(&ui, octree_depth);
                    if ui.button(im_str!("Update Octree"), [0.0, 0.0]) {
                        messages.push(Message::new(UpdateOctree { octree_depth: *octree_depth }));
                    };

                    ui.new_line();
                }

                if ui.collapsing_header(im_str!("Profiling")).default_open(true).build() {
                    ui.new_line();
                    // Plot Frame Times
                    ui.plot_lines(im_str!("Frame Times"), &f_times[..])
                        .graph_size([0.0, 50.0])
                        .overlay_text(&im_str!("{} ms", delta_time.as_millis()))
                        .build();

                    ui.separator();

                    ui.text(im_str!("Rendered Nodes: {}", rendered_nodes));

                    ui.text(im_str!("Render Time Nodes: {:.2} ms", render_time));

                    ui.separator();

                    let mouse_pos = ui.io().mouse_pos;
                    ui.text(format!(
                        "Mouse Position: ({:.1},{:.1})",
                        mouse_pos[0], mouse_pos[1]
                    ));
                    ui.new_line();
                }
            });

        // construct the UI
        self.platform.prepare_render(&ui, &gl_window.window()); // step 5
        // render the UI with a renderer
        let draw_data = ui.render();
        // application-specific rendering *over the UI*

        let mut target = self.display.draw();
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

#[derive(Debug, Clone)]
pub struct UpdateOctree {
    pub octree_depth: i32
}

impl Payload for UpdateOctree {}

#[derive(Debug, Clone, Default)]
pub struct ProfilingData {
    pub rendered_nodes: Option<u32>,
    pub instance_data_generation: Option<u64>,
    pub render_time: Option<u64>, // in nano seconds
}

impl Payload for ProfilingData {}