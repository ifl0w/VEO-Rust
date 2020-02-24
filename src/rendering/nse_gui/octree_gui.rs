use imgui::*;
use winit::event::Event;
use std::sync::{Mutex, Arc};
use std::time::Duration;

use crate::core::{System, Filter, Message};

use crate::NSE;
use crate::rendering::RenderSystem;
//use imgui_gfx_renderer::Shaders;
//use imgui_rs_vulkan_renderer::Renderer;
//use imgui_glium_renderer::glium::Display;
use imgui_glium_renderer::glium::*;
//use imgui_glium_renderer::glium::Display;

use glium::glutin;
use glium::glutin::event::{WindowEvent};
use glium::glutin::event_loop::{ControlFlow, EventLoop};
use glium::glutin::window::WindowBuilder;
use glium::{Display, Surface};
use imgui::{Context, FontConfig, FontGlyphRanges, FontSource, Ui};
use imgui_glium_renderer::Renderer;
use imgui_winit_support::{HiDpiMode, WinitPlatform};
use vulkano_win::VkSurfaceBuild;
use winit::platform::desktop::EventLoopExtDesktop;
use std::cell::{Cell, RefCell};
use imgui::StyleColor::Button;
use std::thread;

pub struct OctreeGuiSystem {
    imgui: Context,
    platform: WinitPlatform,
    render_system: Arc<Mutex<RenderSystem>>,
    renderer: Renderer,
    display: glium::Display,

    event_loop: EventLoop<()>
}

impl OctreeGuiSystem {
    pub fn new(nse: &NSE, render_system: Arc<Mutex<RenderSystem>>) -> Arc<Mutex<Self>> {

        let mut event_loop = EventLoop::new();
//        let render_system_lock = render_system.lock().unwrap();
//        let mut main_window = render_system_lock.surface.window();

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

//        drop(render_system_lock);

        Arc::new(Mutex::new(OctreeGuiSystem {
            imgui,
            platform,
            render_system,
            renderer,
            display,

            event_loop,
        }))
    }
}

impl System for OctreeGuiSystem {
    fn get_filter(&mut self) -> Vec<Filter> { vec![] }
    fn handle_input(&mut self, _event: &Event<()>) {
        let el = &mut self.event_loop;
        let platform = &mut self.platform;
        let gl_window = self.display.gl_window();
        let imgui = &mut self.imgui;

//        el.run_return(|event, _, control_flow| match event {
//            Event::MainEventsCleared => {
//                platform.prepare_frame(imgui.io_mut(), &gl_window.window()) // step 4
//                    .expect("Failed to prepare frame");
//
//                gl_window.window().request_redraw();
//            }
//            Event::WindowEvent {
//                event: WindowEvent::CloseRequested,
//                ..
//            } => *control_flow = ControlFlow::Exit,
//            event => {
//                platform.handle_event(imgui.io_mut(), &gl_window.window(), &event); // step 3
//            }
//        });

        match _event {
            Event::MainEventsCleared => {
                platform.prepare_frame(imgui.io_mut(), &gl_window.window()) // step 4
                    .expect("Failed to prepare frame");

                gl_window.window().request_redraw();
            },
            _ => ()
        }

        platform.handle_event(imgui.io_mut(), &gl_window.window(), &_event); // step 3


    }

    fn consume_messages(&mut self, _: &Vec<Message>) {
    }

    fn execute(&mut self, _: &Vec<Arc<Mutex<Filter>>>, _delta_time: Duration) {
        let ui = self.imgui.frame();
//        let render_system_lock = self.render_system.lock().unwrap();
//        let window = render_system_lock.surface.window();
        let gl_window = self.display.gl_window();
        // application-specific rendering *under the UI*

        Window::new(im_str!("Hello world"))
            .size([300.0, 100.0], Condition::FirstUseEver)
            .build(&ui, || {
                ui.text(im_str!("Hello world! {}", _delta_time.as_millis()));
                ui.text(im_str!("This...is...imgui-rs!"));
                ui.button(im_str!("Test"), [50.0, 10.0]);
                ui.separator();
                let mouse_pos = ui.io().mouse_pos;
                ui.text(format!(
                    "Mouse Position: ({:.1},{:.1})",
                    mouse_pos[0], mouse_pos[1]
                ));
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
    fn get_messages(&mut self) -> Vec<Message> { vec![] }
}