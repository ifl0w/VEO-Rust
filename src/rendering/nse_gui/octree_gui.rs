use imgui::*;
use imgui_winit_support::{WinitPlatform, HiDpiMode};
use winit::event::Event;
use std::sync::{Mutex, Arc};
use std::time::Duration;

use crate::core::{System, Filter, Message};

use crate::NSE;
use crate::rendering::RenderSystem;

pub struct OctreeGuiSystem {
    imgui: Context,
    platform: WinitPlatform,
    render_system: Arc<Mutex<RenderSystem>>
}

impl OctreeGuiSystem {
    pub fn new(nse: &NSE, render_system: Arc<Mutex<RenderSystem>>) -> Arc<Mutex<Self>> {

        let mut event_loop = &nse.event_loop;
        let render_system_lock = render_system.lock().unwrap();
        let mut window = render_system_lock.surface.window();

        let mut imgui = Context::create();
        // configure imgui-rs Context if necessary

        let mut platform = WinitPlatform::init(&mut imgui); // step 1
        platform.attach_window(imgui.io_mut(), &window, HiDpiMode::Default); // step 2

        drop(render_system_lock);

        Arc::new(Mutex::new(OctreeGuiSystem {
            imgui,
            platform,
            render_system
        }))
    }
}

impl System for OctreeGuiSystem {
    fn get_filter(&mut self) -> Vec<Filter> { vec![] }
    fn handle_input(&mut self, event: &Event<()>) {
        match event {
            Event::MainEventsCleared => {
                // other application-specific logic
                let render_system_lock = self.render_system.lock().unwrap();
                let window = render_system_lock.surface.window();
                self.platform.prepare_frame(self.imgui.io_mut(), &window) // step 4
                    .expect("Failed to prepare frame");
            }
            event => {
                let render_system_lock = self.render_system.lock().unwrap();
                let window = render_system_lock.surface.window();
                self.platform.handle_event(self.imgui.io_mut(), &window, &event); // step 3
            }
        }
    }

    fn consume_messages(&mut self, _: &Vec<Message>) {
    }

    fn execute(&mut self, _: &Vec<Arc<Mutex<Filter>>>, _delta_time: Duration) {
        let ui = self.imgui.frame();
        let render_system_lock = self.render_system.lock().unwrap();
        let window = render_system_lock.surface.window();
        // application-specific rendering *under the UI*

        // construct the UI
        self.platform.prepare_render(&ui, &window); // step 5
        // render the UI with a renderer
        let draw_data = ui.render();
        // renderer.render(..., draw_data).expect("UI rendering failed");

        // application-specific rendering *over the UI*
    }
    fn get_messages(&mut self) -> Vec<Message> { vec![] }
}