use crate::core::{Entity, System};
use crate::NSE;

// Constants
const WINDOW_TITLE: &'static str = "NSE";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

#[derive(Debug)]
pub struct RenderSystem {
    window: winit::window::Window,
}

impl System for RenderSystem {
    fn execute(&mut self, _: &Vec<&Box<Entity>>) {}
}

impl RenderSystem {
    pub fn new(nse: &NSE) -> Self {
        let window = RenderSystem::init_window(&nse.event_loop);

        RenderSystem {
            window,
        }
    }

    fn init_window(event_loop: &winit::event_loop::EventLoop<()>) -> winit::window::Window {
        winit::window::WindowBuilder::new()
            .with_title(WINDOW_TITLE)
            .with_inner_size((WINDOW_WIDTH, WINDOW_HEIGHT).into())
            .build(event_loop)
            .expect("Failed to create window.")
    }
}
