use crate::core::{System, Entity, Message, Exit};
use winit::event::{WindowEvent, Event};
use winit::platform::desktop::EventLoopExtDesktop;
use winit::event_loop::ControlFlow;

// Constants
const WINDOW_TITLE: &'static str = "00.Base Code";
const WINDOW_WIDTH: u32 = 800;
const WINDOW_HEIGHT: u32 = 600;

#[derive(Debug)]
pub struct RenderSystem {
    window: winit::window::Window,
    exit: bool
}

impl System for RenderSystem {
    fn execute(&mut self, _: &Vec<&Box<Entity>>) {
        // TODO: do it differently
//        self.event_loop.run_return(|event, _, control_flow| {
//            match event {
//                | Event::WindowEvent { event, .. } => {
//                    match event {
//                        | WindowEvent::CloseRequested => {
//                            self.exit = true;
//                            *control_flow = ControlFlow::Exit
//                        },
////                        | WindowEvent::KeyboardInput { input, .. } => {
////                            match input {
////                                | KeyboardInput { virtual_keycode, state, .. } => {
////                                    match (virtual_keycode, state) {
////                                        | (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
////                                            dbg!();
////                                            *control_flow = ControlFlow::Exit
////                                        },
////                                        | _ => {},
////                                    }
////                                },
////                            }
////                        },
//                        | _ => {},
//                    }
//                },
//                _ => (),
//            }
//        });

    }

    fn get_messages(&mut self) -> Vec<Message> {
        if self.exit {
            return vec![Message::new(Box::new(Exit {}))]
        }

        vec![]
    }
}

impl RenderSystem {

    pub fn new(event_loop: &winit::event_loop::EventLoop<()>) -> Self {

        let window = RenderSystem::init_window(event_loop);

        RenderSystem {
            window,
            exit: false
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
