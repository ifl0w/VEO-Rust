#[macro_use]
extern crate mopa;

use std::iter::FromIterator;

use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;

use crate::core::{Entity, EntityManager, Exit, Message};
use crate::core::MessageManager;
use crate::core::SystemManager;

pub mod core;
pub mod rendering;

pub struct NSE {
    pub message_manager: MessageManager,
    pub entity_manager: EntityManager,
    pub system_manager: SystemManager,
    pub event_loop: winit::event_loop::EventLoop<()>,
}

impl NSE {
    pub fn new() -> Self {
        let nse = NSE {
            message_manager: MessageManager::new(),
            entity_manager: EntityManager::new(),
            system_manager: SystemManager::new(),
            event_loop: winit::event_loop::EventLoop::new(),
        };

        nse
    }

    pub fn init() {
        println!("Initializing...")
    }

    pub fn run(self) {
        let mm = self.message_manager;
        let mut sm = self.system_manager;
        let em = self.entity_manager;

        self.event_loop.run(move |event, _, control_flow| {
            match event {
                | Event::WindowEvent { event, .. } => {
                    match event {
                        | WindowEvent::CloseRequested => {
                            *control_flow = ControlFlow::Exit
                        }
//                        | WindowEvent::KeyboardInput { input, .. } => {
//                            match input {
//                                | KeyboardInput { virtual_keycode, state, .. } => {
//                                    match (virtual_keycode, state) {
//                                        | (Some(VirtualKeyCode::Escape), ElementState::Pressed) => {
//                                            dbg!();
//                                            *control_flow = ControlFlow::Exit
//                                        },
//                                        | _ => {},
//                                    }
//                                },
//                            }
//                        },
                        | _ => {}
                    }
                }
                _ => (),
            }

            let v: Vec<_> = mm.receiver.try_iter().collect();
            for msg in v.iter() {
                if msg.is_type::<Exit>() {
                    println!("Exiting NSE");
                    *control_flow = ControlFlow::Exit;
                    return;
                }
            }

            let iter = sm.systems.iter_mut();
            let mut msgs: Vec<Message> = vec![];

            let entities: Vec<&Box<Entity>> = Vec::from_iter(em.entities.iter());

            for (_, sys) in iter {
                sys.consume_messages(&v);
                sys.execute(&entities);
                msgs.append(&mut sys.get_messages());
            }

            for msg in msgs.iter() {
                let result = mm.sender.send(msg.clone());
                result.expect("Sending message failed");
            }
        });
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
