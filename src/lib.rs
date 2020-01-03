extern crate cgmath;
#[macro_use]
extern crate mopa;
#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;


use std::iter::FromIterator;

use winit::{Event, EventsLoop, WindowEvent};

use crate::core::{EntityManager, Exit, Message};
use crate::core::MessageManager;
use crate::core::SystemManager;

pub mod core;
pub mod rendering;

pub struct NSE {
    pub message_manager: MessageManager,
    pub entity_manager: EntityManager,
    pub system_manager: SystemManager,
    pub event_loop: EventsLoop,
}

impl NSE {
    pub fn new() -> Self {
        let nse = NSE {
            message_manager: MessageManager::new(),
            entity_manager: EntityManager::new(),
            system_manager: SystemManager::new(),
            event_loop: EventsLoop::new(),
        };

        nse
    }

    pub fn init() {
        println!("Initializing...")
    }

    pub fn run(&mut self) {
        loop {
            let mut exit = false;

            self.event_loop.poll_events(|event| {
                match event {
                    | Event::WindowEvent { event, .. } => {
                        match event {
                            | WindowEvent::CloseRequested => {
                                exit = true;
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
            });

            let v: Vec<_> = self.message_manager.receiver.try_iter().collect();
            for msg in v.iter() {
                if msg.is_type::<Exit>() {
//                *control_flow = ControlFlow::Exit;
                    exit = true;
                }
            }

            if exit {
                println!("Exiting NSE");
                return;
            }


            let iter = self.system_manager.systems.iter_mut();
            let mut msgs: Vec<Message> = vec![];

//            let entities = em.entities.iter();

            for (_, sys) in iter {
                let sys_entities;
                match sys.get_filter() {
                    Some(f) => {
                        sys_entities = self.entity_manager.entities.iter()
                            .filter(|e| e.match_filter(&f))
                            .collect::<Vec<_>>();
                    }
                    None => {
                        sys_entities = Vec::from_iter(self.entity_manager.entities.iter());
                    }
                };

                sys.consume_messages(&v);
                sys.execute(&sys_entities);
                msgs.append(&mut sys.get_messages());
            }

            for msg in msgs.iter() {
                let result = self.message_manager.sender.send(msg.clone());
                result.expect("Sending message failed");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
