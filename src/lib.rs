extern crate cgmath;
#[macro_use]
extern crate mopa;
#[macro_use]
extern crate vulkano;
extern crate vulkano_win;
extern crate winit;

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use winit::{Event, EventsLoop, WindowEvent};

use crate::core::{EntityManager, Exit, Message, System};
use crate::core::MessageManager;
use crate::core::SystemManager;

pub mod core;
pub mod rendering;

pub struct NSE {
    pub message_manager: MessageManager,
    pub entity_manager: EntityManager,
    pub system_manager: SystemManager,
    pub event_loop: EventsLoop,

    delta_time: Duration,
}

impl NSE {
    pub fn new() -> Self {
        let nse = NSE {
            message_manager: MessageManager::new(),
            entity_manager: EntityManager::new(),
            system_manager: SystemManager::new(),
            event_loop: EventsLoop::new(),

            delta_time: Duration::new(0, 0),
        };

        nse
    }

    pub fn init() {
        println!("Initializing...")
    }

    pub fn run(&mut self) {
        loop {
            let frame_start = Instant::now();
            let mut exit = false;
            let systems = &self.system_manager.systems.values().cloned().collect::<Vec<Arc<Mutex<dyn System>>>>();

            self.event_loop.poll_events(|event| {
                for sys in systems {
                    sys.lock().unwrap().handle_input(&event);
                }
                match event {
                    | Event::WindowEvent { event, .. } => {
                        match event {
                            | WindowEvent::CloseRequested => {
                                exit = true;
                            }
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


            let sys_iterator = self.system_manager.systems.iter();
            let mut msgs: Vec<Message> = vec![];

            let entities = self.entity_manager.entities.values().cloned().collect();

            for (typeid, sys) in sys_iterator {
                let filter = self.system_manager.get_filter(&typeid).unwrap();
                filter.iter().for_each(|f| f.lock().unwrap().update(&entities));

                sys.lock().unwrap().consume_messages(&v);
                sys.lock().unwrap().execute(filter, self.delta_time);
                msgs.append(&mut sys.lock().unwrap().get_messages());
            }

            for msg in msgs.iter() {
                let result = self.message_manager.sender.send(msg.clone());
                result.expect("Sending message failed");
            }

            let frame_end = Instant::now();
            self.delta_time = frame_end - frame_start;

//            println!("Frame time: {} ", self.delta_time.as_millis())
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
