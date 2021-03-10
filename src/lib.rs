extern crate cgmath;
#[macro_use]
extern crate gfx_hal;
#[macro_use]
extern crate mopa;
extern crate winit;

use std::any::TypeId;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use winit::event::Event;
use winit::event_loop::{ControlFlow, EventLoop};

use crate::core::{EntityManager, EntityRef, Exit, Message, System};
use crate::core::MessageManager;
use crate::core::SystemManager;
use crate::rendering::nse_gui::octree_gui::ProfilingData;

//use winit::{Event, EventsLoop, WindowEvent};

pub mod core;
pub mod rendering;

pub struct NSE {
    message_manager: Arc<Mutex<MessageManager>>,
    entity_manager: Arc<Mutex<EntityManager>>,
    system_manager: Arc<Mutex<SystemManager>>,

    event_loop: EventLoop<()>,

    delta_time: Arc<Mutex<Duration>>,
}

impl NSE {
    pub fn new() -> Self {
        #[cfg(debug_assertions)]
        println!("Debugging enabled ... ");

        env_logger::init();

        let nse = NSE {
            message_manager: Arc::new(Mutex::new(MessageManager::new())),
            entity_manager: Arc::new(Mutex::new(EntityManager::new())),
            system_manager: Arc::new(Mutex::new(SystemManager::new())),

            event_loop: EventLoop::new(),

            delta_time: Arc::new(Mutex::new(Duration::new(0, 0))),
        };

        nse
    }

    pub fn init() {
        println!("Initializing...")
    }

    pub fn run(self) {
        let system_manager = self.system_manager.clone();
        let message_manager = self.message_manager.clone();
        let delta_time = self.delta_time.clone();

        let mut profiling_data = ProfilingData::default();

        self.event_loop.run(move |event, _, control_flow| {
            let system_manager_lock = system_manager.lock().unwrap();

            let frame_start = Instant::now();
            let mut exit = false;
            let systems = system_manager_lock.systems.clone();

            for (tid, sys) in systems {
                sys.lock().unwrap().handle_input(&event);
            }
            match event {
                Event::RedrawRequested(_) => {
                    let v: Vec<_> = message_manager
                        .lock()
                        .unwrap()
                        .receiver
                        .try_iter()
                        .collect();

                    for msg in v.iter() {
                        if msg.is_type::<Exit>() {
                            exit = true;
                        }
                    }

                    if exit {
                        *control_flow = ControlFlow::Exit;
                        println!("Exiting NSE");
                        return;
                    }

                    let sys_iterator = system_manager_lock.systems.iter();
                    let mut msgs: Vec<Message> = vec![];

                    let mut sys_times = Vec::with_capacity(sys_iterator.len());

                    for (system_id, sys) in sys_iterator {
                        let filter = system_manager_lock.get_filter(&system_id).unwrap();

                        // everything of the system should be run here
                        let system_start = Instant::now();

                        sys.lock().unwrap().consume_messages(&v);
                        sys.lock()
                            .unwrap()
                            .execute(filter, *delta_time.lock().unwrap());
                        msgs.append(&mut sys.lock().unwrap().get_messages());

                        // system has completed execution
                        let system_end = Instant::now();
                        sys_times.push((
                            system_manager_lock.get_system_name(system_id).into(),
                            system_end - system_start,
                        ));
                    }

                    // store profiling data
                    profiling_data.system_times = Some(sys_times);
                    msgs.push(Message::new(profiling_data.clone()));

                    for msg in msgs.iter() {
                        let result = message_manager.lock().unwrap().sender.send(msg.clone());
                        result.expect("Sending message failed");
                    }

                    let frame_end = Instant::now();

                    *delta_time.lock().unwrap() = frame_end - frame_start
                }
                _ => (),
            }
        });
    }

    pub fn add_system<T: 'static + System>(&mut self, sys: &Arc<Mutex<T>>) {
        self.add_system_with_name(std::any::type_name::<T>(), sys)
    }

    pub fn add_system_with_name<T: 'static + System>(&mut self, name: &str, sys: &Arc<Mutex<T>>) {
        let typeid = TypeId::of::<T>();
        let mut system_manager_lock = self.system_manager.lock().unwrap();

        system_manager_lock.add_system_with_name(name.into(), sys);

        let entities = &self
            .entity_manager
            .lock()
            .unwrap()
            .entities
            .values()
            .cloned()
            .collect();
        let filter = system_manager_lock.get_filter(&typeid).unwrap();
        filter
            .iter()
            .for_each(|f| f.lock().unwrap().update(entities));
    }

    pub fn add_entity(&mut self, e: EntityRef) {
        self.entity_manager.lock().unwrap().add_entity(e.clone());
        self.system_manager
            .lock()
            .unwrap()
            .filter
            .iter()
            .for_each(|(_, filters)| {
                for filter in filters {
                    filter.lock().unwrap().add(e.clone());
                }
            })
    }

    pub fn remove_entity(&mut self, e: EntityRef) {
        self.entity_manager.lock().unwrap().remove_entity(e.clone());
        self.system_manager
            .lock()
            .unwrap()
            .filter
            .iter()
            .for_each(|(_, filters)| {
                for filter in filters {
                    filter.lock().unwrap().remove(e.clone());
                }
            })
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }
}
