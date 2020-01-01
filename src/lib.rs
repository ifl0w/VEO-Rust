use std::iter::FromIterator;

use crate::core::{Entity, EntityManager, Exit, Message};
use crate::core::MessageManager;
use crate::core::SystemManager;

#[macro_use]
extern crate mopa;

pub mod core;

pub struct NSE {
    pub message_manager: MessageManager,
    pub entity_manager: EntityManager,
    pub system_manager: SystemManager,
}

impl NSE {
    pub fn new() -> Self {
        let nse = NSE {
            message_manager: MessageManager::new(),
            entity_manager: EntityManager::new(),
            system_manager: SystemManager::new(),
        };

        nse
    }

    pub fn init() {
        println!("Initializing...")
    }

    pub fn run(&mut self) {
        loop {
            let v: Vec<_> = self.message_manager.receiver.try_iter().collect();
            for msg in v.iter() {
                if msg.is_type::<Exit>() {
                    println!("Exiting NSE");
                    return;
                }
            }

            let iter = self.system_manager.systems.iter_mut();
            let mut msgs: Vec<Message> = vec![];

            let entities: Vec<&Box<Entity>> = Vec::from_iter(self.entity_manager.entities.iter());

            for (_, sys) in iter {
                sys.consume_messages(&v);
                sys.execute(&entities);
                msgs.append(&mut sys.get_messages());
            }

            self.send_messages(&msgs)
        }
    }

    fn send_messages(&mut self, msgs: &Vec<Message>) {
        for msg in msgs.iter() {
            let result = self.message_manager.sender.send(msg.clone());
            result.expect("Sending message failed");
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
