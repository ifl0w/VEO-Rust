use nse;
use nse::NSE;
use nse::core::{Entity, System, Message, ExitMessage};
use std::any::TypeId;

#[derive(Debug)]
struct NoopSystem {
    counter: u64, // how often noop should be executed
}

impl NoopSystem {
    fn new(count: u64) -> Self {
        NoopSystem {
            counter: count
        }
    }
}

impl System for NoopSystem {
    fn execute(&mut self, entities: &Vec<&Box<Entity>>) {
        if self.counter == 0 {
            println!("Stop");
        } else {
            println!("Noop - Entities: {} - Counter {}", entities.len(), self.counter);
            self.counter -= 1;
        }
    }

    fn get_messages(&mut self) -> Vec<Message> {
        if self.counter == 0 {
            return vec![Message{ code: TypeId::of::<ExitMessage>() }];
        } else {
            return vec![];
        }
    }
}

fn main() {
    let mut engine: NSE = NSE::new();

    let noop_system = Box::new(NoopSystem::new(100));

    engine.system_manager.add_system(noop_system);

    let mut e1 = Box::new(Entity::new());
    e1.name = String::from("Entity 1");
    engine.entity_manager.remove_entity(&e1);

    for _i in 1 .. 10 {
        let mut e= Box::new(Entity::new());
        e.name = String::from(format!("Entity {}", e.id));
        engine.entity_manager.add_entity(&e);
    }

    engine.run();
}
