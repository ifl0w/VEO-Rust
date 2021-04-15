use std::sync::{Arc, Mutex};
use std::time::Duration;

use nse;
use nse::core::{Entity, Exit, Filter, Message, System};
use nse::NSE;

#[derive(Debug)]
struct NoopSystem {
    counter: u64, // how often noop should be executed
}

impl NoopSystem {
    fn new(count: u64) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(NoopSystem { counter: count }))
    }
}

impl System for NoopSystem {
    fn execute(&mut self, entities: &Vec<Arc<Mutex<Filter>>>, _: Duration) {
        if self.counter == 0 {
            println!("Stop");
        } else {
            println!(
                "Noop - Entities: {} - Counter {}",
                entities.len(),
                self.counter
            );
            self.counter -= 1;
        }
    }

    fn get_messages(&mut self) -> Vec<Message> {
        return if self.counter == 0 {
            vec![Message::new(Exit {})]
        } else {
            vec![]
        };
    }
}

fn main() {
    let mut engine: NSE = NSE::new();

    let noop_system = NoopSystem::new(100);

    engine.add_system(&noop_system);

    let e1 = Entity::new();
    e1.lock().unwrap().name = String::from("Entity 1");
    engine.remove_entity(e1);

    for _i in 1..10 {
        let e = Entity::new();

        {
            let mut e_ref = e.lock().unwrap();
            e_ref.name = String::from(format!("Entity {}", e_ref.id));
        }

        engine.add_entity(e);
    }

    engine.run();
}
