use std::sync::{Arc, Mutex};
use std::time::Duration;

use nse;
use nse::core::{Component, Entity, Filter, System};
use nse::NSE;
use nse::rendering::RenderSystem;

#[derive(Debug)]
struct TestSystem {}

impl TestSystem {
    fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(TestSystem {}))
    }
}

impl System for TestSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![nse::filter!(Position)]
    }

    fn execute(&mut self, entities: &Vec<Arc<Mutex<Filter>>>, _: Duration) {
        println!("{}", entities.iter().count())
    }
}

#[derive(Debug, Copy, Clone)]
struct Position {
    x: f64,
    y: f64,
    z: f64,
}

impl Component for Position {}

fn main() {
    let mut engine: NSE = NSE::new();

    // Render System
    let render_system = Box::new(RenderSystem::new(&engine));
    engine.add_system(&render_system);

    // Test System
    let test_system = Box::new(TestSystem::new());
    engine.add_system(&test_system);

    for _i in 1..10 {
        let ent1 = Entity::new();
        {
            let mut e = ent1.lock().unwrap();
            e.name = String::from(format!("Entity {}", e.id));
        }
        engine.add_entity(ent1);

        let ent2 = Entity::new();
        {
            let mut e = ent2.lock().unwrap();
            e.name = String::from(format!("Entity {}", e.id));
        }
        engine.add_entity(ent2);
    }

    engine.run();
}
