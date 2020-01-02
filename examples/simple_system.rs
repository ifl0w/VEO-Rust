use nse;
use nse::core::{Component, Entity, Filter, System};
use nse::NSE;
use nse::rendering::RenderSystem;

#[derive(Debug)]
struct TestSystem {}

impl TestSystem {
    fn new() -> Self {
        TestSystem {}
    }
}

impl System for TestSystem {
    fn get_filter(&mut self) -> Option<Filter> {
        nse::filter!(Position)
    }

    fn execute(&mut self, entities: &Vec<&Box<Entity>>) {
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
    engine.system_manager.add_system(render_system);

    // Test System
    let test_system = Box::new(TestSystem::new());
    engine.system_manager.add_system(test_system);

    for _i in 1..10 {
        let mut ent1 = Box::new(Entity::new());
        ent1.name = String::from(format!("Entity {}", ent1.id));
        ent1.add_component(Position { x: 0.0, y: 0.0, z: 0.0 });
        engine.entity_manager.add_entity(&ent1);

        let mut ent2 = Box::new(Entity::new());
        ent2.name = String::from(format!("Entity {}", ent2.id));
        engine.entity_manager.add_entity(&ent2);
    }

    engine.run();
}
