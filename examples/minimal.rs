use nse;
use nse::NSE;
use nse::core::Entity;

fn main() {
    let mut engine: NSE = NSE::new();

    let mut e1 = Box::new(Entity::new());
    e1.name = String::from("Entity 1");

    for i in 1 .. 10 {
        let mut e= Box::new(Entity::new());
        e.name = String::from(format!("Entity {}", e.id));
        engine.entity_manager.add_entity(&e);
    }

    engine.entity_manager.list_entities();

    engine.entity_manager.remove_entity(&e1);

    engine.entity_manager.list_entities();
}
