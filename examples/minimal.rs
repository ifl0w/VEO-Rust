use nse;
use nse::NSE;
use nse::core::Entity;

fn main() {
    let mut engine: NSE = NSE::new();

    let mut e1: Entity = Entity::new();
    e1.name = String::from("Entity 1");

    let mut e2: Entity = Entity::new();
    e2.name = String::from("Entity 2");

    engine.entity_manager.add_entity(&e1);
    engine.entity_manager.add_entity(&e2);

    engine.entity_manager.list_entities();

    engine.entity_manager.remove_entity(&e1);

    engine.entity_manager.list_entities();
}
