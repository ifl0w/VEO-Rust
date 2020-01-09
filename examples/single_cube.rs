use nse;
use nse::core::{Entity};
use nse::NSE;
use nse::rendering::{Cube, Mesh, RenderSystem};

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = RenderSystem::new(&engine);

    engine.system_manager.add_system(render_system.clone());

    let entity = Entity::new();
    entity.lock().unwrap()
        .add_component(Mesh::new::<Cube>(&render_system.lock().unwrap()));

    engine.entity_manager.add_entity(entity);

    engine.run();
}
