use cgmath::{Deg, Euler, Quaternion, Vector3};

use nse;
use nse::core::{Entity};
use nse::NSE;
use nse::rendering::{Camera, Cube, Mesh, RenderSystem, Transformation};

mod shared;

use crate::shared::fps_camera_system::FPSCameraSystem;

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = RenderSystem::new(&engine);
    let fps_camera_system = FPSCameraSystem::new();

    engine.system_manager.add_system(render_system.clone());
    engine.system_manager.add_system(fps_camera_system.clone());

    let entity = Entity::new();
    entity.lock().unwrap()
        .add_component(Mesh::new::<Cube>(&render_system.lock().unwrap()))
        .add_component(Transformation{ .. Default::default() });
    engine.entity_manager.add_entity(entity);

    let camera = Entity::new();
    camera.lock().unwrap()
        .add_component(Camera::new(0.1, 1000.0, 90.0, [800.0, 600.0]))
        .add_component(Transformation {
            position: Vector3::new(0.0, 0.0, 3.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            rotation: Quaternion::from(Euler {
                x: Deg(0.0),
                y: Deg(0.0),
                z: Deg(0.0),
            }),
        });
    engine.entity_manager.add_entity(camera);

    engine.run();
}
