use cgmath::Vector3;

use nse;
use nse::core::Entity;
use nse::rendering::{Camera, Cube, Mesh, RenderSystem, Transformation};
use nse::NSE;

use crate::shared::fps_camera_system::FPSCameraSystem;

mod shared;

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = RenderSystem::new(&engine);
    let fps_camera_system = FPSCameraSystem::new();
    fps_camera_system.lock().unwrap().set_mouse_speed(1.0);
    fps_camera_system.lock().unwrap().set_movement_speed(30.0);

    engine.add_system(&render_system);
    engine.add_system(&fps_camera_system);

    // add camera
    let camera = Entity::new();
    camera
        .lock()
        .unwrap()
        .add_component(Camera::new(0.1, 1000.0, 90.0, [800.0, 600.0]))
        .add_component(Transformation::new().position(Vector3::new(0.0, 0.0, 3.0)));
    engine.add_entity(camera);

    // add cubes
    let cube_mesh = Mesh::new::<Cube>(&render_system);

    let num_cubes = 0..25;
    let offset = Vector3::new(
        -num_cubes.end as f32,
        -num_cubes.end as f32,
        -num_cubes.end as f32 * 4.0,
    );
    for x in num_cubes.clone() {
        for y in num_cubes.clone() {
            for z in num_cubes.clone() {
                let mut position = Vector3::new(x as f32 * 2.0, y as f32 * 2.0, z as f32 * 2.0);
                position += offset;

                let entity = Entity::new();
                entity
                    .lock()
                    .unwrap()
                    .add_component(cube_mesh.clone())
                    .add_component(Transformation::new().position(position));
                engine.add_entity(entity);
            }
        }
    }

    engine.run();
}
