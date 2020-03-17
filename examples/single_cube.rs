use cgmath::{Deg, Euler, Quaternion, Vector3, Rad, vec3, Matrix4, SquareMatrix};

use nse;
use nse::core::Entity;
use nse::NSE;
use nse::rendering::{Camera, Mesh, RenderSystem, Transformation, utility::Cube, Frustum, AABB};

use crate::shared::fps_camera_system::FPSCameraSystem;

mod shared;

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = RenderSystem::new(&engine);
    let fps_camera_system = FPSCameraSystem::new();

    fps_camera_system.lock().unwrap().set_mouse_speed(2.0);

    let f = Frustum::new(Rad::from(Deg(90.0f32)), Rad::from(Deg(90.0f32)), 1.0, 10.0);
    let mut aabb = AABB::new(vec3(-1.0,-1.0, -6.0), vec3(1.0,1.0, -3.0));
    aabb.update_debug_mesh(&render_system);

    let intersect = f.transformed(Matrix4::identity()).intersect(&aabb);

    engine.add_system(render_system.clone());
    engine.add_system(fps_camera_system.clone());

    let mut entity = Entity::new();
    entity.lock().unwrap()
        .add_component(Mesh::new::<Cube>(&render_system))
        .add_component(Transformation::new());
    engine.add_entity(entity);

    entity = Entity::new();
    entity.lock().unwrap()
        .add_component(aabb);
    engine.add_entity(entity);

    let camera = Entity::new();
    camera.lock().unwrap()
        .add_component(Camera::new(0.1, 1000.0, 90.0, [800.0, 600.0]))
        .add_component(Transformation::new().position(Vector3::new(0.0, 0.0, 3.0)));
    engine.add_entity(camera);

    engine.run();
}
