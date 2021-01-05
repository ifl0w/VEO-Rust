use cgmath::{vec3, Deg, Euler, Matrix4, Quaternion, Rad, SquareMatrix, Vector3};

use nse;
use nse::core::Entity;
use nse::rendering::{utility::Cube, Camera, Frustum, Mesh, RenderSystem, Transformation, AABB};
use nse::NSE;

use crate::shared::fps_camera_system::FPSCameraSystem;
use glium::RawUniformValue::Vec3;

mod shared;

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = RenderSystem::new(&engine);
    let fps_camera_system = FPSCameraSystem::new();

    fps_camera_system.lock().unwrap().set_mouse_speed(2.0);

    //    let aspect = 1024.0 / 768.0;
    //
    //    let fov_y = Rad::from(Deg(90.0 / aspect));
    //    let fov_x = Rad::from(Deg(90.0));
    //
    //    let mut f = Frustum::new(fov_x, fov_y, 1.0, 10.0)
    //        .transformed(Matrix4::identity()); //Matrix4::from_translation(vec3(10.0,10.0,10.0)));
    //
    //    let mut aabb = AABB::new(vec3(-1.0,-10.0, -6.0), vec3(1.0,-5.0, -2.0));
    //    aabb.update_debug_mesh(&render_system);
    //    f.update_debug_mesh(&render_system);
    //
    //    let intersect = f.intersect(&aabb);
    //
    //    println!("intersect: {}", intersect);

    engine.add_system(&render_system);
    engine.add_system(&fps_camera_system);

    let mut entity = Entity::new();
    entity
        .lock()
        .unwrap()
        .add_component(Mesh::new::<Cube>(&render_system))
        .add_component(Transformation::new());
    engine.add_entity(entity);

    let camera = Entity::new();
    camera
        .lock()
        .unwrap()
        .add_component(Camera::new(0.1, 1000.0, 90.0, [800.0, 600.0]))
        .add_component(Transformation::new().position(Vector3::new(0.0, 0.0, 3.0)));
    engine.add_entity(camera);

    engine.run();
}
