use cgmath::{Vector3, Quaternion, Euler, Deg};

use nse;
use nse::core::Entity;
use nse::NSE;
use nse::rendering::{Camera, Cube, Mesh, Octree, OctreeGuiSystem, OctreeSystem, RenderSystem, Transformation};

use crate::shared::fps_camera_system::FPSCameraSystem;

pub mod shared;

fn main() {
    let mut engine: NSE = NSE::new();

    let render_system = RenderSystem::new(&engine);
    let fps_camera_system = FPSCameraSystem::new();
    fps_camera_system.lock().unwrap().set_mouse_speed(3.0);
    fps_camera_system.lock().unwrap().set_movement_speed(5.0);

    let octree_sys = OctreeSystem::new(render_system.clone());
    let octree_gui_system = OctreeGuiSystem::new(&engine, render_system.clone());

    engine.add_system(render_system.clone());
    engine.add_system(fps_camera_system.clone());
    engine.add_system(octree_sys.clone());
    engine.add_system(octree_gui_system.clone());

    // add camera
    let camera = Entity::new();
    camera.lock().unwrap()
        .add_component(Camera::new(0.1, 1000.0, 90.0, [800.0, 600.0]))
        .add_component(Transformation::new()
            .position(Vector3::new(0.0, 10.0, 30.0))
            .rotation(Quaternion::from(Euler::new(Deg(-15.0), Deg(0.0), Deg(0.0))))
        );
    engine.add_entity(camera);

    // add octree
    let octree = Entity::new();
    let size = Vector3::new(25.0, 25.0, 25.0);
    octree.lock().unwrap()
        .add_component(Octree::new(&render_system, 4, Some(size)))
        .add_component(Transformation::new().position(size / 2.0))
        .add_component(Mesh::new::<Cube>(&render_system));
    engine.add_entity(octree);

    engine.run();
}
