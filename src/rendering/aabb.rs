#[cfg(feature = "dx11")]
pub extern crate gfx_backend_dx11 as Backend;
#[cfg(feature = "dx12")]
pub extern crate gfx_backend_dx12 as Backend;
#[cfg(not(any(
    feature = "vulkan",
    feature = "dx12",
    feature = "metal",
    feature = "gl",
    feature = "wgl"
)))]
pub extern crate gfx_backend_empty as Backend;
#[cfg(any(feature = "gl", feature = "wgl"))]
pub extern crate gfx_backend_gl as Backend;
#[cfg(feature = "metal")]
pub extern crate gfx_backend_metal as Backend;
#[cfg(feature = "vulkan")]
pub extern crate gfx_backend_vulkan as Backend;

use std::fmt::{Debug, Error, Formatter};
use std::sync::{Arc, Mutex};

use cgmath::num_traits::{Float};
use cgmath::Vector3;

use crate::core::Component;
use crate::rendering::{GPUMesh, MeshGenerator, MeshID, RenderSystem, Vertex};

#[derive(Clone)]
pub struct AABB {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,

    pub debug_mesh: Option<(MeshID, Arc<GPUMesh<Backend::Backend>>)>,
}

impl Debug for AABB {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        f.debug_tuple("").field(&self.min).field(&self.max).finish()
    }
}

impl Component for AABB {}

impl AABB {
    pub fn new(min: Vector3<f32>, max: Vector3<f32>) -> Self {
        AABB {
            min: Vector3 {
                x: min.x,
                y: min.y,
                z: min.z,
            },
            max: Vector3 {
                x: max.x,
                y: max.y,
                z: max.z,
            },
            debug_mesh: None,
        }
    }

    pub fn intersect(&self, other: &AABB) -> bool {
        return (self.min.x <= other.min.x && self.max.x >= other.max.x)
            && (self.min.y <= other.min.y && self.max.y >= other.max.y)
            && (self.min.z <= other.min.z && self.max.z >= other.max.z);
    }

    pub fn update_debug_mesh(&mut self, render_system: &Arc<Mutex<RenderSystem>>) -> MeshID {
        let rend_lock = render_system.lock().unwrap();
        let mut rm_lock = rend_lock.resource_manager.lock().unwrap();
        let gpu_mesh = GPUMesh::new_dynamic(&rm_lock.device, &rm_lock.adapter, self);
        let (id, mesh) = rm_lock.add_mesh(gpu_mesh);

        self.debug_mesh = Some((id, mesh.clone()));

        id
    }
}

impl MeshGenerator for AABB {
    fn get_vertices_dynamic(&self) -> Vec<Vertex> {
        vec![
            Vertex::new(
                [self.min.x, self.min.y, self.min.z],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [self.min.x, self.min.y, self.max.z],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [self.min.x, self.max.y, self.min.z],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [self.min.x, self.max.y, self.max.z],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [self.max.x, self.min.y, self.min.z],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [self.max.x, self.max.y, self.min.z],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [self.max.x, self.min.y, self.max.z],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
            Vertex::new(
                [self.max.x, self.max.y, self.max.z],
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
            ),
        ]
    }

    fn get_indices_dynamic(&self) -> Vec<u32> {
        vec![
            // left
            0, 1, 2, 2, 1, 3, // right
            4, 5, 6, 5, 7, 6, // front
            1, 6, 7, 1, 7, 3, // back
            0, 2, 5, 4, 0, 5, // top
            3, 7, 2, 2, 7, 5, // bottom
            0, 6, 1, 0, 4, 6,
        ]
    }
}
