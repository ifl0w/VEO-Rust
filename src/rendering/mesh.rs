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
#[cfg(feature = "vulkan")]
pub extern crate gfx_backend_vulkan as Backend;

use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicU64;

use crate::core::Component;
use crate::rendering::{GPUMesh, RenderSystem};
use crate::rendering::utility::MeshGenerator;
use crate::rendering::utility::resources::MeshID;

static mut LAST_MESH_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct Mesh {
    pub id: MeshID,
    pub mesh: Arc<GPUMesh<Backend::Backend>>,
}

impl Component for Mesh {}

impl Hash for Mesh {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.id.hash(state);
    }
}

impl PartialEq for Mesh {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl Eq for Mesh {}

impl Mesh {
    pub fn new<T: MeshGenerator>(render_system: &Arc<Mutex<RenderSystem>>) -> Self {
        let render_system = render_system.lock().unwrap();
        let mut resource_manager = render_system.resource_manager.lock().unwrap();

        let gpu_mesh = GPUMesh::new::<T>(&resource_manager.device, &resource_manager.adapter);
        let (id, mesh) = resource_manager.add_mesh(gpu_mesh);

        Mesh { id, mesh }
    }

    pub fn new_dynamic<T: MeshGenerator>(
        _generator_instance: T,
        render_system: &Arc<Mutex<RenderSystem>>,
    ) -> Self {
        let render_system = render_system.lock().unwrap();
        let mut resource_manager = render_system.resource_manager.lock().unwrap();

        let gpu_mesh = GPUMesh::new::<T>(&resource_manager.device, &resource_manager.adapter);
        let (id, mesh) = resource_manager.add_mesh(gpu_mesh);

        Mesh { id, mesh }
    }
}
