use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use crate::core::Component;
use crate::rendering::RenderSystem;
use crate::rendering::renderer::Renderer;

use gfx_hal::adapter::PhysicalDevice;
use std::mem::ManuallyDrop;
use std::{mem, ptr, iter};
use gfx_hal::device::Device;

use gfx_hal::Backend as m;
use std::any::Any;
use std::rc::Rc;
use gfx_hal::buffer::IndexBufferView;
use crate::rendering::utility::{MeshGenerator, ResourceManager};

static mut LAST_MESH_ID: AtomicU64 = AtomicU64::new(0);

pub type MeshID = u64;

#[derive(Clone)]
pub struct Mesh {
    pub id: MeshID
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
        let id;
        unsafe {
            id = LAST_MESH_ID.fetch_add(1, Relaxed);
        }

        let resource_manager = &mut render_system.lock().unwrap().resource_manager;
        T::generate::<T, _>(id, resource_manager);

        Mesh {
            id,
        }
    }

}