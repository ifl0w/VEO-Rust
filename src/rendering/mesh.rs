use std::{iter, mem, ptr};
use std::any::Any;
use std::borrow::{Borrow, BorrowMut};
use std::hash::{Hash, Hasher};
use std::mem::ManuallyDrop;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use gfx_hal::adapter::PhysicalDevice;
use gfx_hal::Backend as m;
use gfx_hal::buffer::IndexBufferView;
use gfx_hal::device::Device;

use crate::core::Component;
use crate::rendering::renderer::Renderer;
use crate::rendering::{RenderSystem, GPUMesh};
use crate::rendering::utility::{MeshGenerator, ResourceManager};
use crate::rendering::utility::resources::MeshID;

#[cfg(feature = "dx11")]
pub extern crate gfx_backend_dx11 as Backend;
#[cfg(feature = "dx12")]
pub extern crate gfx_backend_dx12 as Backend;
#[cfg(
not(any(
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

static mut LAST_MESH_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone)]
pub struct Mesh {
    pub id: MeshID,
    pub mesh: Arc<GPUMesh<Backend::Backend>>
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
        let (id, mesh) = T::generate::<T, _>(resource_manager.borrow_mut());

        Mesh {
            id,
            mesh,
        }
    }
}