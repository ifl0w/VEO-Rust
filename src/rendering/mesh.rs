use std::sync::Arc;

use cgmath::Vector3;
use vulkano::buffer::{
    BufferAccess,
    BufferUsage,
    immutable::ImmutableBuffer,
    TypedBufferAccess,
};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;

use crate::core::Component;
use crate::rendering::RenderSystem;

use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::Relaxed;

use std::hash::{Hash, Hasher};

#[derive(Copy, Clone, Default)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn new(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self { position, normal, color }
    }
}

#[allow(clippy: ref_in_deref)]
vulkano::impl_vertex!(Vertex, position, normal, color);

#[derive(Default, Debug, Copy, Clone)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4]
}
vulkano::impl_vertex!(InstanceData, model_matrix);

pub trait MeshGenerator {
    fn get_vertices() -> Vec<Vertex>;
    fn get_indices() -> Vec<u16>;
}

static mut LAST_MESH_ID: AtomicU64 = AtomicU64::new(0);

pub type MeshID = u64;

#[derive(Clone)]
pub struct Mesh {
    pub id: MeshID,
    pub vertex_buffer: Arc<dyn BufferAccess + Send + Sync>,
    pub index_buffer: Arc<dyn TypedBufferAccess<Content=[u16]> + Send + Sync>,
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
    pub fn new<T: MeshGenerator>(render_system: &RenderSystem) -> Self {
        let vertex_buffer = Self::create_vertex_buffer::<T>(&render_system.graphics_queue);
        let index_buffer = Self::create_index_buffer::<T>(&render_system.graphics_queue);

        let id;
        unsafe {
            id = LAST_MESH_ID.fetch_add(1, Relaxed);
        }

        Mesh {
            id,
            vertex_buffer,
            index_buffer,
        }
    }

    fn create_vertex_buffer<T: MeshGenerator>(graphics_queue: &Arc<Queue>) -> Arc<dyn BufferAccess + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            T::get_vertices().iter().cloned(), BufferUsage::vertex_buffer(),
            graphics_queue.clone())
            .unwrap();
        future.flush().unwrap();
        buffer
    }

    fn create_index_buffer<T: MeshGenerator>(graphics_queue: &Arc<Queue>) -> Arc<dyn TypedBufferAccess<Content=[u16]> + Send + Sync> {
        let (buffer, future) = ImmutableBuffer::from_iter(
            T::get_indices().iter().cloned(), BufferUsage::index_buffer(),
            graphics_queue.clone())
            .unwrap();
        future.flush().unwrap();
        buffer
    }
}

pub struct Plane {}

impl MeshGenerator for Plane {
    fn get_vertices() -> Vec<Vertex> {
        vec![
            Vertex::new([-0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            Vertex::new([-0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            Vertex::new([0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            Vertex::new([0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0])
        ]
    }

    fn get_indices() -> Vec<u16> {
        vec![0, 1, 2, 2, 3, 0]
    }
}

pub struct Cube {}

impl MeshGenerator for Cube {
    fn get_vertices() -> Vec<Vertex> {
        vec![
            // top
            Vertex::new([-0.5, 0.5, 0.5], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, 0.5, 0.5], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, 0.5, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, 0.5, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
            // bottom
            Vertex::new([0.5, -0.5, 0.5], [0.0, -1.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, -0.5, 0.5], [0.0, -1.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, -0.5, -0.5], [0.0, -1.0, 0.0], [1.0, 1.0, 1.0]),
            // left
            Vertex::new([-0.5, 0.5, 0.5], [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, 0.5, -0.5], [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, -0.5, 0.5], [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, -0.5, -0.5], [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            // right
            Vertex::new([0.5, 0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, -0.5, -0.5], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, -0.5, 0.5], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
            // front
            Vertex::new([-0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, -0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, 0.5, 0.5], [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]),
            // back
            Vertex::new([0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, -0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]),
            Vertex::new([0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]),
            Vertex::new([-0.5, 0.5, -0.5], [0.0, 0.0, -1.0], [1.0, 1.0, 1.0]),
        ]
    }

    fn get_indices() -> Vec<u16> {
        vec![
            0, 1, 2, 1, 3, 2,
            4, 5, 6, 5, 7, 6,
            8, 9, 10, 9, 11, 10,
            12, 13, 14, 13, 15, 14,
            16, 17, 18, 17, 19, 18,
            20, 21, 22, 21, 23, 22,
        ]
    }
}