use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, Weak};
use std::{iter, mem, ptr};

use gfx_hal::adapter::PhysicalDevice;
use gfx_hal::adapter::{Adapter};
use gfx_hal::{
    buffer, command, format as f,
    format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle},
    image as i, memory as m, pass,
    pass::Subpass,
    pool,
    prelude::*,
    pso,
    pso::{PipelineStage, ShaderStageFlags, VertexInputRate},
    queue::{QueueGroup, Submission},
    window, IndexType,
};

use crate::rendering::renderer::Renderer;
use crate::rendering::utility::{GPUBuffer, Index, Vertex};


pub type MeshID = usize;
pub type BufferID = usize;

pub struct ResourceManager<B>
where
    B: gfx_hal::Backend,
{
    pub(in crate::rendering) device: Arc<B::Device>,
    pub(in crate::rendering) adapter: Arc<Adapter<B>>,

    last_mesh_id: MeshID,
    last_buffer_id: MeshID,

    pub(in crate::rendering::utility) meshes: HashMap<MeshID, Weak<GPUMesh<B>>>,
    pub(in crate::rendering::utility) buffers: HashMap<BufferID, Weak<Mutex<GPUBuffer<B>>>>,
    //    pub(in crate::rendering) shaders: HashMap<ShaderID, Arc<Shader<B>>>,
}

impl<B> ResourceManager<B>
where
    B: gfx_hal::Backend,
{
    pub fn new(renderer: &Renderer<B>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(ResourceManager {
            device: renderer.device.clone(),
            adapter: renderer.adapter.clone(),

            last_mesh_id: 0,
            last_buffer_id: 0,

            meshes: HashMap::new(),
            buffers: HashMap::new(),
        }))
    }

    pub fn get_mesh(&self, id: &MeshID) -> Arc<GPUMesh<B>> {
        match self.meshes.get(id) {
            Some(weak_mesh) => match weak_mesh.upgrade() {
                Some(mesh) => mesh,
                None => panic!("Resource not valid."),
            },
            None => panic!("Mesh not stored on GPU"),
        }
    }

    pub fn add_mesh(&mut self, mesh: GPUMesh<B>) -> (MeshID, Arc<GPUMesh<B>>) {
        let id = self.last_mesh_id + 1;

        let arc = Arc::new(mesh);
        // store in map
        self.meshes.insert(id, Arc::downgrade(&arc));
        self.last_mesh_id = id;

        (id, arc)
    }

    pub fn get_buffer(&self, id: BufferID) -> Arc<Mutex<GPUBuffer<B>>> {
        match self.buffers.get(&id) {
            Some(weak_buffer) => match weak_buffer.upgrade() {
                Some(buffer) => buffer,
                None => panic!("Resource not valid."),
            },
            None => panic!("Buffer not stored on GPU"),
        }
    }

    pub fn add_buffer(&mut self, buffer: GPUBuffer<B>) -> (BufferID, Arc<Mutex<GPUBuffer<B>>>) {
        let id = self.last_buffer_id + 1;

        let arc = Arc::new(Mutex::new(buffer));
        // store in map
        self.buffers.insert(id, Arc::downgrade(&arc));
        self.last_buffer_id = id;

        (id, arc)
    }
}

pub trait MeshGenerator {
    fn get_vertices() -> Vec<Vertex> {
        vec![]
    }
    fn get_indices() -> Vec<Index> {
        vec![]
    }

    fn get_vertices_dynamic(&self) -> Vec<Vertex> {
        vec![]
    }
    fn get_indices_dynamic(&self) -> Vec<Index> {
        vec![]
    }
}

pub struct GPUMesh<B>
where
    B: gfx_hal::Backend,
{
    device: Arc<B::Device>,

    pub(in crate::rendering) num_vertices: u32,
    pub(in crate::rendering) vertex_buffer: Arc<ManuallyDrop<B::Buffer>>,
    pub(in crate::rendering) vertex_memory: Arc<ManuallyDrop<B::Memory>>,

    pub(in crate::rendering) num_indices: u32,
    pub(in crate::rendering) index_type: IndexType,
    pub(in crate::rendering) index_buffer: Arc<ManuallyDrop<B::Buffer>>,
    pub(in crate::rendering) index_memory: Arc<ManuallyDrop<B::Memory>>,
}

impl<B> GPUMesh<B>
where
    B: gfx_hal::Backend,
{
    pub fn new<T: MeshGenerator>(device: &Arc<B::Device>, adapter: &Arc<Adapter<B>>) -> Self {
        let vertices = T::get_vertices();
        let indices = T::get_indices();

        let vertex_buffer_info = Self::create_vertex_buffer::<T>(device, adapter, &vertices);
        let index_buffer_info = Self::create_index_buffer::<T>(device, adapter, &indices);

        GPUMesh {
            device: device.clone(),

            num_vertices: vertex_buffer_info.0,
            vertex_buffer: Arc::new(vertex_buffer_info.1),
            vertex_memory: Arc::new(vertex_buffer_info.2),

            num_indices: index_buffer_info.0,
            index_type: IndexType::U32,
            index_buffer: Arc::new(index_buffer_info.1),
            index_memory: Arc::new(index_buffer_info.2),
        }
    }

    pub fn new_dynamic<T: MeshGenerator>(
        device: &Arc<B::Device>,
        adapter: &Arc<Adapter<B>>,
        generator_instance: &T,
    ) -> Self {
        let vertices = generator_instance.get_vertices_dynamic();
        let indices = generator_instance.get_indices_dynamic();

        let vertex_buffer_info = Self::create_vertex_buffer::<T>(device, adapter, &vertices);
        let index_buffer_info = Self::create_index_buffer::<T>(device, adapter, &indices);

        GPUMesh {
            device: device.clone(),

            num_vertices: vertex_buffer_info.0,
            vertex_buffer: Arc::new(vertex_buffer_info.1),
            vertex_memory: Arc::new(vertex_buffer_info.2),

            num_indices: index_buffer_info.0,
            index_type: IndexType::U32,
            index_buffer: Arc::new(index_buffer_info.1),
            index_memory: Arc::new(index_buffer_info.2),
        }
    }

    fn create_vertex_buffer<T: MeshGenerator>(
        device: &B::Device,
        adapter: &Adapter<B>,
        vertices: &Vec<Vertex>,
    ) -> (u32, ManuallyDrop<B::Buffer>, ManuallyDrop<B::Memory>) {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let buffer_stride = mem::size_of::<Vertex>() as u64;
        let buffer_len = vertices.len() as u64 * buffer_stride;
        assert_ne!(buffer_len, 0);
        let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1)
            / non_coherent_alignment)
            * non_coherent_alignment;

        let mut vertex_buffer = ManuallyDrop::new(
            unsafe { device.create_buffer(padded_buffer_len, gfx_hal::buffer::Usage::VERTEX) }
                .unwrap(),
        );

        let buffer_req = unsafe { device.get_buffer_requirements(&vertex_buffer) };

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                // type_mask is a bit field where each bit represents a memory type. If the bit is set
                // to 1 it means we can use that type for our buffer. So this code finds the first
                // memory type that has a `1` (or, is allowed), and is visible to the CPU.
                buffer_req.type_mask & (1 << id as u64) != 0
                    && mem_type
                        .properties
                        .contains(gfx_hal::memory::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        // TODO: check transitions: read/write mapping and vertex buffer read
        let buffer_memory = unsafe {
            let memory = device
                .allocate_memory(upload_type, buffer_req.size)
                .unwrap();
            device
                .bind_buffer_memory(&memory, 0, &mut vertex_buffer)
                .unwrap();
            let mapping = device.map_memory(&memory, 0..padded_buffer_len).unwrap();
            ptr::copy_nonoverlapping(vertices.as_ptr() as *const u8, mapping, buffer_len as usize);
            device
                .flush_mapped_memory_ranges(iter::once((&memory, 0..padded_buffer_len)))
                .unwrap();
            device.unmap_memory(&memory);
            ManuallyDrop::new(memory)
        };

        (vertices.len() as u32, vertex_buffer, buffer_memory)
    }

    fn create_index_buffer<T: MeshGenerator>(
        device: &B::Device,
        adapter: &Adapter<B>,
        indices: &Vec<Index>,
    ) -> (u32, ManuallyDrop<B::Buffer>, ManuallyDrop<B::Memory>) {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        println!("Memory types: {:?}", memory_types);
        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let buffer_stride = mem::size_of::<Index>() as u64;
        let buffer_len = indices.len() as u64 * buffer_stride;
        assert_ne!(buffer_len, 0);
        let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1)
            / non_coherent_alignment)
            * non_coherent_alignment;

        let mut index_buffer = ManuallyDrop::new(
            unsafe { device.create_buffer(padded_buffer_len, gfx_hal::buffer::Usage::INDEX) }
                .unwrap(),
        );

        let buffer_req = unsafe { device.get_buffer_requirements(&index_buffer) };

        let upload_type = memory_types
            .iter()
            .enumerate()
            .position(|(id, mem_type)| {
                // type_mask is a bit field where each bit represents a memory type. If the bit is set
                // to 1 it means we can use that type for our buffer. So this code finds the first
                // memory type that has a `1` (or, is allowed), and is visible to the CPU.
                buffer_req.type_mask & (1 << id as u64) != 0
                    && mem_type
                        .properties
                        .contains(gfx_hal::memory::Properties::CPU_VISIBLE)
            })
            .unwrap()
            .into();

        // TODO: check transitions: read/write mapping and vertex buffer read
        let buffer_memory = unsafe {
            let memory = device
                .allocate_memory(upload_type, buffer_req.size)
                .unwrap();
            device
                .bind_buffer_memory(&memory, 0, &mut index_buffer)
                .unwrap();
            let mapping = device.map_memory(&memory, 0..padded_buffer_len).unwrap();
            ptr::copy_nonoverlapping(indices.as_ptr() as *const u8, mapping, buffer_len as usize);
            device
                .flush_mapped_memory_ranges(iter::once((&memory, 0..padded_buffer_len)))
                .unwrap();
            device.unmap_memory(&memory);
            ManuallyDrop::new(memory)
        };

        (indices.len() as u32, index_buffer, buffer_memory)
    }
}

impl<B> Drop for GPUMesh<B>
where
    B: gfx_hal::Backend,
{
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_buffer(ManuallyDrop::into_inner(ptr::read(
                    self.vertex_buffer.as_ref(),
                )));
            self.device.free_memory(ManuallyDrop::into_inner(ptr::read(
                self.vertex_memory.as_ref(),
            )));

            self.device
                .destroy_buffer(ManuallyDrop::into_inner(ptr::read(
                    self.index_buffer.as_ref(),
                )));
            self.device.free_memory(ManuallyDrop::into_inner(ptr::read(
                self.index_memory.as_ref(),
            )));
        }
    }
}

pub struct Plane {}

impl MeshGenerator for Plane {
    fn get_vertices() -> Vec<Vertex> {
        vec![
            Vertex::new([-0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
            Vertex::new([-0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
            Vertex::new([0.5, 0.0, 0.5], [0.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
            Vertex::new([0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]),
        ]
    }

    fn get_indices() -> Vec<Index> {
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

    fn get_indices() -> Vec<Index> {
        vec![
            0, 1, 2, 1, 3, 2, 4, 5, 6, 5, 7, 6, 8, 9, 10, 9, 11, 10, 12, 13, 14, 13, 15, 14, 16,
            17, 18, 17, 19, 18, 20, 21, 22, 21, 23, 22,
        ]
    }
}
