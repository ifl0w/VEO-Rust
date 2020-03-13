use std::{iter, mem, ptr};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, Weak};

use gfx_hal::{
    buffer,
    command,
    format as f,
    format::{AsFormat, ChannelType, Rgba8Srgb as ColorFormat, Swizzle},
    image as i,
    IndexType,
    memory as m,
    pass,
    pass::Subpass,
    pool,
    prelude::*,
    pso,
    pso::{
        PipelineStage,
        ShaderStageFlags,
        VertexInputRate,
    },
    queue::{QueueGroup, Submission},
    window,
};
use gfx_hal::adapter::{Adapter, MemoryType};
use gfx_hal::adapter::PhysicalDevice;

use crate::rendering::renderer::Renderer;
use crate::rendering::utility::{GPUBuffer, Index, Vertex};
use crate::rendering::BufferID;

pub struct ResourceManager<B>
    where B: gfx_hal::Backend {
    device: Arc<B::Device>,
    adapter: Arc<Adapter<B>>,

    pub(in crate::rendering::utility) meshes: HashMap<MeshID, Arc<GPUMesh<B>>>,
    pub(in crate::rendering::utility) buffers: HashMap<BufferID, Arc<Mutex<GPUBuffer<B>>>>,
//    pub(in crate::rendering) shaders: HashMap<ShaderID, Arc<Shader<B>>>,
}

impl<B> ResourceManager<B>
    where B: gfx_hal::Backend {
    pub fn new(renderer: &Renderer<B>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(ResourceManager {
            device: renderer.device.clone(),
            adapter: renderer.adapter.clone(),

            meshes: HashMap::new(),
            buffers: HashMap::new(),
        }))
    }

    pub fn get_mesh(&self, id: &MeshID) -> &GPUMesh<B> {
        match self.meshes.get(id) {
            Some(m) => m,
            None => panic!("Mesh not stored on GPU")
        }
    }

    pub fn get_buffer(&self, id: BufferID) -> &Arc<Mutex<GPUBuffer<B>>> {
        match self.buffers.get(&id) {
            Some(buffer) => buffer,
            None => panic!("Buffer not stored on GPU")
        }
    }

    pub fn add_buffer(&mut self, buffer: GPUBuffer<B>) -> BufferID {
        let id = self.buffers.len();

        // store in map
        self.buffers.insert(id, Arc::new(Mutex::new(buffer)));

        id
    }
}

pub type MeshID = u64;

pub trait MeshGenerator {
    fn get_vertices() -> Vec<Vertex>;
    fn get_indices() -> Vec<Index>;

    fn generate<T: MeshGenerator, B: gfx_hal::Backend>(rm: &mut ResourceManager<B>)
                                                       -> (MeshID, Weak<GPUMesh<B>>) {
        let device = rm.device.clone();
        let adapter = rm.adapter.clone();

        let m = Arc::new(GPUMesh::new::<T>(device, adapter));

        let id = rm.meshes.len() as u64;

        // store in map
        rm.meshes.insert(id, m.clone());

        // return non owning pointer
        (id, Arc::downgrade(&m))
    }
}

pub struct GPUMesh<B>
    where B: gfx_hal::Backend {
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
    where B: gfx_hal::Backend {
    pub fn new<T: MeshGenerator>(device: Arc<B::Device>, adapter: Arc<Adapter<B>>) -> Self {
        let vertex_buffer_info = Self::create_vertex_buffer::<T>(&device, &adapter);
        let index_buffer_info = Self::create_index_buffer::<T>(&device, &adapter);

        GPUMesh {
            device,

            num_vertices: vertex_buffer_info.0,
            vertex_buffer: Arc::new(vertex_buffer_info.1),
            vertex_memory: Arc::new(vertex_buffer_info.2),

            num_indices: index_buffer_info.0,
            index_type: IndexType::U32,
            index_buffer: Arc::new(index_buffer_info.1),
            index_memory: Arc::new(index_buffer_info.2),
        }
    }

    fn create_vertex_buffer<T: MeshGenerator>(device: &B::Device, adapter: &Adapter<B>)
                                              -> (u32,
                                                  ManuallyDrop<B::Buffer>,
                                                  ManuallyDrop<B::Memory>) {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let vertices = T::get_vertices();

        let buffer_stride = mem::size_of::<Vertex>() as u64;
        let buffer_len = vertices.len() as u64 * buffer_stride;
        assert_ne!(buffer_len, 0);
        let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1)
            / non_coherent_alignment)
            * non_coherent_alignment;

        let mut vertex_buffer = ManuallyDrop::new(
            unsafe { device.create_buffer(padded_buffer_len, gfx_hal::buffer::Usage::VERTEX) }.unwrap(),
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
                    && mem_type.properties.contains(gfx_hal::memory::Properties::CPU_VISIBLE)
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

    fn create_index_buffer<T: MeshGenerator>(device: &B::Device, adapter: &Adapter<B>)
                                             -> (u32,
                                                 ManuallyDrop<B::Buffer>,
                                                 ManuallyDrop<B::Memory>) {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let limits = adapter.physical_device.limits();

        println!("Memory types: {:?}", memory_types);
        let non_coherent_alignment = limits.non_coherent_atom_size as u64;

        let indices = T::get_indices();

        let buffer_stride = mem::size_of::<Index>() as u64;
        let buffer_len = indices.len() as u64 * buffer_stride;
        assert_ne!(buffer_len, 0);
        let padded_buffer_len = ((buffer_len + non_coherent_alignment - 1)
            / non_coherent_alignment)
            * non_coherent_alignment;

        let mut index_buffer = ManuallyDrop::new(
            unsafe { device.create_buffer(padded_buffer_len, gfx_hal::buffer::Usage::INDEX) }.unwrap(),
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
                    && mem_type.properties.contains(gfx_hal::memory::Properties::CPU_VISIBLE)
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
    where B: gfx_hal::Backend {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(ManuallyDrop::into_inner(ptr::read(self.vertex_buffer.as_ref())));
            self.device.free_memory(ManuallyDrop::into_inner(ptr::read(self.vertex_memory.as_ref())));

            self.device.destroy_buffer(ManuallyDrop::into_inner(ptr::read(self.index_buffer.as_ref())));
            self.device.free_memory(ManuallyDrop::into_inner(ptr::read(self.index_memory.as_ref())));
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
            Vertex::new([0.5, 0.0, -0.5], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0])
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
            0, 1, 2, 1, 3, 2,
            4, 5, 6, 5, 7, 6,
            8, 9, 10, 9, 11, 10,
            12, 13, 14, 13, 15, 14,
            16, 17, 18, 17, 19, 18,
            20, 21, 22, 21, 23, 22,
        ]
    }
}