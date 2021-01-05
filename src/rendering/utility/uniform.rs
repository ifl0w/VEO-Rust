use std::{iter, ptr};
use std::mem::{ManuallyDrop, size_of};
use std::sync::Arc;

use gfx_hal::{Backend, buffer, memory};
use gfx_hal::adapter::Adapter;
use gfx_hal::adapter::PhysicalDevice;
use gfx_hal::device::Device;
use gfx_hal::pso::{Descriptor, DescriptorSetWrite};

use crate::rendering::renderer::Renderer;

pub struct GPUBuffer<B: Backend> {
    device: Arc<B::Device>,
    // adapter: Arc<Adapter<B>>,

    buffer: ManuallyDrop<B::Buffer>,
    memory: ManuallyDrop<B::Memory>,

    element_count: usize,
    size: usize,
}

impl<B: Backend> GPUBuffer<B> {
    pub(in crate::rendering) fn get_buffer(&self) -> &B::Buffer {
        &self.buffer
    }

    pub fn new<T>(
        device: &Arc<B::Device>,
        adapter: &Arc<Adapter<B>>,
        data_source: &[T],
        usage: buffer::Usage,
    ) -> Self
        where
            T: Copy,
    {
        let stride = size_of::<T>();
        let upload_size = data_source.len() * stride;

        let mut buf = Self::new_with_size(device, adapter, upload_size, usage);
        buf.update_data(0, data_source);
        buf.element_count = data_source.len();

        buf
    }

    pub fn new_with_size(
        device: &Arc<B::Device>,
        adapter: &Arc<Adapter<B>>,
        byte_size: usize,
        usage: buffer::Usage,
    ) -> Self {
        let memory_types = adapter.physical_device.memory_properties().memory_types;
        let memory: B::Memory;
        let mut buffer: B::Buffer;
        let size: usize;

        let upload_size = byte_size;

        unsafe {
            buffer = device.create_buffer(upload_size as u64, usage).unwrap();
            let mem_req = device.get_buffer_requirements(&buffer);

            // A note about performance: Using CPU_VISIBLE memory is convenient because it can be
            // directly memory mapped and easily updated by the CPU, but it is very slow and so should
            // only be used for small pieces of data that need to be updated very frequently. For something like
            // a vertex buffer that may be much larger and should not change frequently, you should instead
            // use a DEVICE_LOCAL buffer that gets filled by copying data from a CPU_VISIBLE staging buffer.
            let upload_type = memory_types
                .iter()
                .enumerate()
                .position(|(id, mem_type)| {
                    mem_req.type_mask & (1 << id) != 0
                        && mem_type.properties.contains(
                        memory::Properties::CPU_VISIBLE | memory::Properties::COHERENT,
                    )
                })
                .unwrap()
                .into();

            memory = device.allocate_memory(upload_type, mem_req.size).unwrap();
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
            size = mem_req.size as usize;
        }

        GPUBuffer {
            device: device.clone(),
            // adapter: adapter.clone(),
            buffer: ManuallyDrop::new(buffer),
            memory: ManuallyDrop::new(memory),
            element_count: 0,
            size,
        }
    }

    /// updates a section of the buffer.
    /// does not change the stored length. The user has to ensure that elements in the buffer are
    /// updated correctly
    pub fn update_data<T>(&self, offset: usize, data_source: &[T])
        where
            T: Copy,
    {
        let device = &self.device;

        let stride = size_of::<T>();
        let upload_size = data_source.len() * stride;

        assert!(offset + upload_size <= self.size);
        let memory = &self.memory;

        unsafe {
            let mapping = device
                .map_memory(memory, offset as u64..self.size as u64)
                .unwrap();
            ptr::copy_nonoverlapping(data_source.as_ptr() as *const u8, mapping, upload_size);
            device.unmap_memory(memory);
        }
    }

    /// invalidates the buffer content and replaces the data.
    /// Changes the number of elements tracked in the buffer.
    pub fn replace_data<T>(&mut self, data_source: &[T])
        where
            T: Copy,
    {
        self.update_data(0, data_source);
        self.element_count = data_source.len();
    }

    pub fn len(&self) -> usize {
        self.element_count
    }
}

impl<B: Backend> Drop for GPUBuffer<B> {
    fn drop(&mut self) {
        let device = &self.device;
        unsafe {
            device.destroy_buffer(ManuallyDrop::into_inner(ptr::read(&self.buffer)));
            device.free_memory(ManuallyDrop::into_inner(ptr::read(&self.memory)));
        }
    }
}

pub type UniformID = u64;

pub struct Uniform<B: Backend> {
    device: Arc<B::Device>,
    adapter: Arc<Adapter<B>>,

    pub(in crate::rendering) buffers: Vec<GPUBuffer<B>>,
}

impl<B: Backend> Uniform<B> {
    pub fn new<T>(
        renderer: &Renderer<B>,
        data: &[T],
        binding: u32,
        desc_sets: &Vec<B::DescriptorSet>,
    ) -> Self
        where
            T: Copy,
    {
        let mut buffers: Vec<GPUBuffer<B>> = Vec::new();
        for idx in 0..renderer.frames_in_flight {
            let buffer = GPUBuffer::new(
                &renderer.device,
                &renderer.adapter,
                &data,
                buffer::Usage::UNIFORM,
            );

            unsafe {
                renderer
                    .device
                    .write_descriptor_sets(iter::once(DescriptorSetWrite {
                        set: &desc_sets[idx],
                        binding,
                        array_offset: 0,
                        descriptors: iter::once(Descriptor::Buffer(
                            buffer.get_buffer(),
                            None..None,
                        )),
                    }));
            }

            buffers.push(buffer);
        }

        Uniform {
            device: renderer.device.clone(),
            adapter: renderer.adapter.clone(),
            buffers,
        }
    }
}
