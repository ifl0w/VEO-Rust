use gfx_hal::{Backend, memory, buffer, pso};
use std::sync::Arc;
use gfx_hal::device::Device;
use gfx_hal::adapter::MemoryType;
use std::mem::{size_of, ManuallyDrop};
use std::ptr;
use std::borrow::{Borrow, BorrowMut};
use gfx_hal::pso::Descriptor;


pub struct GPUBuffer<B: Backend> {
    buffer: ManuallyDrop<B::Buffer>,
    memory: ManuallyDrop<B::Memory>,
    device: Arc<B::Device>,
    size: u64,
}

impl<B: Backend> GPUBuffer<B> {
    pub (in crate::rendering)  fn get_buffer(&self) -> &B::Buffer {
        &self.buffer
    }

    pub unsafe fn new<T>(
        device: Arc<B::Device>,
        data_source: &[T],
        usage: buffer::Usage,
        memory_types: Vec<MemoryType>,
    ) -> Self
        where
            T: Copy,
    {
        let memory: B::Memory;
        let mut buffer: B::Buffer;
        let size: u64;

        let stride = size_of::<T>();
        let upload_size = data_source.len() * stride;

        {
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
                        && mem_type
                        .properties
                        .contains(memory::Properties::CPU_VISIBLE | memory::Properties::COHERENT)
                })
                .unwrap()
                .into();

            memory = device.allocate_memory(upload_type, mem_req.size).unwrap();
            device.bind_buffer_memory(&memory, 0, &mut buffer).unwrap();
            size = mem_req.size;

            // TODO: check transitions: read/write mapping and vertex buffer read
            let mapping = device.map_memory(&memory, 0 .. size).unwrap();
            ptr::copy_nonoverlapping(data_source.as_ptr() as *const u8, mapping, upload_size);
            device.unmap_memory(&memory);
        }

        GPUBuffer {
            buffer: ManuallyDrop::new(buffer),
            memory: ManuallyDrop::new(memory),
            device,
            size,
        }
    }

    pub fn update_data<T>(&mut self, offset: u64, data_source: &[T])
        where
            T: Copy,
    {
        let device = &self.device;

        let stride = size_of::<T>();
        let upload_size = data_source.len() * stride;

        assert!(offset + upload_size as u64 <= self.size);
        let memory = &self.memory;

        unsafe {
            let mapping = device.map_memory(memory, offset .. self.size).unwrap();
            ptr::copy_nonoverlapping(data_source.as_ptr() as *const u8, mapping, upload_size);
            device.unmap_memory(memory);
        }
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

//pub struct Uniform<B: Backend> {
//    buffers: Vec<GPUBuffer<B>>,
//    desc: Vec<B::Descriptor>,
//}
//
//impl<B: Backend> Uniform<B> {
//    unsafe fn new<T>(
//        device: Arc<dyn Device<B>>,
//        memory_types: &[MemoryType],
//        data: &[T],
//        mut desc: DescSet<B>,
//        binding: u32,
//        frames: u32,
//    ) -> Self
//        where
//            T: Copy,
//    {
//        let buffer = GPUBuffer::new(
//            device,
//            &data,
//            buffer::Usage::UNIFORM,
//            memory_types,
//        );
//        let buffer = Some(buffer);
//
//        desc.write_to_state(
//            vec![DescSetWrite {
//                binding,
//                array_offset: 0,
//                descriptors: Some(pso::Descriptor::Buffer(
//                    buffer.as_ref().unwrap().get_buffer(),
//                    None .. None,
//                )),
//            }],
//            &mut device,
//        );
//
//        Uniform {
//            buffer,
//            desc: Some(desc),
//        }
//    }
//
//    fn get_layout(&self) -> &B::DescriptorSetLayout {
//        self.desc.as_ref().unwrap().get_layout()
//    }
//}