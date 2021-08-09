use std::mem::ManuallyDrop;
use std::sync::Arc;

use gfx_hal::{
    Backend,
    format::{Aspects, Format}, MemoryTypeId,
};
use gfx_hal::adapter::Adapter;
use gfx_hal::adapter::PhysicalDevice;
use gfx_hal::device::Device;
use gfx_hal::image::SubresourceRange;
use gfx_hal::image::Usage;
use gfx_hal::memory::Properties;
use gfx_hal::memory::Requirements;
use gfx_hal::window::Extent2D;

/// Parts for a depth buffer image
pub struct Image<B: Backend, D: Device<B>> {
    device: Arc<D>,

    pub image: ManuallyDrop<B::Image>,
    pub requirements: Requirements,
    pub memory: ManuallyDrop<B::Memory>,
    pub image_view: ManuallyDrop<B::ImageView>,

    pub extent: Extent2D,
}

impl<B: Backend, D: Device<B>> Image<B, D> {
    pub fn new(
        adapter: &Adapter<B>,
        device: &Arc<D>,
        extent: Extent2D,
        usage: Usage,
        format: Format,
    ) -> Result<Self, &'static str> {
        unsafe {
            let mut the_image = device
                .create_image(
                    gfx_hal::image::Kind::D2(extent.width, extent.height, 1, 1),
                    1,
                    format,
                    gfx_hal::image::Tiling::Optimal,
                    usage,
                    gfx_hal::image::ViewCapabilities::MUTABLE_FORMAT,
                )
                .map_err(|_| "Couldn't crate the image!")?;

            let requirements = device.get_image_requirements(&the_image);

            let memory_type_id = adapter
                .physical_device
                .memory_properties()
                .memory_types
                .iter()
                .enumerate()
                .find(|&(id, memory_type)| {
                    // BIG NOTE: THIS IS DEVICE LOCAL NOT CPU VISIBLE
                    requirements.type_mask & (1 << id) != 0
                        && memory_type.properties.contains(Properties::DEVICE_LOCAL)
                })
                .map(|(id, _)| MemoryTypeId(id))
                .ok_or("Couldn't find a memory type to support the image!")?;

            let memory = device
                .allocate_memory(memory_type_id, requirements.size)
                .map_err(|_| "Couldn't allocate image memory!")?;

            device
                .bind_image_memory(&memory, 0, &mut the_image)
                .map_err(|_| "Couldn't bind the image memory!")?;

            let image_view = device
                .create_image_view(
                    &the_image,
                    gfx_hal::image::ViewKind::D2,
                    format,
                    gfx_hal::format::Swizzle::NO,
                    SubresourceRange {
                        aspects: Aspects::COLOR,
                        level_start: 0,
                        level_count: Some(1),
                        layer_start: 0,
                        layer_count: Some(1),
                    },
                )
                .map_err(|_| "Couldn't create the image view!")?;

            Ok(Self {
                device: device.clone(),
                image: ManuallyDrop::new(the_image),
                requirements,
                memory: ManuallyDrop::new(memory),
                image_view: ManuallyDrop::new(image_view),
                extent,
            })
        }
    }
}

impl<B: Backend, D: Device<B>> Drop for Image<B, D> {
    fn drop(&mut self) {
        use core::ptr::read;

        unsafe {
            self.device
                .destroy_image_view(ManuallyDrop::into_inner(read(&self.image_view)));
            self.device
                .destroy_image(ManuallyDrop::into_inner(read(&self.image)));
            self.device
                .free_memory(ManuallyDrop::into_inner(read(&self.memory)));
        }
    }
}
