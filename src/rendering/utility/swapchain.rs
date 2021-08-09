use std::marker::PhantomData;
use std::sync::Arc;

use gfx_hal::adapter::Adapter;
use gfx_hal::Backend;
use gfx_hal::device::Device;
use gfx_hal::format::{ChannelType, Format};
use gfx_hal::image::Extent;
use gfx_hal::image::Usage;
use gfx_hal::prelude::PresentationSurface;
use gfx_hal::window::{Extent2D, SwapchainConfig};
use gfx_hal::window::Surface;

pub struct SwapchainWrapper<B: Backend, D: Device<B>> {
    device: Arc<B::Device>,
    adapter: Arc<Adapter<B>>,

    pub extent: Extent,
    pub format: Format,

    phantom_data: std::marker::PhantomData<D>,
}

impl<B: Backend, D: Device<B>> SwapchainWrapper<B, D> {
    pub unsafe fn new(
        device: &Arc<B::Device>,
        adapter: &Arc<Adapter<B>>,
        surface: &mut B::Surface,
        extent: Extent2D,
    ) -> Self {
        let caps = surface.capabilities(&adapter.physical_device);
        let formats = surface.supported_formats(&adapter.physical_device);

        println!("formats: {:?}", formats);
        let format = formats.map_or(Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        println!("Surface format: {:?}", format);
        let mut swap_config = SwapchainConfig::from_caps(&caps, format, extent)
            .with_present_mode(gfx_hal::window::PresentMode::IMMEDIATE);
        swap_config.image_usage = swap_config.image_usage | Usage::TRANSFER_DST;

        let extent = swap_config.extent.to_extent();
        surface.configure_swapchain(device, swap_config).expect("Can't create swapchain");

        let mut acquire_fences = Vec::new();
        let mut acquire_semaphores = Vec::new();
        for _ in 0..3 {
            acquire_fences.push(device.create_fence(false).expect("Failed to create fence."));
            acquire_semaphores.push(
                device
                    .create_semaphore()
                    .expect("Failed to create semaphore."),
            );
        }

        Self {
            device: device.clone(),
            adapter: adapter.clone(),

            extent,
            format,

            phantom_data: PhantomData,
        }
    }

    pub fn recreate(&mut self, surface: &mut B::Surface, extent: Extent2D) {
        let caps = surface.capabilities(&self.adapter.physical_device);
        let formats = surface.supported_formats(&self.adapter.physical_device);

        let format = formats.map_or(Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let mut swap_config = SwapchainConfig::from_caps(&caps, format, extent)
            .with_present_mode(gfx_hal::window::PresentMode::IMMEDIATE);
        swap_config.image_usage = swap_config.image_usage | Usage::TRANSFER_DST;

        unsafe {
            surface.configure_swapchain(&self.device, swap_config).expect("Can't create swapchain");
        }
    }
}