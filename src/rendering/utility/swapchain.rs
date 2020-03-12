use std::{iter, ptr};
use std::borrow::BorrowMut;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::sync::{Arc, Mutex, RwLock};

use gfx_hal::adapter::Adapter;
use gfx_hal::Backend;
use gfx_hal::device::Device;
use gfx_hal::format::{ChannelType, Format};
use gfx_hal::image::Extent;
use gfx_hal::image::Usage;
use gfx_hal::window::{Extent2D, SwapchainConfig, SwapImageIndex};
use gfx_hal::window::Surface;
use gfx_hal::window::Swapchain;

pub struct SwapchainWrapper<B: Backend, D: Device<B>> {
    device: Arc<B::Device>,
    adapter: Arc<Adapter<B>>,

    swapchain: RwLock<ManuallyDrop<B::Swapchain>>,
    backbuffer: Option<Vec<B::Image>>,
    acquire_fences: Vec<B::Fence>,
    acquire_semaphores: Vec<B::Semaphore>,

    extent: Extent,
    format: Format,

    phantomData: std::marker::PhantomData<D>,
}

impl<B: Backend, D: Device<B>> SwapchainWrapper<B, D> {
    pub unsafe fn new(device: &Arc<B::Device>, adapter: &Arc<Adapter<B>>, surface: &mut B::Surface, extent: Extent2D) -> Self {
        let caps = surface
            .capabilities(&adapter.physical_device);
        let formats = surface
            .supported_formats(&adapter.physical_device);

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
            .with_present_mode(gfx_hal::window::PresentMode::FIFO);
        swap_config.image_usage = swap_config.image_usage | Usage::TRANSFER_DST;

        let extent = swap_config.extent.to_extent();
        let (swapchain, backbuffer) = device
            .create_swapchain(surface, swap_config, None)
            .expect("Can't create swapchain");

        let mut acquire_fences = Vec::with_capacity(backbuffer.len());
        let mut acquire_semaphores = Vec::with_capacity(backbuffer.len());
        for _ in 0..3 {
            acquire_fences.push(device.create_fence(false).expect("Failed to create fence."));
            acquire_semaphores.push(device.create_semaphore().expect("Failed to create semaphore."));
        };

        let swapchain = SwapchainWrapper {
            device: device.clone(),
            adapter: adapter.clone(),

            swapchain: RwLock::new(ManuallyDrop::new(swapchain)),
            backbuffer: Some(backbuffer),

            acquire_fences,
            acquire_semaphores,

            extent,
            format,

            phantomData: PhantomData,
        };
        swapchain
    }

    pub fn recreate(&mut self, surface: &mut B::Surface, extent: Extent2D) {
        let caps = surface
            .capabilities(&self.adapter.physical_device);
        let formats = surface
            .supported_formats(&self.adapter.physical_device);

        let format = formats.map_or(Format::Rgba8Srgb, |formats| {
            formats
                .iter()
                .find(|format| format.base_format().1 == ChannelType::Srgb)
                .map(|format| *format)
                .unwrap_or(formats[0])
        });

        let mut swap_config = SwapchainConfig::from_caps(&caps, format, extent)
            .with_present_mode(gfx_hal::window::PresentMode::FIFO);
        swap_config.image_usage = swap_config.image_usage | Usage::TRANSFER_DST;

        unsafe {
            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(ptr::read(self.swapchain.write().unwrap().deref())));

            let (swapchain, backbuffer) = self.device
                .create_swapchain(surface, swap_config, None)
                .expect("Can't create swapchain");

            self.swapchain = RwLock::new(ManuallyDrop::new(swapchain));
            self.backbuffer = Some(backbuffer);
        }
    }

    pub fn acquire_image(&self, frame_idx: usize) -> (SwapImageIndex, &B::Image) {
        let mut image_index: SwapImageIndex = 0;

        unsafe {
            let (idx, _) = self.swapchain.write().unwrap()
                .acquire_image(!0,
                               Some(&self.acquire_semaphores[frame_idx]),
                               None)
                .expect("Failed to aquire swapchain image.");

            image_index = idx;
        }

        let image = self.backbuffer.as_ref().unwrap().get(image_index as usize).unwrap();
        (image_index, image)
    }

    pub fn get_semaphore(&self, frame_idx: usize) -> &B::Semaphore {
        &self.acquire_semaphores[frame_idx]
    }

    pub fn present(&self, queue: &mut B::CommandQueue,
                   swap_image_index: SwapImageIndex,
                   present_semaphores: Option<Vec<&B::Semaphore>>) {
        let result = unsafe {
            let wait_semaphores = match present_semaphores {
                Some(sem) => sem,
                None => vec![]
            };

            self.swapchain.write().unwrap().present(
                queue,
                swap_image_index,
                wait_semaphores.iter(),
            )
        };

//        if result.is_err() {
//            self.recreate();
//        }
    }
}

impl<B: Backend, D: Device<B>> Drop for SwapchainWrapper<B, D> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_swapchain(ManuallyDrop::into_inner(ptr::read(self.swapchain.write().unwrap().deref())));

            for acquire_fence in self.acquire_fences.drain(0..) {
                self.device.destroy_fence(acquire_fence);
            }
        }
    }
}