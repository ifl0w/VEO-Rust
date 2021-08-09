use std::mem::ManuallyDrop;
use std::sync::Arc;

use gfx_hal::{Backend, image};
use gfx_hal::adapter::Adapter;
use gfx_hal::device::Device;
use gfx_hal::format::Format;
use gfx_hal::image::{Extent, ViewCapabilities};
use gfx_hal::image::Usage;
use gfx_hal::pool::CommandPool;
use gfx_hal::pool::CommandPoolCreateFlags;
use gfx_hal::queue::QueueGroup;
use gfx_hal::window::Extent2D;

use crate::rendering::{DepthImage, Image};

pub struct Framebuffer<B: Backend, D: Device<B>> {
    device: Arc<D>,

    framebuffers: Vec<ManuallyDrop<B::Framebuffer>>,
    framebuffer_fences: Vec<ManuallyDrop<B::Fence>>,
    command_pools: Vec<ManuallyDrop<B::CommandPool>>,
    command_buffer_lists: Vec<Vec<B::CommandBuffer>>,
    frame_images: Vec<Image<B, D>>,
    depth_images: Vec<DepthImage<B, D>>,
    acquire_semaphores: Vec<ManuallyDrop<B::Semaphore>>,
    present_semaphores: Vec<ManuallyDrop<B::Semaphore>>,
}

impl<B: Backend, D: Device<B>> Framebuffer<B, D> {
    pub fn new(
        device: &Arc<D>,
        adapter: &Arc<Adapter<B>>,
        queue_group: &QueueGroup<B>,
        render_pass: &B::RenderPass,
        extent: Extent2D,
        usage: Usage,
        color_format: Format,
        depth_format: Format,
        frames: usize,
    ) -> Result<Self, &'static str> {
        let extend_2d = extent;

        let extent = Extent {
            width: extent.width as _,
            height: extent.height as _,
            depth: 1,
        };

        let mut framebuffers = Vec::with_capacity(frames);
        let mut frame_images = Vec::with_capacity(frames);
        let mut depth_images = Vec::with_capacity(frames);
        let mut fences = Vec::with_capacity(frames);
        let mut command_pools = Vec::with_capacity(frames);
        let mut command_buffer_lists = Vec::with_capacity(frames);
        let mut acquire_semaphores = Vec::with_capacity(frames);
        let mut present_semaphores = Vec::with_capacity(frames);

        for _ in 0..frames {
            unsafe {
                let fb_image = Image::new(adapter, device, extend_2d, usage, color_format)
                    .expect("Image creation failed!");

                let fb_depth_image =
                    DepthImage::new(adapter, device, extend_2d).expect("Image creation failed!");

                let fb = device
                    .create_framebuffer(
                        &render_pass,
                        vec![
                            image::FramebufferAttachment {
                                usage: usage,
                                view_caps: ViewCapabilities::MUTABLE_FORMAT,
                                format: color_format,
                            },
                            image::FramebufferAttachment {
                                usage: Usage::DEPTH_STENCIL_ATTACHMENT,
                                view_caps: ViewCapabilities::MUTABLE_FORMAT,
                                format: depth_format,
                            },
                        ].into_iter(),
                        extent,
                    )
                    .expect("Framebuffer creation failed!");

                let fence = device.create_fence(true).expect("Fence creation failed!");

                let cmd_pool = device
                    .create_command_pool(queue_group.family, CommandPoolCreateFlags::empty())
                    .expect("Command pool creation failed!");

                let cmd_buffers = Vec::with_capacity(frames);

                let a_s = device
                    .create_semaphore()
                    .expect("Semaphore creation failed!");
                let p_s = device
                    .create_semaphore()
                    .expect("Semaphore creation failed!");

                frame_images.push(fb_image);
                depth_images.push(fb_depth_image);
                framebuffers.push(ManuallyDrop::new(fb));
                fences.push(ManuallyDrop::new(fence));
                command_pools.push(ManuallyDrop::new(cmd_pool));
                command_buffer_lists.push(cmd_buffers);
                acquire_semaphores.push(ManuallyDrop::new(a_s));
                present_semaphores.push(ManuallyDrop::new(p_s));
            }
        }

        Ok(Framebuffer {
            device: device.clone(),
            frame_images,
            depth_images,
            framebuffers,
            framebuffer_fences: fences,
            command_pools,
            command_buffer_lists,
            present_semaphores,
            acquire_semaphores,
        })
    }

    pub fn get_frame_data(
        &mut self,
        frame_id: usize,
    ) -> (
        &mut B::Fence,
        &mut Image<B, D>,
        &mut DepthImage<B, D>,
        &mut B::Framebuffer,
        &mut B::CommandPool,
        &mut Vec<B::CommandBuffer>,
        &mut B::Semaphore,
        &mut B::Semaphore,
    ) {
        (
            &mut self.framebuffer_fences[frame_id],
            &mut self.frame_images[frame_id],
            &mut self.depth_images[frame_id],
            &mut self.framebuffers[frame_id],
            &mut self.command_pools[frame_id],
            &mut self.command_buffer_lists[frame_id],
            &mut self.acquire_semaphores[frame_id],
            &mut self.present_semaphores[frame_id],
        )
    }

    pub fn get_frame_semaphore(&self, frame_id: usize) -> &B::Semaphore {
        &self.present_semaphores[frame_id]
    }
}

impl<B: Backend, D: Device<B>> Drop for Framebuffer<B, D> {
    fn drop(&mut self) {
        unsafe {
            for idx in 0..self.command_pools.len() {
                self.device.wait_for_fence(&mut self.framebuffer_fences[idx], !0).unwrap();

                self.device.destroy_fence(ManuallyDrop::take(&mut self.framebuffer_fences[idx]));
                self.device.destroy_semaphore(ManuallyDrop::take(&mut self.acquire_semaphores[idx]));
                self.device.destroy_semaphore(ManuallyDrop::take(&mut self.present_semaphores[idx]));
                self.device.destroy_framebuffer(ManuallyDrop::take(&mut self.framebuffers[idx]));

                let cmds = self.command_buffer_lists[idx].drain(..);
                if cmds.len() > 0 {
                    self.command_pools[idx].reset(true);
                    self.command_pools[idx].free(cmds);

                    // destroying the command pool outside of this condition results in a segfault
                    // during device deletion. Further investigation is required.
                    self.device.destroy_command_pool(ManuallyDrop::take(&mut self.command_pools[idx]));
                }
            }
        }
    }
}
