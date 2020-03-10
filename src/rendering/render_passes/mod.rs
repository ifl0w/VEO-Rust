use std::fmt::Debug;
use std::mem::ManuallyDrop;

use gfx_hal::Backend;
use gfx_hal::queue::CommandQueue;
use gfx_hal::queue::Submission;
use gfx_hal::window::PresentationSurface;

pub use pipelines::*;
pub use forward_render_pass::ForwardRenderPass;
pub use shader::ShaderCode;

use crate::rendering::{Framebuffer, ResourceManager};
use crate::rendering::renderer::Renderer;
use gfx_hal::pso::Viewport;
use std::sync::{Arc, Mutex};

mod shader;
mod forward_render_pass;
mod pipelines;

pub trait RenderPass<B: Backend> {
    fn sync(&mut self, frame_idx: usize);
    fn submit(&mut self, frame_idx: usize, queue: &mut B::CommandQueue) -> Arc<Mutex<Framebuffer<B, B::Device>>>;
    fn get_render_pass(&self) -> &ManuallyDrop<B::RenderPass>;
    fn fill_command_buffer(&self, framebuffer: &mut B::Framebuffer, command_buffer: &mut B::CommandBuffer, frame_idx: usize);
    fn get_descriptor_set(&self, frame_index: usize) -> &B::DescriptorSet;
    fn blit_to_surface(&mut self, queue: &mut B::CommandQueue, surface_image: &B::Image, frame_idx: usize)
                       -> Arc<Mutex<Framebuffer<B, B::Device>>>;
    fn render(&mut self, queue: &mut B::CommandQueue, frame_idx: usize) -> Arc<Mutex<Framebuffer<B, B::Device>>>;
}

