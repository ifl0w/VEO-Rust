use std::mem::ManuallyDrop;

use gfx_hal::Backend;

pub use forward_render_pass::ForwardRenderPass;
pub use forward_render_pass::PipelineOptions;
pub use pipelines::*;
pub use shader::ShaderCode;

mod forward_render_pass;
mod pipelines;
mod shader;

pub trait RenderPass<B: Backend> {
    fn sync(&mut self, frame_idx: usize);
    fn submit(
        &mut self,
        frame_idx: usize,
        queue: &mut B::CommandQueue,
        wait_semaphores: Vec<&B::Semaphore>,
    ) -> &B::Semaphore;
    fn get_render_pass(&self) -> &ManuallyDrop<B::RenderPass>;
    fn get_descriptor_set(&self, frame_index: usize) -> &B::DescriptorSet;
    fn blit_to_surface(
        &mut self,
        queue: &mut B::CommandQueue,
        surface_image: &B::Image,
        frame_idx: usize,
        // acquire_semaphore: &B::Semaphore,
    ) -> &mut B::Semaphore;
    fn record(&mut self, frame_idx: usize);

    fn execution_time(&mut self, frame_idx: usize) -> u64;
}
