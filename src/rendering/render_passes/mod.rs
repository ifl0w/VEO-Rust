use std::mem::ManuallyDrop;

use gfx_hal::Backend;

pub use forward_render_pass::ForwardRenderPass;
pub use shader::ShaderCode;

use crate::rendering::renderer::Renderer;
use crate::rendering::ResourceManager;

mod shader;
mod forward_render_pass;

pub trait RenderPass<B: Backend> {
    fn get_render_pass(&self) -> &ManuallyDrop<B::RenderPass>;
    fn generate_command_buffer(&self, renderer: &mut Renderer<B>,
                               resource_manager: &ResourceManager<B>,
                               framebuffer: &B::Framebuffer) -> B::CommandBuffer;
    fn get_descriptor_set(&self, frame_index: usize) -> &B::DescriptorSet;
}

