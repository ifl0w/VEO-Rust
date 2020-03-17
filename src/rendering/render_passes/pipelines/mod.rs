use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::Arc;

use gfx_hal::Backend;

pub use forward_pipeline::ForwardPipeline;
use gfx_hal::pso::PolygonMode;

mod forward_pipeline;

/* Constants */
pub const ENTRY_NAME: &str = "main";


pub trait Pipeline<B: Backend> {
    fn new(device: &Arc<B::Device>,
           render_pass: &B::RenderPass,
           set_layout: &B::DescriptorSetLayout,
           polygon_mode: PolygonMode)
           -> Self;

    fn get_pipeline(&self, instanced: bool) -> &B::GraphicsPipeline;
    fn get_layout(&self, instanced: bool) -> &B::PipelineLayout;
}