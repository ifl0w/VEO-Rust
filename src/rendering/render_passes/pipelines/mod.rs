use std::mem::ManuallyDrop;
use std::sync::Arc;

use gfx_hal::Backend;

pub use forward_pipeline::ForwardPipeline;
pub use resolve_pipeline::ResolvePipeline;
use std::ptr;

mod forward_pipeline;
mod resolve_pipeline;

/* Constants */
pub const ENTRY_NAME: &str = "main";


pub trait Pipeline<B: Backend> {
    fn new(device: &Arc<B::Device>,
           render_pass: &B::RenderPass,
           set_layout: &B::DescriptorSetLayout)
           -> Self;

    fn get_pipeline(&self) -> &B::GraphicsPipeline;
    fn get_layout(&self) -> &B::PipelineLayout;

    fn create_pipeline(device: &Arc<B::Device>,
                       render_pass: &B::RenderPass,
                       set_layout: &B::DescriptorSetLayout)
                       -> Option<(ManuallyDrop<B::GraphicsPipeline>, ManuallyDrop<B::PipelineLayout>)>;

}