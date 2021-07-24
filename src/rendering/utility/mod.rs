pub use framebuffer::Framebuffer;
pub use resources::Cube;
pub use resources::GPUMesh;
pub use resources::MeshGenerator;
pub use resources::MeshID;
pub use resources::Plane;
pub use resources::ResourceManager;
pub use swapchain::SwapchainWrapper;
pub use uniform::GPUBuffer;
pub use uniform::Uniform;
pub use uniform::UniformID;

//pub use uniform::Uniform;
pub mod uniform;

pub mod framebuffer;
pub mod resources;
pub mod swapchain;

pub type Index = u32;

#[derive(Copy, Clone, Default)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
    pub fn new(position: [f32; 3], normal: [f32; 3], _color: [f32; 3]) -> Self {
        Self {
            position,
            normal,
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct InstanceData {
    pub transformation: [f32; 4], // x,y,z: position, w: scale
    pub color: [f32; 4], // x,y,z: color, w: unused
}
