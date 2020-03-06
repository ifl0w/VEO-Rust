
//pub use uniform::Uniform;
pub use uniform::GPUBuffer;
mod uniform;

mod resources;
pub use resources::ResourceManager;
pub use resources::GPUMesh;

pub use resources::MeshGenerator;
pub use resources::Plane;
pub use resources::Cube;

pub type Index = u32;

#[derive(Copy, Clone, Default)]
pub struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl Vertex {
    fn new(position: [f32; 3], normal: [f32; 3], color: [f32; 3]) -> Self {
        Self {
            position,
            normal,
            color
        }
    }
}

#[derive(Default, Debug, Copy, Clone)]
pub struct InstanceData {
    pub model_matrix: [[f32; 4]; 4]
}
