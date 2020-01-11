use cgmath::{Deg, Rad, Matrix4, Quaternion, SquareMatrix, Vector3, Transform};

use crate::core::Component;

#[derive(Clone)]
pub struct Camera {
    pub near: f32,
    pub far: f32,
    pub fov: f32,
    pub projection: Matrix4<f32>,
}

impl Component for Camera {}

impl Camera {
    pub fn new(near: f32, far: f32, fov: f32, resolution: [f32; 2]) -> Self {
        let aspect = resolution[0] as f32 / resolution[1] as f32;
        let mut proj = cgmath::perspective(
            Rad::from(Deg(fov / aspect)),
            aspect,
            near,
            far,
        );

        Camera {
            near,
            far,
            fov,
            projection: proj,
        }
    }
}

#[derive(Clone)]
pub struct Transformation {
    pub position: Vector3<f32>,
    pub scale: Vector3<f32>,
    pub rotation: Quaternion<f32>,
}

impl Component for Transformation {}

impl Transformation {

    pub fn get_model_matrix(&self) -> Matrix4<f32> {
        let mut ret = Matrix4::identity();
        ret = ret * Matrix4::from_translation(self.position);
        ret = ret * Matrix4::from(self.rotation);
        ret = ret * Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);

        ret
    }
}

#[derive(Copy, Clone)]
pub struct CameraDataUbo {
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
}

impl CameraDataUbo {
    pub fn new(camera: &Camera, transform: &Transformation) -> Self {

        let mut p = camera.projection;
        p.y.y *= -1.0;

        CameraDataUbo {
            view: transform.get_model_matrix().inverse_transform().unwrap(),
            proj: p,
        }
    }
}

impl Default for CameraDataUbo {
    fn default() -> Self {
        CameraDataUbo {
            view: Matrix4::identity(),
            proj: Matrix4::identity(),
        }
    }
}