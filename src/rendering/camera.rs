use std::sync::{Arc, Mutex};

use cgmath::{Deg, Euler, Matrix4, Quaternion, Rad, SquareMatrix, Transform, Vector3, Vector4, Vector2, InnerSpace, vec3};

use crate::core::Component;
use crate::rendering::{RenderSystem, AABB};
use crate::rendering::utility::Uniform;
use crate::rendering::camera::FrustumPlanes::BottomPlane;
use std::collections::HashMap;
use cgmath::num_traits::{FromPrimitive, Float, ToPrimitive};
use cgmath::num_traits::real::Real;
use std::convert::TryInto;

#[derive(Clone)]
pub struct Camera {
    pub near: f32,
    pub far: f32,
    pub fov: f32,
    pub projection: Matrix4<f32>,

    pub frustum: Frustum,
}

impl Component for Camera {}

impl Camera {
    pub fn new(near: f32, far: f32, fov: f32, resolution: [f32; 2]) -> Self {
        let aspect = resolution[0] as f32 / resolution[1] as f32;

        let fov_y = Rad::from(Deg(fov / aspect));
        let fov_x = Rad::from(Deg(fov * aspect));

        let proj = cgmath::perspective(
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

            frustum: Frustum::new(fov_x, fov_y, near, far)
        }
    }
}

#[derive(Copy, Clone)]
pub struct Transformation {
    pub position: Vector3<f32>,
    pub scale: Vector3<f32>,
    pub rotation: Quaternion<f32>,

    pub model_matrix: Matrix4<f32>,
}

impl Transformation {}

impl Default for Transformation {
    fn default() -> Self {
        Transformation {
            position: Vector3::new(0.0, 0.0, 0.0),
            scale: Vector3::new(1.0, 1.0, 1.0),
            rotation: Quaternion::from(Euler {
                x: Deg(0.0),
                y: Deg(0.0),
                z: Deg(0.0),
            }),

            model_matrix: Matrix4::identity(),
        }
    }
}

impl Component for Transformation {}

impl Transformation {
    pub fn new() -> Self {
        Transformation {
            ..Default::default()
        }
    }

    pub fn position(mut self, position: Vector3<f32>) -> Self {
        self.position = position;
        self.update()
    }
    pub fn scale(mut self, scale: Vector3<f32>) -> Self {
        self.scale = scale;
        self.update()
    }
    pub fn rotation(mut self, rotation: Quaternion<f32>) -> Self {
        self.rotation = rotation;
        self.update()
    }

    pub fn update(mut self) -> Self {
        self.model_matrix = self.calculate_model_matrix();

        self
    }

    fn calculate_model_matrix(&self) -> Matrix4<f32> {
        let mut ret = Matrix4::identity();
        ret = ret * Matrix4::from_translation(self.position);
        ret = ret * Matrix4::from(self.rotation);
        ret = ret * Matrix4::from_nonuniform_scale(self.scale.x, self.scale.y, self.scale.z);

        ret
    }

    pub fn get_model_matrix(&self) -> Matrix4<f32> {
        return self.model_matrix;
    }
}

#[derive(Copy, Clone)]
pub struct CameraData {
    pub view: Matrix4<f32>,
    pub proj: Matrix4<f32>,
    pub position: Vector4<f32>,
}

impl CameraData {
    pub fn new(camera: &Camera, transform: &Transformation) -> Self {
        let mut p = camera.projection;
        p.y.y *= -1.0;

        CameraData {
            view: transform.get_model_matrix().inverse_transform().unwrap(),
            proj: p,
            position: transform.position.extend(0.0),
        }
    }
}

impl Default for CameraData {
    fn default() -> Self {
        CameraData {
            view: Matrix4::identity(),
            proj: Matrix4::identity(),
            position: Vector4 { x: 0.0, y: 0.0, z: 0.0, w: 0.0 },
        }
    }
}

#[derive(Hash, Eq, PartialEq)]
pub enum FrustumPlanes {
    NearPlane,
    FarPlane,
    LeftPlane,
    RightPlane,
    TopPlane,
    BottomPlane
}

#[derive(Clone)]
pub struct Frustum {
    /// (plane normalvector, point on plane)
    planes: [(Vector3<f32>, Vector3<f32>); 6],

    fov_x: Rad<f32>,
    fov_y: Rad<f32>,
    near_distance: f32,
    far_distance: f32,

    near_dimensions: Vector2<f32>,
    far_dimensions: Vector2<f32>,
}

impl Frustum {

    pub fn new(fov_x: Rad<f32>, fov_y: Rad<f32>, near_distance: f32, far_distance: f32) -> Self {
        let mut frustum = Frustum {
            planes: [(vec3(0.0,0.0,0.0), vec3(0.0,0.0,0.0)); 6],


            near_distance,
            far_distance,
            fov_x,
            fov_y,

            near_dimensions: Vector2 {
                x: near_distance * (fov_x.0 / 2.0).tan(),
                y: near_distance * (fov_y.0 / 2.0).tan(),
            },
            far_dimensions: Vector2 {
                x: far_distance * (fov_x.0 / 2.0).tan(),
                y: far_distance * (fov_y.0 / 2.0).tan(),
            }
        };

        frustum
    }

    pub fn intersect(&self, aabb: &AABB) -> bool {
        //for each plane do ...
        self.planes.iter().all(|(normal, point)| {
            let mut p = aabb.min.clone();
            p.x = if normal.x > 0.0 { aabb.max.x } else { p.x };
            p.y = if normal.y > 0.0 { aabb.max.y } else { p.y };
            p.z = if normal.z > 0.0 { aabb.max.z } else { p.z };

            let mut n = aabb.max.clone();
            n.x = if normal.x < 0.0 { aabb.min.x } else { n.x };
            n.y = if normal.y < 0.0 { aabb.min.y } else { n.y };
            n.z = if normal.z < 0.0 { aabb.min.z } else { n.z };

            let plane_to_min = p - point;
            let plane_to_max = n - point;

            let d1 = plane_to_min.dot(*normal);
            let d2 = plane_to_max.dot(*normal);

            if d1 < 0.0 && d2 < 0.0 {
                false // no intersection with current plane
            } else {
                true // intersecting with current plane
            }
        })
    }

    pub fn transformed(&self, camera_transform: Matrix4<f32>) -> Frustum {
        let mut new = Frustum::new(self.fov_x,self.fov_y, self.near_distance, self.far_distance);

        // base axis of camera space
        let x: Vector3<f32> = (camera_transform * cgmath::vec4(1.0, 0.0, 0.0, 0.0)).normalize().truncate();
        let y: Vector3<f32> = (camera_transform * cgmath::vec4(0.0, 1.0, 0.0, 0.0)).normalize().truncate();
        let z: Vector3<f32> = (camera_transform * cgmath::vec4(0.0, 0.0, 1.0, 0.0)).normalize().truncate();

        let cam_pos = (camera_transform * cgmath::vec4(0.0, 0.0, 0.0, 1.0)).truncate();
        let near_center = &cam_pos - &z * self.near_distance;
        let far_center = &cam_pos - &z * self.far_distance;

        new.planes[0] = (-z.clone(), near_center.clone());
        new.planes[1] = (z.clone(), far_center.clone());

        let point_on_plane = near_center + &y * self.near_dimensions.y;
        let normal = (point_on_plane - cam_pos).normalize().cross(x).normalize();
        new.planes[2] = (normal, point_on_plane);

        let point_on_plane = near_center - &y * self.near_dimensions.y;
        let normal = (point_on_plane - cam_pos).normalize().cross(x).normalize();
        new.planes[3] = (normal, point_on_plane);

        let point_on_plane = near_center - &x * self.near_dimensions.x;
        let normal = (point_on_plane - cam_pos).normalize().cross(y).normalize();
        new.planes[4] = (normal, point_on_plane);

        let point_on_plane = near_center + &x * self.near_dimensions.x;
        let normal = (point_on_plane - cam_pos).normalize().cross(y).normalize();
        new.planes[5] = (normal, point_on_plane);

        new
    }

}