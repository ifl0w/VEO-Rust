use cgmath::Vector3;
use crate::core::Component;
use cgmath::num_traits::{Float, ToPrimitive};

#[derive(Clone, Debug)]
pub struct AABB {
    pub min: Vector3<f32>,
    pub max: Vector3<f32>,
}

impl Component for AABB {}

impl AABB {
    pub fn new(min: Vector3<f32>, max: Vector3<f32>) -> Self {
        AABB {
            min: Vector3 {
                x: min.x,
                y: min.y,
                z: min.z,
            },
            max: Vector3 {
                x: max.x,
                y: max.y,
                z: max.z,
            }
        }
    }

    pub fn intersect(&self, other: &AABB) -> bool {
        return (self.min.x <= other.min.x && self.max.x >= other.max.x) &&
            (self.min.y <= other.min.y && self.max.y >= other.max.y) &&
            (self.min.z <= other.min.z && self.max.z >= other.max.z);
    }
}