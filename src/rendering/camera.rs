use crate::core::Component;
use cgmath::Vector3;

#[derive(Clone)]
pub struct Camera {
    pub fov: f32,
    pub near: f32,
    pub far: f32,
}

impl Component for Camera {}

#[derive(Clone)]
pub struct Position {
    pub coords: Vector3<f32>,
}

impl Component for Position {}