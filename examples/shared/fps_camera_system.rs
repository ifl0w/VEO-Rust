use std::sync::{Arc, Mutex};
use std::time::Duration;

use winit::{Event, WindowEvent, VirtualKeyCode, ElementState};
use winit::dpi::LogicalPosition;

use cgmath::{Vector3, Deg, Matrix3, Quaternion};

use nse::core::{System, Filter};
use nse::rendering::{Camera, Transformation};

pub struct FPSCameraSystem {
    mouse_delta_x: f32,
    mouse_delta_y: f32,
    last_position: Option<LogicalPosition>,

    move_left: bool,
    move_right: bool,
    move_forward: bool,
    move_back: bool,

    movement_speed: f32,
    mouse_speed: f32,
}

impl FPSCameraSystem {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(FPSCameraSystem {
            mouse_delta_x: 0.0,
            mouse_delta_y: 0.0,
            last_position: None,

            move_left: false,
            move_right: false,
            move_forward: false,
            move_back: false,

            movement_speed: 3.0,
            mouse_speed: 0.25
        }))
    }

    pub fn set_mouse_speed(&mut self, mouse_speed: f32) {
        self.mouse_speed = mouse_speed;
    }

    pub fn set_movement_speed(&mut self, movement_speed: f32) {
        self.movement_speed = movement_speed;
    }
}

impl System for FPSCameraSystem {

    fn get_filter(&mut self) -> Vec<Filter> { vec![nse::filter!(Camera, Transformation)] }

    fn handle_input(&mut self, event: &Event) {
        match event {
            | Event::WindowEvent { event, .. } => {
                match event {
                    | WindowEvent::KeyboardInput { input, .. } => {
                        match input {
                            | winit::KeyboardInput { virtual_keycode, state, .. } => {
                                match (virtual_keycode, state) {
                                    | (Some(VirtualKeyCode::W), ElementState::Pressed) => {
                                        self.move_forward = true;
                                    }
                                    | (Some(VirtualKeyCode::W), ElementState::Released) => {
                                        self.move_forward = false;
                                    }
                                    | (Some(VirtualKeyCode::A), ElementState::Pressed) => {
                                        self.move_left = true;
                                    }
                                    | (Some(VirtualKeyCode::A), ElementState::Released) => {
                                        self.move_left = false;
                                    }
                                    | (Some(VirtualKeyCode::S), ElementState::Pressed) => {
                                        self.move_back = true;
                                    }
                                    | (Some(VirtualKeyCode::S), ElementState::Released) => {
                                        self.move_back = false;
                                    }
                                    | (Some(VirtualKeyCode::D), ElementState::Pressed) => {
                                        self.move_right = true;
                                    }
                                    | (Some(VirtualKeyCode::D), ElementState::Released) => {
                                        self.move_right = false;
                                    }
                                    | _ => {}
                                }
                            }
                        }
                    }
                    | WindowEvent::CursorMoved { position, .. } => {
                        match self.last_position {
                            Some(last) => {
                                self.mouse_delta_x = (position.x - last.x) as f32;
                                self.mouse_delta_y = (position.y - last.y) as f32;
                            }
                            None => ()
                        }

                        self.last_position = Option::Some(*position);
                    }
                    _ => ()
                }
            }
            _ => ()
        }
    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, delta_time: Duration) {
        let filter = filter[0].lock().unwrap();
        let mut camera = filter.entities.get(0)
            .expect("No camera provided").lock().unwrap();

        let mut transform = camera.get_component::<Transformation>().ok().unwrap().clone();

        let mut axis_aligned_translation = Vector3 {x: 0.0, y: 0.0, z: 0.0};
        if self.move_forward {
            axis_aligned_translation.z -= self.movement_speed * delta_time.as_secs_f32();
        }
        if self.move_back {
            axis_aligned_translation.z += self.movement_speed * delta_time.as_secs_f32();
        }
        if self.move_left {
            axis_aligned_translation.x -= self.movement_speed * delta_time.as_secs_f32();
        }
        if self.move_right {
            axis_aligned_translation.x += self.movement_speed * delta_time.as_secs_f32();
        }

        match self.last_position {
            Some(_) => {
                let angle_x = Deg(- self.mouse_delta_y * self.mouse_speed);
                let angle_y = Deg(- self.mouse_delta_x * self.mouse_speed);

                let camera_y = Vector3{ x: 0.0, y:1.0, z:0.0 };
                let camera_x = transform.rotation * Vector3{ x: 1.0, y:0.0, z:0.0 };

                let x = Quaternion::from(Matrix3::from_axis_angle(camera_x, angle_x));
                let y = Quaternion::from(Matrix3::from_axis_angle(camera_y, angle_y));

                transform = transform.rotation(x * y * transform.rotation);
            }
            None => ()
        }

        transform = transform.position(transform.position + transform.rotation * axis_aligned_translation);

        camera.add_component(transform);

        self.mouse_delta_x = 0.0;
        self.mouse_delta_y = 0.0;
    }

}