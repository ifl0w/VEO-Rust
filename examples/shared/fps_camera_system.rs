use std::sync::{Arc, Mutex};
use std::time::Duration;

use cgmath::{Deg, Matrix3, Quaternion, Vector3};
use winit::event::{ElementState, Event, KeyboardInput, MouseButton, VirtualKeyCode, WindowEvent};
use winit::event::DeviceEvent::MouseMotion;
use winit::event::ElementState::Pressed;
use winit::window::WindowId;

use nse::core::{Filter, MainWindow, Message, System};
use nse::rendering::{Camera, Transformation};

pub struct FPSCameraSystem {
    mouse_delta: (f32, f32),
    active: bool,

    move_left: bool,
    move_right: bool,
    move_forward: bool,
    move_back: bool,

    movement_speed: f32,
    mouse_speed: f32,

    main_window: Option<WindowId>,
}

impl FPSCameraSystem {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(FPSCameraSystem {
            mouse_delta: (0.0, 0.0),
            active: false,

            move_left: false,
            move_right: false,
            move_forward: false,
            move_back: false,

            movement_speed: 3.0,
            mouse_speed: 0.25,

            main_window: None,
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

    fn handle_input(&mut self, event: &Event<()>) {
        if self.main_window.is_none() {
            return;
        }

        match event {
            | Event::WindowEvent { event, window_id } => {
                if *window_id != self.main_window.expect("No main window exists.") {
                    return;
                }

                match event {
                    | WindowEvent::KeyboardInput { input, .. } => {
                        match input {
                            | KeyboardInput { virtual_keycode, state, .. } => {
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
                    | WindowEvent::MouseInput { button, state, .. } => {
                        match button {
                            MouseButton::Left => {
                                self.active = *state == Pressed;
                                self.mouse_delta = (0.0, 0.0);
                            }
                            _ => ()
                        }
                    }
                    _ => ()
                }
            }
            | Event::DeviceEvent { event, .. } => {
                match event {
                    | MouseMotion { delta, .. } => {
                        // sum up delta. Per frame there might be more than one MouseMotion event
                        self.mouse_delta = (self.mouse_delta.0 + delta.0 as f32,
                                            self.mouse_delta.1 + delta.1 as f32);
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

        let mut axis_aligned_translation = Vector3 { x: 0.0, y: 0.0, z: 0.0 };
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

        if self.active {
            let angle_x = Deg(-self.mouse_delta.1 / 10.0 * self.mouse_speed);
            let angle_y = Deg(-self.mouse_delta.0 / 10.0 * self.mouse_speed);

            let camera_y = Vector3 { x: 0.0, y: 1.0, z: 0.0 };
            let camera_x = transform.rotation * Vector3 { x: 1.0, y: 0.0, z: 0.0 };

            let x = Quaternion::from(Matrix3::from_axis_angle(camera_x, angle_x));
            let y = Quaternion::from(Matrix3::from_axis_angle(camera_y, angle_y));

            transform = transform.rotation(y * x * transform.rotation);

            self.mouse_delta = (0.0, 0.0);
        }

        transform = transform.position(transform.position + transform.rotation * axis_aligned_translation);

        camera.add_component(transform);
    }

    fn consume_messages(&mut self, messages: &Vec<Message>) {
        for msg in messages {
            if msg.is_type::<MainWindow>() {
                self.main_window = Some(msg.get_payload::<MainWindow>().ok().unwrap().window_id)
            }
        }
    }
}