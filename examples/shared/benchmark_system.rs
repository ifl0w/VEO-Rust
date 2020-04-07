use std::f32::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cgmath::{vec3, Vector3};

use nse::core::{Filter, Message, System};
use nse::rendering::{Camera, Transformation};
use nse::rendering::nse_gui::octree_gui::ProfilingData;

pub struct BenchmarkSystem {
    store: Vec<ProfilingData>,
    active: bool,
    state: f32, // 0 .. 1 interpolation constant of curve movement
}

impl BenchmarkSystem {
    pub fn new() -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(BenchmarkSystem {
            store: Vec::new(),
            active: true,
            state: 0.0,
        }))
    }

    fn circular_motion(center: Vector3<f32>, radius: f32, state: f32) -> Transformation {
        let arc = state * 4.0 * PI;
        let mut pos = center + vec3(arc.cos(), 0.0, arc.sin()) * radius;

        pos.y = 30.0;

        Transformation::new().position(pos)
    }
}

impl System for BenchmarkSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![
            nse::filter!(Camera, Transformation),
        ]
    }

//    fn handle_input(&mut self, _event: &Event<()>) {}

    fn consume_messages(&mut self, messages: &Vec<Message>) {
        for msg in messages {
            if msg.is_type::<ProfilingData>() {
                self.store.push(msg.get_payload::<ProfilingData>().unwrap().clone());
            }
        }
    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, delta_time: Duration) {
        if self.active {
            let camera_filter = filter[0].lock().unwrap();
            let mut camera_entity = camera_filter.entities[0].lock().unwrap();

            let transform = BenchmarkSystem::circular_motion(vec3(0.0, 0.0, 0.0), 100.0, self.state);

            camera_entity.add_component(transform);

            self.state += delta_time.as_secs_f32() as f32 / 30.0;

            if self.state >= 1.0 {
                self.state = 0.0;
            }
        }
    }
}