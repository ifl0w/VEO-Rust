use std::f32::consts::PI;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cgmath::{vec3, Vector3};
use winit::event::{Event, KeyboardInput, VirtualKeyCode, WindowEvent};

use nse::core::{Filter, Message, System};
use nse::rendering::nse_gui::octree_gui::ProfilingData;
use nse::rendering::{Camera, OctreeConfig, OctreeOptimizations, Transformation};

use crate::shared::benchmark_system::Benchmark::CircularMotion;
use winit::event::ElementState::Pressed;

enum Benchmark {
    CircularMotion,
}

pub struct BenchmarkSystem {
    active: bool,
    // number of frames. if none & active = true => will start the benchmark
    initialized: Option<i64>,
    state: f32, // 0 .. 1 interpolation constant of curve movement

    selected_benchmark: Benchmark,
    store: Vec<(f32, ProfilingData)>,

    stages: Vec<(OctreeOptimizations, OctreeConfig)>,
    stage_id: usize,

    frame_profile: ProfilingData,
    messages: Vec<Message>,
}

impl BenchmarkSystem {
    pub fn new() -> Arc<Mutex<Self>> {
        let config = OctreeConfig {
            max_rendered_nodes: Some(4e6 as u64),
            depth: Some(10),
        };

        let stages = vec![
            (
                OctreeOptimizations {
                    depth_threshold: 10.0,
                    ..Default::default()
                },
                config.clone(),
            ),
            (
                OctreeOptimizations {
                    frustum_culling: true,
                    depth_threshold: 10.0,
                    ..Default::default()
                },
                config.clone(),
            ),
            (
                OctreeOptimizations {
                    depth_culling: true,
                    depth_threshold: 10.0,
                    ..Default::default()
                },
                config.clone(),
            ),
            (
                OctreeOptimizations {
                    frustum_culling: true,
                    depth_culling: true,
                    depth_threshold: 10.0,
                    ..Default::default()
                },
                config.clone(),
            ),
        ];

        Arc::new(Mutex::new(BenchmarkSystem {
            store: Vec::new(),
            active: false,
            initialized: None,
            state: 0.0,

            selected_benchmark: CircularMotion,
            stages,
            stage_id: 0,

            frame_profile: ProfilingData::default(),
            messages: vec![],
        }))
    }

    fn initialize_stage(&mut self, benchmark: Benchmark, stage: usize) {
        self.store.clear();
        self.frame_profile = ProfilingData::default();
        self.state = 0.0;
        self.selected_benchmark = benchmark;
        self.initialized = Some(2);

        let stage = &self.stages[stage];
        let m1 = Message::new(stage.0.clone());
        let m2 = Message::new(stage.1.clone());

        self.messages.push(m1);
        self.messages.push(m2);
    }

    fn end_benchmark(&mut self) {
        self.active = false;
    }

    fn next_stage(&mut self) -> usize {
        self.stage_id = self.stage_id + 1;

        if self.stage_id >= self.stages.len() {
            self.stage_id = 0;
            self.active = false;
        }

        self.initialize_stage(CircularMotion, self.stage_id);

        self.stage_id
    }

    fn start_stage(&mut self) {
        self.store.clear();
        self.frame_profile = ProfilingData::default();
        self.state = 0.0;
        self.active = true;
        self.initialized = None;
    }

    fn circular_motion(center: Vector3<f32>, radius: f32, state: f32) -> Transformation {
        let arc = state * 16.0 * PI;
        let mut pos = center + vec3(arc.cos(), 0.0, arc.sin()) * radius;

        pos.y = 30.0;

        Transformation::new().position(pos)
    }

    fn write_to_file(&mut self, filename: &str) -> std::io::Result<()> {
        let file = File::create(filename)?;
        let mut buf_writer = BufWriter::new(file);

        let mut header_string = String::from(
            "benchmark_progress, rendered_nodes, instance_data_generation, render_time",
        );

        self.store.iter().take(1).for_each(|(_, p)| {
            for system_time in p.system_times.as_ref().unwrap_or(&vec![]) {
                header_string.push_str(format!(", {}", system_time.0).as_str());
            }
        });

        writeln!(buf_writer, "{}", header_string)?;

        for (progress, p) in &self.store {
            let mut frame_str = format!(
                "{}, {}, {}, {}",
                progress,
                p.rendered_nodes.unwrap_or(0),
                p.instance_data_generation.unwrap_or(0),
                p.render_time.unwrap_or(0) / (1e6 as u64)
            ); // conversion to ms

            for system_time in p.system_times.as_ref().unwrap_or(&vec![]) {
                frame_str.push_str(format!(", {}", system_time.1.as_millis()).as_str());
            }

            writeln!(buf_writer, "{}", frame_str)?;
        }

        self.store.clear();

        Ok(())
    }
}

impl System for BenchmarkSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![nse::filter!(Camera, Transformation)]
    }

    fn handle_input(&mut self, event: &Event<()>) {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::KeyboardInput { input, .. } => match input {
                    KeyboardInput {
                        virtual_keycode,
                        state,
                        ..
                    } => match (virtual_keycode, state) {
                        (Some(VirtualKeyCode::Numpad1), Pressed) => {
                            if !self.active {
                                self.initialize_stage(CircularMotion, 0);
                            } else {
                                self.end_benchmark();
                            }
                        }
                        _ => {}
                    },
                },
                _ => (),
            },
            _ => (),
        }
    }

    fn consume_messages(&mut self, messages: &Vec<Message>) {
        self.frame_profile = ProfilingData::default();

        for msg in messages {
            if msg.is_type::<ProfilingData>() {
                self.frame_profile
                    .replace(msg.get_payload::<ProfilingData>().unwrap());
            }
        }

        self.store.push((self.state, self.frame_profile.clone()));
    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, delta_time: Duration) {
        match self.initialized {
            Some(n) => {
                if n != 0 {
                    self.initialized.replace(n - 1);
                } else {
                    self.start_stage();
                }
            }
            None => {
                if self.active {
                    let camera_filter = filter[0].lock().unwrap();
                    let mut camera_entity = camera_filter.entities[0].lock().unwrap();

                    let transform =
                        BenchmarkSystem::circular_motion(vec3(0.0, 0.0, 0.0), 100.0, self.state);

                    camera_entity.add_component(transform);

                    self.state += delta_time.as_secs_f32() as f32 / 60.0;

                    if self.state >= 1.0 {
                        if self.stage_id < self.stages.len() {
                            self.write_to_file(format!("stage_{}.csv", self.stage_id).as_str())
                                .expect("Could not write to file.");
                            self.next_stage();
                        } else {
                            self.end_benchmark();
                        }
                    }
                }
            }
        }
    }

    fn get_messages(&mut self) -> Vec<Message> {
        if self.messages.is_empty() {
            return vec![];
        }

        let result = self.messages.clone();

        self.messages.clear();

        result
    }
}
