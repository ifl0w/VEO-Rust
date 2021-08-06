#[cfg(not(any(
feature = "vulkan",
feature = "dx12",
feature = "metal",
feature = "gl",
feature = "wgl"
)))]
pub extern crate gfx_backend_empty as Backend;
#[cfg(any(feature = "gl", feature = "wgl"))]
pub extern crate gfx_backend_gl as Backend;
#[cfg(feature = "vulkan")]
pub extern crate gfx_backend_vulkan as Backend;
extern crate rand;

use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use cgmath::{Array, InnerSpace, Matrix4, Rotation, Transform, vec3, Vector3, SquareMatrix};
use cgmath::num_traits::Pow;
use gfx_hal::buffer;
use rayon::prelude::*;
use winit::event::Event;

use crate::core::{Component, Filter, Message, Payload, System, Exit};
use crate::rendering::{Camera, Frustum, GPUBuffer, InstanceData, Mesh, RenderSystem, Transformation, fractal_generators};
use crate::rendering::nse_gui::octree_gui::ProfilingData;

use std::thread::{sleep, JoinHandle};
use riffy::MpscQueue;

use std::collections::VecDeque;

use shared_arena::{SharedArena, ArenaBox};
use crate::rendering::fractal_generators::FractalSelection;

pub const TREE_SUBDIVISIONS: usize = 2;
const DEFAULT_DEPTH: u64 = 4;

#[derive(Clone)]
pub struct Octree {
    pub root: Arc<Mutex<Node>>,

    pub instance_data_buffer: Vec<Arc<Mutex<GPUBuffer<Backend::Backend>>>>,
    /// points into the instance_data_buffer vec
    pub active_instance_buffer_idx: Arc<AtomicUsize>,

    pub config: OctreeConfig,
    pub info: OctreeInfo,

    pub node_pool: Arc<SharedArena<NodeChildren>>
}

#[derive(Clone, Debug, Copy)]
pub struct OctreeConfig {
    pub max_rendered_nodes: Option<u64>,
    pub subdiv_threshold: Option<f64>,
    pub distance_scale: Option<f64>,
    pub depth: Option<u64>,
    pub fractal: Option<FractalSelection>,
    pub continuous_update: Option<bool>,
    pub reset: Option<bool>,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        OctreeConfig {
            max_rendered_nodes: Some(4e6 as u64),
            subdiv_threshold: Some(20.0),
            distance_scale: Some(1.0),
            depth: Some(DEFAULT_DEPTH),
            fractal: Some(FractalSelection::MandelBulb),
            continuous_update: Some(true),
            reset: None,
        }
    }
}

impl OctreeConfig {
    pub fn merge(&mut self, other: &OctreeConfig) {
        if other.depth.is_some() {
            self.depth = other.depth
        };

        if other.max_rendered_nodes.is_some() {
            self.max_rendered_nodes = other.max_rendered_nodes
        };
    }
}

impl Payload for OctreeConfig {}

#[derive(Default, Clone)]
pub struct OctreeInfo {
    pub render_count: usize,
    pub max_num_nodes: usize,

    pub byte_size: usize,
    // Size of the data structure in RAM
    pub max_byte_size: usize,
    // size of the data structure if filled completely in RAM
    pub gpu_byte_size: usize, // allocated storage on the GPU
}

impl Component for Octree {}

impl Octree {
    pub fn new(render_system: &Arc<Mutex<RenderSystem>>, config: OctreeConfig) -> Self {
        let rm = render_system.lock().unwrap().resource_manager.clone();

        let dev = render_system.lock().unwrap().renderer.device.clone();
        let adapter = render_system.lock().unwrap().renderer.adapter.clone();

        // read config
        let max_rendered_nodes = config.max_rendered_nodes.unwrap_or(4e6 as u64);

        // byte size calculations
        let max_gpu_byte_size = std::mem::size_of::<InstanceData>() * max_rendered_nodes as usize;

        let ring_buffer_length = 2;
        let mut instance_data_buffer = Vec::with_capacity(ring_buffer_length);
        for _ in 0..ring_buffer_length {
            let (_, buffer) = rm.lock().unwrap().add_buffer(GPUBuffer::new_with_size(
                &dev,
                &adapter,
                max_gpu_byte_size,
                buffer::Usage::STORAGE | buffer::Usage::VERTEX,
            ));

            instance_data_buffer.push(buffer);
        }

        let arena = Arc::new(SharedArena::new());
        let mut oct = Octree {
            root: Arc::new(Mutex::new(Node::new_inner(vec3(0.0, 0.0, 0.0), 1.0))),
            instance_data_buffer,
            active_instance_buffer_idx: Arc::new(AtomicUsize::new(0)),

            config,
            info: OctreeInfo::default(),

            node_pool: arena
        };

        oct.reconfigure(render_system, config);

        fractal_generators::build_tree(
            &mut oct.root.lock().unwrap(),
            oct.config,
            0,
            1,
            &oct.node_pool
        );

        oct.info.byte_size = self::Octree::size_in_bytes(&oct);

        oct
    }

    pub fn reconfigure(&mut self, render_system: &Arc<Mutex<RenderSystem>>, config: OctreeConfig) {
        let rm = render_system.lock().unwrap().resource_manager.clone();
        let mut rm_lock = rm.lock().unwrap();

        let dev = render_system.lock().unwrap().renderer.device.clone();
        let adapter = render_system.lock().unwrap().renderer.adapter.clone();

        // read config
        self.config = config;
        let depth = config.depth.unwrap_or(DEFAULT_DEPTH);
        let max_rendered_nodes = config.max_rendered_nodes.unwrap_or(4e6 as u64);

        // byte size calculations
        let max_num_nodes = (TREE_SUBDIVISIONS as i64).pow(3 * depth as u32) as usize;
        let _max_byte_size = std::mem::size_of::<Node>() * max_num_nodes;
        let max_gpu_byte_size = std::mem::size_of::<InstanceData>() * max_rendered_nodes as usize;

        let ring_buffer_length = 2;

        self.instance_data_buffer = Vec::with_capacity(ring_buffer_length);
        for _ in 0..ring_buffer_length {
            let (_, buffer) = rm_lock.add_buffer(GPUBuffer::new_with_size(
                &dev,
                &adapter,
                max_gpu_byte_size,
                buffer::Usage::STORAGE | buffer::Usage::VERTEX,
            ));

            self.instance_data_buffer.push(buffer);
        }

        let octree_info = OctreeInfo {
            render_count: 0,
            byte_size: 0,
            max_num_nodes,
            max_byte_size: 0,
            gpu_byte_size: max_gpu_byte_size * ring_buffer_length,
        };

        self.info = octree_info;
    }

    pub fn get_instance_buffer(&self) -> Option<&Arc<Mutex<GPUBuffer<Backend::Backend>>>> {
        let idx = self.active_instance_buffer_idx.load(Ordering::SeqCst);
        self.instance_data_buffer.get(idx)
    }

    pub fn count_leaves(&self) -> i64 {
        let root = self.root.lock().unwrap();
        root.count_leaves()
    }

    pub fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<Node>() * self.count_nodes(&self.root.lock().unwrap()) as usize
    }

    fn count_nodes(&self, node: &Node) -> i64 {
        let mut count = 0;

        if node.children.is_some() {
            node.children.as_ref().unwrap().iter()
                .for_each(|child| {
                    count += self.count_nodes(child);
                });
        }

        count + 1
    }
}

pub type NodeChildren = [Node; TREE_SUBDIVISIONS * TREE_SUBDIVISIONS * TREE_SUBDIVISIONS];

#[derive(Debug)]
pub struct Node {
    pub children: Option<ArenaBox<NodeChildren>>,
    // position of the node
    pub position: Vector3<f32>,
    // length of a side of the node
    pub scale: f32,
    // whether this node should be refined further
    pub refine: Option<bool>,
    pub solid: bool,
    pub color: Vector3<f32>,
    pub height_values: [f32; 4], // only used for terrain generation
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Node {
            position: self.position,
            color: self.color,
            scale: self.scale,
            solid: self.solid,
            height_values: self.height_values,
            refine: self.refine,
            children: None, // never clone children!,
        }
    }
}

impl Node {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn new_inner(position: Vector3<f32>, scale: f32) -> Self {
        let tmp = Node {
            children: None,
            position,
            color: vec3(1.0, 1.0, 1.0),
            scale,
            solid: false,
            refine: None,
            height_values: [0.0; 4],
        };

        tmp
    }

    pub fn populate(&mut self, node_pool: &SharedArena<NodeChildren>) {
        if self.children.is_some() {
            return; // nothing to do
        }

        let scale_factor = 1.0 / TREE_SUBDIVISIONS as f32;

        let mut chil = node_pool.alloc(Default::default());

        for id_x in 0..TREE_SUBDIVISIONS {
            for id_y in 0..TREE_SUBDIVISIONS {
                for id_z in 0..TREE_SUBDIVISIONS {
                    let s = self.scale * scale_factor;

                    let mut t = self.position;
                    t -= Vector3::from_value(self.scale * 0.5 - s * 0.5);
                    t += vec3(s * id_x as f32, s * id_y as f32, s * id_z as f32);

                    let idx = TREE_SUBDIVISIONS.pow(0) * id_x
                        + TREE_SUBDIVISIONS.pow(1) * id_y
                        + TREE_SUBDIVISIONS.pow(2) * id_z;

                    chil[idx].solid = false;
                    chil[idx].children = None;
                    chil[idx].position = t;
                    chil[idx].scale = s;
                    chil[idx].refine = None;
                }
            }
        }

        self.children = Some(chil);

    }

    pub fn count_nodes(&self) -> usize {
        let mut count = 0;

        if self.children.is_some() {
            self.children.as_ref().unwrap()
                .iter()
                .for_each(|child| {
                    count += child.count_nodes();
                });
        }

        count + 1
    }

    pub fn count_leaves(&self) -> i64 {
        let mut count = 0i64;

        if self.children.is_some() {
            self.children.as_ref().unwrap()
                .iter()
                .for_each(|child| {
                    count += child.count_leaves();
                });
        } else {
            count += 1
        }

        count
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }
}

impl Default for Node {
    fn default() -> Self {
        Node {
            children: None,
            position: vec3(0.0, 0.0, 0.0),
            color: vec3(0.0, 0.0, 0.0),
            height_values: [0.0; 4],
            scale: 1.0,
            solid: false,
            refine: None
        }
    }
}

pub struct OctreeSystem {
    render_sys: Arc<Mutex<RenderSystem>>,

    update_config: Option<OctreeConfig>,

    // optimization flags
    optimizations: OctreeOptimizations,

    messages: Vec<Message>,

    collecting_data: Arc<AtomicBool>,
    collected_nodes: Arc<AtomicUsize>,

    generate_handle: Option<JoinHandle<()>>,
    upload_handle: Option<JoinHandle<()>>,

    data_queue: Arc<MpscQueue<Result<InstanceData, i8>>>,
    last_viewpoint: Matrix4<f32>,
    dirty: bool,
}

unsafe impl Send for OctreeSystem {}

#[derive(Debug, Copy, Clone)]
pub struct OctreeOptimizations {
    pub frustum_culling: bool,
    pub depth_threshold: f64,
    pub ignore_full: bool,
    pub ignore_inner: bool,
    pub depth_culling: bool,
}

impl Default for OctreeOptimizations {
    fn default() -> Self {
        OctreeOptimizations {
            frustum_culling: false,
            depth_threshold: 50.0,
            ignore_full: false,
            ignore_inner: false,
            depth_culling: false,
        }
    }
}

impl Payload for OctreeOptimizations {}

impl Drop for OctreeSystem {
    fn drop(&mut self) {
        match self.generate_handle.take() {
            Some(handle) => { handle.join().unwrap(); },
            _ => {}
        }

        self.data_queue.enqueue(Err(127)).unwrap(); // inform thread to exit
        match self.upload_handle.take() {
            Some(handle) => { handle.join().unwrap(); },
            _ => {}
        }

        return;
    }
}

impl OctreeSystem {
    pub fn new(render_sys: Arc<Mutex<RenderSystem>>) -> Arc<Mutex<Self>> {
        let mut collect_buf = Vec::new();
        collect_buf.resize(1 as usize,
                           InstanceData::default());

        Arc::new(Mutex::new(OctreeSystem {
            render_sys,

            update_config: None,

            optimizations: OctreeOptimizations::default(),

            messages: Vec::new(),

            collecting_data: Arc::new(AtomicBool::new(false)),
            collected_nodes: Arc::new(AtomicUsize::new(0)),

            generate_handle: None,
            upload_handle: None,

            data_queue: Arc::new(MpscQueue::new()),
            last_viewpoint: Matrix4::identity(),
            dirty: true
        }))
    }

    fn init_data_generation(collecting_data: Arc<AtomicBool>,
                            optimization_data: OptimizationData,
                            config: OctreeConfig,
                            root: Arc<Mutex<Node>>,
                            collected_nodes: Arc<AtomicUsize>,
                            data_queue: Arc<MpscQueue<Result<InstanceData, i8>>>,
                            node_pool: &SharedArena<NodeChildren>
    ) {
        let mut root_lock = root.lock().unwrap();

        collecting_data.store(true, Ordering::SeqCst);

        OctreeSystem::generate_instance_data(
            &optimization_data,
            config,
            &mut root_lock,
            &collected_nodes,
            &collecting_data,
            &data_queue,
            node_pool,
            1
        );

        data_queue.enqueue(Err(0)).unwrap(); // finished or canceled traversal

        // let stats = node_pool.stats();
        // let allocation = stats.0 + stats.1;
        // let b = allocation * std::mem::size_of::<Node>();
        // println!("Arena stats: {:?}; Arena size: {:?}mb", stats,  b / (1024*1024));
        collecting_data.store(false, Ordering::SeqCst);
    }

    fn generate_instance_data(
        optimization_data: &OptimizationData,
        config: OctreeConfig,
        node: &mut Node,
        atomic_counter: &AtomicUsize,
        collecting_data: &AtomicBool,
        data_queue: &Arc<MpscQueue<Result<InstanceData, i8>>>,
        node_pool: &SharedArena<NodeChildren>,
        depth: u64
    ) {
        if node.children.is_none() { return; }

        if !collecting_data.load(Ordering::SeqCst) {
            return;
        }

        let camera_pos = optimization_data.camera_transform.position;
        let camera_mag = camera_pos.magnitude();
        let camera_dir = optimization_data.camera_transform.rotation.rotate_vector(-Vector3::unit_z());

        node.children.as_mut().unwrap().as_mut()
            .sort_unstable_by(|a, b| {
            let dist_a = camera_dir.extend(camera_mag)
                .dot((a.position).extend(1.0));
            let dist_b = camera_dir.extend(camera_mag)
                .dot((b.position).extend(1.0));

            dist_a.partial_cmp(&dist_b).unwrap()
        });

        let children = node.children.as_mut().unwrap().as_mut();
        children
            .par_iter_mut()
            .for_each(|child| {
                let limit_depth_reached = limit_depth_traversal(optimization_data, child);

                // add transformation data
                let include = (limit_depth_reached || !child.refine.unwrap_or(false)) && child.solid;

                if include {
                    data_queue.enqueue(Ok(InstanceData {
                        transformation: child.position.extend(child.scale).into(),
                        color: child.color.extend(0.0).into(),
                    })).unwrap();
                }

                // check weather to traverse further
                // IMPORTANT NOTE: it seems that the transformation of the frustum into the octree
                // space causes problems with floating point precision. Thus the blocks are culled
                // too early at the near plane. Either use f64 instead of f32 for frustum culling or
                // embed the octree directly in world space.
                let intersect_frustum = cull_frustum(optimization_data, child);
                let continue_traversal = intersect_frustum && !limit_depth_reached;

                if continue_traversal {
                    if child.is_leaf() {
                        fractal_generators::build_tree(child, config,depth, depth+1, node_pool);
                    }

                    OctreeSystem::generate_instance_data(
                        optimization_data,
                        config,
                        child,
                        atomic_counter,
                        collecting_data,
                        data_queue,
                        node_pool,
                        depth + 1
                    );
                } else {
                    child.children.take(); // drop children
                }
            });
    }
}

impl System for OctreeSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![
            crate::filter!(Octree, Mesh, Transformation),
            crate::filter!(Camera, Transformation),
        ]
    }

    fn handle_input(&mut self, _event: &Event<()>) {}

    fn consume_messages(&mut self, messages: &Vec<Message>) {
        messages.iter().for_each(|msg| {
            if msg.is_type::<OctreeConfig>() {
                self.update_config = Some(msg.get_payload::<OctreeConfig>().unwrap().clone())
            }
            if msg.is_type::<OctreeOptimizations>() {
                self.optimizations = msg.get_payload::<OctreeOptimizations>().unwrap().clone();
            }
            if msg.is_type::<Exit>() {
                if self.generate_handle.is_some() {
                    self.collecting_data.store(false, Ordering::SeqCst); // stop generating nodes
                    let handle = self.generate_handle.take().unwrap();
                    handle.join().unwrap();
                }

                if self.upload_handle.is_some() {
                    let handle = self.upload_handle.take().unwrap();
                    self.data_queue.enqueue(Err(127)).unwrap(); // exit code
                    handle.join().unwrap();
                }
            }
        });
    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, _delta_time: Duration) {
        let octree_entities = &filter[0].lock().unwrap().entities;
        let camera_entities = &filter[1].lock().unwrap().entities;

        for entity in octree_entities {
            let mut entitiy_mutex = entity.lock().unwrap();
            let octree_transform = entitiy_mutex
                .get_component::<Transformation>()
                .ok()
                .unwrap();
            let mut octree = entitiy_mutex
                .get_component::<Octree>()
                .ok()
                .unwrap()
                .clone();

            let mut settings_modified = false;
            if self.update_config.is_some() {

                if self.generate_handle.is_some() {
                    self.collecting_data.store(false, Ordering::SeqCst); // stop generating nodes
                    let handle = self.generate_handle.take().unwrap();
                    handle.join().unwrap();
                }

                if self.upload_handle.is_some() {
                    let handle = self.upload_handle.take().unwrap();
                    self.data_queue.enqueue(Err(127)).unwrap(); // exit code
                    handle.join().unwrap();
                }

                let mut conf = self.update_config.take().unwrap();
                let reset = conf.reset.take().unwrap_or(false);
                if reset {
                    // need to create new octree
                    octree = Octree::new(&self.render_sys, conf);
                } else {
                    // replacing root to initialize clean traversal
                    octree.reconfigure(&self.render_sys, conf);
                }

                settings_modified = true;
            }

            if !camera_entities.is_empty() {
                // camera data
                let camera_mutex = camera_entities[0].lock().unwrap();
                let camera_transform = camera_mutex
                    .get_component::<Transformation>()
                    .ok()
                    .unwrap()
                    .clone();
                let camera = camera_mutex.get_component::<Camera>().ok().unwrap().clone();

                let view_changed = camera_transform.model_matrix != self.last_viewpoint;
                self.last_viewpoint = camera_transform.model_matrix;

                // filter config
                let mut traversal_fnc: Vec<&TraversalFunction> = Vec::new();
                if self.optimizations.frustum_culling {
                    traversal_fnc.push(&cull_frustum); // TODO Fix broken culling at near plane
                }

                let mut filter_fnc: Vec<&FilterFunction> = Vec::new();
                // filter_fnc.push(&generate_leaf_model_matrix);
                filter_fnc.push(&filter_is_solid);

                if self.optimizations.depth_culling {
                    traversal_fnc.push(&limit_depth_traversal);
                    filter_fnc.push(&limit_depth_filter);
                }

                let optimization_data = OptimizationData {
                    camera: camera.clone(),
                    camera_transform: camera_transform.clone(),
                    octree_mvp: camera.projection
                        * Matrix4::inverse_transform(&camera_transform.get_model_matrix()).unwrap()
                        * octree_transform.get_model_matrix(),
                    frustum: camera.frustum.transformed(
                        // TODO: investigate whether this space transformation is the a good approach for frustum culling
                        Matrix4::inverse_transform(&octree_transform.get_model_matrix()).unwrap()
                            * camera_transform.get_model_matrix(),
                    ),
                    octree_scale: octree_transform.scale.x,
                    config: octree.config.clone(),
                };

                let instance_data_start = Instant::now();

                if view_changed {
                    self.dirty = true;
                }

                let gen_data = (!view_changed && self.dirty && octree.config.continuous_update.unwrap_or(true))
                    || settings_modified;

                if gen_data { //|| !self.collecting_data.load(Ordering::SeqCst) {

                    self.dirty = false;

                    if self.generate_handle.is_some() {
                        self.collecting_data.store(false, Ordering::SeqCst);
                    }

                    let collecting_data = self.collecting_data.clone();
                    let collected_nodes = self.collected_nodes.clone();
                    let data_queue = self.data_queue.clone();
                    let octree_root = octree.root.clone();
                    let octree_config = octree.config.clone();
                    let node_pool = octree.node_pool.clone();

                    self.generate_handle = Some(std::thread::spawn(move || {
                        OctreeSystem::init_data_generation(
                            collecting_data,
                            optimization_data,
                            octree_config,
                            octree_root,
                            collected_nodes,
                            data_queue,
                            &node_pool
                        )
                    }));

                }

                if self.upload_handle.is_none() {
                    let active_buffer = octree.active_instance_buffer_idx.clone();
                    let ring_buffer = octree.instance_data_buffer.clone();
                    let ring_buffer_len = octree.instance_data_buffer.len();
                    let collected_nodes = self.collected_nodes.clone();
                    let max_len = octree.config.max_rendered_nodes.unwrap_or(1e6 as u64) as usize;

                    let data_queue = self.data_queue.clone();

                    self.upload_handle = Some(std::thread::spawn(move || {

                        let mut upload_buffer = VecDeque::new();
                        let mut since_empty = 0;
                        let mut prev_blocks = 0;

                        loop {
                            sleep(Duration::from_millis(50));

                            while let Some(data) = data_queue.dequeue() {
                                match data {
                                    Ok(data) => {
                                        since_empty += 1;

                                        upload_buffer.push_back(data);

                                        if prev_blocks > 0 {
                                            upload_buffer.pop_front();
                                            prev_blocks -= 1;
                                        }
                                    }
                                    Err(0) => { // complete or canceled
                                        for _i in 0 .. prev_blocks {
                                            upload_buffer.pop_front();
                                        }

                                        prev_blocks = since_empty;
                                        since_empty = 0;
                                    }
                                    Err(127) => {
                                        return; // special thread exit code
                                    }
                                    _ => ()
                                }
                            }

                            while upload_buffer.len() > max_len {
                                upload_buffer.pop_back();
                                since_empty -= 1;
                            }

                            let buffer_idx = (active_buffer.load(Ordering::SeqCst) + 1)
                                % ring_buffer_len;
                            let gpu_buffer = ring_buffer[buffer_idx].clone();

                            let upload_len = upload_buffer.len().min(max_len);

                            gpu_buffer.lock().unwrap().replace_data(&upload_buffer.make_contiguous()[0..upload_len]);

                            active_buffer.store(buffer_idx, Ordering::SeqCst);
                            collected_nodes.store(upload_len, Ordering::SeqCst);
                        }
                    }));
                }

                // octree.info.render_count = 4e6 as usize;//self.collected_nodes.load(Ordering::SeqCst);
                octree.info.render_count = self.collected_nodes.load(Ordering::SeqCst);

                let instance_data_end = Instant::now();
                let instance_data_duration = instance_data_end - instance_data_start;
                self.messages.push(Message::new(ProfilingData {
                    instance_data_generation: Some(instance_data_duration.as_millis() as u64),
                    ..Default::default()
                }));

                self.messages.push(Message::new(ProfilingData {
                    rendered_nodes: Some(octree.info.render_count as u32),
                    ..Default::default()
                }));
            } else {
                // drop locks
                println!("no camera provided");
            }

            entitiy_mutex.add_component(octree);
        }
    }

    fn get_messages(&mut self) -> Vec<Message> {
        let mut ret = vec![];

        if !self.messages.is_empty() {
            ret = self.messages.clone();
            self.messages.clear();
        }

        ret
    }
}

type FilterFunction = dyn Fn(&OptimizationData, &Node) -> bool;
type TraversalFunction = dyn Fn(&OptimizationData, &Node) -> bool;

struct OptimizationData {
    camera: Camera,
    camera_transform: Transformation,
    octree_mvp: Matrix4<f32>,
    octree_scale: f32,
    frustum: Frustum,
    config: OctreeConfig,
}

fn cull_frustum(optimization_data: &OptimizationData, node: &Node) -> bool {
    // aabb test
    // generate aabb on demand to save memory
    // let min = node.position - Vector3::from_value(node.scale) / 2.0;
    // let max = node.position + Vector3::from_value(node.scale) / 2.0;
    // let aabb = AABB::new(min, max);
    //
    // optimization_data.frustum.intersect(&aabb)

    // sphere test
    let radius = (node.scale * 0.5 * node.scale * 0.5 * 3.0).sqrt();
    optimization_data.frustum.intersect_sphere(node.position, radius)
}

fn limit_depth_filter(optimization_data: &OptimizationData, node: &Node) -> bool {
    !limit_depth_traversal(optimization_data, node) && node.solid
}

fn limit_depth_traversal(optimization_data: &OptimizationData, node: &Node) -> bool {
    let proj_matrix = optimization_data.octree_mvp;

    let projected_position = proj_matrix * node.position.extend(1.0);
    if projected_position.w <= 1.0 {
        true;
    }

    // let mut projected_scale = node.scale / projected_position.w; // fixed linear scaling
    let exponent = optimization_data.config.distance_scale.unwrap(); // 1 = linear scaling
    let mut projected_scale = node.scale / projected_position.w; // * 2.0.pow(1.0/exponent as f32) as f32);
    let depth = (projected_position.z / projected_position.w).max(0.0);
    // let dist = (projected_position.truncate() / projected_position.w).magnitude().max(0.0);
    // let exp = (dist as f32 / exponent as f32).max(1.0);
    let exp = (depth as f32 / exponent as f32).max(1.0);
    projected_scale = projected_scale.pow(exp as f32);
    projected_scale *= optimization_data.camera.resolution[0] / 2.0;


    let target_size = optimization_data.config.subdiv_threshold.unwrap() as f32 / optimization_data.octree_scale;
    return if projected_scale.abs() < target_size as f32 {
        true
    } else {
        false
    };
}

fn filter_is_solid(_: &OptimizationData, node: &Node) -> bool {
    return node.solid && node.is_leaf();
}
