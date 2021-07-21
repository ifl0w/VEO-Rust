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

use num_derive::{FromPrimitive, ToPrimitive};
use num_traits::{FromPrimitive, ToPrimitive};

use crate::core::{Component, Filter, Message, Payload, System};
use crate::rendering::{Camera, Frustum, GPUBuffer, InstanceData, Mesh, RenderSystem, Transformation};
use crate::rendering::nse_gui::octree_gui::ProfilingData;

use std::thread::{sleep, JoinHandle};
use riffy::MpscQueue;

use std::collections::VecDeque;
use std::fmt::Debug;
use std::fmt;
use crate::rendering::FractalSelection::MandelBulb;
use std::fs::read;
use self::rand::random;
use std::ops::{Deref, DerefMut};
use shared_arena::{SharedArena, ArenaBox, ArenaArc, ArenaRc};

const SUBDIVISIONS: usize = 2;
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

#[derive(Clone, Copy, Debug, PartialEq, FromPrimitive, ToPrimitive)]
pub enum FractalSelection {
    MandelBulb = 0,
    MandelBrot,
    SierpinskyPyramid,
    MengerSponge
}

impl fmt::Display for FractalSelection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

#[derive(Clone, Debug, Copy)]
pub struct OctreeConfig {
    pub max_rendered_nodes: Option<u64>,
    pub subdiv_threshold: Option<f64>,
    pub threshold_scale: Option<f64>,
    pub depth: Option<u64>,
    pub fractal: Option<FractalSelection>,
    pub continuous_update: Option<bool>,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        OctreeConfig {
            max_rendered_nodes: Some(4e6 as u64),
            subdiv_threshold: Some(20.0),
            threshold_scale: Some(1.0),
            depth: Some(DEFAULT_DEPTH),
            fractal: Some(MandelBulb),
            continuous_update: Some(true)
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
        let mut rm_lock = rm.lock().unwrap();

        let dev = render_system.lock().unwrap().renderer.device.clone();
        let adapter = render_system.lock().unwrap().renderer.adapter.clone();

        // read config
        let depth = config.depth.unwrap_or(DEFAULT_DEPTH);
        let max_rendered_nodes = config.max_rendered_nodes.unwrap_or(4e6 as u64);

        // byte size calculations
        let max_num_nodes = (SUBDIVISIONS as i64).pow(3 * depth as u32) as usize;
        let max_byte_size = std::mem::size_of::<Node>() * max_num_nodes;
        let max_gpu_byte_size = std::mem::size_of::<InstanceData>() * max_rendered_nodes as usize;

        let ring_buffer_length = 2;
        let mut instance_data_buffer = Vec::with_capacity(ring_buffer_length);
        for _ in 0..ring_buffer_length {
            let (_, buffer) = rm_lock.add_buffer(GPUBuffer::new_with_size(
                &dev,
                &adapter,
                max_gpu_byte_size,
                buffer::Usage::STORAGE | buffer::Usage::VERTEX,
            ));

            instance_data_buffer.push(buffer);
        }

        let octree_info = OctreeInfo {
            render_count: 0,
            byte_size: 0,
            max_num_nodes,
            max_byte_size,
            gpu_byte_size: max_gpu_byte_size * ring_buffer_length,
        };

        let arena = Arc::new(SharedArena::new());
        let mut oct = Octree {
            root: Arc::new(Mutex::new(Node::new_inner(vec3(0.0, 0.0, 0.0), 1.0))),
            instance_data_buffer,
            active_instance_buffer_idx: Arc::new(AtomicUsize::new(0)),

            config,
            info: octree_info,

            node_pool: arena
        };

        let start = std::time::Instant::now();
        {
            Octree::build_tree(
                &mut oct.root.lock().unwrap(),
                oct.config,
                0,
                1,
                &oct.node_pool
            );
        }
        let end = std::time::Instant::now();

        println!("Octree build time: {:?}", end - start);

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
        let max_num_nodes = (SUBDIVISIONS as i64).pow(3 * depth as u32) as usize;
        let max_byte_size = std::mem::size_of::<Node>() * max_num_nodes;
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

    fn build_tree(node: &mut Node, config: OctreeConfig, current_depth: u64, target_depth: u64, node_pool: &SharedArena<NodeChildren>) {
        node.populate(node_pool);

        node.children.as_mut().unwrap().as_mut()
            .iter_mut()
            .for_each(|child| {
                let new_depth = current_depth + 1;

                let traverse = {
                    // let child = id; //node_pool.get_mut(*id).unwrap();

                    let zoom = 2.5;

                    match config.fractal {
                        Some(FractalSelection::MandelBulb) =>
                            Octree::generate_mandelbulb(child, zoom, current_depth + 1),
                        Some(FractalSelection::MandelBrot) =>
                            Octree::generate_mandelbrot(child, zoom, current_depth + 1),
                        Some(FractalSelection::SierpinskyPyramid) =>
                            Octree::generate_sierpinsky(child, zoom, current_depth),
                        Some(FractalSelection::MengerSponge) =>
                            Octree::generate_menger(child, zoom, current_depth),
                        None => false,
                        _ => false,
                    }
                };

                if current_depth < target_depth && traverse {
                    Octree::build_tree(child, config, new_depth, target_depth, node_pool)
                }
            });
    }

    fn generate_mandelbulb(child: &mut Node, zoom: f32, depth: u64) -> bool {
        let origin = &child.position;
        let scale = child.scale;

        let position = origin * zoom;

        let escape_radius = 3.0 as f64;
        let mut iter = 10 * ((depth as f64).log2() as i32 + 1);

        fn to_spherical(a: Vector3<f64>) -> Vector3<f64> {
            let r = a.magnitude();
            let mut phi = (a.y / a.x).atan();
            let mut theta = (a.z / r).acos();

            // handle 0/0
            if a.y == 0.0 && a.x == 0.0 { phi = 0.0; };
            if a.z == 0.0 && r == 0.0 { theta = 0.0; };

            return vec3(r, phi, theta);
        }
        ;

        fn to_cartesian(a: Vector3<f64>) -> Vector3<f64> {
            let x = a.z.sin() * a.y.cos();
            let y = a.y.sin() * a.z.sin();
            let z = a.z.cos();

            return a.x * vec3(x, y, z);
        }
        ;

        // nth power in polar coordinates
        fn spherical_pow(a: Vector3<f64>, n: f64) -> Vector3<f64> {
            let r = a.x.pow(n);
            let phi = n * a.y;
            let theta = n * a.z;
            return vec3(r, phi, theta);
        }

        let c = vec3(position.x as f64, position.y as f64, position.z as f64);

        // z_0 = 0 + i0
        let mut v = vec3(0.0, 0.0, 0.0);
        let mut r = 0.0;

        // z_0' = 1 + 0i
        let mut dr = 1.0;

        let n = 8.0;
        while iter > 0 {
            let v_p = to_spherical(v);

            r = v_p.x;
            if r as f64 > escape_radius {
                break;
            }

            // scalar distance estimation
            // source: http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
            dr = r.pow(n - 1.0) * n * dr + 1.0;

            let v_next = spherical_pow(v_p, n);
            v = to_cartesian(v_next) + c;

            iter -= 1;
        }

        // values
        let distance = 0.5 * r * r.ln() / dr;

        let half_length = (scale * 0.5 * zoom) as f64;
        let radius = (half_length * half_length * 3.0).sqrt();

        let mut traverse = false;

        if distance.abs() <= radius {
            traverse = true;
            child.solid = true;
        }

        return traverse;
    }

    fn generate_mandelbrot(child: &mut Node, zoom: f32, _depth: u64) -> bool {
        let origin = &child.position;
        let scale = child.scale;

        // only a slice
        let thickness = 0.0;
        if origin.y + child.scale * 0.5 < -thickness
            || origin.y - child.scale * 0.5 > thickness {
            return false;
        };

        let position = origin * zoom - vec3(0.5, 0.0, 0.0);

        let escape_radius = 4.0 * 10000.0 as f64;
        let mut iter = 100;

        let c_re = position.x as f64;
        let c_im = position.z as f64;

        // z_0 = 0 + i0
        let mut z_re = 0.0;
        let mut z_im = 0.0;
        let mut z_re2 = z_re * z_re;
        let mut z_im2 = z_im * z_im;

        // z_0' = 1 + 0i
        let mut zp_re = 1.0;
        let mut zp_im = 0.0;

        while iter > 0 {
            zp_re = 2.0 * (z_re * zp_re - z_im * zp_im) + 1.0;
            zp_im = 2.0 * (z_re * zp_im + z_im * zp_re);

            let z_re_new = z_re2 - z_im2 + c_re;
            let z_im_new = 2.0 * z_re * z_im + c_im;
            z_re = z_re_new;
            z_im = z_im_new;
            z_re2 = z_re * z_re;
            z_im2 = z_im * z_im;

            let val2: f64 = z_re2 + z_im2;

            if val2 > (escape_radius * escape_radius) {
                return false;
            }

            iter -= 1;
        }

        // values
        let z_val = (z_re2 + z_im2).sqrt();
        let zp_val = (zp_re * zp_re + zp_im * zp_im).sqrt();

        // let dist = 2.0 * z_val * z_val.ln() / zp_val;
        // let dist = z_val * z_val.ln() / zp_val;
        let distance = 0.5 * z_val * z_val.ln() / zp_val;

        let half_length: f64 = (scale * 0.5 * zoom) as f64;
        let radius = (half_length * half_length * 2.0).sqrt() as f64;

        let mut traverse = false;

        if distance <= radius as f64 {
            traverse = true;

            if distance <= 0.0 {
                child.solid = true;
            }
        }

        return traverse;
    }

    fn generate_menger(child: &mut Node, _zoom: f32, depth: u64) -> bool {
        let s = child.scale;
        let p = child.position;

        fn iterate(p: Vector3<f32>, s: f32, bb_center: Vector3<f32>, bb_size: f32, n: i32) -> bool {
            if n == 0 { return true; }

            // bounding box of the current iteration/contraction
            let bb_min = bb_center - Vector3::from_value(bb_size);
            let bb_max = bb_center + Vector3::from_value(bb_size);

            // bounding box of node
            let node_min = p - Vector3::from_value(s);
            let node_max = p + Vector3::from_value(s);

            // test node bb and iteration bb intersection
            if node_max.x > bb_min.x && node_min.x < bb_max.x
                && node_max.y > bb_min.y && node_min.y < bb_max.y
                && node_max.z > bb_min.z && node_min.z < bb_max.z {

                // calculate contraction bounding size
                let c_size = bb_size * 1.0 / 3.0;
                let offset = bb_size * 2.0 / 3.0;

                // calculate next contraction
                // note: the actual iteration of the IFS
                let mut bounding = [vec3(0.0,0.0,0.0); 20];

                let mut i = 0;
                for x in -1..=1 {
                    for y in -1..=1 {
                        for z in -1..=1 {
                            let mut axis_count = 0;
                            if x == 0 { axis_count += 1; }
                            if y == 0 { axis_count += 1; }
                            if z == 0 { axis_count += 1; }

                            if axis_count != 2 && axis_count != 3 {
                                bounding[i] = bb_center + vec3(
                                    x as f32 * offset,
                                    y as f32 * offset,
                                    z as f32 * offset
                                );
                                i += 1;
                            }
                        }
                    }
                }

                // check if any of the next bounding volumes intersects with the node
                // if none does then we definitely have a node that we do not need
                // to consider anymore
                let inside = bounding.iter().any(|bb| {
                    iterate(p, s, *bb, c_size, n - 1)
                });

                return inside;
            }

            // they do not intersect
            return false;
        }

        let inside = iterate(p, s / SUBDIVISIONS as f32, vec3(0.0, 0.0, 0.0), 0.5, 15);

        if inside {
            child.solid = true;
        }

        return inside;
    }

    fn generate_sierpinsky(child: &mut Node, _zoom: f32, _depth: u64) -> bool {
        let s = child.scale;
        let p = child.position;

        fn iterate(p: Vector3<f32>, s: f32, bb_center: Vector3<f32>, bb_size: f32, n: i32) -> bool {
            if n == 0 { return true; }

            // bounding box of the current iteration/contraction
            let bb_min = bb_center - Vector3::from_value(bb_size);
            let bb_max = bb_center + Vector3::from_value(bb_size);

            // bounding box of node
            let node_min = p - Vector3::from_value(s);
            let node_max = p + Vector3::from_value(s);

            // test node bb and iteration bb intersection
            if node_max.x >= bb_min.x && node_min.x <= bb_max.x
                && node_max.y >= bb_min.y && node_min.y <= bb_max.y
                && node_max.z >= bb_min.z && node_min.z <= bb_max.z {

                // calculate contraction bounding size
                let c_size = bb_size * 0.5;

                // calculate next contraction positions
                // note: the actual iteration of the IFS
                // Pyramid (numerically more stable in octree)
                let bounding = [
                    bb_center + vec3(-c_size, -c_size, c_size),
                    bb_center + vec3(-c_size, -c_size, -c_size),
                    bb_center + vec3(c_size, -c_size, -c_size),
                    bb_center + vec3(c_size, -c_size, c_size),
                    bb_center + vec3(0.0, c_size, 0.0),
                ];
                // Tetrahedron (less stable in octree)
                /*let c_size_diag = (c_size * c_size * 0.5).sqrt();
                let bounding = [
                    bb_center + vec3(-c_size_diag, -c_size, c_size_diag),
                    bb_center + vec3(c_size_diag, -c_size, c_size_diag),
                    bb_center + vec3(0.0, -c_size, -c_size),
                    bb_center + vec3(0.0, c_size, 0.0),
                ];*/

                // check if any of the next bounding volumes intersects with the node
                // if none does then we definitely have a node that we do not need
                // to consider anymore
                let inside = bounding.iter().any(|bb| {
                    iterate(p, s, *bb, c_size, n - 1)
                });

                return inside;
            }

            // they do not intersect
            return false;
        }

        let d = iterate(p, s / SUBDIVISIONS as f32, vec3(0.0, 0.0, 0.0), 0.5, 15);

        if d {
            child.solid = true;
        }

        return d;
    }
}

type NodeChildren = [Node; SUBDIVISIONS * SUBDIVISIONS * SUBDIVISIONS];

#[derive(Debug)]
pub struct Node {
    children: Option<ArenaBox<NodeChildren>>,
    position: Vector3<f32>,
    // position of the node
    scale: f32,
    // length of a side of the node
    solid: bool,
}

impl Clone for Node {
    fn clone(&self) -> Self {
        Node {
            position: self.position,
            scale: self.scale,
            solid: self.solid,
            children: None // never clone children!
        }
    }
}

impl Node {
    pub fn new() -> Self {
        let tmp = Node {
            children: None,
            position: vec3(0.0, 0.0, 0.0),
            scale: 1.0,
            solid: false,
        };


        tmp
    }

    pub fn new_inner(position: Vector3<f32>, scale: f32) -> Self {
        let tmp = Node {
            children: None,
            position,
            scale,
            solid: false,
        };

        tmp
    }

    pub fn populate(&mut self, node_pool: &SharedArena<NodeChildren>) {
        if self.children.is_some() {
            return; // nothing to do
        }

        let scale_factor = 1.0 / SUBDIVISIONS as f32;

        let mut chil = node_pool.alloc(Default::default());

        for id_x in 0..SUBDIVISIONS {
            for id_y in 0..SUBDIVISIONS {
                for id_z in 0..SUBDIVISIONS {
                    let s = self.scale * scale_factor;

                    let mut t = self.position;
                    t -= Vector3::from_value(self.scale * 0.5 - s * 0.5);
                    t += vec3(s * id_x as f32, s * id_y as f32, s * id_z as f32);

                    let idx = SUBDIVISIONS.pow(0) * id_x
                        + SUBDIVISIONS.pow(1) * id_y
                        + SUBDIVISIONS.pow(2) * id_z;

                    chil[idx].solid = false;
                    chil[idx].children = None;
                    chil[idx].position = t;
                    chil[idx].scale = s;
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
            scale: 1.0,
            // aabb: None,
            solid: false,
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
            Some(handle) => { handle.join(); },
            _ => {}
        }

        self.data_queue.enqueue(Err(127)); // inform thread to exit
        match self.upload_handle.take() {
            Some(handle) => { handle.join(); },
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
            node_pool
        );

        if collecting_data.load(Ordering::SeqCst) { // really finished
            data_queue.enqueue(Err(0)); // completely finished traversal
        } else {
            data_queue.enqueue(Err(1)); // cancelled traversal
        }

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
        node_pool: &SharedArena<NodeChildren>
    ) {
        if node.children.is_none() { return; }

        if !collecting_data.load(Ordering::SeqCst) {
            return;
        }

        let mut traverse_children = Vec::with_capacity(SUBDIVISIONS.pow(3));

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

        node.children.as_mut().unwrap()
            .as_mut()
            .iter_mut()
            .for_each(|child| {
                let limit_depth_reached = limit_depth_traversal(optimization_data, child);

                // add transformation data
                let include = limit_depth_reached && child.solid;

                if include {

                    data_queue.enqueue(Ok(InstanceData {
                        transformation: child.position.extend(child.scale).into(),
                    }));
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
                        Octree::build_tree(child, config,0, 1, node_pool);
                    }

                    traverse_children.push(child);
                } else {
                    child.children.take(); // drop children
                }
            });

        traverse_children
            .par_iter_mut()
            .for_each(|child| OctreeSystem::generate_instance_data(
                    optimization_data,
                    config,
                    child,
                    atomic_counter,
                    collecting_data,
                    data_queue,
                    node_pool
                )
            );
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
                let conf = self.update_config.take().unwrap();
                if conf.fractal.unwrap() != octree.config.fractal.unwrap() {
                    // need to create new octree
                    octree = Octree::new(&self.render_sys, conf);
                } else {
                    // octree = Octree::new(&self.render_sys, conf);
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

                if settings_modified && self.generate_handle.is_some() {
                    self.collecting_data.store(false, Ordering::SeqCst); // stop generating nodes
                    let handle = self.generate_handle.take().unwrap();
                    handle.join();
                }

                if settings_modified && self.upload_handle.is_some() {
                    let handle = self.upload_handle.take().unwrap();
                    self.data_queue.enqueue(Err(127)); // exit code
                    handle.join();
                }

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
                    let mut node_pool = octree.node_pool.clone();

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
                    let _collecting_data = self.collecting_data.clone();
                    let collected_nodes = self.collected_nodes.clone();
                    let max_len = octree.config.max_rendered_nodes.unwrap_or(1e6 as u64) as usize;

                    let data_queue = self.data_queue.clone();

                    self.upload_handle = Some(std::thread::spawn(move || {

                        let mut upload_buffer = VecDeque::new();
                        let mut since_empty = 0;
                        let mut block_sizes = VecDeque::new();
                        let mut exit = false;

                        loop {
                            sleep(Duration::from_millis(50));

                            while let Some(data) = data_queue.dequeue() {
                                match data {
                                    Ok(data) => {
                                        since_empty += 1;

                                        let mut val = block_sizes.pop_front().unwrap_or(0);

                                        upload_buffer.push_back(data);

                                        if val > 1 {
                                            upload_buffer.pop_front();
                                            val -= 1;
                                        }

                                        if val > 0 {
                                            block_sizes.push_front(val);
                                        }
                                    }
                                    Err(0) => { // complete
                                        for _i in 0 .. block_sizes.pop_front().unwrap_or(0) {
                                            upload_buffer.pop_front();
                                        }

                                        block_sizes.push_back(since_empty);
                                        since_empty = 0;
                                    }
                                    Err(1) => {
                                        let val = block_sizes.pop_front().unwrap_or(0);
                                        block_sizes.push_front(val + since_empty);
                                        since_empty = 0;
                                    }
                                    Err(127) => {
                                        exit = true; // special thread exit code
                                    }
                                    _ => ()
                                }
                            }

                            while upload_buffer.len() > max_len {
                                upload_buffer.pop_front();
                            }

                            let buffer_idx = (active_buffer.load(Ordering::SeqCst) + 1)
                                % ring_buffer_len;
                            let gpu_buffer = ring_buffer[buffer_idx].clone();

                            let upload_len = upload_buffer.len().min(max_len);

                            gpu_buffer.lock().unwrap().replace_data(&upload_buffer.make_contiguous()[0..upload_len]);

                            active_buffer.store(buffer_idx, Ordering::SeqCst);
                            collected_nodes.store(upload_len, Ordering::SeqCst);

                            if exit {
                                return;
                            }
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
    let exponent = optimization_data.config.threshold_scale.unwrap(); // 1 = linear scaling
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

fn limit_solid_traversal(_: &OptimizationData, node: &Node) -> bool {
    return !node.solid;
}

fn generate_leaf_model_matrix(_: &OptimizationData, node: &Node) -> bool {
    node.is_leaf()
}

fn filter_is_solid(_: &OptimizationData, node: &Node) -> bool {
    return node.solid && node.is_leaf();
}
