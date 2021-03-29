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

use std::convert::{TryFrom, TryInto};
use std::result::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cgmath::{Array, Matrix4, Transform, vec3, Vector3, Rotation, InnerSpace};
use gfx_hal::buffer;
use winit::event::Event;

use crate::core::{Component, Filter, Message, Payload, System};
use crate::rendering::{Camera, Frustum, GPUBuffer, InstanceData, Mesh, RenderSystem, Transformation};
use crate::rendering::nse_gui::octree_gui::ProfilingData;
use cgmath::num_traits::Pow;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::mem::size_of;

enum NodePosition {
    Flt = 0,
    Frt = 1,
    Flb = 2,
    Frb = 3,
    Blt = 4,
    Brt = 5,
    Blb = 6,
    Brb = 7,
}

impl TryFrom<i32> for NodePosition {
    type Error = ();

    fn try_from(v: i32) -> Result<Self, Self::Error> {
        match v {
            x if x == NodePosition::Flt as i32 => Ok(NodePosition::Flt),
            x if x == NodePosition::Frt as i32 => Ok(NodePosition::Frt),
            x if x == NodePosition::Flb as i32 => Ok(NodePosition::Flb),
            x if x == NodePosition::Frb as i32 => Ok(NodePosition::Frb),
            x if x == NodePosition::Blt as i32 => Ok(NodePosition::Blt),
            x if x == NodePosition::Brt as i32 => Ok(NodePosition::Brt),
            x if x == NodePosition::Blb as i32 => Ok(NodePosition::Blb),
            x if x == NodePosition::Brb as i32 => Ok(NodePosition::Brb),
            _ => Err(()),
        }
    }
}

#[derive(Clone)]
pub struct Octree {
    pub root: Arc<Mutex<Node>>,

    pub instance_data_buffer: Vec<Arc<Mutex<GPUBuffer<Backend::Backend>>>>,
    /// points into the instance_data_buffer vec
    pub active_instance_buffer_idx: Option<usize>,

    pub config: OctreeConfig,
    pub info: OctreeInfo,

    collect_buffer: Arc<Vec<InstanceData>>,
}

#[derive(Clone, Debug)]
pub struct OctreeConfig {
    pub max_rendered_nodes: Option<u64>,
    pub depth: Option<u64>,
}

impl Default for OctreeConfig {
    fn default() -> Self {
        OctreeConfig {
            max_rendered_nodes: Some(5e5 as u64),
            depth: Some(5),
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
        let depth = config.depth.unwrap_or(5);
        let max_rendered_nodes = config.max_rendered_nodes.unwrap_or(5e5 as u64);

        // byte size calculations
        let max_num_nodes = 8_i64.pow(depth.try_into().unwrap()) as usize;
        let max_byte_size = std::mem::size_of::<Octree>() * max_num_nodes;
        let max_gpu_byte_size = std::mem::size_of::<InstanceData>() * max_rendered_nodes as usize;

        let ring_buffer_length = 3;
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

        let mut collect_buf = Vec::new();
        collect_buf.resize(config.max_rendered_nodes.unwrap_or(0) as usize,
                           InstanceData::default());

        let mut oct = Octree {
            root: Arc::new(Mutex::new(
                Node::new_inner(vec3(0.0, 0.0, 0.0), 1.0)
            )),
            instance_data_buffer,
            active_instance_buffer_idx: Some(0),

            config,
            info: octree_info,

            collect_buffer: Arc::new(collect_buf)
        };

        oct.root.lock().unwrap().populate();

        let start = std::time::Instant::now();
        Arc::new(Mutex::new(Octree::build_tree(
            &mut oct.root.lock().unwrap(),
            0,
            depth,
        )));
        let end = std::time::Instant::now();

        println!("Octree build time: {:?}", end-start);

        oct.info.byte_size = self::Octree::size_in_bytes(&oct);

        oct
    }

    pub fn get_instance_buffer(&self) -> Option<&Arc<Mutex<GPUBuffer<Backend::Backend>>>> {
        if self.active_instance_buffer_idx.is_some() {
            Some(
                self.instance_data_buffer
                    .get(self.active_instance_buffer_idx.unwrap())
                    .unwrap(),
            )
        } else {
            None
        }
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
            node.children.as_ref().unwrap()
                .iter()
                .for_each(|child| {
                    count += self.count_nodes(child);
                });
        }

        count + 1
    }

    fn build_tree(node: &mut Node, current_depth: u64, target_depth: u64) {
        if node.children.is_some() {
            node.children.as_mut().unwrap()
                .par_iter_mut()
                .enumerate()
                .for_each(|(idx, child)| {
                    let new_depth = current_depth + 1;

                    let zoom = 2.5;
                    // let traverse = Octree::generate_mandelbrot(child, zoom, current_depth + 1);
                    let traverse = Octree::generate_mandelbulb(child, zoom, current_depth + 1);
                    // let traverse = Octree::generate_sierpinsky(child, zoom, current_depth);
                    // let traverse = Octree::generate_menger(child, zoom, current_depth);

                    if current_depth < target_depth && traverse {
                        child.populate();
                        Octree::build_tree(child, new_depth, target_depth)
                    }
                });
        }
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
        };

        fn to_cartesian(a: Vector3<f64>) -> Vector3<f64> {
            let x = a.z.sin() * a.y.cos();
            let y = a.y.sin() * a.z.sin();
            let z = a.z.cos();

            return a.x * vec3(x,y,z);
        };

        // nth power in polar coordinates
        fn spherical_pow(a: Vector3<f64>, n: f64) -> Vector3<f64> {
            let r = a.x.pow(n);
            let phi = n * a.y;
            let theta = n * a.z;
            return vec3(r, phi, theta);
        }

        let mut c = vec3(position.x as f64, position.y as f64, position.z as f64);

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

            let mut v_next = spherical_pow(v_p, n);
            v = to_cartesian(v_next) + c;

            iter -= 1;
        }

        // values
        let distance = 0.5 * r * r.ln() / dr;

        let half_length = (scale * 0.5 * zoom) as f64;
        let radius = (half_length * half_length * 3.0).sqrt();

        let mut traverse = false;

        // added a small bias to the radius to ensure deeper traversal while keeping
        // the octree somewhat sparse. Note: there is probably a better way.
        let bias = 1.5;
        if distance.abs() <= radius * bias {
            traverse = true;
        }

        if distance.abs() <= radius {
            child.solid = true;
        }

        return traverse;
    }

    fn generate_mandelbrot(child: &mut Node, zoom: f32, depth: u64) -> bool {
        let origin = &child.position;
        let scale = child.scale;

        // only a slice
        let thickness = 0.01;
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
                break;
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

        if distance < radius as f64 {
            traverse = true;

            if distance <= 0.0 {
                child.solid = true;
            }
        }

        return traverse;
    }

    fn generate_menger(child: &mut Node, zoom: f32, depth: u64) -> bool {
        let t: Vector3<f64> = vec3(
            child.position.x as f64,
            child.position.y as f64,
            child.position.z as f64,
        );
        let s: f64 = child.scale as f64;

        let mut box_points: Vec<Vector3<f64>> = vec![
            t + vec3(-0.5, -0.5, -0.5) * s,
            t + vec3(0.5, -0.5, -0.5) * s,
            t + vec3(-0.5, 0.5, -0.5) * s,
            t + vec3(0.5, 0.5, -0.5) * s,
            t + vec3(-0.5, -0.5, 0.5) * s,
            t + vec3(0.5, -0.5, 0.5) * s,
            t + vec3(-0.5, 0.5, 0.5) * s,
            t + vec3(0.5, 0.5, 0.5) * s,
        ];
        // transform box points into 0 - 1 range
        box_points = box_points.iter()
            .map(|p| { p + vec3(0.50, 0.50, 0.50) }).collect();

        // reduces the effective depth but prevents errors
        // good values are 2 - 4
        let oversampling = 3;
        let e_min = (1.0 / 3.0);
        // let e_min = 0.25;
        let e_max = (2.0 / 3.0);
        // let e_max = 0.75;
        let scale_base = 1.0 / 3.0;
        // let scale_base = 0.25; // 0.5

        let mut in_empty = false;

        let sample_range = (
            (depth as i32 - oversampling * 4).max(0),
            (depth as i32 - oversampling).max(0)
        );
        for i in sample_range.0..sample_range.1 {
            let scale = (scale_base).pow(i as f64);

            in_empty = box_points.iter()
                .all(|p| {
                    // get remainder of linear transform
                    let mut v = (p / scale) - (p / scale).map(|w| { w.floor() });

                    let x = v.x >= e_min && v.x <= e_max;
                    let y = v.y >= e_min && v.y <= e_max;
                    let z = v.z >= e_min && v.z <= e_max;

                    x && y || y && z || x && z
                });

            if in_empty { break; }
        }

        if !in_empty {
            child.solid = true;
        }

        return !in_empty;
    }

    fn generate_sierpinsky(child: &mut Node, zoom: f32, depth: u64) -> bool {
        let s = child.scale;
        let p = child.position;

        fn f1(p: Vector3<f32>, s: f32, max: i32) -> f32 {
            let a = (p.x + 0.5).abs();
            let b = (p.z + 0.5).abs();
            let c = (p.y + 0.5).abs();

            return (a + b + c) - 1.0;
        };

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
                    iterate(p, s, *bb, c_size, n-1)
                });

                return inside;
            }

            // they do not intersect
            return false;
        }

        let d = iterate(p, s * 0.5, vec3(0.0, 0.0,0.0), 0.5,15);

        if d  {
            child.solid = true;
        }

        return d;
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    children: Option<Vec<Node>>,
    position: Vector3<f32>, // position of the node
    scale: f32, // length of a side of the node
    solid: bool,
}

impl Node {
    pub fn new() -> Self {
        let mut tmp = Node {
            children: None,
            position: vec3(0.0, 0.0, 0.0),
            scale: 1.0,
            solid: false,
        };


        tmp
    }

    pub fn new_inner(position: Vector3<f32>, scale: f32) -> Self {
        let mut tmp = Node {
            children: None,
            position,
            scale,
            solid: false,
        };

        tmp
    }

    pub fn populate(&mut self) {
        if self.children.is_some() {
            panic!("Repopulating a octree cell is currently not supported");
        }

        self.children = Some(Vec::with_capacity(8));

        for child_idx in 0..8 {
            let s = self.scale * (0.5);
            let mut t = self.position;

            // calculate correct translation
            match (child_idx as i32).try_into() {
                Ok(NodePosition::Flt) => t += vec3(-0.5, -0.5, -0.5) * s,
                Ok(NodePosition::Frt) => t += vec3(0.5, -0.5, -0.5) * s,
                Ok(NodePosition::Flb) => t += vec3(-0.5, 0.5, -0.5) * s,
                Ok(NodePosition::Frb) => t += vec3(0.5, 0.5, -0.5) * s,
                Ok(NodePosition::Blt) => t += vec3(-0.5, -0.5, 0.5) * s,
                Ok(NodePosition::Brt) => t += vec3(0.5, -0.5, 0.5) * s,
                Ok(NodePosition::Blb) => t += vec3(-0.5, 0.5, 0.5) * s,
                Ok(NodePosition::Brb) => t += vec3(0.5, 0.5, 0.5) * s,
                Err(_) => panic!("Octree node has more than 8 children!"),
            }

            self.children.as_mut().unwrap().push(Node::new_inner(t, s));
        }
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
}

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

impl OctreeSystem {
    pub fn new(render_sys: Arc<Mutex<RenderSystem>>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(OctreeSystem {
            render_sys,

            update_config: None,

            optimizations: OctreeOptimizations::default(),

            messages: Vec::new(),
        }))
    }

    fn generate_instance_data(
        optimization_data: &OptimizationData,
        node: &mut Node,
        collected_data: &Vec<InstanceData>,
        atomic_counter: &AtomicUsize,
    ) {
        let limit_depth_reached = limit_depth_traversal(optimization_data, node);

        // add transformation data
        let mut include = !limit_depth_reached && node.solid;
        include = include || (node.is_leaf() && node.solid);

        if include {
            let idx = atomic_counter.fetch_add(1, Ordering::SeqCst);
            if idx < collected_data.len() {
                unsafe {
                    // IMPORTANT: really not that nice and really unsafe!!
                    let mut data_ptr = collected_data.as_ptr() as *mut InstanceData;
                    *data_ptr.offset(idx as isize) = InstanceData {
                        transformation: node.position.extend(node.scale).into(),
                    };
                }
            }
        }

        // check weather to traverse further
        let mut continue_traversal = cull_frustum(optimization_data, node);
        continue_traversal = continue_traversal && limit_depth_reached;

        if continue_traversal && node.children.is_some() {
            let camera_pos = optimization_data.camera_transform.position;
            let camera_mag = camera_pos.magnitude();
            let camera_dir = optimization_data.camera_transform.rotation.rotate_vector(-Vector3::unit_z());

            node.children.as_mut().unwrap().sort_unstable_by(|a,b| {
                let dist_a = camera_dir.extend(camera_mag)
                    .dot((a.position).extend(1.0));
                let dist_b = camera_dir.extend(camera_mag)
                    .dot((b.position).extend(1.0));

                dist_a.partial_cmp(&dist_b).unwrap()
            });

            node.children
                .as_mut()
                .unwrap()
                .par_iter_mut()
                .for_each(|child| {
                    &mut OctreeSystem::generate_instance_data(
                        optimization_data,
                        child,
                        collected_data,
                        atomic_counter
                    );
                });
        }
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

            if self.update_config.is_some() {
                octree = Octree::new(&self.render_sys, self.update_config.take().unwrap());
            }

            if !camera_entities.is_empty() {
                // scope to enclose mutex
                let mut root = octree.root.lock().unwrap();

                // camera data
                let camera_mutex = camera_entities[0].lock().unwrap();
                let camera_transform = camera_mutex
                    .get_component::<Transformation>()
                    .ok()
                    .unwrap()
                    .clone();
                let camera = camera_mutex.get_component::<Camera>().ok().unwrap().clone();

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
                    camera: &camera,
                    camera_transform: &camera_transform,
                    octree_mvp: camera.projection
                        * Matrix4::inverse_transform(&camera_transform.get_model_matrix()).unwrap()
                        * octree_transform.get_model_matrix(),
                    frustum: camera.frustum.transformed(
                        // TODO: investigate whether this space transformation is the a good approach for frustum culling
                        Matrix4::inverse_transform(&octree_transform.get_model_matrix()).unwrap()
                            * camera_transform.get_model_matrix(),
                    ),
                    depth_threshold: self.optimizations.depth_threshold,
                    octree_scale: octree_transform.scale.x,
                };

                // building matrices
                let max_nodes = octree.collect_buffer.len();
                let mut model_matrix_counter = std::sync::atomic::AtomicUsize::new(0);

                let instance_data_start = Instant::now();

                OctreeSystem::generate_instance_data(
                    &optimization_data,
                    &mut root,
                    &mut octree.collect_buffer,
                    &mut model_matrix_counter
                );

                // store data
                let rm = self.render_sys.lock().unwrap().resource_manager.clone();
                let _rm_lock = rm.lock().unwrap();

                let buffer_idx = (octree.active_instance_buffer_idx.unwrap_or(0) + 1)
                    % octree.instance_data_buffer.len();
                octree.active_instance_buffer_idx = Some(buffer_idx);
                let buffer = &octree.instance_data_buffer[buffer_idx];

                {
                    let mut gpu_buffer_lock = buffer.lock().unwrap();

                    let num_matrices = model_matrix_counter.load(Ordering::SeqCst)
                        .min(max_nodes as usize);

                    gpu_buffer_lock.replace_data(&octree.collect_buffer[0..num_matrices]);

                    octree.info.render_count = num_matrices;
                }

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

struct OptimizationData<'a> {
    camera: &'a Camera,
    camera_transform: &'a Transformation,
    octree_mvp: Matrix4<f32>,
    octree_scale: f32,
    frustum: Frustum,
    depth_threshold: f64,
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
    let mut projected_scale = node.scale / projected_position.w;
    projected_scale *= optimization_data.camera.resolution[0] / 2.0;

    let target_size = optimization_data.depth_threshold as f32 / optimization_data.octree_scale;
    return if projected_scale.abs() < target_size as f32 {
        false
    } else {
        true
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
