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
use std::f32::INFINITY;
use std::result::Result;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use cgmath::{Matrix4, Transform, vec3, Vector3};
use gfx_hal::buffer;
use winit::event::Event;

use crate::core::{Component, Filter, Message, Payload, System};
use crate::rendering::{
    AABB, Camera, Frustum, GPUBuffer, InstanceData, Mesh, RenderSystem, Transformation,
};
use crate::rendering::nse_gui::octree_gui::ProfilingData;

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

        let mut oct = Octree {
            root: Arc::new(Mutex::new(
                Node::new_inner(vec3(0.0, 0.0, 0.0), 1.0)
            )),
            instance_data_buffer,
            active_instance_buffer_idx: None,

            config,
            info: octree_info,
        };

        oct.root.lock().unwrap().populate();

        Arc::new(Mutex::new(Octree::traverse(
            &mut oct.root.lock().unwrap(),
            0,
            depth,
        )));

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

    fn traverse(
        node: &mut Node,
        current_depth: u64,
        target_depth: u64,
    ) {
        if node.children.is_some() {
            node.children.as_mut().unwrap()
                .iter_mut()
                .enumerate()
                .for_each(|(idx, child)| {
                    let new_depth = current_depth + 1;

                    let mandelbrot = |origin: &Vector3<f32>, size: f32, zoom: f32| {
                        // only a single slice
                        if origin.y + size < 0.0 || origin.y - size > 0.0 { return INFINITY as f64; };

                        let position = origin * zoom - vec3(0.5, 0.0, 0.0);

                        let escape_radius = 4.0 * 10000.0 as f64;
                        let mut iter = 1000;

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

                        if iter == 0 {
                            return 0.0f64;
                        }

                        // values
                        let z_val = (z_re2 + z_im2).sqrt();
                        let zp_val = (zp_re * zp_re + zp_im * zp_im).sqrt();

                        // let dist = 2.0 * z_val * z_val.ln() / zp_val;
                        // let dist = z_val * z_val.ln() / zp_val;
                        let dist = 0.5 * z_val * z_val.ln() / zp_val;

                        return dist;
                    };

                    let zoom = 2.5;
                    let size: f64 = (child.scale * 0.5 * zoom) as f64;
                    let radius = (size * size + size * size).sqrt() as f64;

                    let dem: f64 = mandelbrot(&child.position, child.scale, zoom);
                    let mut traverse = false;

                    if dem == 0.0 {
                        child.solid = true;
                        traverse = true;
                    } else if dem < radius as f64 {
                        traverse = true;

                        if dem < zoom as f64 / 10e12 {
                            child.solid = true;
                        }
                    }

                    if current_depth < target_depth && traverse {
                        child.populate();
                        Octree::traverse(child, new_depth, target_depth)
                    }
                });
        }
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    children: Option<Vec<Node>>,
    position: Vector3<f32>,
    scale: f32,
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
        collected_data: &mut Vec<InstanceData>,
        traversal_criteria: &Vec<&TraversalFunction>,
        filter_functions: &Vec<&FilterFunction>,
    ) {
        // add model matrices
        let include = filter_functions
            .iter()
            .any(|fnc| fnc(optimization_data, node));

        if include {
            let mat = Matrix4::from_translation(node.position)
                * Matrix4::from_scale(node.scale);
            collected_data.push(InstanceData {
                model_matrix: mat.into(),
            });
        }

        // traverse
        let continue_traversal = traversal_criteria
            .iter()
            .all(|fnc| fnc(optimization_data, node));

        if continue_traversal && node.children.is_some() {
            node.children
                .as_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
                .for_each(|(_i, child)| {
                        &mut OctreeSystem::generate_instance_data(
                            optimization_data,
                            child,
                            collected_data,
                            traversal_criteria,
                            filter_functions,
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
                let mut model_matrices = Vec::with_capacity(octree.info.render_count);

                let instance_data_start = Instant::now();

                OctreeSystem::generate_instance_data(
                    &optimization_data,
                    &mut root,
                    &mut model_matrices,
                    &traversal_fnc,
                    &filter_fnc,
                );

                // store data
                let rm = self.render_sys.lock().unwrap().resource_manager.clone();
                let _rm_lock = rm.lock().unwrap();

                let buffer_idx = octree.active_instance_buffer_idx.unwrap_or(0);
                let buffer = &octree.instance_data_buffer[buffer_idx];
                let mut gpu_buffer_lock = buffer.lock().unwrap();

                gpu_buffer_lock.replace_data(
                    &model_matrices[0..model_matrices
                        .len()
                        .min(octree.config.max_rendered_nodes.unwrap_or(1e3 as u64) as usize)],
                );
                octree.active_instance_buffer_idx =
                    Some((buffer_idx + 1) % (octree.instance_data_buffer.len() - 1));
                octree.info.render_count = model_matrices.len();

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
