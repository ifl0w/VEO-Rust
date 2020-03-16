#[cfg(feature = "dx11")]
pub extern crate gfx_backend_dx11 as Backend;
#[cfg(feature = "dx12")]
pub extern crate gfx_backend_dx12 as Backend;
#[cfg(
not(any(
feature = "vulkan",
feature = "dx12",
feature = "metal",
feature = "gl",
feature = "wgl"
)))]
pub extern crate gfx_backend_empty as Backend;
#[cfg(any(feature = "gl", feature = "wgl"))]
pub extern crate gfx_backend_gl as Backend;
#[cfg(feature = "metal")]
pub extern crate gfx_backend_metal as Backend;
#[cfg(feature = "vulkan")]
pub extern crate gfx_backend_vulkan as Backend;

use std::borrow::Borrow;
use std::convert::{TryFrom, TryInto};
use std::f32::consts::PI;
use std::iter;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use cgmath::{Matrix4, vec3, Vector3, Vector4};
use gfx_hal::buffer;
use gfx_hal::pso::{Descriptor, DescriptorSetWrite};
use winit::event::Event;

use crate::core::{Component, Filter, Message, System};
use crate::rendering::{Camera, GPUBuffer, InstanceData, Mesh, RenderSystem, Transformation, AABB, Frustum};
use crate::rendering::nse_gui::octree_gui::{ProfilingData, UpdateOctree};
use crate::rendering::utility::resources::BufferID;

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
    pub scale: Vector3<f32>,
    pub root: Arc<Mutex<Option<Node>>>,

    pub instance_data_buffer: Vec<Arc<Mutex<GPUBuffer<Backend::Backend>>>>,
    /// indirect reference to the GPU buffers
    pub active_instance_buffer_idx: Option<usize>,
    /// points into the instance_data_buffer vec

    pub render_count: usize,

    pub depth: i32,
    pub byte_size: Option<usize>,
    pub max_byte_size: usize,
}

impl Component for Octree {}

impl Octree {
    pub fn new(render_system: &Arc<Mutex<RenderSystem>>, depth: i32, size: Option<Vector3<f32>>) -> Self {
        let size = if size.is_none() {
            vec3(0.0, 0.0, 0.0)
        } else {
            size.unwrap()
        };

        let max_num_nodes = 8_i64.pow(depth.try_into().unwrap()) as usize;
        let max_byte_size = std::mem::size_of::<Octree>() * max_num_nodes;
        let max_gpu_byte_size = std::mem::size_of::<InstanceData>() * max_num_nodes;

        let rm = render_system.lock().unwrap().resource_manager.clone();
        let mut rm_lock = rm.lock().unwrap();

        let dev = render_system.lock().unwrap().renderer.device.clone();
        let adapter = render_system.lock().unwrap().renderer.adapter.clone();

        let num_buffers = 2;
        let mut instance_data_buffer = Vec::with_capacity(num_buffers);
        for _ in 0..num_buffers {
            unsafe {
                let (_, buffer) = rm_lock.add_buffer(GPUBuffer::new_with_size(&dev,
                                                                              &adapter,
                                                                              max_gpu_byte_size,
                                                                              buffer::Usage::STORAGE | buffer::Usage::VERTEX));

                instance_data_buffer.push(buffer);
            }
        }

        let mut oct = Octree {
            scale: size,
            root: Arc::new(Mutex::new(None)),
            instance_data_buffer,
            active_instance_buffer_idx: None,
            render_count: 0,
            depth,
            byte_size: None,
            max_byte_size,
        };

        oct.root = Arc::new(Mutex::new(Octree::traverse(
            Some(Node::new()),
            vec3(0.0, 0.0, 0.0),
            0,
            depth)));

        oct.byte_size = Some(self::Octree::size_in_bytes(&oct));

        println!("SIZE: {} MB", (oct.byte_size.unwrap() as f32 / (1024_f32 * 1024_f32)));
        oct
    }

    pub fn get_instance_buffer(&self) -> Option<&Arc<Mutex<GPUBuffer<Backend::Backend>>>> {
        if self.active_instance_buffer_idx.is_some() {
            Some(self.instance_data_buffer.get(self.active_instance_buffer_idx.unwrap()).unwrap())
        } else {
            None
        }
    }

    pub fn count_leaves(&self) -> i64 {
        let root = self.root.lock().unwrap();
        (*root).as_ref().unwrap().count_leaves()
    }

    pub fn size_in_bytes(&self) -> usize {
        std::mem::size_of::<Octree>() * self.count_nodes(&self.root.lock().unwrap()) as usize
    }

    fn count_nodes(&self, node: &Option<Node>) -> i64 {
        let mut count = 0;

        if !node.is_none() {
            let node_copy = node.clone().unwrap();

            if !node_copy.is_leaf() {
                let children = node_copy.children;
                children.iter().enumerate().for_each(|(_i, child)| {
                    count += self.count_nodes(child);
                });
            } else {
                count += 1;
            }
        }
        count
    }

    fn traverse(node: Option<Node>, translate: Vector3<f32>, current_depth: i32, target_depth: i32) -> Option<Node> {
        if node.is_none() || current_depth == target_depth {
            return node;
        }

        let mut node_copy = node.clone().unwrap();

        let children = node_copy.children;
        let new_children: Vec<_> = children.iter().enumerate().map(|(idx, child)| {
            match child {
                Some(node) => { Some(node.clone()) }
                None => {
                    let new_depth = current_depth + 1;

                    let s = (0.5 as f32).powf(new_depth as f32);
                    let mut t = translate;
                    match (idx as i32).try_into() {
                        Ok(NodePosition::Flt) => t += vec3(-0.5, -0.5, -0.5) * s,
                        Ok(NodePosition::Frt) => t += vec3(0.5, -0.5, -0.5) * s,
                        Ok(NodePosition::Flb) => t += vec3(-0.5, 0.5, -0.5) * s,
                        Ok(NodePosition::Frb) => t += vec3(0.5, 0.5, -0.5) * s,
                        Ok(NodePosition::Blt) => t += vec3(-0.5, -0.5, 0.5) * s,
                        Ok(NodePosition::Brt) => t += vec3(0.5, -0.5, 0.5) * s,
                        Ok(NodePosition::Blb) => t += vec3(-0.5, 0.5, 0.5) * s,
                        Ok(NodePosition::Brb) => t += vec3(0.5, 0.5, 0.5) * s,
                        Err(_) => panic!("Octree node has more than 8 children!")
                    }

                    let mut new_child = Node::new_inner(t, s);

                    let node_origin = t;

                    let sinc = |x: f32, y: f32| {
                        let scale = 6.0 * PI;
                        let scale_y = 0.25;

                        let r = f32::sqrt(x * x + y * y);
                        let r_val = if r == 0.0 { 1.0 } else { (r * scale).sin() / (r * scale) };
                        r_val * scale_y
                    };

                    let intersect = |origin: Vector3<f32>, scale: f32, function: &dyn Fn(f32, f32) -> f32| {
                        let box_points: Vec<Vector3<f32>> = vec![
                            origin + vec3(-0.5, -0.5, -0.5) * scale,
                            origin + vec3(0.5, -0.5, -0.5) * scale,
                            origin + vec3(-0.5, 0.5, -0.5) * scale,
                            origin + vec3(0.5, 0.5, -0.5) * scale,
                            origin + vec3(-0.5, -0.5, 0.5) * scale,
                            origin + vec3(0.5, -0.5, 0.5) * scale,
                            origin + vec3(-0.5, 0.5, 0.5) * scale,
                            origin + vec3(0.5, 0.5, 0.5) * scale
                        ];

                        let all_greater = box_points.iter()
                            .all(|val| function(val.x, val.z) >= (val.y));

                        let all_smaller = box_points.iter()
                            .all(|val| function(val.x, val.z) <= (val.y));

                        !(all_greater || all_smaller)
                    };

                    // NOTE: intersect test does not work with every function correctly.
                    // It is possible that all samples at the corner points of the octree node do
                    // not indicate an intersection and thus leading to false negative intersection
                    // tests!
                    if intersect(node_origin, s, &sinc) {
                        Octree::traverse(Option::Some(new_child), t, new_depth, target_depth)
                    } else {
                        None
                    }
                }
            }
        }).collect();

        node_copy.children = new_children;

        Some(node_copy)
    }
}

#[derive(Clone, Debug)]
pub struct Node {
    children: Vec<Option<Node>>,
    position: Vector3<f32>,
    scale: Vector3<f32>,
    aabb: AABB,
}

impl Node {
    pub fn new() -> Self {
        let mut tmp = Node {
            children: vec![],
            position: vec3(0.0, 0.0, 0.0),
            scale: vec3(1.0, 1.0, 1.0),
            aabb: AABB::new(vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0))
        };

        tmp.children.resize(8, None);

        tmp
    }

    pub fn new_inner(position: Vector3<f32>, scale: f32) -> Self {
        let scale_vec = vec3(scale, scale, scale);
        let min = (position - scale_vec / 2.0);
        let max = (position + scale_vec / 2.0);

        let mut tmp = Node {
            children: vec![],
            position,
            scale: scale_vec,
            aabb: AABB::new(min, max)
        };

        tmp.children.resize(8, None);

        tmp
    }

    pub fn count_leaves(&self) -> i64 {
        let mut count = 0i64;

        for child in &self.children {
            match child {
                Some(n) => {
                    count += n.count_leaves();
                }
                None => {
                    count += 1;
                }
            }
        }

        count
    }

    pub fn is_leaf(&self) -> bool {
        self.children.iter().all(|n| n.is_none())
    }
}

pub struct OctreeSystem {
    render_sys: Arc<Mutex<RenderSystem>>,

    update_octrees: bool,
    octree_depth: i32,

    // optimization flags
    frustum_culling: Option<bool>,
    limit_depth: Option<f64>,
    ignore_full: Option<bool>,
    ignore_inner: Option<bool>,

    messages: Vec<Message>,
}

impl OctreeSystem {
    pub fn new(render_sys: Arc<Mutex<RenderSystem>>) -> Arc<Mutex<Self>> {
        Arc::new(Mutex::new(OctreeSystem {
            render_sys,
            update_octrees: false,
            octree_depth: 5,

            frustum_culling: None,
            limit_depth: None,
            ignore_full: None,
            ignore_inner: None,

            messages: Vec::new(),
        }))
    }

    fn generate_instance_data(optimization_data: &OptimizationData,
                              node: &Option<Node>,
                              scale: Vector3<f32>,
                              traversal_criteria: &Vec<&TraversalFunction>,
                              filter_functions: &Vec<&FilterFunction>)
                              -> Vec<InstanceData> {
        if node.is_none() {
            return vec![];
        }

        let node_copy = node.clone().unwrap();

        let mut model_matrices: Vec<InstanceData> = vec![];

        // add model matrices
        let include = filter_functions.iter().all(|fnc| {
            fnc(optimization_data, node, scale)
        });

        if include {
            let mat = Matrix4::from_nonuniform_scale(scale.x, scale.y, scale.z)
                * Matrix4::from_translation(node_copy.position)
                * Matrix4::from_scale(node_copy.scale.x);
            model_matrices.push(InstanceData {
                model_matrix: mat.into()
            });
        }

        // traverse
        let continue_traversal = traversal_criteria.iter().all(|fnc| {
            fnc(optimization_data, node, scale)
        });

        if continue_traversal {
            let children = node_copy.children;
            children.iter().enumerate().for_each(|(_i, child)| {
                match child {
                    Some(real_child) => {
                        let new_mat =
                            &mut OctreeSystem::generate_instance_data(optimization_data,
                                                                      child,
                                                                      scale,
                                                                      traversal_criteria,
                                                                      filter_functions);
                        model_matrices.append(new_mat);
                    }
                    None => {}
                }
            });
        }

        model_matrices
    }
}

impl System for OctreeSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![
            crate::filter!(Octree, Mesh, Transformation),
            crate::filter!(Camera, Transformation)
        ]
    }
    fn handle_input(&mut self, _event: &Event<()>) {}

    fn consume_messages(&mut self, messages: &Vec<Message>) {
        messages.iter().for_each(|msg| {
            if msg.is_type::<UpdateOctree>() {
                self.update_octrees = true;
                self.octree_depth = msg.get_payload::<UpdateOctree>().unwrap().octree_depth
            }
        });
    }

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, _delta_time: Duration) {
        let octree_entities = &filter[0].lock().unwrap().entities;
        let camera_entities = &filter[1].lock().unwrap().entities;


        for entity in octree_entities {
            let mut entitiy_mutex = entity.lock().unwrap();
            let _transform = entitiy_mutex.get_component::<Transformation>().ok().unwrap();
            let mut octree = entitiy_mutex.get_component::<Octree>().ok().unwrap().clone();

            if self.update_octrees {
                octree = Octree::new(&self.render_sys, self.octree_depth, Some(octree.scale));
                self.update_octrees = false;
            }

            if !camera_entities.is_empty() { // scope to enclose mutex
                let root = octree.root.lock().unwrap();

                // camera data
                let mut camera_mutex = camera_entities[0].lock().unwrap();
                let mut camera_transform = camera_mutex.get_component::<Transformation>().ok().unwrap().clone();
                let mut camera = camera_mutex.get_component::<Camera>().ok().unwrap().clone();

                // filter config
                let mut traversal_fnc: Vec<&TraversalFunction> = Vec::new();
                traversal_fnc.push(&continue_to_leaf);

                let mut filter_fnc: Vec<&FilterFunction> = Vec::new();
//                filter_fnc.push(&cull_frustum); // TODO Fix broken culling
                filter_fnc.push(&generate_leaf_model_matrix);

                let optimization_data = OptimizationData {
                    camera: &camera,
                    camera_transform: &camera_transform,
                    frustum: camera.frustum.transformed(camera_transform.get_model_matrix())
                };

                // building matrices
                let model_matrices = OctreeSystem::generate_instance_data(
                    &optimization_data,
                    &root,
                    octree.scale,
                    &traversal_fnc,
                    &filter_fnc
                );

                // store data
                let rm = self.render_sys.lock().unwrap().resource_manager.clone();
                let mut rm_lock = rm.lock().unwrap();

                let buffer = &octree.instance_data_buffer[0]; // TODO select correct idx
                let mut gpu_buffer_lock = buffer.lock().unwrap();

                gpu_buffer_lock.replace_data(&model_matrices);
                octree.active_instance_buffer_idx = Some(0);
                octree.render_count = model_matrices.len();

                self.messages.push(Message::new(
                    ProfilingData {
                        rendered_nodes: Some(octree.render_count as u32),
                        ..Default::default()
                    }));
            } else { // drop locks
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

type FilterFunction = dyn Fn(&OptimizationData, &Option<Node>, Vector3<f32>) -> bool;
type TraversalFunction = dyn Fn(&OptimizationData, &Option<Node>, Vector3<f32>) -> bool;

struct OptimizationData<'a> {
    camera: &'a Camera,
    camera_transform: &'a Transformation,
    frustum: Frustum,
}

fn continue_to_leaf(optimization_data: &OptimizationData, node: &Option<Node>, scale: Vector3<f32>) -> bool {
//    if node.is_some() && node.as_ref().unwrap().is_leaf() {
//        false
//    } else {
//        true
//    }
    true
}

fn cull_frustum(optimization_data: &OptimizationData, node: &Option<Node>, scale: Vector3<f32>) -> bool {
    if node.is_some() {
        let node = node.as_ref().unwrap();
        optimization_data.frustum.intersect(&node.aabb)
    } else {
        false
    }
}

fn generate_leaf_model_matrix(optimization_data: &OptimizationData, node: &Option<Node>, scale: Vector3<f32>) -> bool {
    if node.is_some() && node.as_ref().unwrap().is_leaf() {
        true
    } else {
        false
    }
}