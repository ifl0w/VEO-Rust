use std::sync::{Arc, Mutex};
use std::time::Duration;
use std::convert::{TryFrom, TryInto};

use winit::Event;

use crate::core::{Component, Filter, Message, System};
use crate::rendering::{RenderSystem, Camera, Transformation, InstanceData};
use cgmath::{Vector3, Matrix4, SquareMatrix, vec3};
use vulkano::buffer::cpu_pool::CpuBufferPoolChunk;
use vulkano::memory::pool::StdMemoryPool;

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
    pub size: Vector3<f32>,
    pub root: Option<Node>,
    pub instance_data_buffer: Option<Arc<CpuBufferPoolChunk<InstanceData, Arc<StdMemoryPool>>>>
}

impl Component for Octree {}

impl Octree {
    pub fn new(size: Vector3<f32>) -> Self {
        let oct = Octree {
            size,
            root: Some(Node::new()),
            instance_data_buffer: None
        };

        Octree::fill_octree(oct)
    }

    pub fn count_leaves(&self) -> i64 {
        self.root.clone().unwrap().count_leaves()
    }

    fn fill_octree(mut octree: Octree) -> Octree {
        octree.root = Octree::traverse(
            octree.root,
            vec3(0.0,0.0,0.0),
            0,
            4);

        octree.clone()
    }

    fn traverse(node: Option<Node>, translate: Vector3<f32>, current_depth: i32, target_depth: i32) -> Option<Node> {
        if node.is_none() || current_depth == target_depth {
            return node;
        }

        let mut node_copy = node.clone().unwrap();

        let children = node_copy.children;
        let new_children:Vec<_> = children.iter().enumerate().map(|(idx, child)| {
            match child {
                Some(node) => { Some(node.clone()) },
                None => {
                    let new_depth = current_depth + 1;

                    let mut new_child = Node::new();
                    let s = (0.5 as f32).powf(new_depth as f32);
                    new_child.scale = vec3(s, s, s);

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
                    new_child.position = t;

                    Octree::traverse(Option::Some(new_child), t, new_depth, target_depth)
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
}

impl Node {
    pub fn new() -> Self {
        let mut tmp = Node {
            children: vec![],
            position: vec3(0.0, 0.0,0.0),
            scale: vec3(1.0, 1.0,1.0),
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
                },
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
    render_sys: Arc<Mutex<RenderSystem>>
}

impl OctreeSystem {
    pub fn new(render_sys: Arc<Mutex<RenderSystem>>) -> Arc<Mutex<Self>> {
        let mut data = ();

        Arc::new(Mutex::new(OctreeSystem {
            render_sys
        }))
    }

    fn generate_instance_data(node: &Option<Node>) -> Vec<InstanceData> {
        if node.is_none() {
            return vec![];
        }

        let mut node_copy = node.clone().unwrap();

        let mut model_matrices: Vec<InstanceData> = vec![];

        if !node_copy.is_leaf() {
            let children = node_copy.children;
            children.iter().enumerate().for_each(|(i, child)| {
                match child {
                    Some(node) => {
                        let mut new_mat = &mut OctreeSystem::generate_instance_data(child);
                        model_matrices.append(new_mat);
                    },
                    None => {}
                }
            });
        } else {
            let mat = Matrix4::from_translation(node_copy.position)
                * Matrix4::from_scale(node_copy.scale.x);
            model_matrices.push(InstanceData {
                model_matrix: mat.into()
            });
        }

        model_matrices
    }
}

impl System for OctreeSystem {
    fn get_filter(&mut self) -> Vec<Filter> {
        vec![
            crate::filter!(Octree, Transformation),
            crate::filter!(Camera, Transformation)
        ]
    }
    fn handle_input(&mut self, _event: &Event) {}
    fn consume_messages(&mut self, _: &Vec<Message>) {}

    fn execute(&mut self, filter: &Vec<Arc<Mutex<Filter>>>, _delta_time: Duration) {
        let entities = &filter[0].lock().unwrap().entities;

        if !entities.is_empty() {
            for entity in entities {
                let mut entitiy_mutex = entity.lock().unwrap();
                let transform = entitiy_mutex.get_component::<Transformation>().ok().unwrap();
                let mut octree = entitiy_mutex.get_component::<Octree>().ok().unwrap().clone();

                let model_matrices = OctreeSystem::generate_instance_data(&octree.root);

                octree.instance_data_buffer = Some(Arc::new(
                    self.render_sys.lock().unwrap().instance_buffer_pool.chunk(model_matrices).unwrap()));
                entitiy_mutex.add_component(octree);
            }
        }
    }

    fn get_messages(&mut self) -> Vec<Message> { vec![] }
}