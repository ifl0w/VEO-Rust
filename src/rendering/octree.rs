use std::sync::{Arc, Mutex};
use std::time::Duration;

use winit::Event;

use crate::core::{Component, Filter, Message, System};
use crate::rendering::{RenderSystem, Camera, Transformation};
use cgmath::{Vector3, Matrix4, SquareMatrix};
use vulkano::command_buffer::validity::check_fill_buffer;

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

#[derive(Clone, Debug)]
pub struct Octree {
    pub size: Vector3<f32>,
    pub root: Option<Node>
}

impl Component for Octree {}

impl Octree {
    pub fn new(size: Vector3<f32>) -> Self {
        let oct = Octree {
            size,
            root: Some(Node::new())
        };

        Octree::fill_octree(oct)
    }

    pub fn count_leaves(&self) -> i64 {
        self.root.clone().unwrap().count_leaves()
    }

    fn fill_octree(mut octree: Octree) -> Octree {
        octree.root = Octree::traverse(octree.root, 5);

        octree.clone()
    }

    fn traverse(node: Option<Node>, depth: i32) -> Option<Node> {
        if node.is_none() || depth == 0 {
            return node;
        }

        let mut node_copy = node.clone().unwrap();

        let children = node_copy.children;
        let new_children:Vec<_> = children.iter().map(|child| {
            match child {
                Some(node) => { Some(node.clone()) },
                None => {
                    let new_depth = depth - 1;

                    let mut new_child = Node::new();

                    Octree::traverse(Option::Some(new_child), new_depth)
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
    filled: bool,
}

impl Node {
    pub fn new() -> Self {
        let mut tmp = Node {
            children: vec![],
            filled: true,
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

    fn generate_model_matrices(node: &Option<Node>) -> Vec<Matrix4<f32>> {
        if node.is_none() {
            return vec![];
        }

        let mut node_copy = node.clone().unwrap();

        let mut model_matrices: Vec<Matrix4<f32>> = vec![];

        if node_copy.is_leaf() {
            let children = node_copy.children;
            children.iter().enumerate().for_each(|(i, child)| {
                match child {
                    Some(node) => {
                        let mut new_mat = &mut OctreeSystem::generate_model_matrices(child);
                        model_matrices.append(new_mat);
                    },
                    None => {}
                }
            });
        } else {
            model_matrices.push(Matrix4::identity());
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
                let entitiy_mutex = entity.lock().unwrap();
                let transform = entitiy_mutex.get_component::<Transformation>().ok().unwrap();
                let octree = entitiy_mutex.get_component::<Octree>().ok().unwrap();

                let model_matrices = OctreeSystem::generate_model_matrices(&octree.root);
            }
        }
    }

    fn get_messages(&mut self) -> Vec<Message> { vec![] }
}