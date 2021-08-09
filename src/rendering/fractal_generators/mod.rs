use core::fmt;
use std::fmt::Debug;

use num_derive::{FromPrimitive, ToPrimitive};
use shared_arena::SharedArena;

use mandelbrot::generate_mandelbrot;
use mandelbulb::generate_mandelbulb;
use menger_sponge::generate_menger;
use midpoint_displacement::generate_terrain;
use sierpinski_pyramid::generate_sierpinski_pyramid;
use sierpinski_tetrahedron::generate_sierpinski_tetrahedron;

use crate::rendering::{Node, NodeChildren, OctreeConfig};

mod mandelbrot;
mod mandelbulb;
mod menger_sponge;
mod sierpinski_pyramid;
mod sierpinski_tetrahedron;
mod midpoint_displacement;

#[derive(Clone, Copy, Debug, PartialEq, FromPrimitive, ToPrimitive)]
pub enum FractalSelection {
    MandelBulb = 0,
    MandelBrot,
    SierpinskiPyramid,
    SierpinskiTetrahedron,
    MengerSponge,
    MidpointDisplacement,
}

impl fmt::Display for FractalSelection {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

pub fn build_tree(node: &mut Node, config: OctreeConfig, current_depth: u64, target_depth: u64, node_pool: &SharedArena<NodeChildren>) {
    let zoom = 3.0;
    let traverse_further = match config.fractal {
        Some(FractalSelection::MandelBulb) =>
            generate_mandelbulb(node, zoom, current_depth),
        Some(FractalSelection::MandelBrot) =>
            generate_mandelbrot(node, zoom, current_depth),
        Some(FractalSelection::SierpinskiPyramid) =>
            generate_sierpinski_pyramid(node, zoom, current_depth),
        Some(FractalSelection::SierpinskiTetrahedron) =>
            generate_sierpinski_tetrahedron(node, zoom, current_depth),
        Some(FractalSelection::MengerSponge) =>
            generate_menger(node, zoom, current_depth),
        Some(FractalSelection::MidpointDisplacement) =>
            generate_terrain(node, node_pool, current_depth),
        None => false,
    };

    if node.refine.is_none() {
        node.refine = Some(traverse_further);
    }

    if current_depth < target_depth && traverse_further {
        node.populate(node_pool);

        node.children.as_mut().unwrap().as_mut()
            .iter_mut()
            .for_each(|child| {
                build_tree(child, config, current_depth + 1, target_depth, node_pool)
            });
    }
}
