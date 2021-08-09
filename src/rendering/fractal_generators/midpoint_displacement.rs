use cgmath::Vector3;
use rand::{random, Rng, RngCore};
use rand::rngs::StdRng;
use shared_arena::SharedArena;

use crate::rendering::{Node, NodeChildren, TREE_SUBDIVISIONS};

static mut SEED_OFFSET: f32 = 0.0;

pub fn generate_terrain(node: &mut Node, node_pool: &SharedArena<NodeChildren>, depth: u64) -> bool {
    let origin = node.position;
    let scale = node.scale;

    if depth == 0 {
        // initialization only for the root node
        unsafe { SEED_OFFSET = random::<f32>(); }
        node.height_values = rand::thread_rng().gen::<[f32; 4]>();
        node.height_values.iter_mut().for_each(|v| { *v = *v - scale * 0.5; });
    }

    if node.refine.is_some() && !node.refine.unwrap() {
        return node.refine.unwrap();
    } else {
        node.refine = Some(true);
    }

    // HACK!!: generate same random value for same vertical position
    let seed = unsafe {
        ((origin.x + SEED_OFFSET).to_ne_bytes(), (origin.z + SEED_OFFSET).to_ne_bytes())
    };
    let mut rng: StdRng = rand_seeder::Seeder::from(seed).make_rng();
    let rng_offset = rng.next_u32() as f64 / u32::max_value() as f64;

    // displace mid point
    let mut midpoint: f32 = node.height_values.iter().sum::<f32>() / 4.0 as f32;
    midpoint += (rng_offset as f32 - 0.5) * node.scale;

    node.populate(node_pool);

    let height_vals = node.height_values;
    node.children.as_mut().unwrap().as_mut().iter_mut()
        .for_each(|child| {
            let offset = child.position - origin;

            // calculate cyclical indexing of height values of the child
            let (x, z) = if offset.z > 0.0 {
                if offset.x > 0.0 { (0, 1) } else { (1, 1) }
            } else {
                if offset.x > 0.0 { (1, 0) } else { (0, 0) }
            };

            // map cyclical indices to child indices
            let h_idx = TREE_SUBDIVISIONS.pow(0) * x + TREE_SUBDIVISIONS.pow(1) * z;
            let idx_1 = (h_idx + 1) % 4;
            let mid_idx = (h_idx + 2) % 4;
            let idx_3 = (h_idx + 3) % 4;

            child.height_values[h_idx] = height_vals[h_idx];
            child.height_values[mid_idx] = midpoint;
            child.height_values[idx_1] = (height_vals[idx_1] + height_vals[h_idx]) / 2.0;
            child.height_values[idx_3] = (height_vals[idx_3] + height_vals[h_idx]) / 2.0;

            // Helper values for finding nodes that intersect the surface
            let surface_in_node = child.height_values.iter().any(|val| {
                let min = child.position.y - child.scale * 0.5;
                let max = child.position.y + child.scale * 0.5;

                return *val >= min && *val <= max;
            });

            let higher_max = child.height_values.iter().any(|val| {
                let max = child.position.y + child.scale * 0.5;
                return *val > max;
            });

            let lower_max = child.height_values.iter().any(|val| {
                let max = child.position.y + child.scale * 0.5;
                return *val < max;
            });


            let lower_min = child.height_values.iter().any(|val| {
                let min = child.position.y - child.scale * 0.5;
                return *val < min;
            });

            let higher_min = child.height_values.iter().any(|val| {
                let min = child.position.y - child.scale * 0.5;
                return *val > min;
            });

            // Test whether all height values are above the node.
            let node_below_surface = child.height_values.iter().all(|val| {
                let max = child.position.y + child.scale * 0.5;
                return *val > max;
            });

            // Test whether all height values are below the node.
            let node_above_surface = child.height_values.iter().all(|val| {
                let min = child.position.y - child.scale * 0.5;
                return *val < min;
            });

            child.color = Vector3::new(midpoint, 1.0 - midpoint, 0.0);
            child.solid = true;

            // surface blocks need to be refined
            if surface_in_node || (higher_max && lower_max) || (lower_min && higher_min) {
                child.refine = Some(true);
            } else {
                child.refine = Some(false);
            }

            // "earth" blocks
            if node_below_surface {
                child.color = Vector3::new(0.25, 0.25, 0.0);
            }

            // "air" blocks
            if node_above_surface {
                child.solid = false;
            }
        });

    return true;
}