use crate::rendering::{Node, SUBDIVISIONS, NodeChildren};
use cgmath::{Vector3, Array};
use rand::{random, Rng, RngCore};
use shared_arena::SharedArena;
use rand::rngs::StdRng;

static mut SEED_OFFSET: f32 = 0.0;

pub fn generate_terrain(node: &mut Node, node_pool: &SharedArena<NodeChildren>, depth: u64) -> bool {
    let origin = node.position;
    let scale = node.scale;

    if depth == 0 {
        unsafe {
            SEED_OFFSET = random::<f32>();
        }
        node.height_values = rand::thread_rng().gen::<[f32; 4]>();
        node.height_values.iter_mut().for_each(|v| {
            *v = *v - scale * 0.5;
        });
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
    let rng_offset =  rng.next_u32() as f64 / u32::max_value() as f64;

    let mut midpoint: f32 = node.height_values.iter().sum::<f32>() / 4.0 as f32;
    midpoint += (rng_offset as f32 - 0.5) * node.scale;

    node.populate(node_pool);

    let height_vals = node.height_values;
    node.children.as_mut().unwrap().as_mut().iter_mut()
        .for_each(|child| {
            let offset = child.position - origin;

            let (x, z) = if offset.z > 0.0 {
                if offset.x > 0.0 { (0,1) } else { (1,1) }
            } else {
                if offset.x > 0.0 { (1,0) } else { (0,0) }
            };

            let h_idx = SUBDIVISIONS.pow(0) * x + SUBDIVISIONS.pow(1) * z;
            let idx_1 = (h_idx + 1) % 4;
            let mid_idx = (h_idx + 2) % 4;
            let idx_3 = (h_idx + 3) % 4;

            child.height_values[h_idx] = height_vals[h_idx];
            child.height_values[mid_idx] = midpoint;

            let off_coords = match (x,z) {

                (0,0) => (origin.x - 0.5 * scale, origin.z - 0.5 * scale),
                (1,0) => (origin.x + 0.5 * scale, origin.z - 0.5 * scale),
                (0,1) => (origin.x + 0.5 * scale, origin.z + 0.5 * scale),
                (1,1) => (origin.x - 0.5 * scale, origin.z + 0.5 * scale),
                _ => (0.0, 0.0)
            };

            // HACK!!: generate same random value for same vertical position
            // let mut rng_x: StdRng = rand_seeder::Seeder::from(((child.position.x + fu.x).to_ne_bytes(), child.position.z.to_ne_bytes())).make_rng();
            let mut rng_x: StdRng = rand_seeder::Seeder::from((off_coords.0.to_ne_bytes(), origin.z.to_ne_bytes())).make_rng();
            let _rng_offset_x = ((rng_x.next_u32() as f64 / u32::max_value() as f64) as f32 - 0.5) * child.scale;

            let mut rng_z: StdRng = rand_seeder::Seeder::from((origin.x.to_ne_bytes(), off_coords.1.to_ne_bytes())).make_rng();
            let _rng_offset_z = ((rng_z.next_u32() as f64 / u32::max_value() as f64) as f32 - 0.5) * child.scale;

            // match h_idx {
            //     0 => {
            //         // child.height_values[idx_1] = (hvals[idx_1] + hvals[h_idx] + midpoint) / 3.0 + rng_offset_z;
            //         // child.height_values[idx_3] = (hvals[idx_3] + hvals[h_idx] + midpoint) / 3.0 + rng_offset_x;
            //         //
            //         child.height_values[idx_1] = (hvals[idx_1] + hvals[h_idx]) / 2.0 + rng_offset_z;
            //         child.height_values[idx_3] = (hvals[idx_3] + hvals[h_idx]) / 2.0 + rng_offset_x;
            //     }
            //     1 => {
            //         // child.height_values[idx_1] = (hvals[idx_1] + hvals[h_idx] + midpoint) / 3.0 + rng_offset_x;
            //         // child.height_values[idx_3] = (hvals[idx_3] + hvals[h_idx] + midpoint) / 3.0 + rng_offset_z;
            //         //
            //         child.height_values[idx_1] = (hvals[idx_1] + hvals[h_idx]) / 2.0 + rng_offset_x;
            //         child.height_values[idx_3] = (hvals[idx_3] + hvals[h_idx]) / 2.0 + rng_offset_z;
            //     }
            //     2 => {
            //         // child.height_values[idx_1] = (hvals[idx_1] + hvals[h_idx] + midpoint) / 3.0 + rng_offset_z;
            //         // child.height_values[idx_3] = (hvals[idx_3] + hvals[h_idx] + midpoint) / 3.0 + rng_offset_x;
            //         //
            //         child.height_values[idx_1] = (hvals[idx_1] + hvals[h_idx]) / 2.0 + rng_offset_z;
            //         child.height_values[idx_3] = (hvals[idx_3] + hvals[h_idx]) / 2.0 + rng_offset_x;
            //     }
            //     3 => {
            //         // child.height_values[idx_1] = (hvals[idx_1] + hvals[h_idx] + midpoint) / 3.0 + rng_offset_x;
            //         // child.height_values[idx_3] = (hvals[idx_3] + hvals[h_idx] + midpoint) / 3.0 + rng_offset_z;
            //         //
            //         child.height_values[idx_1] = (hvals[idx_1] + hvals[h_idx]) / 2.0 + rng_offset_x;
            //         child.height_values[idx_3] = (hvals[idx_3] + hvals[h_idx]) / 2.0 + rng_offset_z;
            //     }
            //     _ => ()
            // }
            child.height_values[idx_1] = (height_vals[idx_1] + height_vals[h_idx]) / 2.0;
            child.height_values[idx_3] = (height_vals[idx_3] + height_vals[h_idx]) / 2.0;

            // one of the height values intersects with this child's bounding box
            let refine = child.height_values.iter().any(|val| {
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

            // all height values are over this node. We do not need to refine further.
            let all_lower = child.height_values.iter().all(|val| {
                let max = child.position.y + child.scale * 0.5;
                return *val > max;
            });

            let all_higher = child.height_values.iter().all(|val| {
                let min = child.position.y - child.scale * 0.5;
                return *val < min;
            });

            child.color = Vector3::from_value(1.0);
            child.solid = true;
            if refine || (higher_max && lower_max) || (lower_min && higher_min) {
                child.refine = Some(true);
            } else {
                child.refine = Some(false);
            }

            if all_lower {
                child.color = Vector3::from_value(0.5);
            }

            if all_higher {
                child.solid = false;
            }
        });

    return true;
}