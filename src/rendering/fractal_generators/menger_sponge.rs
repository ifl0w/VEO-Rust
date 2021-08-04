use crate::rendering::{Node, TREE_SUBDIVISIONS};
use cgmath::{Vector3, vec3, Array};

pub fn generate_menger(child: &mut Node, _zoom: f64, depth: u64) -> bool {
    let s = child.scale;
    let p = child.position;

    fn iterate(p: Vector3<f32>, s: f32, bb_center: Vector3<f32>, bb_size: f32, n: u64) -> bool {
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
            // There are 27 blocks and only 20 are potentially "solid"
            let mut bounding = [vec3(0.0,0.0,0.0); 20];
            let mut i = 0;
            for x in -1..=1 {
                for y in -1..=1 {
                    for z in -1..=1 {
                        let mut num_zero_axis = 0;
                        if x == 0 { num_zero_axis += 1; }
                        if y == 0 { num_zero_axis += 1; }
                        if z == 0 { num_zero_axis += 1; }

                        // if 2 or 3 coordinates are 0, then the block is one of the 7 blocks
                        // building the cross in the middle.
                        if num_zero_axis != 2 && num_zero_axis != 3 {
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

    let inside = iterate(p, s / TREE_SUBDIVISIONS as f32, vec3(0.0, 0.0, 0.0), 0.5, depth + 3);

    if inside {
        child.color = Vector3::new(1.0 - (1.0 / s.log2()).abs(), 0.25, 0.25);
        child.solid = true;
    }

    return inside;
}
