use cgmath::{Array, vec3, Vector3};

use crate::rendering::{Node, TREE_SUBDIVISIONS};

pub fn generate_sierpinsky_tetrahedron(node: &mut Node, _zoom: f64, depth: u64) -> bool {
    let s = node.scale;
    let p = node.position;

    fn iterate(p: Vector3<f32>, s: f32, bb_center: Vector3<f32>, bb_size: f32, n: u64) -> bool {
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
            // Tetrahedron (less stable in octree)
            let c_size_diag = (c_size * c_size * 1.0).sqrt();
            let bounding = [
                bb_center + vec3(-c_size_diag, -c_size, c_size_diag),
                bb_center + vec3(c_size_diag, -c_size, c_size_diag),
                bb_center + vec3(0.0, -c_size, -c_size),
                bb_center + vec3(0.0, c_size, c_size / 4.0),
            ];

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

    let d = iterate(p, s / TREE_SUBDIVISIONS as f32, vec3(0.0, 0.0, 0.0), 0.5, 5 + depth);

    if d {
        node.color = Vector3::new(0.25, 0.25, 1.0 + (1.0 / s.log2()).abs());
        node.solid = true;
    }

    return d;
}
