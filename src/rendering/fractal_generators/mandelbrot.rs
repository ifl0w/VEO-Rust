use crate::rendering::Node;
use cgmath::{Vector3, vec3};

pub fn generate_mandelbrot(child: &mut Node, zoom: f64, depth: u64) -> bool {
    let origin = &child.position;
    let scale = child.scale;

    let thickness = 0.0; // only a slice
    if origin.y + child.scale * 0.5 < -thickness
        || origin.y - child.scale * 0.5 > thickness {
        return false;
    };

    let position = origin * zoom as f32 - vec3(0.7, 0.0, 0.0);

    let escape_radius = 4.0 + 1000.0 * scale as f64;
    let iter_start = 100 + 100 * ((depth as f64).log2() + 1.0) as u32;
    let mut iter = iter_start;

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
        // derivative
        zp_re = 2.0 * (z_re * zp_re - z_im * zp_im) + 1.0;
        zp_im = 2.0 * (z_re * zp_im + z_im * zp_re);

        // iteration
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

    // magnitude
    let z_val = (z_re2 + z_im2).sqrt();
    let zp_val = (zp_re * zp_re + zp_im * zp_im).sqrt();

    let distance = 0.5 * z_val * z_val.ln() / zp_val;

    let half_length: f64 = (scale * 0.5) as f64 * zoom;
    let radius = (half_length * half_length * 2.0).sqrt() as f64;

    // inner of mandelbrot
    if distance <= 0.0 || iter == 0 {
        child.solid = true;
        child.color = Vector3::new(0.0, 0.0, 0.0);
        return true;
    }

    // the further away from the zero point, the more chaotic the behaviour.
    // this value is used for normalization of colors and the runaway.
    let chaos_factor = ((c_re * c_re + c_im * c_im).sqrt() / zoom) + 1.0;
    // Limit octree depth by "runaway". The octree is only refined if the runaway is < 1.
    // This metric describes the number of iterations required to reach the escape radius in
    // relation to the current octree depth. This works since the required iteration number
    // increases the closer we get to the border. However, I did not proofe this statement and
    // it is only backed by experimental
    let runaway = (depth as f64) / (((iter_start - iter) as f64) * chaos_factor);
    if runaway < 1.0 {
        child.color = Vector3::new(
            (runaway * chaos_factor) as f32,
            (1.0 - (distance / radius).ln() as f32 / (1.0 / scale).ln()) * chaos_factor as f32,
            ((iter_start - iter) as f32 / iter_start as f32) * chaos_factor as f32
        );
        child.solid = true;
        return true;
    }

    return false;
}