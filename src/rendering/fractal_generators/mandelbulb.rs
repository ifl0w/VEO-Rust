use crate::rendering::Node;
use cgmath::{Vector3, vec3, Array, InnerSpace};
use num_traits::Pow;

pub fn generate_mandelbulb(child: &mut Node, zoom: f64, depth: u64) -> bool {
    let origin = &child.position;
    let scale = child.scale;

    let position = origin * zoom as f32;

    // NOTE: For sample point (0, 0, 0) the iteration would be stuck and the distance estimation
    // would be NaN.
    if position == Vector3::from_value(0.0) {
        child.solid = true;
        return true;
    }

    let escape_radius = 3.0 as f64;
    let iter_start = 10 * ((depth as f64).log2() as i32 + 1);
    let mut iter = iter_start;

    fn to_spherical(a: Vector3<f64>) -> Vector3<f64> {
        let r = a.magnitude();
        let mut phi = (a.y / a.x).atan();
        let mut theta = (a.z / r).acos();

        // handle 0/0
        if a.y == 0.0 && a.x == 0.0 { phi = 0.0; };
        if a.z == 0.0 && r == 0.0 { theta = 0.0; };

        return vec3(r, phi, theta);
    }

    fn to_cartesian(a: Vector3<f64>) -> Vector3<f64> {
        let x = a.z.sin() * a.y.cos();
        let y = a.y.sin() * a.z.sin();
        let z = a.z.cos();

        return a.x * vec3(x, y, z);
    }

    // nth power in polar coordinates
    fn spherical_pow(a: Vector3<f64>, n: f64) -> Vector3<f64> {
        let r = a.x.pow(n);
        let phi = n * a.y;
        let theta = n * a.z;
        return vec3(r, phi, theta);
    }

    let c = vec3(position.x as f64, position.y as f64, position.z as f64);

    // z_0 = 0 + i0
    let mut v = vec3(0.0, 0.0, 0.0);
    let mut r = 0.0;

    // z_0' = 1 + 0i
    let mut dr = 1.0;

    let n = 8.0;
    while iter > 0 {
        let v_p = to_spherical(v);

        r = v_p.x;
        if r as f64 > escape_radius {
            break;
        }

        // scalar distance estimation
        // source: http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
        dr = r.pow(n - 1.0) * n * dr + 1.0;

        let v_next = spherical_pow(v_p, n);
        v = to_cartesian(v_next) + c;

        iter -= 1;
    }

    // values
    let distance = 0.5 * r * r.ln() / dr;

    let half_length = (scale * 0.5) as f64 * zoom;
    let radius = (half_length * half_length * 3.0).sqrt() as f64;

    if distance.abs() <= radius || iter == 0 {
        if distance > -radius {
            child.solid = true;

            // same runaway factor as for the mandelbrot. But only used for color here.
            let runaway = (depth) as f64 / ((iter_start - iter) as f64);

            // color transfer
            child.color = Vector3::new(
                (runaway) as f32,
                (1.0 - (distance / radius).ln() as f32) as f32,
                ((iter_start - iter) as f32 / iter_start as f32) as f32
            );

            // pseudo occlusion factor
            let dampening = position.magnitude() * (scale).exp2();
            child.color *= dampening;
        }
        return true;
    }

    return false;
}