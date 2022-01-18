use ndarray::{Array, Array2, ArrayView};

fn lifetime_approx(kL: f64) -> f64 {
    return 0.0;
}

fn lifetime_exact(kL: f64) -> f64 {
    return 0.0;
}

fn vonkarman_spectrum(ae: f64, k: f64, L: f64) -> f64 {
    return 0.0;
}

fn sqrt_iso_tensor(K: &[f64], ae: f64, L: f64) -> Array2<f64> {
    Array2::zeros((3, 3))
}

fn iso_tensor(K: &[f64], ae: f64, L: f64) -> Array2<f64> {
    Array2::zeros((3, 3))
}

fn sqrt_sheared_tensor(K: &[f64], ae: f64, L: f64, gamma: f64) -> Array2<f64> {
    Array2::zeros((3, 3))
}

fn sheared_transform(K: &[f64], ae: f64, L: f64, gamma: f64) -> Array2<f64> {
    Array2::zeros((3, 3))
}

fn sheared_tensor(K: &[f64], ae: f64, L: f64, gamma: f64) -> Array2<f64> {
    Array2::zeros((3, 3))
}

mod python_interface;
mod tests;
