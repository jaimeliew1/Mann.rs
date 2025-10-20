#![allow(non_snake_case)]
//! Coherence turbulence box generation using the Mann turbulence model.
//!
//! `Mannrs` provides a computationally efficient module for generating Mann
//! turbulence boxes for wind turbine simulations. `Mannrs` is designed to be
//! called from Python, however the underlying functions are also available in
//! rust.
mod python_interface;
mod tensors;
mod tests;
mod utilities;
mod unconstrained;
mod constrained;
mod spectral_impulse;


use std::f32::consts::PI;
use ndarray::prelude::*;



pub use self::tensors::Tensors;
pub use self::unconstrained::{Stencil, StencilParams};
pub use self::constrained::{ConstrainedStencil, Constraint};
pub use self::spectral_impulse::SpectralImpulseResponse;
use self::utilities::trapezoidal_integral_2d;
use tensors::Tensors::{Sheared, TensorGenerator};


pub fn mann_spectra(
    kx: &Array1<f32>,
    ae: f32,
    l: f32,
    gamma: f32,
) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
    let tensor_gen = Sheared::from_params(ae, l, gamma);
    let nr = 150;
    let ntheta = 30;

    let rs = Array1::linspace(-4.0, 7.0, nr).mapv(|x| 10f32.powf(x));
    let thetas = Array1::linspace(0.0, 2.0 * PI, ntheta);

    let mut uu_vals = Array1::zeros(kx.len());
    let mut vv_vals = Array1::zeros(kx.len());
    let mut ww_vals = Array1::zeros(kx.len());
    let mut uw_vals = Array1::zeros(kx.len());

    for (idx, &kx_val) in kx.iter().enumerate() {
        let mut uu_grid = Array::zeros((nr, ntheta));
        let mut vv_grid = Array::zeros((nr, ntheta));
        let mut ww_grid = Array::zeros((nr, ntheta));
        let mut uw_grid = Array::zeros((nr, ntheta));

        for (i, &r) in rs.iter().enumerate() {
            for (j, &theta) in thetas.iter().enumerate() {
                let ky = r * theta.cos();
                let kz = r * theta.sin();
                let tensor = tensor_gen.tensor(&[kx_val, ky, kz]);
                uu_grid[[i, j]] = r * tensor[[0, 0]];
                vv_grid[[i, j]] = r * tensor[[1, 1]];
                ww_grid[[i, j]] = r * tensor[[2, 2]];
                uw_grid[[i, j]] = r * tensor[[0, 2]];
            }
        }

        uu_vals[idx] = trapezoidal_integral_2d(&uu_grid, &rs, &thetas);
        vv_vals[idx] = trapezoidal_integral_2d(&vv_grid, &rs, &thetas);
        ww_vals[idx] = trapezoidal_integral_2d(&ww_grid, &rs, &thetas);
        uw_vals[idx] = trapezoidal_integral_2d(&uw_grid, &rs, &thetas);
    }

    (uu_vals, vv_vals, ww_vals, uw_vals)
}
