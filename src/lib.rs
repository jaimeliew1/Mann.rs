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

pub use self::tensors::Tensors;
pub use self::utilities::Utilities;

use itertools::izip;
use ndarray::linspace;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::{stack, Axis, Zip};
use ndrustfft::Complex;
use ninterp::prelude::*;
use numpy::Complex32;
use std::f32::consts::PI;
use std::mem::drop;
use tensors::Tensors::{Sheared, ShearedSinc, TensorGenerator};

pub struct StencilParams {
    L: f32,
    gamma: f32,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    aperiodic_x: bool,
    aperiodic_y: bool,
    aperiodic_z: bool,
}

impl StencilParams {
    pub fn get_axes(&self) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        (
            linspace(0.0, self.Lx, self.Nx).collect(),
            linspace(0.0, self.Ly, self.Ny).collect(),
            linspace(0.0, self.Lz, self.Nz).collect(),
        )
    }
}
pub struct Stencil {
    p: StencilParams,
    stencil: Array5<f32>,
}

impl Stencil {
    pub fn from_params(
        L: f32,
        gamma: f32,
        Lx: f32,
        Ly: f32,
        Lz: f32,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        aperiodic_x: bool,
        aperiodic_y: bool,
        aperiodic_z: bool,
        sinc_thres: f32,
        parallel: bool,
    ) -> Self {
        let p: StencilParams = StencilParams {
            L: L,
            gamma: gamma,
            Lx: Lx,
            Ly: Ly,
            Lz: Lz,
            Nx: Nx,
            Ny: Ny,
            Nz: Nz,
            aperiodic_x: aperiodic_x,
            aperiodic_y: aperiodic_y,
            aperiodic_z: aperiodic_z,
        };
        let stencil: Array5<f32>;
        if parallel {
            stencil = stencilate_sinc_par(&p, sinc_thres);
        } else {
            stencil = stencilate_sinc(&p, sinc_thres);
        }
        Stencil {
            p: p,
            stencil: stencil,
        }
    }

    pub fn turbulate(
        &self,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        let (U, V, W): (Array3<f32>, Array3<f32>, Array3<f32>);
        if parallel {
            (U, V, W) = turbulate_par(&self.stencil.view(), ae, seed, &self.p);
        } else {
            (U, V, W) = turbulate(&self.stencil.view(), ae, seed, &self.p);
        }
        (
            U.slice(s![..self.p.Nx, ..self.p.Ny, ..self.p.Nz])
                .to_owned(),
            V.slice(s![..self.p.Nx, ..self.p.Ny, ..self.p.Nz])
                .to_owned(),
            W.slice(s![..self.p.Nx, ..self.p.Ny, ..self.p.Nz])
                .to_owned(),
        )
    }
    pub fn partial_turbulate(
        &self,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
        let (U_f, V_f, W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>);
        if parallel {
            (U_f, V_f, W_f) = partial_turbulate_par(&self.stencil.view(), ae, seed, &self.p);
        } else {
            (U_f, V_f, W_f) = partial_turbulate(&self.stencil.view(), ae, seed, &self.p);
        }
        (U_f, V_f, W_f)
    }
    /// Returns the normalized spectral component from a stencil of shape `(Nx, Ny,
    /// Nz, 3, 3)`.
    ///
    /// This function extracts the `Ruu`, `Rvv`, `Rww`, and `Ruw` components from
    /// the stencil. These components are normalized such that their inverse fourier
    /// transform as a maximum value of 1.
    ///
    /// - `Ruu_f`: Normalized spectral energy of the `u` component
    /// - `Rvv_f`: Normalized spectral energy of the `v` component
    /// - `Rww_f`: Normalized spectral energy of the `w` component
    /// - `Ruw_f`: Normalized spectral cross-component `uw`
    ///
    /// # Arguments
    ///
    /// * `stencil` - A 5D array containing the velocity correlation tensor across a
    ///   3D grid.
    ///
    /// # Returns
    ///
    /// A tuple of 3D arrays `(Ruu_f, Rvv_f, Rww_f, Ruw_f)` where each is the real
    /// part of the normalized spectral component for the corresponding correlation
    /// tensor entry.
    pub fn spectral_component_grids(&self) -> (Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>) {
        let mut Ruu_f: Array3<Complex32> = self
            .stencil
            .slice(s![.., .., .., 0, 0])
            .mapv(|x| Complex32::new(x, 0.0));
        let mut Rvv_f: Array3<Complex32> = self
            .stencil
            .slice(s![.., .., .., 1, 1])
            .mapv(|x| Complex32::new(x, 0.0));
        let mut Rww_f: Array3<Complex32> = self
            .stencil
            .slice(s![.., .., .., 2, 2])
            .mapv(|x| Complex32::new(x, 0.0));
        let Ruw_f: Array3<Complex32> = self
            .stencil
            .slice(s![.., .., .., 0, 2])
            .mapv(|x| Complex32::new(x, 0.0));

        let Ruu: Array3<f32> = Utilities::irfft3d(&mut Ruu_f);
        let Rvv: Array3<f32> = Utilities::irfft3d(&mut Rvv_f);
        let Rww: Array3<f32> = Utilities::irfft3d(&mut Rww_f);

        // Normalize frequency components
        (
            Ruu_f.mapv(|x| x.re / Ruu[[0, 0, 0]]),
            Rvv_f.mapv(|x| x.re / Rvv[[0, 0, 0]]),
            Rww_f.mapv(|x| x.re / Rww[[0, 0, 0]]),
            Ruw_f.mapv(|x| x.re / (Ruu[[0, 0, 0]] * Rww[[0, 0, 0]]).sqrt()),
        )
    }

    /// Returns the normalized correlation matrices from a stencil of shape `(Nx,
    /// Ny, Nz, 3, 3)`.
    ///
    /// This function extracts the `Ruu`, `Rvv`, `Rww`, and `Ruw` components from
    /// the stencil and performs an inverse fourier transform to arrive at the
    /// spatial correlation. These components are normalized such that their inverse
    /// fourier transform as a maximum value of 1.
    ///
    /// # Arguments
    ///
    /// * `stencil` - A 5D array containing the velocity correlation tensor across a
    ///   3D grid.
    ///
    /// # Returns
    ///
    /// A tuple of 3D arrays `(Ruu, Rvv, Rww, Ruw)` where each is the spatial
    /// correlation matrix for the U, V and W wind components as well as the cross
    /// correlation between U and W.
    pub fn correlation_grids(&self) -> (Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>) {
        let mut Ruu_f: Array3<Complex32> = self
            .stencil
            .slice(s![.., .., .., 0, 0])
            .mapv(|x| Complex32::new(x, 0.0));
        let mut Rvv_f: Array3<Complex32> = self
            .stencil
            .slice(s![.., .., .., 1, 1])
            .mapv(|x| Complex32::new(x, 0.0));
        let mut Rww_f: Array3<Complex32> = self
            .stencil
            .slice(s![.., .., .., 2, 2])
            .mapv(|x| Complex32::new(x, 0.0));
        let mut Ruw_f: Array3<Complex32> = self
            .stencil
            .slice(s![.., .., .., 0, 2])
            .mapv(|x| Complex32::new(x, 0.0));

        let Ruu: Array3<f32> = Utilities::irfft3d(&mut Ruu_f);
        drop(Ruu_f);
        let Rvv: Array3<f32> = Utilities::irfft3d(&mut Rvv_f);
        drop(Rvv_f);
        let Rww: Array3<f32> = Utilities::irfft3d(&mut Rww_f);
        drop(Rww_f);
        let Ruw: Array3<f32> = Utilities::irfft3d(&mut Ruw_f);
        drop(Ruw_f);

        (
            Ruu.mapv(|x| x / Ruu[[0, 0, 0]]),
            Rvv.mapv(|x| x / Rvv[[0, 0, 0]]),
            Rww.mapv(|x| x / Rww[[0, 0, 0]]),
            Ruw.mapv(|x| x / (Ruu[[0, 0, 0]] * Rww[[0, 0, 0]]).sqrt()),
        )
    }

    pub fn constrain(self, constraints: Vec<Constraint>) -> ConstrainedStencil {
        ConstrainedStencil::new(self, constraints)
    }
}

pub struct Constraint {
    x: f32,
    y: f32,
    z: f32,
    u: f32,
    // v: f32,
    // w: f32,
}

pub struct ConstrainedStencil {
    stencil: Stencil,
    constraints: Vec<Constraint>,
    A_factorized: Array2<f32>,
}

impl ConstrainedStencil {
    pub fn new(stencil: Stencil, constraints: Vec<Constraint>) -> Self {
        // Parallel?
        // where is threshold set?
        let p: &StencilParams = &stencil.p;
        let n: usize = constraints.len();
        println!("hello!!!! YOU MADE IT!!!");
        println!("extracting correlation grid...");
        let (Ruu, Rvv, Rww, Ruw): (Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>) =
            stencil.correlation_grids();

        println!("clip correlation data (current shape {:?})...", Ruu.shape());
        let Ruu: Array3<f32> = Ruu.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();
        let Rvv: Array3<f32> = Rvv.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();
        let Rww: Array3<f32> = Rww.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();
        let Ruw: Array3<f32> = Ruw.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();
        println!("clipped to {:?}.", Ruu.shape());
        println!("Calculating distance matrices...");

        let x_dist =
            Utilities::distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.x)));
        let y_dist =
            Utilities::distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.y)));
        let z_dist =
            Utilities::distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.z)));
        println!("x_dist {:?}.", x_dist);
        println!("building interpolator...");

        let (x, y, z) = p.get_axes();
        let interp_uu = Interp3DOwned::new(
            x.clone(),
            y.clone(),
            z.clone(),
            Ruu,
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        let interp_vv = Interp3DOwned::new(
            x.clone(),
            y.clone(),
            z.clone(),
            Rvv,
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        let interp_ww = Interp3DOwned::new(
            x.clone(),
            y.clone(),
            z.clone(),
            Rww,
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        let interp_uw =
            Interp3DOwned::new(x, y, z, Ruw, strategy::Linear, Extrapolate::Error).unwrap();

        println!("Calculating correlation matrix...");
        // Add caching to each interpolator.
        let mut UUcorr: Array2<f32> = Array2::zeros(x_dist.raw_dim());
        let mut VVcorr: Array2<f32> = Array2::zeros(x_dist.raw_dim());
        let mut WWcorr: Array2<f32> = Array2::zeros(x_dist.raw_dim());
        let mut UWcorr: Array2<f32> = Array2::zeros(x_dist.raw_dim());
        for (_x, _y, _z, u, v, w, uw) in izip!(
            &x_dist,
            &y_dist,
            &z_dist,
            &mut UUcorr,
            &mut VVcorr,
            &mut WWcorr,
            &mut UWcorr
        ) {
            *u = interp_uu.interpolate(&[*_x, *_y, *_z]).unwrap();
            *v = interp_vv.interpolate(&[*_x, *_y, *_z]).unwrap();
            *w = interp_ww.interpolate(&[*_x, *_y, *_z]).unwrap();
            *uw = interp_uw.interpolate(&[*_x, *_y, *_z]).unwrap();
        }
        println!("UUcorr {:?}.", UUcorr);

        println!("Applying hard threshold (print threshold) and converting to sparse matrix...");
        println!("factorizing...");
        println!("Done!");

        ConstrainedStencil {
            stencil: stencil,
            constraints: constraints,
            A_factorized: Array2::zeros([2, 3]),
        }
    }
}
pub fn stencilate_par(p: StencilParams) -> Array5<f32> {
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let tensor_gen = Sheared::from_params(1.0, p.L, p.gamma);
    stencil
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    component.assign(&tensor_gen.decomp(K));
                }
            }
        });
    stencil
}

pub fn stencilate_sinc_par(p: &StencilParams, sinc_thres: f32) -> Array5<f32> {
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let tensor_gen_sinc = ShearedSinc::from_params(1.0, p.L, p.gamma, Ly, Lz, 1.0, 2);
    let tensor_gen = Sheared::from_params(1.0, p.L, p.gamma);

    stencil
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    let norm = K.iter().fold(0.0, |acc, &x| acc + x * x);
                    if norm < sinc_thres / p.L {
                        component.assign(&tensor_gen_sinc.decomp(K));
                    } else {
                        component.assign(&tensor_gen.decomp(K));
                    }
                }
            }
        });
    stencil
}

pub fn partial_turbulate_par(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    p: &StencilParams,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);

    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    Zip::from(UVW_f.outer_iter_mut())
        .and(stencil.outer_iter())
        .and(random.outer_iter())
        .par_for_each(|mut UVW_slice, stencil_slice, random_slice| {
            Zip::from(UVW_slice.outer_iter_mut())
                .and(stencil_slice.outer_iter())
                .and(random_slice.outer_iter())
                .par_for_each(|mut UVW_col, stencil_col, random_col| {
                    Zip::from(UVW_col.outer_iter_mut())
                        .and(stencil_col.outer_iter())
                        .and(random_col.outer_iter())
                        .for_each(|mut freq_comp, tensor, n| {
                            let _tensor = tensor.mapv(|elem| Complex32::new(elem, 0.0));
                            freq_comp.assign(&_tensor.dot(&n));
                            freq_comp *= KVolScaleFac;
                        })
                })
        });
    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn turbulate_par(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    p: &StencilParams,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_turbulate_par(stencil, ae, seed, p);

    let U: Array3<f32> = Utilities::irfft3d_par(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = Utilities::irfft3d_par(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = Utilities::irfft3d_par(&mut W_f);
    drop(W_f);
    (U, V, W)
}

pub fn stencilate(
    L: f32,
    gamma: f32,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    Nx: usize,
    Ny: usize,
    Nz: usize,
) -> Array5<f32> {
    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);
    stencil
        .outer_iter_mut()
        .into_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    component.assign(&tensor_gen.decomp(K));
                }
            }
        });
    stencil
}

pub fn stencilate_sinc(p: &StencilParams, sinc_thres: f32) -> Array5<f32> {
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let tensor_gen_sinc = ShearedSinc::from_params(1.0, p.L, p.gamma, Ly, Lz, 1.0, 2);
    let tensor_gen = Sheared::from_params(1.0, p.L, p.gamma);

    stencil
        .outer_iter_mut()
        .into_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    let norm = K.iter().fold(0.0, |acc, &x| acc + x * x);
                    if norm < sinc_thres / p.L {
                        component.assign(&tensor_gen_sinc.decomp(K));
                    } else {
                        component.assign(&tensor_gen.decomp(K));
                    }
                }
            }
        });
    stencil
}

pub fn partial_turbulate(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    p: &StencilParams,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);

    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    Zip::from(UVW_f.outer_iter_mut())
        .and(stencil.outer_iter())
        .and(random.outer_iter())
        .for_each(|mut UVW_slice, stencil_slice, random_slice| {
            Zip::from(UVW_slice.outer_iter_mut())
                .and(stencil_slice.outer_iter())
                .and(random_slice.outer_iter())
                .for_each(|mut UVW_col, stencil_col, random_col| {
                    Zip::from(UVW_col.outer_iter_mut())
                        .and(stencil_col.outer_iter())
                        .and(random_col.outer_iter())
                        .for_each(|mut freq_comp, tensor, n| {
                            let _tensor = tensor.mapv(|elem| Complex32::new(elem, 0.0));
                            freq_comp.assign(&_tensor.dot(&n));
                            freq_comp *= KVolScaleFac;
                        })
                })
        });
    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn turbulate(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    p: &StencilParams,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_turbulate(stencil, ae, seed, p);

    let U: Array3<f32> = Utilities::irfft3d(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = Utilities::irfft3d(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = Utilities::irfft3d(&mut W_f);
    drop(W_f);
    (U, V, W)
}

pub fn partial_turbulate_unit(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_unit(seed, Nx, Ny, Nz / 2 + 1);

    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    Zip::from(UVW_f.outer_iter_mut())
        .and(stencil.outer_iter())
        .and(random.outer_iter())
        .for_each(|mut UVW_slice, stencil_slice, random_slice| {
            Zip::from(UVW_slice.outer_iter_mut())
                .and(stencil_slice.outer_iter())
                .and(random_slice.outer_iter())
                .for_each(|mut UVW_col, stencil_col, random_col| {
                    Zip::from(UVW_col.outer_iter_mut())
                        .and(stencil_col.outer_iter())
                        .and(random_col.outer_iter())
                        .for_each(|mut freq_comp, tensor, n| {
                            let _tensor = tensor.mapv(|elem| Complex32::new(elem, 0.0));
                            freq_comp.assign(&_tensor.dot(&n));
                            freq_comp *= KVolScaleFac;
                        })
                })
        });
    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn turbulate_unit(
    stencil: &ArrayView5<f32>,
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_turbulate_unit(stencil, ae, seed, Nx, Ny, Nz, Lx, Ly, Lz);

    let U: Array3<f32> = Utilities::irfft3d(&mut U_f);
    let V: Array3<f32> = Utilities::irfft3d(&mut V_f);
    let W: Array3<f32> = Utilities::irfft3d(&mut W_f);
    (U, V, W)
}

pub fn partial_forgetful_turbulate_par(
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    L: f32,
    gamma: f32,
    sinc_thres: f32,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    let tensor_gen_sinc = ShearedSinc::from_params(1.0, L, gamma, Ly, Lz, 1.0, 2);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);

    UVW_f
        .outer_iter_mut()
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    let norm = K.iter().fold(0.0, |acc, &x| acc + x * x);

                    let invol: bool = norm < sinc_thres / L;

                    let coef: Array2<f32> = match invol {
                        true => tensor_gen_sinc.decomp(K),
                        false => tensor_gen.decomp(K),
                    };
                    let coef = coef.mapv(|elem| Complex32::new(elem, 0.0));
                    let n = random.slice(s![i, j, k, ..]);
                    component.assign(&coef.dot(&n));
                    component *= KVolScaleFac;
                }
            }
        });

    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn forgetful_turbulate_par(
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    L: f32,
    gamma: f32,
    sinc_thres: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_forgetful_turbulate_par(ae, seed, Nx, Ny, Nz, Lx, Ly, Lz, L, gamma, sinc_thres);

    let U: Array3<f32> = Utilities::irfft3d_par(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = Utilities::irfft3d_par(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = Utilities::irfft3d_par(&mut W_f);
    drop(W_f);
    (U, V, W)
}

pub fn partial_forgetful_turbulate(
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    L: f32,
    gamma: f32,
    sinc_thres: f32,
) -> (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) {
    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = Utilities::complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
    let mut UVW_f: Array4<Complex32> = Array4::zeros((Nx, Ny, (Nz / 2 + 1), 3));

    let tensor_gen_sinc = ShearedSinc::from_params(1.0, L, gamma, Ly, Lz, 1.0, 2);
    let tensor_gen = Sheared::from_params(1.0, L, gamma);

    UVW_f
        .outer_iter_mut()
        .into_iter()
        .enumerate()
        .for_each(|(i, mut slice)| {
            for (j, mut column) in slice.outer_iter_mut().enumerate() {
                for (k, mut component) in column.outer_iter_mut().enumerate() {
                    let K = &[Kx[i], Ky[j], Kz[k]];
                    let norm = K.iter().fold(0.0, |acc, &x| acc + x * x);

                    let invol: bool = norm < sinc_thres / L;

                    let coef: Array2<f32> = match invol {
                        true => tensor_gen_sinc.decomp(K),
                        false => tensor_gen.decomp(K),
                    };
                    let coef = coef.mapv(|elem| Complex32::new(elem, 0.0));
                    let n = random.slice(s![i, j, k, ..]);
                    component.assign(&coef.dot(&n));
                    component *= KVolScaleFac;
                }
            }
        });

    UVW_f[[0, 0, 0, 0]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 1]] = Complex::new(0.0, 0.0);
    UVW_f[[0, 0, 0, 2]] = Complex::new(0.0, 0.0);
    (
        UVW_f.slice(s![.., .., .., 0]).to_owned(),
        UVW_f.slice(s![.., .., .., 1]).to_owned(),
        UVW_f.slice(s![.., .., .., 2]).to_owned(),
    )
}

pub fn forgetful_turbulate(
    ae: f32,
    seed: u64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    L: f32,
    gamma: f32,
    sinc_thres: f32,
) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
    let (mut U_f, mut V_f, mut W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
        partial_forgetful_turbulate(ae, seed, Nx, Ny, Nz, Lx, Ly, Lz, L, gamma, sinc_thres);

    let U: Array3<f32> = Utilities::irfft3d(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = Utilities::irfft3d(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = Utilities::irfft3d(&mut W_f);
    drop(W_f);
    (U, V, W)
}
