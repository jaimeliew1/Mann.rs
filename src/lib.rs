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

use crate::Utilities::roll_1d_array;
use crate::Utilities::roll_3d_array;

pub use self::tensors::Tensors;
pub use self::utilities::Utilities;

use faer::col::generic::Col;
use faer::col::Own;
use faer::prelude::*;
use faer::sparse::linalg::solvers::Llt;
use faer::sparse::*;
use faer::Side;
use faer_ext::{IntoFaer, IntoNdarray};
use itertools::izip;
use itertools::Itertools;
use ndarray::linspace;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::{stack, Axis, Zip};
use ndrustfft::Complex;
use ninterp::prelude::*;
use numpy::Complex32;
use rayon::prelude::*;
use std::f32::consts::PI;
use std::iter::FromIterator;
use std::mem::drop;
use std::sync::Arc;
use std::sync::Mutex;
use tensors::Tensors::{Sheared, ShearedSinc, TensorGenerator};

use ndarray::{ArrayBase, Data, Dimension};
use std::fmt::Debug;

pub fn analyze_array<A, S, D>(array: &ArrayBase<S, D>)
where
    A: Copy + Into<f32> + Debug,
    S: Data<Elem = A>,
    D: Dimension,
{
    println!("Shape: {:?}", array.shape());

    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    let mut count = 0;

    for &val in array.iter() {
        let x: f32 = val.into();
        sum += x;
        sum_sq += x * x;
        count += 1;
    }

    if count == 0 {
        println!("Array is empty.");
        return;
    }

    let mean = sum / count as f32;
    let variance = (sum_sq / count as f32) - (mean * mean);
    let std_dev = variance.sqrt();

    println!("Mean: {}", mean);
    println!("Standard Deviation: {}", std_dev);
    println!("Sum (Checksum): {}", sum);
    println!("");
}

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

    pub fn linear_wave_numbers(&self) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Calculate linear wave number arrays.
        let kxs: Array1<f32> = Utilities::fftfreq(self.Nx, self.Lx / ((self.Nx) as f32));
        let kys: Array1<f32> = Utilities::fftfreq(self.Ny, self.Ly / ((self.Ny) as f32));
        let kzs: Array1<f32> = Utilities::rfftfreq(self.Nz, self.Lz / ((self.Nz) as f32));
        (kxs, kys, kzs)
    }

    pub fn angular_wave_numbers(&self) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Calculate linear wave number arrays.
        let kxs: Array1<f32> = Utilities::fftfreq(self.Nx, self.Lx / (2.0 * PI * (self.Nx) as f32));
        let kys: Array1<f32> = Utilities::fftfreq(self.Ny, self.Ly / (2.0 * PI * (self.Ny) as f32));
        let kzs: Array1<f32> =
            Utilities::rfftfreq(self.Nz, self.Lz / (2.0 * PI * (self.Nz) as f32));
        (kxs, kys, kzs)
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

    pub fn constrain(self, constraints: Vec<Constraint>, corr_thres: f32) -> ConstrainedStencil {
        ConstrainedStencil::new(self, constraints, corr_thres)
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
    A_factorized: Llt<usize, f32>,
}

impl ConstrainedStencil {
    pub fn new(stencil: Stencil, constraints: Vec<Constraint>, corr_thres: f32) -> Self {
        // Parallel?
        // where is threshold set?
        let p: &StencilParams = &stencil.p;
        println!("hello!!!! YOU MADE IT!!!");
        println!("extracting correlation grid...");
        let (Ruu, Rvv, Rww, Ruw): (Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>) =
            stencil.correlation_grids();

        println!("clip correlation data (current shape {:?})...", Ruu.shape());
        let Ruu: Array3<f32> = Ruu.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();
        println!("RUU");
        analyze_array(&Ruu);
        // let Rvv: Array3<f32> = Rvv.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();
        // let Rww: Array3<f32> = Rww.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();
        // let Ruw: Array3<f32> = Ruw.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();
        println!("clipped to {:?}.", Ruu.shape());
        println!("Calculating distance matrices...");

        let x_dist =
            Utilities::distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.x)));
        let y_dist =
            Utilities::distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.y)));
        let z_dist =
            Utilities::distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.z)));
        analyze_array(&x_dist);
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
        // let interp_vv = Interp3DOwned::new(
        //     x.clone(),
        //     y.clone(),
        //     z.clone(),
        //     Rvv,
        //     strategy::Linear,
        //     Extrapolate::Error,
        // )
        // .unwrap();
        // let interp_ww = Interp3DOwned::new(
        //     x.clone(),
        //     y.clone(),
        //     z.clone(),
        //     Rww,
        //     strategy::Linear,
        //     Extrapolate::Error,
        // )
        // .unwrap();
        // let interp_uw =
        //     Interp3DOwned::new(x, y, z, Ruw, strategy::Linear, Extrapolate::Error).unwrap();

        println!("Calculating correlation matrix...");
        // Add caching to each interpolator.
        let mut UUcorr: Array2<f32> = Array2::zeros(x_dist.raw_dim());
        // let mut VVcorr: Array2<f32> = Array2::zeros(x_dist.raw_dim());
        // let mut WWcorr: Array2<f32> = Array2::zeros(x_dist.raw_dim());
        // let mut UWcorr: Array2<f32> = Array2::zeros(x_dist.raw_dim());
        for (_x, _y, _z, u) in izip!(
            &x_dist,
            &y_dist,
            &z_dist,
            &mut UUcorr,
            // &mut VVcorr,
            // &mut WWcorr,
            // &mut UWcorr
        ) {
            *u = interp_uu.interpolate(&[*_x, *_y, *_z]).unwrap();
            // *v = interp_vv.interpolate(&[*_x, *_y, *_z]).unwrap();
            // *w = interp_ww.interpolate(&[*_x, *_y, *_z]).unwrap();
            // *uw = interp_uw.interpolate(&[*_x, *_y, *_z]).unwrap();
        }
        // println!("UUcorr {:?}.", UUcorr);
        analyze_array(&UUcorr);
        println!(
            "Applying hard threshold {} and converting to sparse matrix...",
            corr_thres
        );
        // let mut triplets: Vec<Triplet<usize, usize, f32>> = Vec::new();
        // for ((i, j), v) in UUcorr.indexed_iter() {
        //     if *v > corr_thres {
        //         triplets.push(Triplet::new(i, j, *v))
        //     }
        // }
        let triplets: Vec<Triplet<usize, usize, f32>> = UUcorr
            .indexed_iter()
            // .par_bridge()
            .filter(|(_, &v)| v > corr_thres)
            .map(|((i, j), v)| Triplet::new(i, j, *v))
            .collect();

        println!("n_triplets: {:?}", triplets.len());
        println!(
            "sparsity: {:?}%",
            100.0 - (triplets.len() as f64) / (constraints.len() as f64).powi(2) * 100.0
        );
        println!("creating sparse matrix...");
        let A = SparseColMat::<usize, f32>::try_new_from_triplets(
            constraints.len(),
            constraints.len(),
            &triplets,
        )
        .unwrap();


        println!("factorizing...");
        let llt = A.sp_cholesky(Side::Lower).unwrap();
        println!("Done!");

        ConstrainedStencil {
            stencil: stencil,
            constraints: constraints,
            A_factorized: llt,
        }
    }

    pub fn turbulate(
        &self,
        ae: f32,
        seed: u64,
        impulse_thres: f32,
        parallel: bool,
    ) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        println!(
            "Generating unconstrained box with seed={}, ae={}.",
            seed, ae
        );
        let (U, V, W) = self.stencil.turbulate(ae, seed, parallel);

        let (x, y, z) = self.stencil.p.get_axes();
        println!("making U interpolator...");
        let interp_uu = Interp3DOwned::new(
            x.clone(),
            y.clone(),
            z.clone(),
            U.clone(),
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        println!("interpolating contemporaneous wind speeds...");

        let U_contemp: Vec<f32> = self
            .constraints
            .iter()
            .map(|c| interp_uu.interpolate(&[c.x, c.y, c.z]).unwrap())
            .collect();
        println!("constructing b matrix...");
        let b = faer::col::Col::from_iter(
            U_contemp
                .iter()
                .zip(&self.constraints)
                .map(|(&u, c)| c.u - u),
        );
        println!("U_contemp");
        analyze_array(&Array1::from_iter(U_contemp.into_iter()));
        println!("b");
        analyze_array(&Array1::from_iter(b.iter().map(|x| *x)));
        // println!("b: {:?}", b);
        
        println!("solving linear system...");
        let Uweight: Vec<f32> = self.A_factorized.solve(&b).iter().map(|&x| x).collect();
        
        println!("Uweight");
        analyze_array(&Array1::from_iter(Uweight.iter().map(|x| *x)));
        println!("performing spectral superposition...");

        // Calculate normalized spectral component grids
        let (Ruu_f, Rvv_f, Rww_f, Ruw_f) = self.stencil.spectral_component_grids();

        // Calculate linear wave number arrays and record sizes.
        let (kxs, kys, kzs) = self.stencil.p.linear_wave_numbers();
        let (Nx_exp, Ny_exp, Nz_exp): (usize, usize, usize) = (kxs.len(), kys.len(), kzs.len());
        // TODO I think linear_wave_numbers should be for the true stencil size (including aperiodic doubling)
        // By doing so, the clipping just below is not necessary.
        // clip spectra
        let Ruu_f: Array3<f32> = Ruu_f.slice(s![..Nx_exp, ..Ny_exp, ..Nz_exp]).to_owned();
        let Rvv_f: Array3<f32> = Rvv_f.slice(s![..Nx_exp, ..Ny_exp, ..Nz_exp]).to_owned();
        let Rww_f: Array3<f32> = Rww_f.slice(s![..Nx_exp, ..Ny_exp, ..Nz_exp]).to_owned();
        let Ruw_f: Array3<f32> = Ruw_f.slice(s![..Nx_exp, ..Ny_exp, ..Nz_exp]).to_owned();

        println!("Ruu_f");
        analyze_array(&Ruu_f);
        // Roll arrays
        let (xroll, yroll, zroll): (isize, isize, isize) =
            ((&Nx_exp / 2) as isize, (&Ny_exp / 2) as isize, 0);

        let kxs: Array1<f32> = roll_1d_array(&kxs, &xroll);
        let kys: Array1<f32> = roll_1d_array(&kys, &yroll);
        let kzs: Array1<f32> = roll_1d_array(&kzs, &zroll);

        let Ruu_f: Array3<f32> = roll_3d_array(&Ruu_f, &xroll, &yroll, &zroll);
        let Rvv_f: Array3<f32> = roll_3d_array(&Rvv_f, &xroll, &yroll, &zroll);
        let Rww_f: Array3<f32> = roll_3d_array(&Rww_f, &xroll, &yroll, &zroll);
        let Ruw_f: Array3<f32> = roll_3d_array(&Ruw_f, &xroll, &yroll, &zroll);
        println!("hello2");
        // Reduce arrays TODO
        let (Ruu_f_max, Rvv_f_max, Rww_f_max): (f32, f32, f32) = (
            Ruu_f.iter().copied().fold(f32::NAN, f32::max),
            Rvv_f.iter().copied().fold(f32::NAN, f32::max),
            Rww_f.iter().copied().fold(f32::NAN, f32::max),
        );
        let ixmin: usize = Ruu_f
            .slice(s![.., yroll as usize, zroll as usize])
            .iter()
            .position(|&x| x >= impulse_thres * Ruu_f_max)
            .unwrap_or(0);
        let ixmax: usize = Ruu_f
            .slice(s![.., yroll as usize, zroll as usize])
            .iter()
            .rposition(|&x| x >= impulse_thres * Ruu_f_max)
            .unwrap_or(Nx_exp);

        let iymin: usize = Rvv_f
            .slice(s![xroll as usize, .., zroll as usize])
            .iter()
            .position(|&x| x >= impulse_thres * Rvv_f_max)
            .unwrap_or(0);
        let iymax: usize = Rvv_f
            .slice(s![xroll as usize, .., zroll as usize])
            .iter()
            .rposition(|&x| x >= impulse_thres * Rvv_f_max)
            .unwrap_or(Ny_exp);
        let izmax: usize = Rww_f
            .slice(s![xroll as usize, yroll as usize, ..])
            .iter()
            .rposition(|&x| x >= impulse_thres * Rww_f_max)
            .unwrap_or(Nz_exp);

        println!("ixmin: {ixmin}, ixmax {ixmax}");
        println!("iymin: {iymin}, iymax {iymax}");
        println!("izmax: {izmax}");

        let Ruu_f: Array3<f32> = Ruu_f
            .slice(s![ixmin..ixmax, iymin..iymax, 0..izmax])
            .to_owned();
        let Rvv_f: Array3<f32> = Rvv_f
            .slice(s![ixmin..ixmax, iymin..iymax, 0..izmax])
            .to_owned();
        let Rww_f: Array3<f32> = Rww_f
            .slice(s![ixmin..ixmax, iymin..iymax, 0..izmax])
            .to_owned();
        let Ruw_f: Array3<f32> = Ruw_f
            .slice(s![ixmin..ixmax, iymin..iymax, 0..izmax])
            .to_owned();
        println!("hello3");
        let Ruu_f: Array3<Complex32> = Ruu_f.mapv(|x| Complex32::new(x, 0.0));
        let Rvv_f: Array3<Complex32> = Rvv_f.mapv(|x| Complex32::new(x, 0.0));
        let Rww_f: Array3<Complex32> = Rww_f.mapv(|x| Complex32::new(x, 0.0));
        let Ruw_f: Array3<Complex32> = Ruw_f.mapv(|x| Complex32::new(x, 0.0));
        println!("hello4");
        let kxs: Array1<f32> = kxs.slice(s![ixmin..ixmax]).to_owned();
        let kys: Array1<f32> = kys.slice(s![iymin..iymax]).to_owned();
        let kzs: Array1<f32> = kzs.slice(s![0..izmax]).to_owned();
        println!("hello5");
        // Calculate 3d meshgrid of linear wave numbers.
        let (nx, ny, nz) = (kxs.len(), kys.len(), kzs.len());
        let mut kx_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        let mut ky_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        let mut kz_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    kx_mesh[[i, j, k]] = Complex32::new(kxs[i], 0.0);
                    ky_mesh[[i, j, k]] = Complex32::new(kys[j], 0.0);
                    kz_mesh[[i, j, k]] = Complex32::new(kzs[k], 0.0);
                }
            }
        }
        println!("hello6");
        let U_f: Array3<Complex32>;
        let V_f: Array3<Complex32>;
        let W_f: Array3<Complex32>;
        if !parallel {
            let mut _U_f = Array3::<Complex32>::zeros((nx, ny, nz));
            let mut _V_f = Array3::<Complex32>::zeros((nx, ny, nz));
            let mut _W_f = Array3::<Complex32>::zeros((nx, ny, nz));

            for (i, c) in self.constraints.iter().enumerate() {
                let phase: Array3<Complex32> = (Complex32::new(0.0, -2.0 * std::f32::consts::PI)
                    * (&kx_mesh * c.x + &ky_mesh * c.y + &kz_mesh * c.z))
                    .mapv(|x| x.exp());

                _U_f += &(&phase * (&Ruu_f * Uweight[i]));
                // _U_f += &(&phase * (&Ruu_f * Uweight[i] + &Ruw_f * CConstW[i]));
                // _V_f += &(&phase * (&Rvv_f * CConstV[i]));
                // _W_f += &(&phase * (&Ruw_f * CConstU[i] + &Rww_f * CConstW[i]));
            }
            U_f = _U_f;
            V_f = _V_f;
            W_f = _W_f;
        } else {
            let _U_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
            let _V_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
            let _W_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
            self.constraints.par_iter().enumerate().for_each(|(i, c)| {
                let phase: Array3<Complex32> = (Complex32::new(0.0, -2.0 * std::f32::consts::PI)
                    * (&kx_mesh * c.x + &ky_mesh * c.y + &kz_mesh * c.z))
                    .mapv(|x| x.exp());

                {
                    let mut _U_f = _U_f.lock().unwrap();
                    let mut _V_f = _V_f.lock().unwrap();
                    let mut _W_f = _W_f.lock().unwrap();
                    let to_add = Complex32::new(1.0, 0.0) * &phase * (&Ruu_f * Uweight[i]);
                    _U_f.scaled_add(Complex32::new(1.0, 0.0), &to_add);
                    // let to_add = Complex32::new(1.0, 0.0)
                    //     * &phase
                    //     * (&Ruu_f * CConstU[i] + &Ruw_f * CConstW[i]);
                    // _U_f.scaled_add(Complex32::new(1.0, 0.0), &to_add);

                    // let to_add = Complex32::new(1.0, 0.0) * &phase * (&Rvv_f * CConstV[i]);
                    // _V_f.scaled_add(Complex32::new(1.0, 0.0), &to_add);

                    // let to_add = Complex32::new(1.0, 0.0)
                    //     * &phase
                    //     * (&Ruw_f * CConstU[i] + &Rww_f * CConstW[i]);
                    // _W_f.scaled_add(Complex32::new(1.0, 0.0), &to_add);
                }
            });
            U_f = _U_f.lock().unwrap().to_owned();
            V_f = _V_f.lock().unwrap().to_owned();
            W_f = _W_f.lock().unwrap().to_owned();
        }
        println!("hello7");
        // Expand arrays
        let mut U_f_exp: Array3<Complex32> = Array3::zeros((Nx_exp, Ny_exp, Nz_exp));
        let mut V_f_exp: Array3<Complex32> = Array3::zeros((Nx_exp, Ny_exp, Nz_exp));
        let mut W_f_exp: Array3<Complex32> = Array3::zeros((Nx_exp, Ny_exp, Nz_exp));

        U_f_exp
            .slice_mut(s![ixmin..ixmax, iymin..iymax, 0..izmax])
            .assign(&U_f);
        V_f_exp
            .slice_mut(s![ixmin..ixmax, iymin..iymax, 0..izmax])
            .assign(&V_f);
        W_f_exp
            .slice_mut(s![ixmin..ixmax, iymin..iymax, 0..izmax])
            .assign(&W_f);

        // Unroll arrays
        let mut U_f_exp: Array3<Complex32> =
            roll_3d_array(&U_f_exp, &(-xroll), &(-yroll), &(-zroll));
        let mut V_f_exp: Array3<Complex32> =
            roll_3d_array(&V_f_exp, &(-xroll), &(-yroll), &(-zroll));
        let mut W_f_exp: Array3<Complex32> =
            roll_3d_array(&W_f_exp, &(-xroll), &(-yroll), &(-zroll));

        println!("inverse 3d fourier transform...");
        let U: Array3<f32> = Utilities::irfft3d(&mut U_f_exp) + U;
        let V: Array3<f32> = Utilities::irfft3d(&mut V_f_exp) + V;
        let W: Array3<f32> = Utilities::irfft3d(&mut W_f_exp) + W;

        println!("Done!");
        (U, V, W)
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
