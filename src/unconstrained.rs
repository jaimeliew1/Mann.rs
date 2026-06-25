use crate::spectral_impulse::SpectralImpulseResponse;
use crate::tensors::Tensors::{Sheared, ShearedSinc, TensorGenerator};
use crate::utilities::{
    complex_random_gaussian, fftfreq, freq_components, irfft3d, irfft3d_par, rfftfreq,
};
use ndrustfft::Complex;
use numpy::Complex32;
use std::f32::consts::PI;

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::{linspace, Zip};

#[derive(Debug, Clone)]
pub struct StencilParams {
    pub L: f32,
    pub gamma: f32,
    pub Lx: f32,
    pub Ly: f32,
    pub Lz: f32,
    pub Nx: usize,
    pub Ny: usize,
    pub Nz: usize,
    pub aperiodic_x: bool,
    pub aperiodic_y: bool,
    pub aperiodic_z: bool,
}

impl StencilParams {
    pub fn get_axes(&self) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        (
            Array1::from_iter((0..self.Nx).map(|i| i as f32 * self.Lx / ((self.Nx - 1) as f32))),
            Array1::from_iter((0..self.Ny).map(|j| j as f32 * self.Ly / ((self.Ny - 1) as f32))),
            Array1::from_iter((0..self.Nz).map(|k| k as f32 * self.Lz / ((self.Nz - 1) as f32))),
        )
    }

    pub fn linear_wave_numbers(&self) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Calculate linear wave number arrays.
        let kxs: Array1<f32> = fftfreq(self.Nx, self.Lx / (self.Nx as f32));
        let kys: Array1<f32> = fftfreq(self.Ny, self.Ly / (self.Ny as f32));
        let kzs: Array1<f32> = rfftfreq(self.Nz, self.Lz / (self.Nz as f32));
        (kxs, kys, kzs)
    }

    pub fn aperiodic_linear_wave_numbers(&self) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Calculate linear wave number arrays.
        let Nx: usize = if self.aperiodic_x {
            2 * self.Nx - 1
        } else {
            self.Nx
        };
        let Ny: usize = if self.aperiodic_y {
            2 * self.Ny - 1
        } else {
            self.Ny
        };
        let Nz: usize = if self.aperiodic_z {
            2 * self.Nz - 1
        } else {
            self.Nz
        };

        let Lx: f32 = if self.aperiodic_x {
            2.0 * self.Lx
        } else {
            self.Lx
        };
        let Ly: f32 = if self.aperiodic_y {
            2.0 * self.Ly
        } else {
            self.Ly
        };
        let Lz: f32 = if self.aperiodic_z {
            2.0 * self.Lz
        } else {
            self.Lz
        };
        let kxs: Array1<f32> = fftfreq(Nx, Lx / (Nx as f32));
        let kys: Array1<f32> = fftfreq(Ny, Ly / (Ny as f32));
        let kzs: Array1<f32> = rfftfreq(Nz, Lz / (Nz as f32));
        (kxs, kys, kzs)
    }
    pub fn angular_wave_numbers(&self) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        // Calculate angular wave number arrays consistent with a physical box
        // where L = (N - 1) * dx.
        let kxs: Array1<f32> = 2.0 * PI * fftfreq(self.Nx, self.Lx / ((self.Nx - 1) as f32));
        let kys: Array1<f32> = 2.0 * PI * fftfreq(self.Ny, self.Ly / ((self.Ny - 1) as f32));
        let kzs: Array1<f32> = 2.0 * PI * rfftfreq(self.Nz, self.Lz / ((self.Nz - 1) as f32));
        (kxs, kys, kzs)
    }
}

#[derive(Clone)]
pub struct Stencil {
    pub p: StencilParams,
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

    fn decompose_stencil(
        &self,
    ) -> (
        Array3<Complex32>,
        Array3<Complex32>,
        Array3<Complex32>,
        Array3<Complex32>,
    ) {
        let Nx: usize = if self.p.aperiodic_x {
            2 * self.p.Nx - 1
        } else {
            self.p.Nx
        };
        let Ny: usize = if self.p.aperiodic_y {
            2 * self.p.Ny - 1
        } else {
            self.p.Ny
        };
        let Nz: usize = if self.p.aperiodic_z {
            2 * self.p.Nz - 1
        } else {
            self.p.Nz
        };

        let mut Ruu_f: Array3<Complex32> = Array3::zeros((Nx, Ny, Nz / 2 + 1));
        let mut Rvv_f: Array3<Complex32> = Array3::zeros((Nx, Ny, Nz / 2 + 1));
        let mut Rww_f: Array3<Complex32> = Array3::zeros((Nx, Ny, Nz / 2 + 1));
        let mut Ruw_f: Array3<Complex32> = Array3::zeros((Nx, Ny, Nz / 2 + 1));
        Zip::from(Ruu_f.outer_iter_mut())
            .and(Rvv_f.outer_iter_mut())
            .and(Rww_f.outer_iter_mut())
            .and(Ruw_f.outer_iter_mut())
            .and(self.stencil.outer_iter())
            .par_for_each(
                |mut Ruu_f_slice,
                 mut Rvv_f_slice,
                 mut Rww_f_slice,
                 mut Ruw_f_slice,
                 stencil_slice| {
                    Zip::from(Ruu_f_slice.outer_iter_mut())
                        .and(Rvv_f_slice.outer_iter_mut())
                        .and(Rww_f_slice.outer_iter_mut())
                        .and(Ruw_f_slice.outer_iter_mut())
                        .and(stencil_slice.outer_iter())
                        .par_for_each(
                            |mut Ruu_f_col,
                             mut Rvv_f_col,
                             mut Rww_f_col,
                             mut Ruw_f_col,
                             stencil_col| {
                                Zip::from(Ruu_f_col.outer_iter_mut())
                                    .and(Rvv_f_col.outer_iter_mut())
                                    .and(Rww_f_col.outer_iter_mut())
                                    .and(Ruw_f_col.outer_iter_mut())
                                    .and(stencil_col.outer_iter())
                                    .for_each(
                                        |mut Ruu_comp,
                                         mut Rvv_comp,
                                         mut Rww_comp,
                                         mut Ruw_comp,
                                         decomp| {
                                            let _tensor = decomp.dot(&decomp.t());
                                            Ruu_comp.assign(
                                                &_tensor
                                                    .slice(s![0, 0])
                                                    .mapv(|x| Complex32::new(x, 0.0)),
                                            );
                                            Rvv_comp.assign(
                                                &_tensor
                                                    .slice(s![1, 1])
                                                    .mapv(|x| Complex32::new(x, 0.0)),
                                            );
                                            Rww_comp.assign(
                                                &_tensor
                                                    .slice(s![2, 2])
                                                    .mapv(|x| Complex32::new(x, 0.0)),
                                            );
                                            Ruw_comp.assign(
                                                &_tensor
                                                    .slice(s![0, 2])
                                                    .mapv(|x| Complex32::new(x, 0.0)),
                                            );
                                        },
                                    )
                            },
                        )
                },
            );
        (Ruu_f, Rvv_f, Rww_f, Ruw_f)
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
        let (mut Ruu_f, mut Rvv_f, mut Rww_f, mut Ruw_f) = self.decompose_stencil();

        let Ruu: Array3<f32> = irfft3d(&mut Ruu_f);
        let Rvv: Array3<f32> = irfft3d(&mut Rvv_f);
        let Rww: Array3<f32> = irfft3d(&mut Rww_f);
        // let Ruw: Array3<f32> = irfft3d(&mut Ruw_f);
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
        let (mut Ruu_f, mut Rvv_f, mut Rww_f, mut Ruw_f) = self.decompose_stencil();
        let Ruu: Array3<f32> = irfft3d(&mut Ruu_f);
        drop(Ruu_f);
        let Rvv: Array3<f32> = irfft3d(&mut Rvv_f);
        drop(Rvv_f);
        let Rww: Array3<f32> = irfft3d(&mut Rww_f);
        drop(Rww_f);
        let Ruw: Array3<f32> = irfft3d(&mut Ruw_f);
        drop(Ruw_f);
        (
            Ruu.mapv(|x| x / Ruu[[0, 0, 0]]),
            Rvv.mapv(|x| x / Rvv[[0, 0, 0]]),
            Rww.mapv(|x| x / Rww[[0, 0, 0]]),
            Ruw.mapv(|x| x / (Ruu[[0, 0, 0]] * Rww[[0, 0, 0]]).sqrt()),
        )
    }

    pub fn spectral_impulses(
        &self,
    ) -> (
        SpectralImpulseResponse,
        SpectralImpulseResponse,
        SpectralImpulseResponse,
        SpectralImpulseResponse,
    ) {
        let (kxs, kys, kzs) = self.p.aperiodic_linear_wave_numbers();
        let (Ruu_f, Rvv_f, Rww_f, Ruw_f) = self.spectral_component_grids();
        let impulse_u = SpectralImpulseResponse::new(Ruu_f, kxs.clone(), kys.clone(), kzs.clone());
        let impulse_v = SpectralImpulseResponse::new(Rvv_f, kxs.clone(), kys.clone(), kzs.clone());
        let impulse_w = SpectralImpulseResponse::new(Rww_f, kxs.clone(), kys.clone(), kzs.clone());
        let impulse_uw = SpectralImpulseResponse::new(Ruw_f, kxs, kys, kzs);

        (impulse_u, impulse_v, impulse_w, impulse_uw)
    }
}

pub fn stencilate_par(p: StencilParams) -> Array5<f32> {
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx - 1 } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny - 1 } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz - 1 } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
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
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx - 1 } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny - 1 } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz - 1 } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
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
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx - 1 } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny - 1 } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz - 1 } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);

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

    let U: Array3<f32> = irfft3d_par(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = irfft3d_par(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = irfft3d_par(&mut W_f);
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
        freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
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
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx - 1 } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny - 1 } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz - 1 } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let mut stencil: Array5<f32> = Array5::zeros((Nx, Ny, Nz / 2 + 1, 3, 3));
    let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
        freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
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
    let Nx: usize = if p.aperiodic_x { 2 * p.Nx - 1 } else { p.Nx };
    let Ny: usize = if p.aperiodic_y { 2 * p.Ny - 1 } else { p.Ny };
    let Nz: usize = if p.aperiodic_z { 2 * p.Nz - 1 } else { p.Nz };

    let Lx: f32 = if p.aperiodic_x { 2.0 * p.Lx } else { p.Lx };
    let Ly: f32 = if p.aperiodic_y { 2.0 * p.Ly } else { p.Ly };
    let Lz: f32 = if p.aperiodic_z { 2.0 * p.Lz } else { p.Lz };

    let KVolScaleFac: Complex32 = Complex::new(
        2.0 * (Nx * Ny * (Nz / 2 + 1)) as f32 * ((8.0 * ae * PI.powi(3)) / (Lx * Ly * Lz)).sqrt(),
        0.0,
    );
    let random: Array4<Complex32> = complex_random_gaussian(seed, Nx, Ny, Nz / 2 + 1);

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

    let U: Array3<f32> = irfft3d(&mut U_f);
    drop(U_f);
    let V: Array3<f32> = irfft3d(&mut V_f);
    drop(V_f);
    let W: Array3<f32> = irfft3d(&mut W_f);
    drop(W_f);
    (U, V, W)
}
