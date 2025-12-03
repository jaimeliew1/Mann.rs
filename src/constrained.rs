use crate::spectral_impulse::{
    spectral_superposition_par, spectral_superposition_ser, CompressedSpectralImpulseResponse,
};
use crate::unconstrained::{Stencil, StencilParams};
use crate::utilities::{distance_matrix, irfft3d};

use ninterp::prelude::*;

use itertools::izip;
use std::iter::FromIterator;
use std::mem::drop;

use faer::prelude::*;
use faer::sparse::linalg::solvers::Lu;
use faer::sparse::linalg::solvers::Llt;
use faer::sparse::linalg::solvers::Qr;
use faer::sparse::*;
use faer::Side;

use numpy::Complex32;

use ndarray::prelude::*;

#[derive(Debug)]
pub struct Constraint {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub u: f32,
}

pub enum MatrixSolver {
    Lu,
    Llt,
    Qr
}
pub enum FactorizedMatrix {
    Lu(Lu<usize, f32>),
    Llt(Llt<usize, f32>),
    Qr(Qr<usize, f32>),
}

impl FactorizedMatrix {
    pub fn cholesky(A: SparseColMat<usize, f32>) -> Self {
        let llt = A.sp_cholesky(Side::Lower).unwrap();
        FactorizedMatrix::Llt(llt)
    }

    pub fn LU(A: SparseColMat<usize, f32>) -> Self {
        let lu = A.sp_lu().unwrap();
        FactorizedMatrix::Lu(lu)
    }

    pub fn QR(A: SparseColMat<usize, f32>) -> Self {
        let qr = A.sp_qr().unwrap();
        FactorizedMatrix::Qr(qr)
    }

    pub fn solve(&self, b: &faer::col::Col<f32>) -> faer::col::Col<f32> {
        match self {
            FactorizedMatrix::Lu(lu) => lu.solve(b),
            FactorizedMatrix::Llt(llt) => llt.solve(b),
            FactorizedMatrix::Qr(qr) => qr.solve_lstsq(b),
        }
    }
}

pub struct ConstrainedStencil {
    pub stencil: Stencil,
    pub constraints: Vec<Constraint>,
    A_factorized: FactorizedMatrix,
    pub impulse_u: CompressedSpectralImpulseResponse,
    pub sparsity: f64,
    pub spectral_compression: f64,
}

impl ConstrainedStencil {
    pub fn new(
        stencil: Stencil,
        constraints: Vec<Constraint>,
        corr_thres: f32,
        spectral_compression_target: f64,
        solver: MatrixSolver,
    ) -> Self {
        let p: &StencilParams = &stencil.p;

        let (Ruu, _Rvv, _Rww, _Ruw): (Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>) =
            stencil.correlation_grids();

        // clip correlation data
        let Ruu: Array3<f32> = Ruu.slice(s![..p.Nx, ..p.Ny, ..p.Nz]).to_owned();

        // Calculating distance matrices
        let x_dist = distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.x)));
        let y_dist = distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.y)));
        let z_dist = distance_matrix(&Array1::from_iter(constraints.iter().map(|c| c.z)));

        // building interpolator
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

        //Calculating correlation matrix

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

        // Apply hard threshold and converting to sparse matrix.
        let triplets: Vec<Triplet<usize, usize, f32>> = UUcorr
            .indexed_iter()
            // .par_bridge()
            .filter(|(_, &v)| v > corr_thres)
            .map(|((i, j), v)| Triplet::new(i, j, *v))
            .collect();
        drop(UUcorr);

        let sparsity: f64 = 1.0 - (triplets.len() as f64) / (constraints.len() as f64).powi(2);

        // create sparse matrix
        let A = SparseColMat::<usize, f32>::try_new_from_triplets(
            constraints.len(),
            constraints.len(),
            &triplets,
        )
        .unwrap();

        //print sum of A to monitor stability
        let diag_sum: f32 = A.triplet_iter()
            .map(|x| x.val)
            .sum();
        println!("Sum of A: {}", diag_sum);

        let llt: FactorizedMatrix = match solver {
            MatrixSolver::Lu => FactorizedMatrix::LU(A),
            MatrixSolver::Llt => FactorizedMatrix::cholesky(A),
            MatrixSolver::Qr => FactorizedMatrix::QR(A),
        };


        // Calculate compressed impulse responses
        let (impulse_u, _impulse_v, _impulse_w, _impulse_uw) = stencil.spectral_impulses();
        let impulse_thres = impulse_u.get_thres_from_compression_ratio(spectral_compression_target);
        let compression_indices = impulse_u.get_compression_indices(impulse_thres);
        // Note: use the CompressionIndices.combine_all method to find the max envelope of multiple compressed impulses when needed.
        let spectral_compression = compression_indices.total_compression_ratio();
        let compressed_impulse_u = impulse_u.compress(compression_indices);

        ConstrainedStencil {
            stencil: stencil,
            constraints: constraints,
            A_factorized: llt,
            impulse_u: compressed_impulse_u,
            sparsity: sparsity,
            spectral_compression: spectral_compression,
        }
    }

    pub fn turbulate(
        &self,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (Array3<f32>, Array3<f32>, Array3<f32>) {
        let (U, V, W) = self.stencil.turbulate(ae, seed, parallel);

        let (x, y, z) = self.stencil.p.get_axes();
        // make U interpolator
        let interp_uu = Interp3DOwned::new(
            x.clone(),
            y.clone(),
            z.clone(),
            U.clone(),
            strategy::Linear,
            Extrapolate::Error,
        )
        .unwrap();
        // interpolate contemporaneous wind speeds

        let U_contemp: Vec<f32> = self
            .constraints
            .iter()
            .map(|c| interp_uu.interpolate(&[c.x, c.y, c.z]).unwrap())
            .collect();
        //construct b matrix
        let b = faer::col::Col::from_iter(
            U_contemp
                .iter()
                .zip(&self.constraints)
                .map(|(&u, c)| c.u - u),
        );

        //solve linear system
        let Uweight: Vec<f32> = self.A_factorized.solve(&b).iter().map(|&x| x).collect();
        

        // print the sum of Uweight to monitor stability
        println!("Sum of Uweight: {}", Uweight.iter().sum::<f32>());
        //perform spectral superposition

        // Calculate normalized spectral component grids

        let U_f: CompressedSpectralImpulseResponse;
        if !parallel {
            //superimpose in serial
            U_f = spectral_superposition_ser(&self.constraints, &self.impulse_u, Uweight);
        } else {
            //superimpose in parallel
            U_f = spectral_superposition_par(&self.constraints, &self.impulse_u, &Uweight);
            // let V_f: Array3<Complex32>;
            // let W_f: Array3<Complex32>;
        }

        let mut U_f_exp: Array3<Complex32> = U_f.zero_pad_and_unroll_impulse();
        let mut V_f_exp = Array3::<Complex32>::zeros(U_f_exp.dim());
        let mut W_f_exp = Array3::<Complex32>::zeros(U_f_exp.dim());

        // inverse 3d fourier transform
        let output_slice = s![
            ..self.stencil.p.Nx,
            ..self.stencil.p.Ny,
            ..self.stencil.p.Nz
        ];
        let U: Array3<f32> = irfft3d(&mut U_f_exp).slice(output_slice).to_owned() + U;
        let V: Array3<f32> = irfft3d(&mut V_f_exp).slice(output_slice).to_owned() + V;
        let W: Array3<f32> = irfft3d(&mut W_f_exp).slice(output_slice).to_owned() + W;

        (U, V, W)
    }
}
