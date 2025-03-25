use crate::{
    correlation_grids, forgetful_turbulate, forgetful_turbulate_par, partial_forgetful_turbulate,
    partial_forgetful_turbulate_par, partial_turbulate, partial_turbulate_par,
    spectral_component_grids, stencilate_sinc, stencilate_sinc_par, turbulate, turbulate_par,
    Constraint, Tensors::*, Utilities, Utilities::fftfreq, Utilities::freq_components,
    Utilities::rfftfreq,
};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array3, Array5, Zip};
use numpy::{
    Complex32, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
    ToPyArray,
};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

#[pyclass]
struct RustStencil {
    L: f32,
    gamma: f32,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    _stencil: Array5<f32>,
}

#[pyclass]
struct RustForgetfulStencil {
    L: f32,
    gamma: f32,
    Lx: f32,
    Ly: f32,
    Lz: f32,
    Nx: usize,
    Ny: usize,
    Nz: usize,
}

#[pymethods]
impl RustStencil {
    #[new]
    fn __new__(
        L: f32,
        gamma: f32,
        Lx: f32,
        Ly: f32,
        Lz: f32,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        parallel: bool,
    ) -> Self {
        match parallel {
            true => RustStencil {
                L: L,
                gamma: gamma,
                Lx: Lx,
                Ly: Ly,
                Lz: Lz,
                Nx: Nx,
                Ny: Ny,
                Nz: Nz,
                _stencil: stencilate_sinc_par(L, gamma, Lx, Ly, Lz, Nx, Ny, Nz),
            },
            false => RustStencil {
                L: L,
                gamma: gamma,
                Lx: Lx,
                Ly: Ly,
                Lz: Lz,
                Nx: Nx,
                Ny: Ny,
                Nz: Nz,
                _stencil: stencilate_sinc(L, gamma, Lx, Ly, Lz, Nx, Ny, Nz),
            },
        }
    }

    fn turbulence<'py>(
        &self,
        py: Python<'py>,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (&'py PyArray3<f32>, &'py PyArray3<f32>, &'py PyArray3<f32>) {
        let (U_f, V_f, W_f): (Array3<f32>, Array3<f32>, Array3<f32>) = match parallel {
            true => turbulate_par(
                &self._stencil.view(),
                ae,
                seed,
                self.Nx,
                self.Ny,
                self.Nz,
                self.Lx,
                self.Ly,
                self.Lz,
            ),
            false => turbulate(
                &self._stencil.view(),
                ae,
                seed,
                self.Nx,
                self.Ny,
                self.Nz,
                self.Lx,
                self.Ly,
                self.Lz,
            ),
        };
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    fn partial_turbulence<'py>(
        &self,
        py: Python<'py>,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (
        &'py PyArray3<Complex32>,
        &'py PyArray3<Complex32>,
        &'py PyArray3<Complex32>,
    ) {
        let (U_f, V_f, W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
            match parallel {
                true => partial_turbulate_par(
                    &self._stencil.view(),
                    ae,
                    seed,
                    self.Nx,
                    self.Ny,
                    self.Nz,
                    self.Lx,
                    self.Ly,
                    self.Lz,
                ),
                false => partial_turbulate(
                    &self._stencil.view(),
                    ae,
                    seed,
                    self.Nx,
                    self.Ny,
                    self.Nz,
                    self.Lx,
                    self.Ly,
                    self.Lz,
                ),
            };
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    fn spectral_component_grids<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        &'py PyArray3<f32>,
        &'py PyArray3<f32>,
        &'py PyArray3<f32>,
        &'py PyArray3<f32>,
    ) {
        let (Ruu_f, Rvv_f, Rww_f, Ruw_f) = spectral_component_grids(&self._stencil.view());

        (
            Ruu_f.to_pyarray(py),
            Rvv_f.to_pyarray(py),
            Rww_f.to_pyarray(py),
            Ruw_f.to_pyarray(py),
        )
    }

    fn correlation_grids<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        &'py PyArray3<f32>,
        &'py PyArray3<f32>,
        &'py PyArray3<f32>,
        &'py PyArray3<f32>,
    ) {
        let (Ruu, Rvv, Rww, Ruw) = correlation_grids(&self._stencil.view());

        (
            Ruu.to_pyarray(py),
            Rvv.to_pyarray(py),
            Rww.to_pyarray(py),
            Ruw.to_pyarray(py),
        )
    }

    fn constrain<'py>(
        &self,
        py: Python<'py>,
        constraints: PyReadonlyArray2<'py, f32>,
        CConstU: PyReadonlyArray1<'py, f32>,
        CConstV: PyReadonlyArray1<'py, f32>,
        CConstW: PyReadonlyArray1<'py, f32>,
    ) -> (&'py PyArray3<f32>, &'py PyArray3<f32>, &'py PyArray3<f32>) {
        let CConstU: Array1<Complex32> = CConstU.to_owned_array().mapv(|x| Complex32::new(x, 0.0));
        let CConstV: Array1<Complex32> = CConstV.to_owned_array().mapv(|x| Complex32::new(x, 0.0));
        let CConstW: Array1<Complex32> = CConstW.to_owned_array().mapv(|x| Complex32::new(x, 0.0));

        // Calculate normalized spectral component grids
            // Calculate normalized spectral component grids
            let (Ruu_f, Rvv_f, Rww_f, Ruw_f) = spectral_component_grids(&self._stencil.view());
            let Ruu_f: Array3<Complex32> = Ruu_f.mapv(|x| Complex32::new(x, 0.0));
            let Rvv_f: Array3<Complex32> = Rvv_f.mapv(|x| Complex32::new(x, 0.0));
            let Rww_f: Array3<Complex32> = Rww_f.mapv(|x| Complex32::new(x, 0.0));
            let Ruw_f: Array3<Complex32> = Ruw_f.mapv(|x| Complex32::new(x, 0.0));
        
        // Calculate 3d meshgrid of linear wave numbers.
        let kxs: Array1<f32> = fftfreq(self.Nx, self.Lx / (self.Nx as f32));
        let kys: Array1<f32> = fftfreq(self.Ny, self.Ly / (self.Ny as f32));
        let kzs: Array1<f32> = rfftfreq(self.Nz, self.Lz / (self.Nz as f32));

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

        // let mut U_f = Array3::<Complex32>::zeros((nx, ny, nz));
        // let mut V_f = Array3::<Complex32>::zeros((nx, ny, nz));
        // let mut W_f = Array3::<Complex32>::zeros((nx, ny, nz));

        // for (i, c) in constraints.as_array().outer_iter().enumerate() {
        //     let phase: Array3<Complex32> = (Complex32::new(0.0, -2.0 * std::f32::consts::PI)
        //         * (&kx_mesh * c[0] + &ky_mesh * c[1] + &kz_mesh * c[2]))
        //         .mapv(|x| x.exp());

        //     U_f = &U_f
        //     + Complex32::new(0.5, 0.0) * &phase * (&Ruu_f * CConstU[i] + &Ruw_f * CConstW[i]);
        //     V_f = &V_f + Complex32::new(0.5, 0.0) * &phase * (&Rvv_f * CConstV[i]);
        //     W_f = &W_f
        //     + Complex32::new(0.5, 0.0) * &phase * (&Ruw_f * CConstU[i] + &Rww_f * CConstW[i]);
        // }
        // let U: Array3<f32> = Utilities::irfft3d(&mut U_f);
        // let V: Array3<f32> = Utilities::irfft3d(&mut V_f);
        // let W: Array3<f32> = Utilities::irfft3d(&mut W_f);

        let U_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
        let V_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
        let W_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
        constraints
            .as_array()
            .outer_iter()
            .into_par_iter()
            .enumerate()
            .for_each(|(i, c)| {
                let phase: Array3<Complex32> = (Complex32::new(0.0, -2.0 * std::f32::consts::PI)
                    * (&kx_mesh * c[0] + &ky_mesh * c[1] + &kz_mesh * c[2]))
                    .mapv(|x| x.exp());

                let to_add =
                    Complex32::new(1.0, 0.0) * &phase * (&Ruu_f * CConstU[i] + &Ruw_f * CConstW[i]);
                {
                    let mut U_f = U_f.lock().unwrap();
                    Zip::from(&mut *U_f).and(&to_add).apply(|a, &b| *a += b);
                }
                let to_add = Complex32::new(1.0, 0.0) * &phase * (&Rvv_f * CConstV[i]);
                {
                    let mut V_f = V_f.lock().unwrap();
                    Zip::from(&mut *V_f).and(&to_add).apply(|a, &b| *a += b);
                }
                let to_add =
                    Complex32::new(1.0, 0.0) * &phase * (&Ruw_f * CConstU[i] + &Rww_f * CConstW[i]);
                {
                    let mut W_f = W_f.lock().unwrap();
                    Zip::from(&mut *W_f).and(&to_add).apply(|a, &b| *a += b);
                }
            });
        let U: Array3<f32> = Utilities::irfft3d(&mut U_f.lock().unwrap());
        let V: Array3<f32> = Utilities::irfft3d(&mut V_f.lock().unwrap());
        let W: Array3<f32> = Utilities::irfft3d(&mut W_f.lock().unwrap());

        (U.to_pyarray(py), V.to_pyarray(py), W.to_pyarray(py))
    }
}

#[pymethods]
impl RustForgetfulStencil {
    #[new]
    fn __new__(
        L: f32,
        gamma: f32,
        Lx: f32,
        Ly: f32,
        Lz: f32,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> Self {
        RustForgetfulStencil {
            L: L,
            gamma: gamma,
            Lx: Lx,
            Ly: Ly,
            Lz: Lz,
            Nx: Nx,
            Ny: Ny,
            Nz: Nz,
        }
    }

    fn turbulence<'py>(
        &self,
        py: Python<'py>,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (&'py PyArray3<f32>, &'py PyArray3<f32>, &'py PyArray3<f32>) {
        let (U_f, V_f, W_f): (Array3<f32>, Array3<f32>, Array3<f32>) = match parallel {
            true => forgetful_turbulate_par(
                ae, seed, self.Nx, self.Ny, self.Nz, self.Lx, self.Ly, self.Lz, self.L, self.gamma,
            ),
            false => forgetful_turbulate(
                ae, seed, self.Nx, self.Ny, self.Nz, self.Lx, self.Ly, self.Lz, self.L, self.gamma,
            ),
        };
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    fn partial_turbulence<'py>(
        &self,
        py: Python<'py>,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (
        &'py PyArray3<Complex32>,
        &'py PyArray3<Complex32>,
        &'py PyArray3<Complex32>,
    ) {
        let (U_f, V_f, W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
            match parallel {
                true => partial_forgetful_turbulate_par(
                    ae, seed, self.Nx, self.Ny, self.Nz, self.Lx, self.Ly, self.Lz, self.L,
                    self.gamma,
                ),
                false => partial_forgetful_turbulate(
                    ae, seed, self.Nx, self.Ny, self.Nz, self.Lx, self.Ly, self.Lz, self.L,
                    self.gamma,
                ),
            };
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }
}

#[pymodule]
pub fn mannrs(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<RustStencil>()?;
    module.add_class::<RustForgetfulStencil>()?;

    #[pyfn(module)]
    fn freq_components_f32<'py>(
        py: Python<'py>,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f32,
        Ly: f32,
        Lz: f32,
    ) -> (&'py PyArray1<f32>, &'py PyArray1<f32>, &'py PyArray1<f32>) {
        let (f_x, f_y, f_z): (Array1<f32>, Array1<f32>, Array1<f32>) =
            freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
        (f_x.to_pyarray(py), f_y.to_pyarray(py), f_z.to_pyarray(py))
    }

    #[pyfn(module)]
    fn isotropic_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
    ) -> &'py PyArray2<f32> {
        Isotropic::from_params(ae, L)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }

    #[pyfn(module)]
    fn isotropic_sqrt_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
    ) -> &'py PyArray2<f32> {
        Isotropic::from_params(ae, L)
            .decomp(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    #[pyfn(module)]
    fn sheared_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
        gamma: f32,
    ) -> &'py PyArray2<f32> {
        Sheared::from_params(ae, L, gamma)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }

    #[pyfn(module)]
    fn sheared_sqrt_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
        gamma: f32,
    ) -> &'py PyArray2<f32> {
        Sheared::from_params(ae, L, gamma)
            .decomp(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    #[pyfn(module)]
    fn sheared_sinc_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
        gamma: f32,
        Ly: f32,
        Lz: f32,
        tol: f32,
        min_depth: u64,
    ) -> &'py PyArray2<f32> {
        ShearedSinc::from_params(ae, L, gamma, Ly, Lz, tol, min_depth)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    #[pyfn(module)]
    fn sheared_sinc_info_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
        gamma: f32,
        Ly: f32,
        Lz: f32,
        tol: f32,
        min_depth: u64,
    ) -> (&'py PyArray2<f32>, u64) {
        let (out, neval) = ShearedSinc::from_params(ae, L, gamma, Ly, Lz, tol, min_depth)
            .tensor_info(&K.as_slice().unwrap());

        (out.to_pyarray(py), neval)
    }

    #[pyfn(module)]
    fn sheared_sinc_sqrt_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
        gamma: f32,
        Ly: f32,
        Lz: f32,
        tol: f32,
        min_depth: u64,
    ) -> &'py PyArray2<f32> {
        ShearedSinc::from_params(ae, L, gamma, Ly, Lz, tol, min_depth)
            .decomp(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    Ok(())
}
