use ndarray::{Array1, Array3, Array5};
use numpy::{Complex32, PyArray1, PyArray2, PyArray3, PyArrayMethods, PyReadonlyArray1, ToPyArray};
use pyo3::prelude::*;

use crate::{
    forgetful_turbulate, forgetful_turbulate_par, partial_forgetful_turbulate,
    partial_forgetful_turbulate_par, partial_turbulate, partial_turbulate_par,
    spectra::spectra, stencilate_sinc, stencilate_sinc_par, turbulate,
    turbulate_par, Tensors::*, Utilities::freq_components,
};

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

    #[pyfn(module)]
    fn mann_spectra<'py>(
        py: Python<'py>,
        kx: PyReadonlyArray1<f32>,
        ae: f32,
        l: f32,
        gamma: f32,
    ) -> (
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
        &'py PyArray1<f32>,
    ) {
        let kx = kx.to_owned_array();
        let (uu, vv, ww, uw) = spectra::mann_spectra(&kx, ae, l, gamma);
        (
            uu.to_pyarray(py),
            vv.to_pyarray(py),
            ww.to_pyarray(py),
            uw.to_pyarray(py),
        )
    }
    Ok(())
}
