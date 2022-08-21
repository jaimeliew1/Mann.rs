use ndarray::{Array1, Array3, Array5};
use numpy::{
    c64, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyReadonlyArray1, PyReadonlyArray5,
    ToPyArray,
};
use pyo3::prelude::*;

use crate::{
    partial_turbulate, partial_turbulate_par, stencilate_sinc, stencilate_sinc_par, turbulate,
    turbulate_par, turbulate_unit, Tensors::*, Utilities::complex_random_gaussian,
    Utilities::freq_components,
};

#[pyclass]
struct RustStencil {
    L: f64,
    gamma: f64,
    Lx: f64,
    Ly: f64,
    Lz: f64,
    Nx: usize,
    Ny: usize,
    Nz: usize,
    _stencil: Array5<f64>,
}

#[pymethods]
impl RustStencil {
    #[new]
    fn __new__(
        L: f64,
        gamma: f64,
        Lx: f64,
        Ly: f64,
        Lz: f64,
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
        ae: f64,
        seed: u64,
        parallel: bool,
    ) -> (&'py PyArray3<f64>, &'py PyArray3<f64>, &'py PyArray3<f64>) {
        let (U_f, V_f, W_f): (Array3<f64>, Array3<f64>, Array3<f64>) = match parallel {
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
        ae: f64,
        seed: u64,
        parallel: bool,
    ) -> (&'py PyArray3<c64>, &'py PyArray3<c64>, &'py PyArray3<c64>) {
        let (U_f, V_f, W_f): (Array3<c64>, Array3<c64>, Array3<c64>) = match parallel {
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

#[pymodule]
pub fn RustMann(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_class::<RustStencil>()?;

    #[pyfn(module)]
    fn freq_components_f64<'py>(
        py: Python<'py>,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f64,
        Ly: f64,
        Lz: f64,
    ) -> (&'py PyArray1<f64>, &'py PyArray1<f64>, &'py PyArray1<f64>) {
        let (f_x, f_y, f_z): (Array1<f64>, Array1<f64>, Array1<f64>) =
            freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
        (f_x.to_pyarray(py), f_y.to_pyarray(py), f_z.to_pyarray(py))
    }

    #[pyfn(module)]
    fn isotropic_f64<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f64>,
        ae: f64,
        L: f64,
    ) -> &'py PyArray2<f64> {
        Isotropic::from_params(ae, L)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }

    #[pyfn(module)]
    fn isotropic_sqrt_f64<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f64>,
        ae: f64,
        L: f64,
    ) -> &'py PyArray2<f64> {
        Isotropic::from_params(ae, L)
            .decomp(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    #[pyfn(module)]
    fn sheared_f64<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f64>,
        ae: f64,
        L: f64,
        gamma: f64,
    ) -> &'py PyArray2<f64> {
        Sheared::from_params(ae, L, gamma)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }

    #[pyfn(module)]
    fn sheared_sqrt_f64<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f64>,
        ae: f64,
        L: f64,
        gamma: f64,
    ) -> &'py PyArray2<f64> {
        Sheared::from_params(ae, L, gamma)
            .decomp(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    Ok(())
}
