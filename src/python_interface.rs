use ndarray::{Array1, Array3};
use numpy::{
    c64, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyReadonlyArray1, PyReadonlyArray5,
    ToPyArray,
};
use pyo3::prelude::*;

use crate::{
    partial_turbulate, stencilate, stencilate_sinc, turbulate, turbulate_unit,
    Tensors::*, Utilities::complex_random_gaussian, Utilities::freq_components, stencilate_sinc_par, turbulate_par,
};
#[pymodule]
fn RustMann(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
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

    #[pyfn(m)]
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
    #[pyfn(m)]
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

    #[pyfn(m)]
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

    #[pyfn(m)]
    fn sheared_sinc_f64<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f64>,
        ae: f64,
        L: f64,
        gamma: f64,
        Ly: f64,
        Lz: f64,
    ) -> &'py PyArray2<f64> {
        ShearedSinc::from_params(ae, L, gamma, Ly, Lz)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }

    #[pyfn(m)]
    fn stencilate_f64<'py>(
        py: Python<'py>,
        L: f64,
        gamma: f64,
        Lx: f64,
        Ly: f64,
        Lz: f64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> &'py PyArray5<f64> {
        stencilate(L, gamma, Lx, Ly, Lz, Nx, Ny, Nz).to_pyarray(py)
    }

    #[pyfn(m)]
    fn stencilate_sinc_f64<'py>(
        py: Python<'py>,
        L: f64,
        gamma: f64,
        Lx: f64,
        Ly: f64,
        Lz: f64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> &'py PyArray5<f64> {
        stencilate_sinc(L, gamma, Lx, Ly, Lz, Nx, Ny, Nz).to_pyarray(py)
    }
    #[pyfn(m)]
    fn stencilate_sinc_par_f64<'py>(
        py: Python<'py>,
        L: f64,
        gamma: f64,
        Lx: f64,
        Ly: f64,
        Lz: f64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> &'py PyArray5<f64> {
        stencilate_sinc_par(L, gamma, Lx, Ly, Lz, Nx, Ny, Nz).to_pyarray(py)
    }

    #[pyfn(m)]
    fn complex_random_gaussian_f64<'py>(
        py: Python<'py>,
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> &'py PyArray4<c64> {
        complex_random_gaussian(seed, Nx, Ny, Nz).to_pyarray(py)
    }

    #[pyfn(m)]
    fn partial_turbulate_f64<'py>(
        py: Python<'py>,
        stencil: PyReadonlyArray5<f64>,
        ae: f64,
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f64,
        Ly: f64,
        Lz: f64,
    ) -> (&'py PyArray3<c64>, &'py PyArray3<c64>, &'py PyArray3<c64>) {
        let (U_f, V_f, W_f): (Array3<c64>, Array3<c64>, Array3<c64>) =
            partial_turbulate(&stencil.as_array(), ae, seed, Nx, Ny, Nz, Lx, Ly, Lz);
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    #[pyfn(m)]
    fn turbulate_f64<'py>(
        py: Python<'py>,
        stencil: PyReadonlyArray5<f64>,
        ae: f64,
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f64,
        Ly: f64,
        Lz: f64,
    ) -> (&'py PyArray3<f64>, &'py PyArray3<f64>, &'py PyArray3<f64>) {
        let (U_f, V_f, W_f): (Array3<f64>, Array3<f64>, Array3<f64>) =
            turbulate(&stencil.as_array(), ae, seed, Nx, Ny, Nz, Lx, Ly, Lz);
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    #[pyfn(m)]
    fn turbulate_par_f64<'py>(
        py: Python<'py>,
        stencil: PyReadonlyArray5<f64>,
        ae: f64,
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f64,
        Ly: f64,
        Lz: f64,
    ) -> (&'py PyArray3<f64>, &'py PyArray3<f64>, &'py PyArray3<f64>) {
        let (U_f, V_f, W_f): (Array3<f64>, Array3<f64>, Array3<f64>) =
            turbulate_par(&stencil.as_array(), ae, seed, Nx, Ny, Nz, Lx, Ly, Lz);
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    #[pyfn(m)]
    fn turbulate_unit_f64<'py>(
        py: Python<'py>,
        stencil: PyReadonlyArray5<f64>,
        ae: f64,
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f64,
        Ly: f64,
        Lz: f64,
    ) -> (&'py PyArray3<f64>, &'py PyArray3<f64>, &'py PyArray3<f64>) {
        let (U_f, V_f, W_f): (Array3<f64>, Array3<f64>, Array3<f64>) =
            turbulate_unit(&stencil.as_array(), ae, seed, Nx, Ny, Nz, Lx, Ly, Lz);
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    #[pyfn(m)]
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
    Ok(())
}
