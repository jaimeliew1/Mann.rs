use ndarray::{Array3, Array1};
use numpy::{
    c64, PyArray1, PyArray2, PyArray3, PyArray4, PyArray5, PyReadonlyArray1, PyReadonlyArray5, ToPyArray,
};
use pyo3::prelude::*;

use crate::{
    partial_turbulate, stencilate, turbulate, Utilities::complex_random_gaussian,
    Utilities::freq_components, Tensors, stencilate_sinc,
};
#[pymodule]
fn rustmann(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    #[pyfn(m)]
    fn sheared_tensor_f64<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f64>,
        ae: f64,
        L: f64,
        gamma: f64,
    ) -> &'py PyArray2<f64> {
        Tensors::sheared_tensor(&K.as_array(), ae, L, gamma).to_pyarray(py)
    }

    #[pyfn(m)]
    fn sqrt_sheared_tensor_f64<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f64>,
        ae: f64,
        L: f64,
        gamma: f64,
    ) -> &'py PyArray2<f64> {
        Tensors::sqrt_sheared_tensor(&K.as_array(), ae, L, gamma).to_pyarray(py)
    }

    #[pyfn(m)]
    fn sheared_tensor_sinc_f64<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f64>,
        ae: f64,
        L: f64,
        gamma: f64,
        Lx: f64,
        Ly: f64,
        Lz: f64,
        Ny: usize,
        Nz: usize,
    ) -> &'py PyArray2<f64> {
        Tensors::sheared_tensor_sinc(&K.as_array(), ae, L, gamma, Lx, Ly, Lz, Ny, Nz).to_pyarray(py)
    }

    #[pyfn(m)]
    fn stencilate_f64<'py>(
        py: Python<'py>,
        ae: f64,
        L: f64,
        gamma: f64,
        Lx: f64,
        Ly: f64,
        Lz: f64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> &'py PyArray5<f64> {
        stencilate(ae, L, gamma, Lx, Ly, Lz, Nx, Ny, Nz).to_pyarray(py)
    }

    #[pyfn(m)]
    fn stencilate_sinc_f64<'py>(
        py: Python<'py>,
        ae: f64,
        L: f64,
        gamma: f64,
        Lx: f64,
        Ly: f64,
        Lz: f64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> &'py PyArray5<f64> {
        stencilate_sinc(ae, L, gamma, Lx, Ly, Lz, Nx, Ny, Nz).to_pyarray(py)
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
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f64,
        Ly: f64,
        Lz: f64,
    ) -> (&'py PyArray3<c64>, &'py PyArray3<c64>, &'py PyArray3<c64>) {
        let (U_f, V_f, W_f): (Array3<c64>, Array3<c64>, Array3<c64>) =
            partial_turbulate(&stencil.as_array(), seed, Nx, Ny, Nz, Lx, Ly, Lz);
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    #[pyfn(m)]
    fn turbulate_f64<'py>(
        py: Python<'py>,
        stencil: PyReadonlyArray5<f64>,
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f64,
        Ly: f64,
        Lz: f64,
    ) -> (&'py PyArray3<f64>, &'py PyArray3<f64>, &'py PyArray3<f64>) {
        let (U_f, V_f, W_f): (Array3<f64>, Array3<f64>, Array3<f64>) =
            turbulate(&stencil.as_array(), seed, Nx, Ny, Nz, Lx, Ly, Lz);
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
