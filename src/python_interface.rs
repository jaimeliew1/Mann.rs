use crate::{
    forgetful_turbulate, forgetful_turbulate_par, partial_forgetful_turbulate,
    partial_forgetful_turbulate_par, Tensors::*, Utilities, Utilities::freq_components,
};
use crate::{ConstrainedStencil, Constraint, Stencil};
use ndarray::{Array1, Array3};
use numpy::{
    Complex32, PyArray1, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2, ToPyArray,
};
use pyo3::prelude::*;

#[pyclass]
struct RustStencil {
    stencil: Stencil,
}

#[pyclass]
struct RustConstrainedStencil {
    stencil: ConstrainedStencil,
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
    sinc_thres: f32,
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
        aperiodic_x: bool,
        aperiodic_y: bool,
        aperiodic_z: bool,
        parallel: bool,
        sinc_thres: f32,
    ) -> Self {
        RustStencil {
            stencil: Stencil::from_params(
                L,
                gamma,
                Lx,
                Ly,
                Lz,
                Nx,
                Ny,
                Nz,
                aperiodic_x,
                aperiodic_y,
                aperiodic_z,
                sinc_thres,
                parallel,
            ),
        }
    }

    fn turbulence<'py>(
        &self,
        py: Python<'py>,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
    ) {
        let (U_f, V_f, W_f): (Array3<f32>, Array3<f32>, Array3<f32>) =
            self.stencil.turbulate(ae, seed, parallel);

        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    fn partial_turbulence<'py>(
        &self,
        py: Python<'py>,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (
        Bound<'py, PyArray3<Complex32>>,
        Bound<'py, PyArray3<Complex32>>,
        Bound<'py, PyArray3<Complex32>>,
    ) {
        let (U_f, V_f, W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
            self.stencil.partial_turbulate(ae, seed, parallel);

        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }

    fn spectral_component_grids<'py>(
        &self,
        py: Python<'py>,
    ) -> (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
    ) {
        let (Ruu_f, Rvv_f, Rww_f, Ruw_f) = &self.stencil.spectral_component_grids();

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
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
    ) {
        let (Ruu, Rvv, Rww, Ruw) = &self.stencil.correlation_grids();

        (
            Ruu.to_pyarray(py),
            Rvv.to_pyarray(py),
            Rww.to_pyarray(py),
            Ruw.to_pyarray(py),
        )
    }
}

#[pymethods]
impl RustConstrainedStencil {
    #[new]
    fn __new__<'py>(
        // py: Python<'py>,
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
        constraints: PyReadonlyArray2<'py, f32>,
        parallel: bool,
        corr_thres: f32,
        sinc_thres: f32,
    ) -> Self {
        let mut constraints_new: Vec<Constraint> = Vec::new();
        for row in constraints.as_array().rows() {
            let slice = row.as_slice().unwrap();
            let (x, y, z, u) = (slice[0], slice[1], slice[2], slice[3]);
            constraints_new.push(Constraint {
                x: x,
                y: y,
                z: z,
                u: u,
            });
        }
        RustConstrainedStencil {
            stencil: Stencil::from_params(
                L,
                gamma,
                Lx,
                Ly,
                Lz,
                Nx,
                Ny,
                Nz,
                aperiodic_x,
                aperiodic_y,
                aperiodic_z,
                sinc_thres,
                parallel,
            )
            .constrain(constraints_new, corr_thres),
        }
    }

    fn turbulate<'py>(
        &self,
        py: Python<'py>,
        ae: f32,
        seed: u64,
        impulse_thres: f32,
        parallel: bool,
    ) -> (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
    ) {
        let (U, V, W) = self.stencil.turbulate(ae, seed, impulse_thres, parallel);
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
        sinc_thres: f32,
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
            sinc_thres: sinc_thres,
        }
    }

    fn turbulence<'py>(
        &self,
        py: Python<'py>,
        ae: f32,
        seed: u64,
        parallel: bool,
    ) -> (
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
        Bound<'py, PyArray3<f32>>,
    ) {
        let (U_f, V_f, W_f): (Array3<f32>, Array3<f32>, Array3<f32>) = match parallel {
            true => forgetful_turbulate_par(
                ae,
                seed,
                self.Nx,
                self.Ny,
                self.Nz,
                self.Lx,
                self.Ly,
                self.Lz,
                self.L,
                self.gamma,
                self.sinc_thres,
            ),
            false => forgetful_turbulate(
                ae,
                seed,
                self.Nx,
                self.Ny,
                self.Nz,
                self.Lx,
                self.Ly,
                self.Lz,
                self.L,
                self.gamma,
                self.sinc_thres,
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
        Bound<'py, PyArray3<Complex32>>,
        Bound<'py, PyArray3<Complex32>>,
        Bound<'py, PyArray3<Complex32>>,
    ) {
        let (U_f, V_f, W_f): (Array3<Complex32>, Array3<Complex32>, Array3<Complex32>) =
            match parallel {
                true => partial_forgetful_turbulate_par(
                    ae,
                    seed,
                    self.Nx,
                    self.Ny,
                    self.Nz,
                    self.Lx,
                    self.Ly,
                    self.Lz,
                    self.L,
                    self.gamma,
                    self.sinc_thres,
                ),
                false => partial_forgetful_turbulate(
                    ae,
                    seed,
                    self.Nx,
                    self.Ny,
                    self.Nz,
                    self.Lx,
                    self.Ly,
                    self.Lz,
                    self.L,
                    self.gamma,
                    self.sinc_thres,
                ),
            };
        (U_f.to_pyarray(py), V_f.to_pyarray(py), W_f.to_pyarray(py))
    }
}

#[pymodule]
pub fn mannrs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustStencil>()?;
    m.add_class::<RustConstrainedStencil>()?;
    m.add_class::<RustForgetfulStencil>()?;

    #[pyfn(m)]
    fn freq_components_f32<'py>(
        py: Python<'py>,
        Nx: usize,
        Ny: usize,
        Nz: usize,
        Lx: f32,
        Ly: f32,
        Lz: f32,
    ) -> (
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
    ) {
        let (f_x, f_y, f_z): (Array1<f32>, Array1<f32>, Array1<f32>) =
            freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
        (f_x.to_pyarray(py), f_y.to_pyarray(py), f_z.to_pyarray(py))
    }

    #[pyfn(m)]
    fn isotropic_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
    ) -> Bound<'py, PyArray2<f32>> {
        Isotropic::from_params(ae, L)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }

    #[pyfn(m)]
    fn isotropic_sqrt_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
    ) -> Bound<'py, PyArray2<f32>> {
        Isotropic::from_params(ae, L)
            .decomp(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    #[pyfn(m)]
    fn sheared_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
        gamma: f32,
    ) -> Bound<'py, PyArray2<f32>> {
        Sheared::from_params(ae, L, gamma)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }

    #[pyfn(m)]
    fn sheared_sqrt_f32<'py>(
        py: Python<'py>,
        K: PyReadonlyArray1<'py, f32>,
        ae: f32,
        L: f32,
        gamma: f32,
    ) -> Bound<'py, PyArray2<f32>> {
        Sheared::from_params(ae, L, gamma)
            .decomp(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    #[pyfn(m)]
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
    ) -> Bound<'py, PyArray2<f32>> {
        ShearedSinc::from_params(ae, L, gamma, Ly, Lz, tol, min_depth)
            .tensor(&K.as_slice().unwrap())
            .to_pyarray(py)
    }
    #[pyfn(m)]
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
    ) -> (Bound<'py, PyArray2<f32>>, u64) {
        let (out, neval) = ShearedSinc::from_params(ae, L, gamma, Ly, Lz, tol, min_depth)
            .tensor_info(&K.as_slice().unwrap());

        (out.to_pyarray(py), neval)
    }

    #[pyfn(m)]
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
    ) -> Bound<'py, PyArray2<f32>> {
        ShearedSinc::from_params(ae, L, gamma, Ly, Lz, tol, min_depth)
            .decomp(&K.as_slice().unwrap())
            .to_pyarray(py)
    }

    #[pyfn(m)]
    fn distance_matrix<'py>(
        py: Python<'py>,
        x: PyReadonlyArray1<'py, f32>,
    ) -> Bound<'py, PyArray2<f32>> {
        Utilities::distance_matrix(&x.as_array().to_owned()).to_pyarray(py)
    }

    Ok(())
}
