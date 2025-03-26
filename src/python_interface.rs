use crate::{
    correlation_grids, forgetful_turbulate, forgetful_turbulate_par, partial_forgetful_turbulate,
    partial_forgetful_turbulate_par, partial_turbulate, partial_turbulate_par,
    spectral_component_grids, stencilate_sinc, stencilate_sinc_par, turbulate, turbulate_par,
    Tensors::*, Utilities, Utilities::fftfreq, Utilities::freq_components,
    Utilities::rfftfreq, Utilities::roll_1d_array, Utilities::roll_3d_array,
};
use ndarray::parallel::prelude::*;
use ndarray::{s, Array1, Array3, Array5};
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
        thres: f32,
        parallel: bool,
    ) -> (&'py PyArray3<f32>, &'py PyArray3<f32>, &'py PyArray3<f32>) {
        let CConstU: Array1<Complex32> = CConstU.to_owned_array().mapv(|x| Complex32::new(x, 0.0));
        let CConstV: Array1<Complex32> = CConstV.to_owned_array().mapv(|x| Complex32::new(x, 0.0));
        let CConstW: Array1<Complex32> = CConstW.to_owned_array().mapv(|x| Complex32::new(x, 0.0));

        // Calculate normalized spectral component grids
        // Calculate normalized spectral component grids
        let (Ruu_f, Rvv_f, Rww_f, Ruw_f) = spectral_component_grids(&self._stencil.view());

        // Calculate linear wave number arrays and record sizes.
        let kxs: Array1<f32> = fftfreq(self.Nx, self.Lx / (self.Nx as f32));
        let kys: Array1<f32> = fftfreq(self.Ny, self.Ly / (self.Ny as f32));
        let kzs: Array1<f32> = rfftfreq(self.Nz, self.Lz / (self.Nz as f32));
        let (Nx_exp, Ny_exp, Nz_exp): (usize, usize, usize) = (kxs.len(), kys.len(), kzs.len());

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
        // Reduce arrays TODO
        let (Ruu_f_max, Rvv_f_max, Rww_f_max): (f32, f32, f32) = (
            Ruu_f.iter().copied().fold(f32::NAN, f32::max),
            Rvv_f.iter().copied().fold(f32::NAN, f32::max),
            Rww_f.iter().copied().fold(f32::NAN, f32::max),
        );
        let ixmin: usize = Ruu_f
            .slice(s![.., yroll as usize, zroll as usize])
            .iter()
            .position(|&x| x >= thres * Ruu_f_max)
            .unwrap_or(0);
        let ixmax: usize = Ruu_f
            .slice(s![.., yroll as usize, zroll as usize])
            .iter()
            .rposition(|&x| x >= thres * Ruu_f_max)
            .unwrap_or(Nx_exp);

        let iymin: usize = Rvv_f
            .slice(s![xroll as usize, .., zroll as usize])
            .iter()
            .position(|&x| x >= thres * Rvv_f_max)
            .unwrap_or(0);
        let iymax: usize = Rvv_f
            .slice(s![xroll as usize, .., zroll as usize])
            .iter()
            .rposition(|&x| x >= thres * Rvv_f_max)
            .unwrap_or(Ny_exp);
        let izmax: usize = Rww_f
            .slice(s![xroll as usize, yroll as usize, ..])
            .iter()
            .rposition(|&x| x >= thres * Rww_f_max)
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

        let Ruu_f: Array3<Complex32> = Ruu_f.mapv(|x| Complex32::new(x, 0.0));
        let Rvv_f: Array3<Complex32> = Rvv_f.mapv(|x| Complex32::new(x, 0.0));
        let Rww_f: Array3<Complex32> = Rww_f.mapv(|x| Complex32::new(x, 0.0));
        let Ruw_f: Array3<Complex32> = Ruw_f.mapv(|x| Complex32::new(x, 0.0));

        let kxs: Array1<f32> = kxs.slice(s![ixmin..ixmax]).to_owned();
        let kys: Array1<f32> = kys.slice(s![iymin..iymax]).to_owned();
        let kzs: Array1<f32> = kzs.slice(s![0..izmax]).to_owned();

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
        let U_f: Array3<Complex32>;
        let V_f: Array3<Complex32>;
        let W_f: Array3<Complex32>;
        if !parallel {
            let mut _U_f = Array3::<Complex32>::zeros((nx, ny, nz));
            let mut _V_f = Array3::<Complex32>::zeros((nx, ny, nz));
            let mut _W_f = Array3::<Complex32>::zeros((nx, ny, nz));
            println!("Uf shape: {:?}", _U_f.shape());
            println!("kx_mesh shape: {:?}", kx_mesh.shape());
            println!("Ruu_f shape: {:?}", Ruu_f.shape());
            for (i, c) in constraints.as_array().outer_iter().enumerate() {
                let phase: Array3<Complex32> = (Complex32::new(0.0, -2.0 * std::f32::consts::PI)
                    * (&kx_mesh * c[0] + &ky_mesh * c[1] + &kz_mesh * c[2]))
                    .mapv(|x| x.exp());

                _U_f += &(&phase * (&Ruu_f * CConstU[i] + &Ruw_f * CConstW[i]));
                _V_f += &(&phase * (&Rvv_f * CConstV[i]));
                _W_f += &(&phase * (&Ruw_f * CConstU[i] + &Rww_f * CConstW[i]));
            }
            U_f = _U_f;
            V_f = _V_f;
            W_f = _W_f;
        } else {
            let _U_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
            let _V_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
            let _W_f = Arc::new(Mutex::new(Array3::<Complex32>::zeros((nx, ny, nz))));
            constraints
                .as_array()
                .outer_iter()
                .into_par_iter()
                .enumerate()
                .for_each(|(i, c)| {
                    let phase: Array3<Complex32> =
                        (Complex32::new(0.0, -2.0 * std::f32::consts::PI)
                            * (&kx_mesh * c[0] + &ky_mesh * c[1] + &kz_mesh * c[2]))
                            .mapv(|x| x.exp());

                    {
                        let mut _U_f = _U_f.lock().unwrap();
                        let mut _V_f = _V_f.lock().unwrap();
                        let mut _W_f = _W_f.lock().unwrap();
                        let to_add = Complex32::new(1.0, 0.0)
                            * &phase
                            * (&Ruu_f * CConstU[i] + &Ruw_f * CConstW[i]);
                        _U_f.scaled_add(Complex32::new(1.0, 0.0), &to_add);

                        let to_add = Complex32::new(1.0, 0.0) * &phase * (&Rvv_f * CConstV[i]);
                        _V_f.scaled_add(Complex32::new(1.0, 0.0), &to_add);

                        let to_add = Complex32::new(1.0, 0.0)
                            * &phase
                            * (&Ruw_f * CConstU[i] + &Rww_f * CConstW[i]);
                        _W_f.scaled_add(Complex32::new(1.0, 0.0), &to_add);
                    }
                });
            U_f = _U_f.lock().unwrap().to_owned();
            V_f = _V_f.lock().unwrap().to_owned();
            W_f = _W_f.lock().unwrap().to_owned();
        }

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

        let U: Array3<f32> = Utilities::irfft3d(&mut U_f_exp);
        let V: Array3<f32> = Utilities::irfft3d(&mut V_f_exp);
        let W: Array3<f32> = Utilities::irfft3d(&mut W_f_exp);



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
