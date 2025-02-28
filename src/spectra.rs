pub mod spectra {
    use crate::Tensors::Sheared;
    use crate::Tensors::*;
    use ndarray::{Array, Array1, Array2};
    use std::f32::consts::PI;


    /// Computes the 2D trapezoidal integral using two successive 1D trapezoidal integrations.
    /// First integrates along the x-axis, then along the y-axis.
    ///
    /// # Arguments
    /// * `f` - 2D ndarray representing function values at grid points.
    /// * `x` - 1D array of x-coordinates (non-uniform spacing).
    /// * `y` - 1D array of y-coordinates (non-uniform spacing).
    ///
    /// # Returns
    /// * Approximate integral value.
    fn trapezoidal_integral_2d(f: &Array2<f32>, x: &Array1<f32>, y: &Array1<f32>) -> f32 {
        let nx = x.len();
        let ny = y.len();

        // Step 1: Integrate along the x-axis for each fixed y
        let mut integral_x = Array1::zeros(ny);
        for j in 0..ny {
            let mut sum_x = 0.0;
            for i in 0..nx - 1 {
                let dx = x[i + 1] - x[i];
                sum_x += 0.5 * (f[[i, j]] + f[[i + 1, j]]) * dx;
            }
            integral_x[j] = sum_x;
        }

        // Step 2: Integrate the intermediate result along the y-axis
        let mut integral = 0.0;
        for j in 0..ny - 1 {
            let dy = y[j + 1] - y[j];
            integral += 0.5 * (integral_x[j] + integral_x[j + 1]) * dy;
        }

        integral
    }

    pub fn mann_spectra(
        kx: &Array1<f32>,
        ae: f32,
        l: f32,
        gamma: f32,
    ) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
        let tensor_gen = Sheared::from_params(ae, l, gamma);
        let nr = 150;
        let ntheta = 30;

        let rs = Array1::linspace(-4.0, 7.0, nr).mapv(|x| 10f32.powf(x));
        let thetas = Array1::linspace(0.0, 2.0 * PI, ntheta);

        let mut uu_vals = Array1::zeros(kx.len());
        let mut vv_vals = Array1::zeros(kx.len());
        let mut ww_vals = Array1::zeros(kx.len());
        let mut uw_vals = Array1::zeros(kx.len());

        for (idx, &kx_val) in kx.iter().enumerate() {
            let mut uu_grid = Array::zeros((nr, ntheta));
            let mut vv_grid = Array::zeros((nr, ntheta));
            let mut ww_grid = Array::zeros((nr, ntheta));
            let mut uw_grid = Array::zeros((nr, ntheta));

            for (i, &r) in rs.iter().enumerate() {
                for (j, &theta) in thetas.iter().enumerate() {
                    let ky = r * theta.cos();
                    let kz = r * theta.sin();
                    let tensor = tensor_gen.tensor(&[kx_val, ky, kz]);
                    uu_grid[[i, j]] = r * tensor[[0, 0]];
                    vv_grid[[i, j]] = r * tensor[[1, 1]];
                    ww_grid[[i, j]] = r * tensor[[2, 2]];
                    uw_grid[[i, j]] = r * tensor[[0, 2]];
                }
            }

            uu_vals[idx] = trapezoidal_integral_2d(&uu_grid, &rs, &thetas);
            vv_vals[idx] = trapezoidal_integral_2d(&vv_grid, &rs, &thetas);
            ww_vals[idx] = trapezoidal_integral_2d(&ww_grid, &rs, &thetas);
            uw_vals[idx] = trapezoidal_integral_2d(&uw_grid, &rs, &thetas);
        }

        (uu_vals, vv_vals, ww_vals, uw_vals)
    }
}
