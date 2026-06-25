#![allow(non_snake_case)]
#![cfg(test)]

mod tests {
    use crate::tensors::Tensors::*;
    use crate::*;
    static TOL: f32 = 1e-5;

    #[test]
    fn test_vonkarman_spectrum() {
        let ae = 2.0;
        let k = 2.0;
        let L = 2.0;

        let correct = 0.5305357774587827;
        let ans = tensors::vonkarman_spectrum(ae, k, L);

        assert!((ans - correct).abs() < TOL);
    }

    #[test]
    fn test_lifetime_approx() {
        let kL = 1.0;
        let correct = 1.2341234009393085;

        let ans = tensors::lifetime_approx(kL);

        assert!((ans - correct).abs() < TOL);
    }

    #[test]
    fn test_isotropic_tensor() {
        let ae: f32 = 1.0;
        let L: f32 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [4.81365456e-04, -7.40562240e-05, -1.11084336e-04],
            [-7.40562240e-05, 3.70281120e-04, -2.22168672e-04],
            [-1.11084336e-04, -2.22168672e-04, 1.85140560e-04],
        ];
        let ans: Array2<f32> = Tensors::Isotropic::from_params(ae, L).tensor(K);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sqrt_isotropic_tensor() {
        let ae: f32 = 1.0;
        let L: f32 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [0., 0.01825522, -0.01217015],
            [-0.01825522, 0., 0.00608507],
            [0.01217015, -0.00608507, 0.],
        ];
        let ans: Array2<f32> = Tensors::Isotropic::from_params(ae, L).decomp(K);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared_transform() {
        let gamma = 1.0;
        let ae: f32 = 1.0;
        let L: f32 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [1., 0., -0.40395476],
            [0., 1., 0.12190881],
            [0., 0., 1.195048],
        ];
        let ans: Array2<f32> = Tensors::Sheared::from_params(ae, L, gamma).sheared_transform(K);

        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared_sqrt() {
        let gamma = 1.0;
        let ae: f32 = 1.0;
        let L: f32 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [-0.0038791, 0.01838437, -0.0096028],
            [-0.01527416, -0.00058533, 0.0048014],
            [0.0114758, -0.0057379, 0.],
        ];
        let ans: Array2<f32> = Tensors::Sheared::from_params(ae, L, gamma).decomp(K);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared() {
        let gamma = 1.0;
        let ae: f32 = 1.0;
        let L: f32 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [4.45246082e-04, 2.38208492e-06, -1.50003417e-04],
            [2.38208492e-06, 2.56695868e-04, -1.71924607e-04],
            [-1.50003417e-04, -1.71924607e-04, 1.64617544e-04],
        ];
        let ans: Array2<f32> = Tensors::Sheared::from_params(ae, L, gamma).tensor(K);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }

    #[test]
    fn test_freq_components_even() {
        let (Lx, Ly, Lz) = (10.0, 20.0, 30.0);
        let (Nx, Ny, Nz) = (10, 10, 10);
        let (Kx, Ky, Kz): (Array1<f32>, Array1<f32>, Array1<f32>) =
            utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
        let ans_Kx = [
            0.,
            0.56548667,
            1.13097335,
            1.69646001,
            2.26194668,
            -2.82743339,
            -2.26194668,
            -1.69646001,
            -1.13097335,
            -0.56548667,
        ];
        let ans_Ky = [
            0.,
            0.28274334,
            0.56548668,
            0.84822999,
            1.13097335,
            -1.41371668,
            -1.13097335,
            -0.84822999,
            -0.56548668,
            -0.28274334,
        ];
        let ans_Kz = [
            0.,
            0.18849556,
            0.37699112,
            0.56548668,
            0.75398224,
            0.94247780,
        ];
        Kx.into_iter()
            .zip(ans_Kx.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
        Ky.into_iter()
            .zip(ans_Ky.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
        Kz.into_iter()
            .zip(ans_Kz.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }

    #[test]
    fn test_fftfreq_standard() {
        let arr: Array1<f32> = utilities::fftfreq(10, 1.0);
        let expected = [0.0, 0.1, 0.2, 0.3, 0.4, -0.5, -0.4, -0.3, -0.2, -0.1];
        arr.into_iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }

    #[test]
    fn test_rfftfreq_standard() {
        let arr: Array1<f32> = utilities::rfftfreq(10, 1.0);
        let expected = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
        arr.into_iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }

    #[test]
    fn test_get_axes_physical_box_spacing() {
        let p = crate::unconstrained::StencilParams {
            L: 1.0,
            gamma: 0.0,
            Lx: 10.0,
            Ly: 20.0,
            Lz: 30.0,
            Nx: 10,
            Ny: 10,
            Nz: 10,
            aperiodic_x: false,
            aperiodic_y: false,
            aperiodic_z: false,
        };
        let (x, y, z) = p.get_axes();
        assert_eq!(x[0], 0.0);
        assert_eq!(x[9], 10.0);
        assert_eq!(y[9], 20.0);
        assert_eq!(z[9], 30.0);
    }

    #[test]
    fn test_constrained_stencil_preserves_constraint_point() {
        let stencil = Stencil::from_params(
            1.0,
            0.0,
            10.0,
            10.0,
            10.0,
            8,
            8,
            8,
            false,
            false,
            false,
            0.0,
            false,
        );
        let ae = 1.0;
        let seed = 42;
        let (U, _, _) = stencil.turbulate(ae, seed, false);
        let (x, y, z) = stencil.p.get_axes();

        let ix = 1;
        let iy = 2;
        let iz = 3;
        let constraint = Constraint {
            x: x[ix],
            y: y[iy],
            z: z[iz],
            u: U[[ix, iy, iz]],
        };
        let constrained = ConstrainedStencil::new(stencil.clone(), vec![constraint], 0.0, 0.0);
        let (U_constrained, _, _) = constrained.turbulate(ae, seed, false);

        assert!((U_constrained[[ix, iy, iz]] - U[[ix, iy, iz]]).abs() < 1e-5);
    }

    #[test]
    fn test_axes_physical_box_convention() {
        let p = crate::unconstrained::StencilParams {
            L: 1.0,
            gamma: 0.0,
            Lx: 10.0,
            Ly: 20.0,
            Lz: 30.0,
            Nx: 10,
            Ny: 11,
            Nz: 12,
            aperiodic_x: false,
            aperiodic_y: false,
            aperiodic_z: false,
        };
        let (x, y, z) = p.get_axes();

        assert_eq!(x.len(), p.Nx);
        assert_eq!(y.len(), p.Ny);
        assert_eq!(z.len(), p.Nz);

        assert!((x[0] - 0.0).abs() < TOL);
        assert!((x[p.Nx - 1] - p.Lx).abs() < TOL);
        assert!((x[p.Nx - 1] - x[0] - p.Lx).abs() < TOL);
        assert!((x[1] - x[0] - p.Lx / ((p.Nx - 1) as f32)).abs() < TOL);

        assert!((y[0] - 0.0).abs() < TOL);
        assert!((y[p.Ny - 1] - p.Ly).abs() < TOL);
        assert!((y[p.Ny - 1] - y[0] - p.Ly).abs() < TOL);
        assert!((y[1] - y[0] - p.Ly / ((p.Ny - 1) as f32)).abs() < TOL);

        assert!((z[0] - 0.0).abs() < TOL);
        assert!((z[p.Nz - 1] - p.Lz).abs() < TOL);
        assert!((z[p.Nz - 1] - z[0] - p.Lz).abs() < TOL);
        assert!((z[1] - z[0] - p.Lz / ((p.Nz - 1) as f32)).abs() < TOL);
    }

    #[test]
    fn test_linear_wave_numbers_physical_box() {
        let p = crate::unconstrained::StencilParams {
            L: 1.0,
            gamma: 0.0,
            Lx: 10.0,
            Ly: 20.0,
            Lz: 30.0,
            Nx: 10,
            Ny: 10,
            Nz: 10,
            aperiodic_x: false,
            aperiodic_y: false,
            aperiodic_z: false,
        };
        let (kx, ky, kz) = p.linear_wave_numbers();

        assert_eq!(kx.len(), p.Nx);
        assert_eq!(ky.len(), p.Ny);
        assert_eq!(kz.len(), p.Nz / 2 + 1);
        assert!((kx[1] - 1.0 / p.Lx).abs() < TOL);
        assert!((ky[1] - 1.0 / p.Ly).abs() < TOL);
        assert!((kz[1] - 1.0 / p.Lz).abs() < TOL);
        assert!((kx[p.Nx - 1] + 1.0 / p.Lx).abs() < TOL);
    }

    #[test]
    fn test_aperiodic_linear_wave_numbers_physical_box() {
        let p = crate::unconstrained::StencilParams {
            L: 1.0,
            gamma: 0.0,
            Lx: 10.0,
            Ly: 20.0,
            Lz: 30.0,
            Nx: 10,
            Ny: 10,
            Nz: 10,
            aperiodic_x: true,
            aperiodic_y: false,
            aperiodic_z: false,
        };

        let (kx, ky, kz) = p.aperiodic_linear_wave_numbers();
        let Nx_ext = 2 * p.Nx - 1;
        let Lx_ext = 2.0 * p.Lx;

        assert_eq!(kx.len(), Nx_ext);
        assert_eq!(ky.len(), p.Ny);
        assert_eq!(kz.len(), p.Nz / 2 + 1);
        assert!((kx[1] - 1.0 / Lx_ext).abs() < TOL);
        assert!((kx[Nx_ext - 1] + 1.0 / Lx_ext).abs() < TOL);
        assert!((ky[1] - 1.0 / p.Ly).abs() < TOL);
        assert!((kz[1] - 1.0 / p.Lz).abs() < TOL);
    }

    #[test]
    fn test_distance_matrix() {
        let x: Array1<f32> = array![1.0, 2.0, 4.0];

        let expected: Array2<f32> = array![[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]];
        let ans: Array2<f32> = utilities::distance_matrix(&x);
        ans.into_iter()
            .zip(expected.iter())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
}
