#![allow(non_snake_case)]
#![cfg(test)]

mod tests {
    use crate::tensors::Tensors::*;
    use crate::*;
    static TOL: f64 = 1e-7;

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
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [4.81365456e-04, -7.40562240e-05, -1.11084336e-04],
            [-7.40562240e-05, 3.70281120e-04, -2.22168672e-04],
            [-1.11084336e-04, -2.22168672e-04, 1.85140560e-04],
        ];
        let ans: Array2<f64> = Tensors::Isotropic::from_params(ae, L).tensor(K);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sqrt_isotropic_tensor() {
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [0., 0.01825522, -0.01217015],
            [-0.01825522, 0., 0.00608507],
            [0.01217015, -0.00608507, 0.],
        ];
        let ans: Array2<f64> = Tensors::Isotropic::from_params(ae, L).decomp(K);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared_transform() {
        let gamma = 1.0;
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [1., 0., -0.40395476],
            [0., 1., 0.12190881],
            [0., 0., 1.195048],
        ];
        let ans: Array2<f64> = Tensors::Sheared::from_params(ae, L, gamma).sheared_transform(K);

        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared_sqrt() {
        let gamma = 1.0;
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [-0.0038791, 0.01838437, -0.0096028],
            [-0.01527416, -0.00058533, 0.0048014],
            [0.0114758, -0.0057379, 0.],
        ];
        let ans: Array2<f64> = Tensors::Sheared::from_params(ae, L, gamma).decomp(K);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared() {
        let gamma = 1.0;
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K = &[1.0, 2.0, 3.0];

        let correct = [
            [4.45246082e-04, 2.38208492e-06, -1.50003417e-04],
            [2.38208492e-06, 2.56695868e-04, -1.71924607e-04],
            [-1.50003417e-04, -1.71924607e-04, 1.64617544e-04],
        ];
        let ans: Array2<f64> = Tensors::Sheared::from_params(ae, L, gamma).tensor(K);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }

    #[test]
    fn test_freq_components_even() {
        let (Lx, Ly, Lz) = (10.0, 20.0, 30.0);
        let (Nx, Ny, Nz) = (10, 10, 10);
        let (Kx, Ky, Kz): (Array1<f64>, Array1<f64>, Array1<f64>) =
            Utilities::freq_components(Lx, Ly, Lz, Nx, Ny, Nz);
        println!("{:?}", Kx);
        let ans_Kx = [
            0.,
            0.62831853,
            1.25663706,
            1.88495559,
            2.51327412,
            -3.14159265,
            -2.51327412,
            -1.88495559,
            -1.25663706,
            -0.62831853,
        ];
        let ans_Ky = [
            0.,
            0.31415927,
            0.62831853,
            0.9424778,
            1.25663706,
            -1.57079633,
            -1.25663706,
            -0.9424778,
            -0.62831853,
            -0.31415927,
        ];
        let ans_Kz = [
            0., 0.20943951, 0.41887902, 0.62831853, 0.83775804, 1.04719755,
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
    // #[test]
    // fn test_stencilate() {
    //     let gamma = 1.0;
    //     let ae: f64 = 1.0;
    //     let L: f64 = 1.0;
    //     let (Nx, Ny, Nz) = (8192, 32, 32);
    //     let (Lx, Ly, Lz) = (10.0, 10.0, 10.0);
    //     stencilate(ae, L, gamma, Lx, Ly, Lz, Nx, Ny, Nz);
    //     assert!(false);
    // }
}
