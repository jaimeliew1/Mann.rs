#![allow(non_snake_case)]
#![cfg(test)]

mod tests {
    use crate::*;

    static TOL: f64 = 1e-7;

    #[test]
    fn test_vonkarman_spectrum() {
        let ae = 2.0;
        let k = 2.0;
        let L = 2.0;

        let correct = 0.5305357774587827;
        let ans = vonkarman_spectrum(ae, k, L);

        assert!((ans - correct).abs() < TOL);
    }

    #[test]
    fn test_lifetime_approx() {
        let kL = 1.0;
        let correct = 1.2341234009393085;

        let ans = lifetime_approx(kL);

        assert!((ans - correct).abs() < TOL);
    }

    #[test]
    fn test_lifetime_exact() {
        let kL = 1.0;
        let correct = 1.234430999866;

        let ans = lifetime_exact(kL);

        assert!((ans - correct).abs() < TOL);
    }

    #[test]
    fn test_isotropic_tensor() {
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K: [f64; 3] = [1.0, 2.0, 3.0];

        let correct = [
            [4.81365456e-04, -7.40562240e-05, -1.11084336e-04],
            [-7.40562240e-05, 3.70281120e-04, -2.22168672e-04],
            [-1.11084336e-04, -2.22168672e-04, 1.85140560e-04],
        ];
        let ans: Array2<f64> = iso_tensor(&K, ae, L);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sqrt_isotropic_tensor() {
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K: [f64; 3] = [1.0, 2.0, 3.0];

        let correct = [
            [0., 0.01825522, -0.01217015],
            [-0.01825522, 0., 0.00608507],
            [0.01217015, -0.00608507, 0.],
        ];
        let ans: Array2<f64> = sqrt_iso_tensor(&K, ae, L);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared_transform() {
        let gamma = 0.0;
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K: [f64; 3] = [1.0, 2.0, 3.0];

        let correct = [
            [1., 0., -0.40395476],
            [0., 1., 0.12190881],
            [0., 0., 1.195048],
        ];
        let ans: Array2<f64> = sheared_transform(&K, ae, L, gamma);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared_sqrt_tensor() {
        let gamma = 0.0;
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K: [f64; 3] = [1.0, 2.0, 3.0];

        let correct = [
            [-0.0038791, 0.01838437, -0.0096028],
            [-0.01527416, -0.00058533, 0.0048014],
            [0.0114758, -0.0057379, 0.],
        ];
        let ans: Array2<f64> = sqrt_sheared_tensor(&K, ae, L, gamma);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
    #[test]
    fn test_sheared_tensor() {
        let gamma = 0.0;
        let ae: f64 = 1.0;
        let L: f64 = 1.0;
        let K: [f64; 3] = [1.0, 2.0, 3.0];

        let correct = [
            [4.45246082e-04, 2.38208492e-06, -1.50003417e-04],
            [2.38208492e-06, 2.56695868e-04, -1.71924607e-04],
            [-1.50003417e-04, -1.71924607e-04, 1.64617544e-04],
        ];
        let ans: Array2<f64> = sheared_tensor(&K, ae, L, gamma);
        ans.into_iter()
            .zip(correct.iter().flatten())
            .for_each(|(a, b)| assert!((a - b).abs() < TOL));
    }
}
