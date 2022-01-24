pub use crate::utilities::Utilities;
use ndarray::prelude::*;
use ndarray_linalg::cholesky::*;
use std::f64::consts::PI;

pub fn lifetime_approx(mut kL: f64) -> f64 {
    if kL < 0.005 {
        kL = 0.005;
    }
    let kSqr = kL.powi(2);
    (1.0 + kSqr).powf(1.0 / 6.0) / kL
    * (1.2050983316598936 - 0.04079766636961979 * kL + 1.1050803451576134 * kSqr)
    / (1.0 - 0.04103886513006046 * kL + 1.1050902034670118 * kSqr)
}

pub fn vonkarman_spectrum(ae: f64, k: f64, L: f64) -> f64 {
    ae * L.powf(5.0 / 3.0) * (L * k).powi(4) / (1.0 + (L * k).powi(2)).powf(17.0 / 6.0)
}

/// Spectral tensor calculations
/// 
/// Contains calculations for various spectral tensors, including isotropic,
/// Mann (sheared), and their decompositions.
pub mod Tensors {
    use super::*;

    pub fn sqrt_iso_tensor(K: &ArrayView1<f64>, ae: f64, L: f64) -> Array2<f64> {
        let k_norm = K.dot(K).sqrt();
        if k_norm == 0.0 {
            return Array2::zeros((3, 3));
        }
        let E = vonkarman_spectrum(ae, k_norm, L);
        let mut tensor: Array2<f64> =
            arr2(&[[0.0, K[2], -K[1]], [-K[2], 0.0, K[0]], [K[1], -K[0], 0.0]]);
        tensor *= (E / PI).sqrt() / (2.0 * k_norm.powi(2));
        tensor
    }

    pub fn iso_tensor(K: &ArrayView1<f64>, ae: f64, L: f64) -> Array2<f64> {
        let k_norm = K.dot(K).sqrt();
        if k_norm == 0.0 {
            return Array2::zeros((3, 3));
        }
        let E = vonkarman_spectrum(ae, k_norm, L);
        let mut tensor: Array2<f64> = arr2(&[
            [K[1].powi(2) + K[2].powi(2), -K[0] * K[1], -K[0] * K[2]],
            [-K[0] * K[1], K[0].powi(2) + K[2].powi(2), -K[1] * K[2]],
            [-K[0] * K[2], -K[1] * K[2], K[0].powi(2) + K[1].powi(2)],
        ]);
        tensor *= E / (4.0 * PI * k_norm.powi(4));
        tensor
    }

    pub fn sheared_transform(K: &ArrayView1<f64>, _ae: f64, L: f64, gamma: f64) -> Array2<f64> {
        let k_norm2 = K.dot(K);
        if k_norm2 == 0.0 {
            return Array2::zeros((3, 3));
        }
        let beta = gamma * lifetime_approx((k_norm2).sqrt() * L);

        // Equation (12)
        let K0: Array1<f64> = K + arr1(&[0.0, 0.0, beta * K[0]]);
        let k0_norm2 = K0.dot(&K0);
        let (zeta1, zeta2);
        if K[0] == 0.0 {
            zeta1 = -beta;
            zeta2 = 0.0;
        } else {
            // Equation (15)
            let C1 = beta * K[0].powi(2) * (k0_norm2 - 2.0 * K0[2].powi(2) + beta * K[0] * K0[2])
                / (k_norm2 * (K[0].powi(2) + K[1].powi(2)));
            // Equation (16)
            let C2 = K[1] * k0_norm2 / (K[0].powi(2) + K[1].powi(2)).powf(3.0 / 2.0)
                * ((beta * K[0] * (K[0].powi(2) + K[1].powi(2)).sqrt())
                    / (k0_norm2 - K0[2] * K[0] * beta))
                    .atan();
            // Equation (14)
            zeta1 = C1 - K[1] / K[0] * C2;
            zeta2 = K[1] / K[0] * C1 + C2;
        }
        // Equation (13)
        arr2(&[
            [1.0, 0.0, zeta1],
            [0.0, 1.0, zeta2],
            [0.0, 0.0, k0_norm2 / k_norm2],
        ])
    }

    pub fn sqrt_sheared_tensor(K: &ArrayView1<f64>, ae: f64, L: f64, gamma: f64) -> Array2<f64> {
        let k_norm2 = K.dot(K);
        if k_norm2 == 0.0 {
            return Array2::zeros((3, 3));
        }
        let A: Array2<f64> = sheared_transform(K, ae, L, gamma);
        let beta = gamma * lifetime_approx((k_norm2).sqrt() * L);

        // Equation (12)
        let K0: Array1<f64> = K + arr1(&[0.0, 0.0, beta * K[0]]);
        let iso_tensor = sqrt_iso_tensor(&K0.view(), ae, L);

        A.dot(&iso_tensor)
    }

    pub fn sheared_tensor(K: &ArrayView1<f64>, ae: f64, L: f64, gamma: f64) -> Array2<f64> {
        let k_norm2 = K.dot(K);
        if k_norm2 == 0.0 {
            return Array2::zeros((3, 3));
        }
        let A: Array2<f64> = sheared_transform(K, ae, L, gamma);
        let beta = gamma * lifetime_approx((k_norm2).sqrt() * L);

        // Equation (12)
        let K0: Array1<f64> = K + arr1(&[0.0, 0.0, beta * K[0]]);
        let iso_tensor = iso_tensor(&K0.view(), ae, L);

        A.dot(&iso_tensor).dot(&A.t())
    }

    pub fn sheared_tensor_sinc(
        K: &ArrayView1<f64>,
        ae: f64,
        L: f64,
        gamma: f64,
        _Lx: f64,
        Ly: f64,
        Lz: f64,
        Ny: usize,
        Nz: usize,
    ) -> Array2<f64> {
        let NN = 51;
        let kys: Array1<f64> = Array1::linspace(-1.0, 1.0, NN);
        let kzs: Array1<f64> = Array1::linspace(-1.0, 1.0, NN);

        let mut ans: Array2<f64> = Array2::zeros((3, 3));
        for (i, ky) in kys.iter().enumerate() {
            for (j, kz) in kzs.iter().enumerate() {
                let mut factor = 4.0;
                if i == 0 || i == Ny - 1 {
                    factor /= 2.0
                }
                if j == 0 || j == Nz - 1 {
                    factor /= 2.0
                }
                let sinc = Utilities::sinc2(ky * PI) * Utilities::sinc2(kz * PI);
                ans = ans
                    + factor
                        * sinc
                        * sheared_tensor(
                            &arr1(&[K[0], K[1] + *ky * 2.0 * PI / Ly, K[2] + *kz * 2.0 * PI / Lz])
                                .view(),
                            ae,
                            L,
                            gamma,
                        );
            }
        }
        ans *= 1.22686 / (NN as f64 - 1.0).powi(2);
        ans
    }

    pub fn sqrt_sheared_tensor_sinc(
        K: &ArrayView1<f64>,
        ae: f64,
        L: f64,
        gamma: f64,
        _Lx: f64,
        Ly: f64,
        Lz: f64,
        Ny: usize,
        Nz: usize,
    ) -> Array2<f64> {
        sheared_tensor_sinc(K, ae, L, gamma, _Lx, Ly, Lz, Ny, Nz)
            .cholesky(UPLO::Lower)
            .unwrap()
    }
}
