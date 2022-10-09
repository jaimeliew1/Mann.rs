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
/// sheared (Mann), and their decompositions.
pub mod Tensors {
    use super::*;

    pub struct Isotropic<T> {
        /// $\alpha\epsilon^{2/3}$
        pub ae: T,
        /// Length scale, $L$
        pub L: T,
    }
    impl Isotropic<f64> {
        pub fn from_params(ae: f64, L: f64) -> Isotropic<f64> {
            Isotropic { ae, L }
        }
    }

    pub struct Sheared<T> {
        /// $\alpha\epsilon^{2/3}$
        pub ae: T,
        /// Length scale, $L$
        pub L: T,
        /// Lifetime parameter, $\Gamma$
        pub gamma: T,
    }

    impl Sheared<f64> {
        pub fn from_params(ae: f64, L: f64, gamma: f64) -> Sheared<f64> {
            Sheared { ae, L, gamma }
        }

        /// Isotropic to sheared tensor transformation matrix.
        pub fn sheared_transform(&self, K: &[f64]) -> Array2<f64> {
            let k_norm2 = K.iter().map(|x| x * x).sum::<f64>();
            if k_norm2 == 0.0 {
                return Array2::zeros((3, 3));
            }
            let beta = self.gamma * lifetime_approx((k_norm2).sqrt() * self.L);

            // Equation (12)
            let K0: Array1<f64> = arr1(K) + arr1(&[0.0, 0.0, beta * K[0]]);
            let k0_norm2 = K0.dot(&K0);
            let (zeta1, zeta2);
            if K[0] == 0.0 {
                zeta1 = -beta;
                zeta2 = 0.0;
            } else {
                // Equation (15)
                let C1 =
                    beta * K[0].powi(2) * (k0_norm2 - 2.0 * K0[2].powi(2) + beta * K[0] * K0[2])
                        / (k_norm2 * (K[0].powi(2) + K[1].powi(2)));
                // Equation (16)
                let C2 = K[1] * k0_norm2 / (K[0].powi(2) + K[1].powi(2)).powf(3.0 / 2.0)
                    * (beta * K[0] * (K[0].powi(2) + K[1].powi(2)).sqrt())
                        .atan2(k0_norm2 - K0[2] * K[0] * beta);
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
    }

    pub struct ShearedSinc<T> {
        /// $\alpha\epsilon^{2/3}$
        pub ae: T,
        /// Length scale, $L$
        pub L: T,
        /// Lifetime parameter, $\Gamma$
        pub gamma: T,
        /// Lateral box length, $L\_y$
        pub Ly: T,
        /// Vertical box length, $L\_z$
        pub Lz: T,
    }

    impl ShearedSinc<f64> {
        pub fn from_params(ae: f64, L: f64, gamma: f64, Ly: f64, Lz: f64) -> ShearedSinc<f64> {
            ShearedSinc {
                ae,
                L,
                gamma,
                Ly,
                Lz,
            }
        }

        /// Isotropic to sheared tensor transformation matrix.
        pub fn sheared_transform(&self, K: &[f64]) -> Array2<f64> {
            Sheared {
                ae: self.ae,
                L: self.L,
                gamma: self.gamma,
            }
            .sheared_transform(K)
        }
    }

    pub trait TensorGenerator<T> {
        /// Returns the tensor.
        fn tensor(&self, K: &[T]) -> Array2<T>;

        /// Returns the tensor decomposition.
        fn decomp(&self, K: &[T]) -> Array2<T>;
    }

    impl TensorGenerator<f64> for Isotropic<f64> {
        /// Isotropic spectral tensor
        ///
        /// Generates the incompressible isotropic turbulence spectral tensor as
        /// described in Equation (8). $$
        /// \mathbf{\Phi}^{\text{ISO}}\_{ij}(\mathbf{k}) =
        /// \frac{E(|\mathbf{k}|)}{4\pi|\mathbf{k}|^4}(\delta\_{ij}|\mathbf{k}|^2 -
        /// k\_ik\_j) $$
        ///
        fn tensor(&self, K: &[f64]) -> Array2<f64> {
            let k_norm = K.iter().map(|x| x * x).sum::<f64>().sqrt();
            if k_norm == 0.0 {
                return Array2::zeros((3, 3));
            }
            let E = vonkarman_spectrum(self.ae, k_norm, self.L);
            let mut tensor: Array2<f64> = arr2(&[
                [K[1].powi(2) + K[2].powi(2), -K[0] * K[1], -K[0] * K[2]],
                [-K[0] * K[1], K[0].powi(2) + K[2].powi(2), -K[1] * K[2]],
                [-K[0] * K[2], -K[1] * K[2], K[0].powi(2) + K[1].powi(2)],
            ]);
            tensor *= E / (4.0 * PI * k_norm.powi(4));
            tensor
        }
        /// Decomposition of isotropic spectral tensor
        ///
        /// Generates the decomposition of the isotropic spectral tensor,
        /// $\mathbf{\phi}(\mathbf{k})$, where
        /// $\mathbf{\phi}^\*(\mathbf{k})\mathbf{\phi}(\mathbf{k}) =
        /// \mathbf{\Phi}^{\text{ISO}}(\mathbf{k})$.
        fn decomp(&self, K: &[f64]) -> Array2<f64> {
            let k_norm = K.iter().map(|x| x * x).sum::<f64>().sqrt();
            if k_norm == 0.0 {
                return Array2::zeros((3, 3));
            }
            let E = vonkarman_spectrum(self.ae, k_norm, self.L);
            let mut tensor: Array2<f64> =
                arr2(&[[0.0, K[2], -K[1]], [-K[2], 0.0, K[0]], [K[1], -K[0], 0.0]]);
            tensor *= (E / PI).sqrt() / (2.0 * k_norm.powi(2));
            tensor
        }
    }

    impl TensorGenerator<f64> for Sheared<f64> {
        /// Sheared (Mann) spectral tensor
        fn tensor(&self, K: &[f64]) -> Array2<f64> {
            let k_norm2 = K.iter().map(|x| x * x).sum::<f64>();
            if k_norm2 == 0.0 {
                return Array2::zeros((3, 3));
            }
            let A: Array2<f64> = self.sheared_transform(K);
            let beta = self.gamma * lifetime_approx((k_norm2).sqrt() * self.L);

            // Equation (12)
            let K0: Array1<f64> = arr1(K) + arr1(&[0.0, 0.0, beta * K[0]]);
            let iso_tensor = Isotropic {
                ae: self.ae,
                L: self.L,
            }
            .tensor(&K0.as_slice().unwrap());

            A.dot(&iso_tensor).dot(&A.t())
        }
        /// Decomposition of sheared (Mann) spectral tensor
        fn decomp(&self, K: &[f64]) -> Array2<f64> {
            let k_norm2 = K.iter().map(|x| x * x).sum::<f64>();
            if k_norm2 == 0.0 {
                return Array2::zeros((3, 3));
            }
            let A: Array2<f64> = self.sheared_transform(K);
            let beta = self.gamma * lifetime_approx((k_norm2).sqrt() * self.L);

            // Equation (12)
            let K0: Array1<f64> = arr1(K) + arr1(&[0.0, 0.0, beta * K[0]]);
            let iso_tensor = Isotropic {
                ae: self.ae,
                L: self.L,
            }
            .decomp(&K0.as_slice().unwrap());

            A.dot(&iso_tensor)
        }
    }

    impl TensorGenerator<f64> for ShearedSinc<f64> {
        /// Sheared spectral tensor with sinc correction
        fn tensor(&self, K: &[f64]) -> Array2<f64> {
            let NN = 51;
            let kys: Array1<f64> = Array1::linspace(-1.0, 1.0, NN);
            let kzs: Array1<f64> = Array1::linspace(-1.0, 1.0, NN);

            let mut ans: Array2<f64> = Array2::zeros((3, 3));
            for (i, ky) in kys.iter().enumerate() {
                for (j, kz) in kzs.iter().enumerate() {
                    let mut factor = 4.0;
                    if i == 0 || i == NN - 1 {
                        factor /= 2.0
                    }
                    if j == 0 || j == NN - 1 {
                        factor /= 2.0
                    }
                    let sinc = Utilities::sinc2(ky * PI) * Utilities::sinc2(kz * PI);
                    ans = ans
                        + factor
                            * sinc
                            * Sheared {
                                ae: self.ae,
                                L: self.L,
                                gamma: self.gamma,
                            }
                            .tensor(&[
                                K[0],
                                K[1] + *ky * 2.0 * PI / self.Ly,
                                K[2] + *kz * 2.0 * PI / self.Lz,
                            ]);
                }
            }
            ans *= 1.22686 / (NN as f64 - 1.0).powi(2);
            ans
        }

        /// Decomposition of sheared spectral tensor with sinc correction
        fn decomp(&self, K: &[f64]) -> Array2<f64> {
            self.tensor(K).cholesky(UPLO::Lower).unwrap()
        }
    }
}
