pub use crate::utilities::Utilities;
use ndarray::prelude::*;
use std::f32::consts::PI;

pub fn lifetime_approx(mut kL: f32) -> f32 {
    if kL < 0.005 {
        kL = 0.005;
    }
    let kSqr = kL.powi(2);
    (1.0 + kSqr).powf(1.0 / 6.0) / kL
        * (1.2050983316598936 - 0.04079766636961979 * kL + 1.1050803451576134 * kSqr)
        / (1.0 - 0.04103886513006046 * kL + 1.1050902034670118 * kSqr)
}

pub fn vonkarman_spectrum(ae: f32, k: f32, L: f32) -> f32 {
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
    impl Isotropic<f32> {
        pub fn from_params(ae: f32, L: f32) -> Isotropic<f32> {
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

    impl Sheared<f32> {
        pub fn from_params(ae: f32, L: f32, gamma: f32) -> Sheared<f32> {
            Sheared { ae, L, gamma }
        }

        /// Isotropic to sheared tensor transformation matrix.
        pub fn sheared_transform(&self, K: &[f32]) -> Array2<f32> {
            let k_norm2 = K.iter().map(|x| x * x).sum::<f32>();
            if k_norm2 == 0.0 {
                return Array2::zeros((3, 3));
            }
            let beta = self.gamma * lifetime_approx((k_norm2).sqrt() * self.L);

            // Equation (12)
            let K0: Array1<f32> = arr1(K) + arr1(&[0.0, 0.0, beta * K[0]]);
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
        /// Adaptive integration tolerance
        pub tol: T,
        /// Adaptive integration minimum depth
        pub min_depth: u64,
    }

    impl ShearedSinc<f32> {
        pub fn from_params(
            ae: f32,
            L: f32,
            gamma: f32,
            Ly: f32,
            Lz: f32,
            tol: f32,
            min_depth: u64,
        ) -> ShearedSinc<f32> {
            ShearedSinc {
                ae,
                L,
                gamma,
                Ly,
                Lz,
                tol,
                min_depth,
            }
        }

        /// Sinc-corrected spectral tensor with additional information.
        pub fn tensor_info(&self, K: &[f32]) -> (Array2<f32>, u64) {
            let tensor_gen: Sheared<f32> = Sheared::from_params(self.ae, self.L, self.gamma);

            let func = |y: f32, z: f32| {
                Utilities::sinc2(y * self.Ly / 2.0)
                    * Utilities::sinc2(z * self.Lz / 2.0)
                    * tensor_gen.tensor(&[K[0], K[1] + y, K[2] + z])
            };

            let (mut out, neval): (Array2<f32>, u64) = Utilities::adaptive_quadrature_2d(
                func,
                -2.0 * PI / self.Ly,
                2.0 * PI / self.Ly,
                -2.0 * PI / self.Lz,
                2.0 * PI / self.Lz,
                self.tol,
                self.min_depth,
            );

            out *= 1.22686 * self.Ly * self.Lz / (2.0 * PI).powi(2);
            (out, neval)
        }
    }

    pub trait TensorGenerator<T> {
        /// Returns the tensor.
        fn tensor(&self, K: &[T]) -> Array2<T>;

        /// Returns the tensor decomposition.
        fn decomp(&self, K: &[T]) -> Array2<T>;
    }

    impl TensorGenerator<f32> for Isotropic<f32> {
        /// Isotropic spectral tensor
        ///
        /// Generates the incompressible isotropic turbulence spectral tensor as
        /// described in Equation (8). $$
        /// \mathbf{\Phi}^{\text{ISO}}\_{ij}(\mathbf{k}) =
        /// \frac{E(|\mathbf{k}|)}{4\pi|\mathbf{k}|^4}(\delta\_{ij}|\mathbf{k}|^2 -
        /// k\_ik\_j) $$
        ///
        fn tensor(&self, K: &[f32]) -> Array2<f32> {
            let k_norm = K.iter().map(|x| x * x).sum::<f32>().sqrt();
            if k_norm == 0.0 {
                return Array2::zeros((3, 3));
            }
            let E = vonkarman_spectrum(self.ae, k_norm, self.L);
            let mut tensor: Array2<f32> = arr2(&[
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
        fn decomp(&self, K: &[f32]) -> Array2<f32> {
            let k_norm = K.iter().map(|x| x * x).sum::<f32>().sqrt();
            if k_norm == 0.0 {
                return Array2::zeros((3, 3));
            }
            let E = vonkarman_spectrum(self.ae, k_norm, self.L);
            let mut tensor: Array2<f32> =
                arr2(&[[0.0, K[2], -K[1]], [-K[2], 0.0, K[0]], [K[1], -K[0], 0.0]]);
            tensor *= (E / PI).sqrt() / (2.0 * k_norm.powi(2));
            tensor
        }
    }

    impl TensorGenerator<f32> for Sheared<f32> {
        /// Sheared (Mann) spectral tensor
        fn tensor(&self, K: &[f32]) -> Array2<f32> {
            let k_norm2 = K.iter().map(|x| x * x).sum::<f32>();
            if k_norm2 == 0.0 {
                return Array2::zeros((3, 3));
            }
            let A: Array2<f32> = self.sheared_transform(K);
            let beta = self.gamma * lifetime_approx((k_norm2).sqrt() * self.L);

            // Equation (12)
            let K0: Array1<f32> = arr1(K) + arr1(&[0.0, 0.0, beta * K[0]]);
            let iso_tensor = Isotropic {
                ae: self.ae,
                L: self.L,
            }
            .tensor(&K0.as_slice().unwrap());

            A.dot(&iso_tensor).dot(&A.t())
        }
        /// Decomposition of sheared (Mann) spectral tensor
        fn decomp(&self, K: &[f32]) -> Array2<f32> {
            let k_norm2 = K.iter().map(|x| x * x).sum::<f32>();
            if k_norm2 == 0.0 {
                return Array2::zeros((3, 3));
            }
            let A: Array2<f32> = self.sheared_transform(K);
            let beta = self.gamma * lifetime_approx((k_norm2).sqrt() * self.L);

            // Equation (12)
            let K0: Array1<f32> = arr1(K) + arr1(&[0.0, 0.0, beta * K[0]]);
            let iso_tensor = Isotropic {
                ae: self.ae,
                L: self.L,
            }
            .decomp(&K0.as_slice().unwrap());

            A.dot(&iso_tensor)
        }
    }

    impl TensorGenerator<f32> for ShearedSinc<f32> {
        /// Sheared spectral tensor with sinc correction

        fn tensor(&self, K: &[f32]) -> Array2<f32> {
            let (out, _): (Array2<f32>, u64) = self.tensor_info(K);
            out
        }

        /// Decomposition of sheared spectral tensor with sinc correction using a Cholesky decomposition.
        fn decomp(&self, K: &[f32]) -> Array2<f32> {
            let mut l: Array2<f32> = Array2::<f32>::zeros((3, 3));
            let tensor: Array2<f32> = self.tensor(K);

            for i in 0..3 {
                for j in 0..=i {
                    let mut sum = if i == j {
                        let mut s = 0.0;
                        for k in 0..j {
                            s += l[[j, k]] * l[[j, k]];
                        }
                        (tensor[[j, j]] - s).sqrt()
                    } else {
                        let mut s = 0.0;
                        for k in 0..j {
                            s += l[[i, k]] * l[[j, k]];
                        }
                        (1.0 / l[[j, j]]) * (tensor[[i, j]] - s)
                    };

                    if i == j {
                        if sum <= 0.0 {
                            panic!(); // Matrix is not positive definite
                        }
                    } else {
                        if l[[j, j]] <= 0.0 {
                            panic!(); // Matrix is not positive definite
                        }
                    }

                    l[[i, j]] = sum;
                }
            }

            l
        }
    }
}
