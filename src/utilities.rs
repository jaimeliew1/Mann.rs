use ndarray::{concatenate, prelude::*};
use ndarray_rand::{
    rand::SeedableRng,
    rand_distr::{Normal, Uniform},
    RandomExt,
};
use ndrustfft::{
    ndfft, ndfft_par, ndfft_r2c, ndfft_r2c_par, ndifft, ndifft_par, ndifft_r2c, ndifft_r2c_par,
    Complex, FftHandler, R2cFftHandler,
};
use numpy::Complex32;
use std::f32::consts::{PI, SQRT_2};
use std::fmt::Debug;

/// Various mathematical function implementations.
pub mod Utilities {

    use ndarray::Data;
    use rayon::prelude::*;

    use crate::Constraint;

    use super::*;

    /// Unnormalized sinc squared function
    pub fn sinc2(x: f32) -> f32 {
        match x {
            x if x == 0.0 => 1.0,
            x => (x.sin() / x).powi(2),
        }
    }

    /// Returns the frequency components for a fft given a signal length (N) and a
    /// sampling distance. This function replicates the behaviour of
    /// `numpy.fft.fftfreq`.
    pub fn fftfreq(N: usize, dx: f32) -> Array1<f32> {
        let df = 1.0 / (N as f32 * dx);
        let _N = (N as i32 - 1) / 2 + 1;
        let f1: Array1<f32> = Array1::from_iter(0.._N).mapv(|elem| elem as f32);
        let f2: Array1<f32> = Array1::from_iter(-(N as i32) / 2..0).mapv(|elem| elem as f32);
        df * concatenate![Axis(0), f1, f2]
    }

    pub fn rfft3d_par(input: &mut Array3<f32>) -> Array3<Complex32> {
        let (nx, ny, nz) = input.dim();
        let mut vhat: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));

        let mut handler: R2cFftHandler<f32> = R2cFftHandler::new(nz);
        ndfft_r2c_par(input, &mut vhat, &mut handler, 2);

        let mut vhat2: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f32> = FftHandler::new(nx);

        ndfft_par(&mut vhat, &mut vhat2, &mut handler, 0);

        let mut vhat3: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f32> = FftHandler::new(ny);

        ndfft_par(&mut vhat2, &mut vhat3, &mut handler, 1);

        vhat3
    }
    pub fn irfft3d_par(input: &mut Array3<Complex32>) -> Array3<f32> {
        let (nx, ny, _nz) = input.dim();
        let nz = (_nz - 1) * 2;

        let mut handler: FftHandler<f32> = FftHandler::new(nx);
        let mut vhat2: Array3<Complex32> = Array3::zeros((nx, ny, _nz));
        ndifft_par(input, &mut vhat2, &mut handler, 0);

        let mut handler: FftHandler<f32> = FftHandler::new(ny);
        let mut vhat: Array3<Complex32> = Array3::zeros((nx, ny, _nz));
        ndifft_par(&mut vhat2, &mut vhat, &mut handler, 1);

        let mut output: Array3<f32> = Array3::zeros((nx, ny, nz));
        let mut handler: R2cFftHandler<f32> = R2cFftHandler::new(nz);
        ndifft_r2c_par(&mut vhat, &mut output, &mut handler, 2);

        output
    }
    pub fn rfft3d(input: &mut Array3<f32>) -> Array3<Complex32> {
        let (nx, ny, nz) = input.dim();
        let mut vhat: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));

        let mut handler: R2cFftHandler<f32> = R2cFftHandler::new(nz);
        ndfft_r2c(input, &mut vhat, &mut handler, 2);

        let mut vhat2: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f32> = FftHandler::new(nx);

        ndfft(&mut vhat, &mut vhat2, &mut handler, 0);

        let mut vhat3: Array3<Complex32> = Array3::zeros((nx, ny, nz / 2 + 1));
        let mut handler: FftHandler<f32> = FftHandler::new(ny);

        ndfft(&mut vhat2, &mut vhat3, &mut handler, 1);

        vhat3
    }
    pub fn irfft3d(input: &mut Array3<Complex32>) -> Array3<f32> {
        let (nx, ny, _nz) = input.dim();
        let nz = (_nz - 1) * 2;

        let mut handler: FftHandler<f32> = FftHandler::new(nx);
        let mut vhat2: Array3<Complex32> = Array3::zeros((nx, ny, _nz));
        ndifft(input, &mut vhat2, &mut handler, 0);

        let mut handler: FftHandler<f32> = FftHandler::new(ny);
        let mut vhat: Array3<Complex32> = Array3::zeros((nx, ny, _nz));
        ndifft(&mut vhat2, &mut vhat, &mut handler, 1);

        let mut output: Array3<f32> = Array3::zeros((nx, ny, nz));
        let mut handler: R2cFftHandler<f32> = R2cFftHandler::new(nz);
        ndifft_r2c(&mut vhat, &mut output, &mut handler, 2);

        output
    }
    /// Returns the frequency components for a real fft given a signal length (N)
    /// and a sampling distance. This function replicates the behaviour of
    /// `numpy.fft.rfftfreq`.
    pub fn rfftfreq(N: usize, dx: f32) -> Array1<f32> {
        let df = 1.0 / (N as f32 * dx);
        let _N = (N as i32) / 2 + 1;
        let f: Array1<f32> = Array1::from_iter(0.._N).mapv(|elem| elem as f32);
        df * f
    }

    pub fn cholesky(A: Array2<f32>) -> Array2<f32> {
        let mut l: Array2<f32> = Array2::<f32>::zeros((3, 3));

        for i in 0..3 {
            for j in 0..=i {
                let sum = if i == j {
                    let mut s = 0.0;
                    for k in 0..j {
                        s += l[[j, k]] * l[[j, k]];
                    }
                    (A[[j, j]] - s).sqrt()
                } else {
                    let mut s = 0.0;
                    for k in 0..j {
                        s += l[[i, k]] * l[[j, k]];
                    }
                    (1.0 / l[[j, j]]) * (A[[i, j]] - s)
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

    /// Returns wave numbers for a turbulence box specification.
    pub fn freq_components(
        Lx: f32,
        Ly: f32,
        Lz: f32,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> (Array1<f32>, Array1<f32>, Array1<f32>) {
        (
            fftfreq(Nx, Lx / (2.0 * PI * Nx as f32)),
            fftfreq(Ny, Ly / (2.0 * PI * Ny as f32)),
            rfftfreq(Nz, Lz / (2.0 * PI * Nz as f32)),
        )
    }

    /// Returns Array3 of of complex, gaussian distributed random numbers with
    /// unit variance.
    pub fn complex_random_gaussian(
        seed: u64,
        Nx: usize,
        Ny: usize,
        Nz: usize,
    ) -> Array4<Complex32> {
        let mut rng = ndarray_rand::rand::rngs::SmallRng::seed_from_u64(seed);
        let dist = Normal::new(0.0, SQRT_2.recip()).unwrap();
        let real: Array4<Complex32> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng)
            .mapv(|elem| Complex::new(elem, 0.0));
        let imag: Array4<Complex32> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng)
            .mapv(|elem| Complex::new(0.0, elem));

        real + imag
    }

    /// Returns Array3 of of complex, random numbers with unit length.
    pub fn complex_random_unit(seed: u64, Nx: usize, Ny: usize, Nz: usize) -> Array4<Complex32> {
        let mut rng = ndarray_rand::rand::rngs::SmallRng::seed_from_u64(seed);
        let dist = Uniform::new(0.0, 2.0 * PI);
        let phase: Array4<f32> = Array4::random_using((Nx, Ny, Nz, 3), dist, &mut rng);
        let out: Array4<Complex32> = phase.mapv(|elem| Complex::new(elem.cos(), elem.sin()));
        out
    }

    #[derive(Debug)]
    struct SimpsonRange {
        a: f32,
        m: f32,
        b: f32,
        fa: Array2<f32>,
        fm: Array2<f32>,
        fb: Array2<f32>,
        depth: u64,
    }

    pub fn adaptive_quadrature<F>(
        mut func: F,
        A: f32,
        B: f32,
        tol: f32,
        min_depth: u64,
    ) -> (Array2<f32>, u64)
    where
        F: FnMut(f32) -> Array2<f32>,
    {
        let m: f32 = (A + B) / 2.0;
        let mut I: Array2<f32> = Array2::zeros((3, 3));

        let mut S: Vec<SimpsonRange> = Vec::new();
        S.push(SimpsonRange {
            a: A,
            m: m,
            b: B,
            fa: func(A),
            fm: func(m),
            fb: func(B),
            depth: 1,
        });
        let mut neval: u64 = 3;

        while let Some(SimpsonRange {
            a,
            m,
            b,
            fa,
            fm,
            fb,
            depth,
        }) = S.pop()
        {
            // simpson rule
            let I1: Array2<f32> = (b - a) / 6.0 * (&fa + 4.0 * &fm + &fb);
            let m1: f32 = (a + m) / 2.0;
            let m3: f32 = (m + b) / 2.0;
            let fm1: Array2<f32> = func(m1);
            let fm3: Array2<f32> = func(m3);
            neval += 2;

            // Composite trapeszoidal rule with 2 equidistant intervals
            let I2: Array2<f32> =
                (b - a) / 12.0 * (&fa.view() + 4.0 * &fm1 + 2.0 * &fm + 4.0 * &fm3 + &fb);
            if depth >= min_depth && (&I2 - &I1).iter().all(|&x| x.abs() < 15.0 * tol) {
                I += &I2;
            } else {
                S.push(SimpsonRange {
                    a: a,
                    m: m1,
                    b: m,
                    fa: fa,
                    fm: fm1,
                    fb: fm.clone(),
                    depth: depth + 1,
                });
                S.push(SimpsonRange {
                    a: m,
                    m: m3,
                    b: b,
                    fa: fm,
                    fm: fm3,
                    fb: fb,
                    depth: depth + 1,
                });
            }
        }
        (I, neval)
    }

    pub fn adaptive_quadrature_2d<F>(
        mut func: F,
        x0: f32,
        x1: f32,
        y0: f32,
        y1: f32,
        tol: f32,
        min_depth: u64,
    ) -> (Array2<f32>, u64)
    where
        F: FnMut(f32, f32) -> Array2<f32>,
    {
        let mut neval: u64 = 0;
        let g = |x: f32| {
            let f = |y: f32| func(x, y);
            let (I, _neval): (Array2<f32>, u64) = adaptive_quadrature(f, y0, y1, tol, min_depth);
            neval += _neval;
            I
        };
        let (I, _): (Array2<f32>, u64) = adaptive_quadrature(g, x0, x1, tol, min_depth);
        (I, neval)
    }
    pub fn roll_1d_array(arr: &Array1<f32>, roll: &isize) -> Array1<f32> {
        let n = arr.len() as isize;
        let roll = (roll % n + n) % n;

        let mut rolled = arr.clone();

        for i in 0..n {
            let new_i = (i + roll) % n;
            rolled[new_i as usize] = arr[i as usize].clone();
        }

        rolled
    }

    pub fn roll_3d_array<T>(
        arr: &Array3<T>,
        xroll: &isize,
        yroll: &isize,
        zroll: &isize,
    ) -> Array3<T>
    where
        T: Clone + Copy,
    {
        let shape = arr.shape();
        let (nx, ny, nz) = (shape[0] as isize, shape[1] as isize, shape[2] as isize);

        let xroll = (xroll % nx + nx) % nx;
        let yroll = (yroll % ny + ny) % ny;
        let zroll = (zroll % nz + nz) % nz;

        let mut rolled = arr.clone();

        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    let new_i = (i + xroll) % nx;
                    let new_j = (j + yroll) % ny;
                    let new_k = (k + zroll) % nz;
                    rolled[[new_i as usize, new_j as usize, new_k as usize]] =
                        arr[[i as usize, j as usize, k as usize]];
                }
            }
        }

        rolled
    }

    /// Computes the pairwise distance matrix from a 1D array of x locations.
    ///
    /// # Arguments
    ///
    /// * `x` - A 1D ndarray of f64 values representing x coordinates.
    ///
    /// # Returns
    ///
    /// * A 2D ndarray where the element at (i, j) is the absolute distance between x[i] and x[j].
    pub fn distance_matrix(x: &Array1<f32>) -> Array2<f32> {
        let x_row: ArrayView2<f32> = x.view().insert_axis(ndarray::Axis(0)); // shape (1, n)
        let x_col: ArrayView2<f32> = x.view().insert_axis(ndarray::Axis(1)); // shape (n, 1)
        (&x_row - &x_col).mapv(f32::abs)
    }

    pub struct SpectralImpulseResponse {
        pub impulse: Array3<Complex32>,
        pub kx: Array1<f32>,
        pub ky: Array1<f32>,
        pub kz: Array1<f32>,
        xroll: isize,
        yroll: isize,
        zroll: isize,
    }

    pub struct CompressedSpectralImpulseResponse {
        pub impulse: Array3<Complex32>,
        pub kx: Array1<f32>,
        pub ky: Array1<f32>,
        pub kz: Array1<f32>,
        xroll: isize,
        yroll: isize,
        zroll: isize,
        indices: CompressionIndices,
    }
    pub struct CompressionIndices {
        Nx_exp: usize,
        Ny_exp: usize,
        Nz_exp: usize,
        ixmin: usize,
        ixmax: usize,
        iymin: usize,
        iymax: usize,
        izmax: usize,
    }

    impl CompressionIndices {
        pub fn combine_all<I>(iter: I) -> Option<Self>
        where
            I: IntoIterator<Item = CompressionIndices>,
        {
            iter.into_iter().fold(None, |acc, item| {
                Some(match acc {
                    None => item,
                    Some(current) => CompressionIndices {
                        Nx_exp: current.Nx_exp,
                        Ny_exp: current.Ny_exp,
                        Nz_exp: current.Nz_exp,
                        ixmin: current.ixmin.min(item.ixmin),
                        ixmax: current.ixmax.max(item.ixmax),
                        iymin: current.iymin.min(item.iymin),
                        iymax: current.iymax.max(item.iymax),
                        izmax: current.izmax.max(item.izmax),
                    },
                })
            })
        }
    }

    impl SpectralImpulseResponse {
        pub fn new(
            impulse: Array3<f32>,
            kx: Array1<f32>,
            ky: Array1<f32>,
            kz: Array1<f32>,
        ) -> Self {
            let (Nx_exp, Ny_exp, Nz_exp): (usize, usize, usize) = (kx.len(), ky.len(), kz.len());
            let (xroll, yroll, zroll): (isize, isize, isize) =
                ((&Nx_exp / 2) as isize, (&Ny_exp / 2) as isize, 0);

            let kxs: Array1<f32> = roll_1d_array(&kx, &xroll);
            let kys: Array1<f32> = roll_1d_array(&ky, &yroll);
            let kzs: Array1<f32> = roll_1d_array(&kz, &zroll);

            let impulse_out: Array3<Complex32> =
                roll_3d_array(&impulse, &xroll, &yroll, &zroll).mapv(|x| Complex32::new(x, 0.0));

            SpectralImpulseResponse {
                impulse: impulse_out,
                kx: kxs,
                ky: kys,
                kz: kzs,
                xroll: xroll,
                yroll: yroll,
                zroll: zroll,
            }
        }
        pub fn get_compression_indices(&self, impulse_thres: f32) -> CompressionIndices {
            let (Nx_exp, Ny_exp, Nz_exp): (usize, usize, usize) =
                (self.kx.len(), self.ky.len(), self.kz.len());

            let impulse_max: f32 = self.impulse.iter().map(|x| x.re).fold(f32::NAN, f32::max);
            let ixmin: usize = self
                .impulse
                .slice(s![.., self.yroll as usize, self.zroll as usize])
                .iter()
                .position(|&x| x.re >= impulse_thres * impulse_max)
                .unwrap_or(0);
            let ixmax: usize = self
                .impulse
                .slice(s![.., self.yroll as usize, self.zroll as usize])
                .iter()
                .rposition(|&x| x.re >= impulse_thres * impulse_max)
                .unwrap_or(Nx_exp);

            let iymin: usize = self
                .impulse
                .slice(s![self.xroll as usize, .., self.zroll as usize])
                .iter()
                .position(|&x| x.re >= impulse_thres * impulse_max)
                .unwrap_or(0);
            let iymax: usize = self
                .impulse
                .slice(s![self.xroll as usize, .., self.zroll as usize])
                .iter()
                .rposition(|&x| x.re >= impulse_thres * impulse_max)
                .unwrap_or(Ny_exp);
            let izmax: usize = self
                .impulse
                .slice(s![self.xroll as usize, self.yroll as usize, ..])
                .iter()
                .rposition(|&x| x.re >= impulse_thres * impulse_max)
                .unwrap_or(Nz_exp);

            CompressionIndices {
                Nx_exp: Nx_exp,
                Ny_exp: Ny_exp,
                Nz_exp: Nz_exp,
                ixmin: ixmin,
                ixmax: ixmax,
                iymin: iymin,
                iymax: iymax,
                izmax: izmax,
            }
        }
        pub fn compress(self, indices: CompressionIndices) -> CompressedSpectralImpulseResponse {
            // println!("ixmin: {ixmin}, ixmax {ixmax}");
            // println!("iymin: {iymin}, iymax {iymax}");
            // println!("izmax: {izmax}");

            let impulse: Array3<Complex32> = self
                .impulse
                .slice(s![
                    indices.ixmin..indices.ixmax,
                    indices.iymin..indices.iymax,
                    0..indices.izmax
                ])
                .to_owned();
            // println!("hello3");
            // let self.impulse: Array3<Complex32> = self.impulse;
            // println!("hello4");

            let kx: Array1<f32> = self.kx.slice(s![indices.ixmin..indices.ixmax]).to_owned();
            let ky: Array1<f32> = self.ky.slice(s![indices.iymin..indices.iymax]).to_owned();
            let kz: Array1<f32> = self.kz.slice(s![0..indices.izmax]).to_owned();

            CompressedSpectralImpulseResponse {
                impulse: impulse,
                kx: kx,
                ky: ky,
                kz: kz,
                xroll: self.xroll,
                yroll: self.yroll,
                zroll: self.zroll,
                indices: indices,
            }
        }
    }

    impl CompressedSpectralImpulseResponse {
        pub fn zero_pad_and_unroll_impulse(&self) -> Array3<Complex32> {
            let mut out: Array3<Complex32> = Array3::zeros((
                self.indices.Nx_exp,
                self.indices.Ny_exp,
                self.indices.Nz_exp,
            ));

            out.slice_mut(s![
                self.indices.ixmin..self.indices.ixmax,
                self.indices.iymin..self.indices.iymax,
                0..self.indices.izmax
            ])
            .assign(&self.impulse);

            let out: Array3<Complex32> =
                roll_3d_array(&out, &(-self.xroll), &(-self.yroll), &(-self.zroll));
            out
        }
    }

    pub fn spectral_superposition_ser<I2>(
        constraints: &Vec<Constraint>,
        impulse_u: CompressedSpectralImpulseResponse,
        weights: I2,
    ) -> CompressedSpectralImpulseResponse
    where
        I2: IntoIterator<Item = f32>,
    {
        let (nx, ny, nz) = (impulse_u.kx.len(), impulse_u.ky.len(), impulse_u.kz.len());
        let mut kx_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        let mut ky_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        let mut kz_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    kx_mesh[[i, j, k]] = Complex32::new(impulse_u.kx[i], 0.0);
                    ky_mesh[[i, j, k]] = Complex32::new(impulse_u.ky[j], 0.0);
                    kz_mesh[[i, j, k]] = Complex32::new(impulse_u.kz[k], 0.0);
                }
            }
        }

        let two_pi_i = Complex32::new(0.0, -2.0 * std::f32::consts::PI);

        let U_f = constraints.iter().zip(weights).fold(
            Array3::<Complex32>::zeros((nx, ny, nz)),
            |mut acc, (c, w)| {
                let phase: Array3<Complex32> = (two_pi_i
                    * (&kx_mesh * c.x + &ky_mesh * c.y + &kz_mesh * c.z))
                    .mapv(|x| x.exp());

                acc += &(&phase * (&impulse_u.impulse * w));
                acc
            },
        );

        CompressedSpectralImpulseResponse {
            impulse: U_f,
            kx: impulse_u.kx,
            ky: impulse_u.ky,
            kz: impulse_u.kz,
            xroll: impulse_u.xroll,
            yroll: impulse_u.yroll,
            zroll: impulse_u.zroll,
            indices: impulse_u.indices,
        }
    }

    pub fn spectral_superposition_par(
        constraints: &Vec<Constraint>,
        impulse_u: CompressedSpectralImpulseResponse,
        weights: &Vec<f32>,
    ) -> CompressedSpectralImpulseResponse {
        let (nx, ny, nz) = (impulse_u.kx.len(), impulse_u.ky.len(), impulse_u.kz.len());
        let mut kx_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        let mut ky_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        let mut kz_mesh: Array3<Complex32> = Array3::zeros((nx, ny, nz));
        for i in 0..nx {
            for j in 0..ny {
                for k in 0..nz {
                    kx_mesh[[i, j, k]] = Complex32::new(impulse_u.kx[i], 0.0);
                    ky_mesh[[i, j, k]] = Complex32::new(impulse_u.ky[j], 0.0);
                    kz_mesh[[i, j, k]] = Complex32::new(impulse_u.kz[k], 0.0);
                }
            }
        }

        let two_pi_i = Complex32::new(0.0, -2.0 * std::f32::consts::PI);

        let U_f = constraints
            .iter()
            .zip(weights)
            .par_bridge()
            .fold(
                || Array3::<Complex32>::zeros((nx, ny, nz)),
                |mut acc, (c, w)| {
                    acc += &(&(two_pi_i * (&kx_mesh * c.x + &ky_mesh * c.y + &kz_mesh * c.z))
                        .mapv(|x| x.exp())
                        * (&impulse_u.impulse * *w));
                    acc
                },
            )
            .reduce(
                || Array3::<Complex32>::zeros((nx, ny, nz)),
                |mut acc, v| {
                    acc += &v;
                    acc
                },
            );
        CompressedSpectralImpulseResponse {
            impulse: U_f,
            kx: impulse_u.kx,
            ky: impulse_u.ky,
            kz: impulse_u.kz,
            xroll: impulse_u.xroll,
            yroll: impulse_u.yroll,
            zroll: impulse_u.zroll,
            indices: impulse_u.indices,
        }
    }

    pub fn analyze_array<A, S, D>(array: &ArrayBase<S, D>)
    where
        A: Copy + Into<f32> + Debug,
        S: Data<Elem = A>,
        D: Dimension,
    {
        println!("Shape: {:?}", array.shape());

        let mut sum = 0.0;
        let mut sum_sq = 0.0;
        let mut count = 0;

        for &val in array.iter() {
            let x: f32 = val.into();
            sum += x;
            sum_sq += x * x;
            count += 1;
        }

        if count == 0 {
            println!("Array is empty.");
            return;
        }

        let mean = sum / count as f32;
        let variance = (sum_sq / count as f32) - (mean * mean);
        let std_dev = variance.sqrt();

        println!("Mean: {}", mean);
        println!("Standard Deviation: {}", std_dev);
        println!("Sum (Checksum): {}", sum);
        println!("");
    }
}
