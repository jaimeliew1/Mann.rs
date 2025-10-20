use ndarray::prelude::*;

use numpy::Complex32;
use std::fmt::Debug;

/// Various mathematical function implementations.
use rayon::prelude::*;

use crate::utilities::{bisect_positive, roll_1d_array, roll_3d_array};
use crate::Constraint;

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
    pub indices: CompressionIndices,
}

#[derive(Clone, Debug)]
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
    /// Returns the bounding `CompressionIndices` that encloses all given indices.
    ///
    /// Combines multiple `CompressionIndices` into one by taking the element-wise
    /// min/max across the iterator:
    /// - `ixmin`, `iymin`: minimum of all values
    /// - `ixmax`, `iymax`, `izmax`: maximum of all values
    /// - `Nx_exp`, `Ny_exp`, `Nz_exp`: taken from the first element (not checked for consistency)
    ///
    /// # Parameters
    /// - `iter`: Any iterator of `CompressionIndices`
    ///
    /// # Returns
    /// - `Some(CompressionIndices)` if input is non-empty
    /// - `None` if input is empty      
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

    /// Returns the compression ratios in each dimension (x, y, z).
    ///
    /// Each ratio is the size of the compressed range divided by the full dimension size.
    /// Values are in the range `(0.0, 1.0]`.
    pub fn compression_ratios(&self) -> (f64, f64, f64) {
        let x_ratio = (self.ixmax - self.ixmin) as f64 / self.Nx_exp as f64;
        let y_ratio = (self.iymax - self.iymin) as f64 / self.Ny_exp as f64;
        let z_ratio = self.izmax as f64 / self.Nz_exp as f64;
        (1.0 - x_ratio, 1.0 - y_ratio, 1.0 - z_ratio)
    }

    /// Returns the total compression ratio as the product of per-axis ratios.
    pub fn total_compression_ratio(&self) -> f64 {
        let (rx, ry, rz) = self.compression_ratios();
        1.0 - (1.0 - rx) * (1.0 - ry) * (1.0 - rz)
    }
}

impl SpectralImpulseResponse {
    pub fn new(impulse: Array3<f32>, kx: Array1<f32>, ky: Array1<f32>, kz: Array1<f32>) -> Self {
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

    pub fn get_thres_from_compression_ratio(&self, cr_target: f64) -> f32 {
        if cr_target == 0.0 {
            return 0.0;
        }
        let f = |thres: f32| {
            self.get_compression_indices(thres)
                .total_compression_ratio()
                - cr_target
        };
        bisect_positive(0.0, 1.0, f).unwrap()
    }
    pub fn get_compression_indices(&self, impulse_thres: f32) -> CompressionIndices {
        let (Nx_exp, Ny_exp, Nz_exp): (usize, usize, usize) =
            (self.kx.len(), self.ky.len(), self.kz.len());

        if impulse_thres == 0.0 {
            return CompressionIndices {
                Nx_exp: Nx_exp,
                Ny_exp: Ny_exp,
                Nz_exp: Nz_exp,
                ixmin: 0,
                ixmax: Nx_exp,
                iymin: 0,
                iymax: Ny_exp,
                izmax: Nz_exp,
            };
        }
        let impulse_max: f32 = self.impulse.iter().map(|x| x.re).fold(f32::NAN, f32::max);

        let ixmin: usize = self
            .impulse
            .axis_iter(Axis(0))
            .position(|x| {
                x.map(|v| v.norm())
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(*v))
                    >= impulse_thres * impulse_max
            })
            .unwrap_or(0);

        let ixmax: usize = self
            .impulse
            .axis_iter(Axis(0))
            .rposition(|x| {
                x.map(|v| v.norm())
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(*v))
                    >= impulse_thres * impulse_max
            })
            .unwrap_or(Nx_exp);

        let iymin: usize = self
            .impulse
            .axis_iter(Axis(1))
            .position(|x| {
                x.map(|v| v.norm())
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(*v))
                    >= impulse_thres * impulse_max
            })
            .unwrap_or(0);

        let iymax: usize = self
            .impulse
            .axis_iter(Axis(1))
            .rposition(|x| {
                x.map(|v| v.norm())
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(*v))
                    >= impulse_thres * impulse_max
            })
            .unwrap_or(Ny_exp);

        let izmax: usize = self
            .impulse
            .axis_iter(Axis(2))
            .rposition(|x| {
                x.map(|v| v.norm())
                    .fold(f32::NEG_INFINITY, |acc, v| acc.max(*v))
                    >= impulse_thres * impulse_max
            })
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
    impulse_u: &CompressedSpectralImpulseResponse,
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
            let phase: Array3<Complex32> =
                (two_pi_i * (&kx_mesh * c.x + &ky_mesh * c.y + &kz_mesh * c.z)).mapv(|x| x.exp());

            acc += &(&phase * (&impulse_u.impulse * w));
            acc
        },
    );

    CompressedSpectralImpulseResponse {
        impulse: U_f,
        kx: impulse_u.kx.clone(),
        ky: impulse_u.ky.clone(),
        kz: impulse_u.kz.clone(),
        xroll: impulse_u.xroll,
        yroll: impulse_u.yroll,
        zroll: impulse_u.zroll,
        indices: impulse_u.indices.clone(),
    }
}

pub fn spectral_superposition_par(
    constraints: &Vec<Constraint>,
    impulse_u: &CompressedSpectralImpulseResponse,
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
        kx: impulse_u.kx.clone(),
        ky: impulse_u.ky.clone(),
        kz: impulse_u.kz.clone(),
        xroll: impulse_u.xroll,
        yroll: impulse_u.yroll,
        zroll: impulse_u.zroll,
        indices: impulse_u.indices.clone(),
    }
}
