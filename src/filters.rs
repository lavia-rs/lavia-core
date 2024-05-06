use ndarray::{self, Array, Array1, ArrayBase, Data, Dimension, LinalgScalar, RawDataClone};

use crate::convolutions::{self, ConvPaddingStrategy, ExecutionMode};

/// Create a 1D Gaussian kernel centered at the middle of the kernel.
pub fn generate_gaussian_kernel_1d(sigma: f64, size: Option<usize>) -> Array<f64, ndarray::Ix1> {
    let size = match size {
        Some(size) => size,
        None => (sigma * 6.0).ceil() as usize,
    };

    // to be checked if this is correct
    let center = match size % 2 {
        0 => size as f64 / 2.0 - 0.5,
        _ => size as f64 / 2.0 - 0.5,
    };
    let two_sigma_sq = 2.0 * sigma * sigma;
    let normalizer = 1.0 / (std::f64::consts::PI * two_sigma_sq).sqrt();

    (0..size)
        .map(|i| {
            let x = i as f64 - center;
            normalizer * (-x * x / two_sigma_sq).exp()
        })
        .collect()
}

pub fn gaussian_filter2d<A, S, D>(
    data: &ArrayBase<S, D>,
    sigma: f64,
    size: Option<usize>,
    padding_strategy: &ConvPaddingStrategy<A>,
    execution_mode: &ExecutionMode,
) -> Array<A, D>
where
    S: Data<Elem = A> + std::marker::Sync + RawDataClone,
    A: LinalgScalar
        + Send
        + Sync
        + num_traits::FromPrimitive
        + num_traits::Num
        + std::cmp::PartialOrd,
    D: Dimension,
{
    let kernel: Array1<f64> = generate_gaussian_kernel_1d(sigma, size);
    let kernel: Array1<A> = kernel.mapv(|x| num_traits::FromPrimitive::from_f64(x).unwrap());
    let size = kernel.shape()[0];

    let new_kernel_x = kernel.clone().into_shape((1, size)).unwrap();
    let new_kernel_y = kernel.into_shape((size, 1)).unwrap();
    convolutions::separable_convolve(
        data,
        &new_kernel_x,
        &new_kernel_y,
        None,
        execution_mode,
        padding_strategy,
    )
}

pub fn gaussian_filter3d<A, S, D>(
    data: &ArrayBase<S, D>,
    sigma: f64,
    size: Option<usize>,
    padding_strategy: &ConvPaddingStrategy<A>,
    execution_mode: &ExecutionMode,
) -> Array<A, D>
where
    S: Data<Elem = A> + std::marker::Sync + RawDataClone,
    A: LinalgScalar
        + Send
        + Sync
        + num_traits::FromPrimitive
        + num_traits::Num
        + std::cmp::PartialOrd,
    D: Dimension,
{
    let kernel: Array1<f64> = generate_gaussian_kernel_1d(sigma, size);
    let kernel: Array1<A> = kernel.mapv(|x| num_traits::FromPrimitive::from_f64(x).unwrap());
    let size = kernel.shape()[0];

    let new_kernel_x = kernel.clone().into_shape((1, 1, size)).unwrap();
    let new_kernel_y = kernel.clone().into_shape((1, size, 1)).unwrap();
    let new_kernel_z = kernel.into_shape((1, 1, size)).unwrap();
    convolutions::separable_convolve(
        data,
        &new_kernel_x,
        &new_kernel_y,
        Some(&new_kernel_z),
        execution_mode,
        padding_strategy,
    )
}

#[cfg(test)]
mod tests {

    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_generate_gaussian_kernel_1d() {
        let expected_kernel = Array1::from(vec![
            0.24197072451914337,
            0.3989422804014327,
            0.24197072451914337,
        ]);
        let kernel = generate_gaussian_kernel_1d(1.0, Some(3));
        println!("{:?}", kernel);
        assert!(kernel
            .iter()
            .zip(expected_kernel.iter())
            .all(|(a, b)| (f64::abs(a - b) < 1e-6)));
    }

    #[test]
    fn test_gaussian_filter() {
        let data = Array2::from(vec![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]);

        let expected_result = Array2::from(vec![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.0,
                0.05854983152431917,
                0.09653235263005391,
                0.05854983152431917,
                0.0,
            ],
            [
                0.0,
                0.09653235263005391,
                0.15915494309189535,
                0.09653235263005391,
                0.0,
            ],
            [
                0.0,
                0.05854983152431917,
                0.09653235263005391,
                0.05854983152431917,
                0.0,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]);

        let sigma = 1.0;
        let size = Some(3);
        let padding_strategy = ConvPaddingStrategy::Same(ndarray_ndimage::PadMode::Symmetric);
        let execution_mode = ExecutionMode::Parallel;
        let result = gaussian_filter2d(&data, sigma, size, &padding_strategy, &execution_mode);
        assert!(result
            .iter()
            .zip(expected_result.iter())
            .all(|(a, b)| (f64::abs(a - b) < 1e-6)));
    }
}
