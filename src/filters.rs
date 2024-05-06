use ndarray::{self, Array, Array1, ArrayBase, Data, Dimension, LinalgScalar};

use crate::convolutions::{ConvPaddingStrategy, ExecutionMode};

/// Create a 1D Gaussian kernel centered at the middle of the kernel.
fn generate_gaussian_kernel_1d(sigma: f64, size: Option<usize>) -> Array<f64, ndarray::Ix1> {
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

fn gaussian_filter<A, S, D>(
    data: &ArrayBase<S, D>,
    sigma: f64,
    size: Option<usize>,
    padding_strategy: &ConvPaddingStrategy<A>,
    execution_mode: &ExecutionMode,
) -> Array<A, D>
where
    S: Data<Elem = A> + std::marker::Sync,
    A: LinalgScalar
        + Send
        + Sync
        + num_traits::FromPrimitive
        + num_traits::Num
        + std::cmp::PartialOrd,
    D: Dimension,
{
    let kernel: Array1<f64> = generate_gaussian_kernel_1d(sigma, size);
    let size = kernel.len();
    let new_kernel = kernel.into_shape((1, size)).unwrap();
    println!("{:?}", data.shape());
    println!("{:?}", new_kernel.shape());

    !todo!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_generate_gaussian_kernel_1d() {
        let kernel = generate_gaussian_kernel_1d(1.0, None);
        println!("{:?}", kernel);
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

        let sigma = 1.0;
        let size = Some(3);
        let padding_strategy = ConvPaddingStrategy::Same(ndarray_ndimage::PadMode::Symmetric);
        let execution_mode = ExecutionMode::Parallel;
        let result = gaussian_filter(&data, sigma, size, &padding_strategy, &execution_mode);
        true;
    }
}
