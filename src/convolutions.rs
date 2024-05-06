use ndarray::{self, Array, ArrayBase, Data, Dimension, LinalgScalar, RawDataClone, Zip};
use ndarray_ndimage::{pad, PadMode};

pub enum ExecutionMode {
    Serial,
    Parallel,
}

pub enum ConvPaddingStrategy<T> {
    Valid,
    Same(PadMode<T>),
}

fn kernel_flip<A, S, D>(kernel: &ArrayBase<S, D>) -> ArrayBase<S, D>
where
    D: Dimension,
    S: Data<Elem = A> + RawDataClone,
{
    /*
    Kernel flip is required to match the convention of convolution.
    The kernel is flipped along all axes before convolution.
    Example:
    Kernel: [[0, 0, 0], [0, 0, 1], [0, 0, 0]]
    Flipped Kernel: [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
    */

    let mut new_kernel = kernel.clone();
    for ax in kernel.axes().into_iter() {
        new_kernel.slice_axis_inplace(
            ax.axis,
            ndarray::Slice {
                start: 0,
                end: None,
                step: -1,
            },
        );
    }
    new_kernel
}

fn _convolve<A, S1, S2, D>(
    data: &ArrayBase<S1, D>,
    kernel: &ArrayBase<S2, D>,
    execution_mode: &ExecutionMode,
) -> Array<A, D>
where
    S1: Data<Elem = A> + std::marker::Sync,
    S2: Data<Elem = A> + std::marker::Sync + RawDataClone,
    A: LinalgScalar + Send + Sync,
    D: Dimension,
{
    let kernel = &kernel_flip(kernel);
    let base_iter = data.windows(kernel.dim());
    match execution_mode {
        ExecutionMode::Serial => Zip::from(base_iter).map_collect(|w| (&w * kernel).sum()),
        ExecutionMode::Parallel => Zip::from(base_iter).par_map_collect(|w| (&w * kernel).sum()),
    }
}

pub fn convolve<A, S1, S2, D>(
    data: &ArrayBase<S1, D>,
    kernel: &ArrayBase<S2, D>,
    padding_strategy: &ConvPaddingStrategy<A>,
    execution_mode: &ExecutionMode,
) -> Array<A, D>
where
    S1: Data<Elem = A> + std::marker::Sync,
    S2: Data<Elem = A> + std::marker::Sync + RawDataClone,
    A: LinalgScalar
        + Send
        + Sync
        + num_traits::FromPrimitive
        + num_traits::Num
        + std::cmp::PartialOrd,
    D: Dimension,
{
    match padding_strategy {
        ConvPaddingStrategy::Valid => _convolve(data, kernel, execution_mode),
        ConvPaddingStrategy::Same(pad_mode) => {
            let padding: Vec<[usize; 2]> = kernel
                .shape()
                .iter()
                .map(|x| match x % 2 {
                    0 => [x / 2 - 1, x / 2],
                    _ => [x / 2, x / 2],
                })
                .collect();
            let padded_image = pad(data, &padding, *pad_mode);
            _convolve(&padded_image, kernel, execution_mode)
        }
    }
}


/*
fn seperable_convolve2d<A, S1, S2, S3, D>(
    data: &ArrayBase<S1, D>,
    kernel1: &ArrayBase<S2, D>,
    kernel2: &ArrayBase<S3, D>,
    execution_mode: &ExecutionMode,
    padding_strategy: &ConvPaddingStrategy<A>,
) -> Array<A, D>
where
    S1: Data<Elem = A> + std::marker::Sync,
    S2: Data<Elem = A> + std::marker::Sync + RawDataClone,
    S3: Data<Elem = A> + std::marker::Sync + RawDataClone,
    A: LinalgScalar
        + Send
        + Sync
        + num_traits::FromPrimitive
        + num_traits::Num
        + std::cmp::PartialOrd,
    D: Dimension,
{
    let intermediate = convolve(data, kernel1, padding_strategy, execution_mode);
    convolve(&intermediate, kernel2, padding_strategy, execution_mode)
}

fn seperable_convolve3d<A, S1, S2, S3, S4, D>(
    data: &ArrayBase<S1, D>,
    kernel1: &ArrayBase<S2, D>,
    kernel2: &ArrayBase<S3, D>,
    kernel3: &ArrayBase<S4, D>,
    execution_mode: &ExecutionMode,
    padding_strategy: &ConvPaddingStrategy<A>,
) -> Array<A, D>
where
    S1: Data<Elem = A> + std::marker::Sync,
    S2: Data<Elem = A> + std::marker::Sync + RawDataClone,
    S3: Data<Elem = A> + std::marker::Sync + RawDataClone,
    S4: Data<Elem = A> + std::marker::Sync + RawDataClone,
    A: LinalgScalar
        + Send
        + Sync
        + num_traits::FromPrimitive
        + num_traits::Num
        + std::cmp::PartialOrd,
    D: Dimension,
{
    let intermediate = convolve(data, kernel1, padding_strategy, execution_mode);
    let intermediate = convolve(&intermediate, kernel2, padding_strategy, execution_mode);
    convolve(&intermediate, kernel3, padding_strategy, execution_mode)
}

 */

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;

    #[test]
    fn test_convolve() {
        let data = Array2::from(vec![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]);

        let kernel = Array2::from(vec![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]);

        let expected_result = Array2::from(vec![
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]);

        for mode in vec![ExecutionMode::Serial, ExecutionMode::Parallel] {
            let result = convolve(
                &data,
                &kernel,
                &ConvPaddingStrategy::Same(PadMode::Symmetric),
                &mode,
            );
            assert_eq!(result, expected_result);
        }

        let valid_result = Array2::from(vec![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]);

        let result = convolve(
            &data,
            &kernel,
            &ConvPaddingStrategy::Valid,
            &ExecutionMode::Serial,
        );
        assert_eq!(result, valid_result);
    }
}
