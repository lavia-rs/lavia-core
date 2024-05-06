use ndarray::{self, Array, ArrayBase, Data, Dimension, Ix2, Ix3, LinalgScalar, RawDataClone, Zip};
use ndarray_ndimage::{pad, PadMode};

/// Execution mode for the convolution operation.
pub enum ExecutionMode {
    /// Execute the convolution operation in serial.
    Serial,
    /// Execute the convolution operation in parallel using rayon.
    Parallel,
}

/// Padding strategy for the convolution operation.
pub enum ConvPaddingStrategy<T> {
    /// The output size will be smaller than the input size and no padding will be applied.
    Valid,
    /// The output size will be the same as the input size and padding will be applied.
    /// The padding strategy can be specified using the PadMode enum.
    Same(PadMode<T>),
}

/// Kernel flip [::-1] for all axis is required to match the convention of convolution.
fn kernel_flip<A, S, D>(kernel: &ArrayBase<S, D>) -> ArrayBase<S, D>
where
    D: Dimension,
    S: Data<Elem = A> + RawDataClone,
{
    // Kernel flip is required to match the convention of convolution.
    // The kernel is flipped along all axes before convolution.
    // Example:
    // Kernel:
    //        [[0, 0, 0],
    //         [0, 0, 1],
    //         [0, 0, 0]]
    //
    // Flipped Kernel:
    //        [[0, 0, 0],
    //         [1, 0, 0],
    //         [0, 0, 0]]

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

/// Extend the kernel to match the shape of the input data.
/// This is used for broadcasting the kernels during Nd convolutions.
fn extend_kernel<A, S1, S2, D, D2>(
    data: &ArrayBase<S1, D>,
    kernel: &ArrayBase<S2, D2>,
) -> Array<A, D>
where
    S1: Data<Elem = A>,
    S2: Data<Elem = A> + RawDataClone,
    A: LinalgScalar,
    D: Dimension,
    D2: Dimension,
{
    let shape_diff = data.ndim() - kernel.ndim();
    let mut kernel_shape_padding = vec![1; shape_diff];
    kernel_shape_padding.extend(kernel.shape());

    let kernel = kernel.clone().into_shape(kernel_shape_padding).unwrap();
    let kernel: ArrayBase<S2, D> = kernel.into_dimensionality::<D>().unwrap();
    kernel.into_owned()
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

/// Convolve the input data with the kernel using the specified padding strategy and execution mode.
///
/// # Examples
/// ```
/// let data = Array2::from(vec![[0.0, 0.0, 0.0, 0.0, 0.0],
///                              [0.0, 0.0, 0.0, 0.0, 0.0],
///                              [0.0, 0.0, 1.0, 0.0, 0.0],
///                              [0.0, 0.0, 0.0, 0.0, 0.0],
///                              [0.0, 0.0, 0.0, 0.0, 0.0]]);
/// let kernel = Array2::from(vec![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]);
///
/// let result = convolve(&data, &kernel, &ConvPaddingStrategy::Same(PadMode::Symmetric), &ExecutionMode::Serial);
/// ```
///
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

/// Convolve the input data with the kernel using the specified padding strategy and execution mode.
/// This function broadcasts the kernel to match the shape of the input data.
/// By convention, the kernel is broadcasted along all fists axes to match the shape of the input data.
pub fn broadcast_convolve<A, S1, S2, D1, D2>(
    data: &ArrayBase<S1, D1>,
    kernel: &ArrayBase<S2, D2>,
    padding_strategy: &ConvPaddingStrategy<A>,
    execution_mode: &ExecutionMode,
) -> Array<A, D1>
where
    S1: Data<Elem = A> + std::marker::Sync,
    S2: Data<Elem = A> + std::marker::Sync + RawDataClone,
    A: LinalgScalar
        + Send
        + Sync
        + num_traits::FromPrimitive
        + num_traits::Num
        + std::cmp::PartialOrd,
    D1: Dimension,
    D2: Dimension,
{
    let kernel = &extend_kernel(data, kernel);
    convolve(data, kernel, padding_strategy, execution_mode)
}

/// Wrapper function around broadcast_convolve for 2D data.
pub fn convolve2d<A, S1, S2, D>(
    data: &ArrayBase<S1, D>,
    kernel: &ArrayBase<S2, Ix2>,
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
    broadcast_convolve(data, kernel, padding_strategy, execution_mode)
}

/// Wrapper function around broadcast_convolve for 3D data.
pub fn convolve3d<A, S1, S2, D>(
    data: &ArrayBase<S1, D>,
    kernel: &ArrayBase<S2, Ix3>,
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
    let kernel = extend_kernel(data, kernel);
    convolve(data, &kernel, padding_strategy, execution_mode)
}

// Separable convolutions are a special case of convolutions where the kernel is separable into multiple 1D kernels.
// This is can be more efficient than convolving with a single 2D kernel (in particular for large kernels).
// This is implemented only for 2D and 3D kernels.
pub fn separable_convolve<A, S1, S2, D>(
    data: &ArrayBase<S1, D>,
    kernels: &[ArrayBase<S2, D>],
    execution_mode: &ExecutionMode,
    padding_strategy: &ConvPaddingStrategy<A>,
) -> Array<A, D>
where
    S1: Data<Elem = A> + std::marker::Sync + RawDataClone,
    S2: Data<Elem = A> + std::marker::Sync + RawDataClone,
    A: LinalgScalar
        + Send
        + Sync
        + num_traits::FromPrimitive
        + num_traits::Num
        + std::cmp::PartialOrd,
    D: Dimension,
{
    assert!(
        kernels.len() == 2 || kernels.len() == 3,
        "Kernel length must be 2 or 3"
    );
    let intermediate = convolve(data, &kernels[0], padding_strategy, execution_mode);
    let intermediate = convolve(&intermediate, &kernels[1], padding_strategy, execution_mode);
    if kernels.len() == 2 {
        return intermediate;
    }

    convolve(&intermediate, &kernels[2], padding_strategy, execution_mode)
}


#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, Array3};

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

    #[test]
    fn test_convolve2d() {
        let data = Array3::from(vec![
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]);

        let kernel = Array2::from(vec![[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]);

        let expected_result = Array3::from(vec![
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
        ]);

        let result = convolve2d(
            &data,
            &kernel,
            &ConvPaddingStrategy::Same(PadMode::Symmetric),
            &ExecutionMode::Serial,
        );
        assert_eq!(result, expected_result);
    }
}
