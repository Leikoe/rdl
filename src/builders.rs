use crate::error::LayoutError;
use crate::layout::{DigitSpec, RankedDigitLayout};
use crate::spaces::Space;
use crate::transforms::Transform;

pub fn identity_layout(
    space: Space,
    name: impl Into<String>,
) -> Result<RankedDigitLayout, LayoutError> {
    let spec = DigitSpec::canonical(&space.shape)?;
    let t = Transform::identity(spec.digit_radices())?;
    RankedDigitLayout::with_specs(space.clone(), spec.clone(), t, spec, space, name)
}

pub fn refactor_layout(
    src: Space,
    dst: Space,
    name: impl Into<String>,
) -> Result<RankedDigitLayout, LayoutError> {
    if src.total_size() != dst.total_size() {
        return Err(LayoutError::MathError(
            "refactor_layout requires equal total size".to_string(),
        ));
    }
    let src_spec = DigitSpec::canonical(&src.shape)?;
    let dst_spec = DigitSpec::canonical(&dst.shape)?;
    let t = Transform::refactor(src_spec.digit_radices(), dst_spec.digit_radices())?;
    RankedDigitLayout::with_specs(src, src_spec, t, dst_spec, dst, name)
}

pub fn reshape_layout(
    src: Space,
    dst_shape: &[usize],
    dst_labels: &[&str],
    name: impl Into<String>,
) -> Result<RankedDigitLayout, LayoutError> {
    let dst = Space::named(dst_shape, dst_labels)?;
    refactor_layout(src, dst, name)
}

pub fn transpose_layout(
    src: Space,
    perm: &[usize],
    name: impl Into<String>,
) -> Result<RankedDigitLayout, LayoutError> {
    let n = src.ndim();
    let mut sorted = perm.to_vec();
    sorted.sort_unstable();
    if sorted != (0..n).collect::<Vec<_>>() {
        return Err(LayoutError::InvalidAxisOrder);
    }

    let dst_shape: Vec<usize> = perm.iter().map(|&i| src.shape[i]).collect();
    let dst_labels: Vec<String> = if src.labels.is_empty() {
        vec![]
    } else {
        perm.iter().map(|&i| src.labels[i].clone()).collect()
    };
    let dst = Space::new(dst_shape, dst_labels)?;

    let src_spec = DigitSpec::canonical(&src.shape)?;
    let dst_facs: Vec<Vec<usize>> = perm.iter().map(|&i| src_spec.factorizations[i].clone()).collect();
    let dst_spec = DigitSpec {
        factorizations: dst_facs,
        axis_order: (0..n).rev().collect(),
    };

    let src_slices = src_spec.axis_digit_slices(n);
    let dst_slices = dst_spec.axis_digit_slices(n);
    let total_digits = src_spec.digit_radices().len();
    let mut order = vec![0usize; total_digits];
    for (dst_axis, &src_axis) in perm.iter().enumerate() {
        for (d, s) in dst_slices[dst_axis].clone().zip(src_slices[src_axis].clone()) {
            order[d] = s;
        }
    }

    let t = Transform::permute(src_spec.digit_radices(), order)?;
    RankedDigitLayout::with_specs(src, src_spec, t, dst_spec, dst, name)
}

pub fn project_layout(
    src: Space,
    keep_digit_indices: &[usize],
    dst: Space,
    name: impl Into<String>,
) -> Result<RankedDigitLayout, LayoutError> {
    let src_spec = DigitSpec::canonical(&src.shape)?;
    let dst_spec = DigitSpec::canonical(&dst.shape)?;
    let t = Transform::project(src_spec.digit_radices(), keep_digit_indices.to_vec())?;
    RankedDigitLayout::with_specs(src, src_spec, t, dst_spec, dst, name)
}

pub fn embed_layout(
    src: Space,
    dst: Space,
    positions: &[usize],
    fill: Option<&[usize]>,
    name: impl Into<String>,
) -> Result<RankedDigitLayout, LayoutError> {
    let src_spec = DigitSpec::canonical(&src.shape)?;
    let dst_spec = DigitSpec::canonical(&dst.shape)?;
    let t = Transform::embed(
        src_spec.digit_radices(),
        dst_spec.digit_radices(),
        positions.to_vec(),
        fill.map(|f| f.to_vec()),
    )?;
    RankedDigitLayout::with_specs(src, src_spec, t, dst_spec, dst, name)
}

pub fn block_affine_layout(
    space: Space,
    positions: &[usize],
    matrix: Vec<Vec<usize>>,
    offset: Option<Vec<usize>>,
    name: impl Into<String>,
) -> Result<RankedDigitLayout, LayoutError> {
    let spec = DigitSpec::canonical(&space.shape)?;
    let t = Transform::block_affine(spec.digit_radices(), positions.to_vec(), matrix, offset)?;
    RankedDigitLayout::with_specs(space.clone(), spec.clone(), t, spec, space, name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math_utils::iterate_coords;

    fn space(shape: &[usize]) -> Space {
        Space::of(shape).unwrap()
    }

    #[test]
    fn test_identity_layout() {
        let s = space(&[2, 3]);
        let l = identity_layout(s.clone(), "id").unwrap();
        for coord in iterate_coords(&[2, 3]) {
            assert_eq!(l.map_coord(&coord).unwrap(), coord);
        }
    }

    #[test]
    fn test_reshape_layout() {
        let src = space(&[4]);
        let l = reshape_layout(src, &[2, 2], &[], "reshape").unwrap();
        assert_eq!(l.map_coord(&[0]).unwrap(), vec![0, 0]);
        assert_eq!(l.map_coord(&[3]).unwrap(), vec![1, 1]);
    }

    #[test]
    fn test_transpose_layout() {
        let src = space(&[2, 3]);
        let l = transpose_layout(src, &[1, 0], "T").unwrap();
        assert_eq!(l.map_coord(&[0, 2]).unwrap(), vec![2, 0]);
        assert_eq!(l.map_coord(&[1, 1]).unwrap(), vec![1, 1]);
    }

    #[test]
    fn test_block_affine_swizzle() {
        let s = Space::of(&[4]).unwrap();
        let l = block_affine_layout(s, &[0, 1], vec![vec![0, 1], vec![1, 0]], None, "xor").unwrap();
        assert_eq!(l.map_coord(&[2]).unwrap(), vec![1]);
    }
}
