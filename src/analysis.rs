use crate::error::LayoutError;
use crate::layout::RankedDigitLayout;
use crate::transforms::Transform;

/// Runtime contiguity check: how many consecutive steps along `axis` from `base_coord`
/// map to consecutive flat destination addresses.
///
/// Prefer `analytical_contiguity` for compile-time analysis.
pub fn contiguous_run_along_axis(
    layout: &RankedDigitLayout,
    axis: usize,
    base_coord: Option<&[usize]>,
    max_steps: Option<usize>,
) -> Result<usize, LayoutError> {
    let src = &layout.src;

    if axis >= src.ndim() {
        return Err(LayoutError::OutOfBounds(format!(
            "axis {} out of range [0,{})",
            axis,
            src.ndim()
        )));
    }

    let mut base = match base_coord {
        Some(c) => c.to_vec(),
        None => vec![0; src.ndim()],
    };

    if base[axis] >= src.shape[axis] {
        return Err(LayoutError::OutOfBounds(format!(
            "base_coord[{}]={} out of range [0,{})",
            axis, base[axis], src.shape[axis]
        )));
    }

    let limit = {
        let remaining = src.shape[axis] - base[axis];
        max_steps.map(|m| m.min(remaining)).unwrap_or(remaining)
    };

    if limit == 0 {
        return Ok(0);
    }

    let mut run = 1;
    let mut prev = layout.flat_index(&base)? as isize;
    for _ in 1..limit {
        base[axis] += 1;
        let now = layout.flat_index(&base)? as isize;
        if now != prev + 1 {
            break;
        }
        run += 1;
        prev = now;
    }
    Ok(run)
}

/// Whether layout `a`'s transform chain structurally begins with layout `b`'s chain.
pub fn factor_through_refactor_prefix(
    layout: &RankedDigitLayout,
    tile: &RankedDigitLayout,
) -> bool {
    fn pieces(t: &Transform) -> Vec<&Transform> {
        match t {
            Transform::Compose(ps) => ps.iter().collect(),
            other => vec![other],
        }
    }

    let a = layout.simplify();
    let b = tile.simplify();

    if a.transform.src_radices() != b.transform.src_radices() {
        return false;
    }

    let a_pieces = pieces(&a.transform);
    let b_pieces = pieces(&b.transform);

    if b_pieces.len() > a_pieces.len() {
        return false;
    }

    for (x, y) in a_pieces.iter().zip(b_pieces.iter()) {
        if std::mem::discriminant(*x) != std::mem::discriminant(*y) {
            return false;
        }
        match x.ext_equal(y) {
            Ok(true) => {}
            _ => return false,
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builders::{identity_layout, transpose_layout};
    use crate::spaces::Space;

    fn space(shape: &[usize]) -> Space {
        Space::of(shape).unwrap()
    }

    #[test]
    fn test_contiguous_run_identity() {
        let l = identity_layout(space(&[4]), "id").unwrap();
        assert_eq!(contiguous_run_along_axis(&l, 0, None, None).unwrap(), 4);
    }

    #[test]
    fn test_contiguous_run_transposed() {
        let l = transpose_layout(space(&[2, 4]), &[1, 0], "T").unwrap();
        assert_eq!(contiguous_run_along_axis(&l, 0, None, None).unwrap(), 2);
        assert_eq!(contiguous_run_along_axis(&l, 1, None, None).unwrap(), 1);
    }

    #[test]
    fn test_factor_through_prefix_identity() {
        let a = identity_layout(space(&[4]), "a").unwrap();
        let b = identity_layout(space(&[4]), "b").unwrap();
        assert!(factor_through_refactor_prefix(&a, &b));
    }
}
