use crate::error::LayoutError;
use crate::math_utils::{
    check_radices, digits_to_int, int_to_digits, invert_permutation, mat_inv_mod, mat_vec_mod,
};

/// A closed-set enum of all digit-level transformations.
///
/// Each variant preserves the mixed-radix structure of the digit stream.
/// Constructors validate all invariants and return `Result`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Transform {
    Identity {
        radices: Vec<usize>,
    },
    Refactor {
        src_radices: Vec<usize>,
        dst_radices: Vec<usize>,
    },
    Permute {
        src_radices: Vec<usize>,
        /// `order[i]` = source digit index that feeds output position `i`.
        order: Vec<usize>,
    },
    /// Projects (drops) digits not in `keep`. Non-invertible — use `kernel_size` for the fiber.
    Project {
        src_radices: Vec<usize>,
        keep: Vec<usize>,
    },
    /// Embeds src digits into a larger dst digit stream at `positions`, filling the rest with `fill`.
    Embed {
        src_radices: Vec<usize>,
        dst_radices: Vec<usize>,
        positions: Vec<usize>,
        fill: Vec<usize>,
    },
    /// Applies an invertible affine map over GF(p) on the selected digit positions.
    BlockAffine {
        radices: Vec<usize>,
        positions: Vec<usize>,
        matrix: Vec<Vec<usize>>,
        offset: Vec<usize>,
    },
    /// Sequential composition. Always non-empty; use `Transform::compose` to construct.
    Compose(Vec<Transform>),
}

// ── Constructors ─────────────────────────────────────────────────────────────

impl Transform {
    pub fn identity(radices: Vec<usize>) -> Result<Self, LayoutError> {
        if !radices.is_empty() {
            check_radices(&radices)?;
        }
        Ok(Transform::Identity { radices })
    }

    pub fn refactor(
        src_radices: Vec<usize>,
        dst_radices: Vec<usize>,
    ) -> Result<Self, LayoutError> {
        let src_prod: usize = if src_radices.is_empty() {
            1
        } else {
            check_radices(&src_radices)?;
            src_radices.iter().product()
        };
        let dst_prod: usize = if dst_radices.is_empty() {
            1
        } else {
            check_radices(&dst_radices)?;
            dst_radices.iter().product()
        };
        if src_prod != dst_prod {
            return Err(LayoutError::MathError(
                "Refactor requires equal total cardinality".to_string(),
            ));
        }
        Ok(Transform::Refactor { src_radices, dst_radices })
    }

    pub fn permute(src_radices: Vec<usize>, order: Vec<usize>) -> Result<Self, LayoutError> {
        if !src_radices.is_empty() {
            check_radices(&src_radices)?;
        }
        let n = src_radices.len();
        let mut sorted = order.clone();
        sorted.sort_unstable();
        if sorted != (0..n).collect::<Vec<_>>() {
            return Err(LayoutError::InvalidAxisOrder);
        }
        Ok(Transform::Permute { src_radices, order })
    }

    pub fn project(src_radices: Vec<usize>, keep: Vec<usize>) -> Result<Self, LayoutError> {
        if !src_radices.is_empty() {
            check_radices(&src_radices)?;
        }
        let n = src_radices.len();
        let mut seen = std::collections::HashSet::new();
        for &i in &keep {
            if i >= n {
                return Err(LayoutError::OutOfBounds(format!(
                    "keep index {} out of range [0,{})",
                    i, n
                )));
            }
            if !seen.insert(i) {
                return Err(LayoutError::MathError(format!(
                    "keep index {} is not unique",
                    i
                )));
            }
        }
        Ok(Transform::Project { src_radices, keep })
    }

    pub fn embed(
        src_radices: Vec<usize>,
        dst_radices: Vec<usize>,
        positions: Vec<usize>,
        fill: Option<Vec<usize>>,
    ) -> Result<Self, LayoutError> {
        if !src_radices.is_empty() {
            check_radices(&src_radices)?;
        }
        if !dst_radices.is_empty() {
            check_radices(&dst_radices)?;
        }
        if positions.len() != src_radices.len() {
            return Err(LayoutError::LengthMismatch(
                "positions length must equal src_radices length".to_string(),
            ));
        }
        let m = dst_radices.len();
        let mut seen = std::collections::HashSet::new();
        for (src_i, &dst_p) in positions.iter().enumerate() {
            if dst_p >= m {
                return Err(LayoutError::OutOfBounds(format!(
                    "position {} out of range [0,{})",
                    dst_p, m
                )));
            }
            if !seen.insert(dst_p) {
                return Err(LayoutError::MathError(format!(
                    "position {} is not unique",
                    dst_p
                )));
            }
            if src_radices[src_i] != dst_radices[dst_p] {
                return Err(LayoutError::MathError(format!(
                    "Radix mismatch at src[{}]={} vs dst[{}]={}",
                    src_i, src_radices[src_i], dst_p, dst_radices[dst_p]
                )));
            }
        }
        let fill = fill.unwrap_or_else(|| vec![0; m]);
        if fill.len() != m {
            return Err(LayoutError::LengthMismatch("fill length mismatch".to_string()));
        }
        for (&d, &r) in fill.iter().zip(dst_radices.iter()) {
            if d >= r {
                return Err(LayoutError::OutOfBounds(format!(
                    "fill digit {} out of range for radix {}",
                    d, r
                )));
            }
        }
        Ok(Transform::Embed { src_radices, dst_radices, positions, fill })
    }

    pub fn block_affine(
        radices: Vec<usize>,
        positions: Vec<usize>,
        matrix: Vec<Vec<usize>>,
        offset: Option<Vec<usize>>,
    ) -> Result<Self, LayoutError> {
        if !radices.is_empty() {
            check_radices(&radices)?;
        }
        if positions.is_empty() {
            return Err(LayoutError::MathError("positions cannot be empty".to_string()));
        }
        let n = radices.len();
        let mut seen = std::collections::HashSet::new();
        for &p in &positions {
            if p >= n {
                return Err(LayoutError::OutOfBounds(format!(
                    "position {} out of range [0,{})",
                    p, n
                )));
            }
            if !seen.insert(p) {
                return Err(LayoutError::MathError(format!(
                    "position {} is not unique",
                    p
                )));
            }
        }
        let radix_set: std::collections::HashSet<usize> =
            positions.iter().map(|&p| radices[p]).collect();
        if radix_set.len() != 1 {
            return Err(LayoutError::MathError(
                "All selected positions must have the same radix".to_string(),
            ));
        }
        let radix = radices[positions[0]];
        let k = positions.len();
        if matrix.len() != k || matrix.iter().any(|row| row.len() != k) {
            return Err(LayoutError::MathError(format!(
                "matrix must be {}x{}",
                k, k
            )));
        }
        // Verify invertibility upfront.
        mat_inv_mod(&matrix, radix)?;

        let offset = offset.unwrap_or_else(|| vec![0; k]);
        if offset.len() != k {
            return Err(LayoutError::LengthMismatch("offset length mismatch".to_string()));
        }

        Ok(Transform::BlockAffine { radices, positions, matrix, offset })
    }

    pub fn compose(pieces: Vec<Transform>) -> Result<Self, LayoutError> {
        if pieces.is_empty() {
            return Err(LayoutError::MathError(
                "Compose requires at least one piece".to_string(),
            ));
        }
        for pair in pieces.windows(2) {
            if pair[0].dst_radices() != pair[1].src_radices() {
                return Err(LayoutError::LengthMismatch(
                    "Transform chain radices mismatch".to_string(),
                ));
            }
        }
        Ok(Transform::Compose(pieces))
    }
}

// ── Core interface ────────────────────────────────────────────────────────────

impl Transform {
    pub fn src_radices(&self) -> Vec<usize> {
        match self {
            Transform::Identity { radices } => radices.clone(),
            Transform::Refactor { src_radices, .. } => src_radices.clone(),
            Transform::Permute { src_radices, .. } => src_radices.clone(),
            Transform::Project { src_radices, .. } => src_radices.clone(),
            Transform::Embed { src_radices, .. } => src_radices.clone(),
            Transform::BlockAffine { radices, .. } => radices.clone(),
            Transform::Compose(pieces) => pieces[0].src_radices(),
        }
    }

    pub fn dst_radices(&self) -> Vec<usize> {
        match self {
            Transform::Identity { radices } => radices.clone(),
            Transform::Refactor { dst_radices, .. } => dst_radices.clone(),
            Transform::Permute { src_radices, order } => {
                order.iter().map(|&i| src_radices[i]).collect()
            }
            Transform::Project { src_radices, keep } => {
                keep.iter().map(|&i| src_radices[i]).collect()
            }
            Transform::Embed { dst_radices, .. } => dst_radices.clone(),
            Transform::BlockAffine { radices, .. } => radices.clone(),
            Transform::Compose(pieces) => pieces.last().unwrap().dst_radices(),
        }
    }

    pub fn apply(&self, digits: &[usize]) -> Result<Vec<usize>, LayoutError> {
        match self {
            Transform::Identity { radices } => {
                if digits.len() != radices.len() {
                    return Err(LayoutError::LengthMismatch("digit length mismatch".to_string()));
                }
                Ok(digits.to_vec())
            }

            Transform::Refactor { src_radices, dst_radices } => {
                let x = if src_radices.is_empty() {
                    0
                } else {
                    digits_to_int(digits, src_radices)?
                };
                if dst_radices.is_empty() {
                    Ok(vec![])
                } else {
                    int_to_digits(x, dst_radices)
                }
            }

            Transform::Permute { src_radices, order } => {
                if digits.len() != src_radices.len() {
                    return Err(LayoutError::LengthMismatch("digit length mismatch".to_string()));
                }
                Ok(order.iter().map(|&i| digits[i]).collect())
            }

            Transform::Project { src_radices, keep } => {
                if digits.len() != src_radices.len() {
                    return Err(LayoutError::LengthMismatch("digit length mismatch".to_string()));
                }
                Ok(keep.iter().map(|&i| digits[i]).collect())
            }

            Transform::Embed { src_radices, dst_radices: _, positions, fill } => {
                if digits.len() != src_radices.len() {
                    return Err(LayoutError::LengthMismatch("digit length mismatch".to_string()));
                }
                let mut out = fill.clone();
                for (src_i, &dst_p) in positions.iter().enumerate() {
                    out[dst_p] = digits[src_i];
                }
                Ok(out)
            }

            Transform::BlockAffine { radices, positions, matrix, offset } => {
                if digits.len() != radices.len() {
                    return Err(LayoutError::LengthMismatch("digit length mismatch".to_string()));
                }
                let radix = radices[positions[0]];
                let x: Vec<usize> = positions.iter().map(|&p| digits[p]).collect();
                let mut y = mat_vec_mod(matrix, &x, radix);
                for (yi, &ci) in y.iter_mut().zip(offset.iter()) {
                    *yi = (*yi + ci) % radix;
                }
                let mut out = digits.to_vec();
                for (&p, v) in positions.iter().zip(y) {
                    out[p] = v;
                }
                Ok(out)
            }

            Transform::Compose(pieces) => {
                let mut x = digits.to_vec();
                for p in pieces {
                    x = p.apply(&x)?;
                }
                Ok(x)
            }
        }
    }

    pub fn invert(&self) -> Result<Transform, LayoutError> {
        match self {
            Transform::Identity { .. } => Ok(self.clone()),

            Transform::Refactor { src_radices, dst_radices } => {
                Ok(Transform::Refactor {
                    src_radices: dst_radices.clone(),
                    dst_radices: src_radices.clone(),
                })
            }

            Transform::Permute { src_radices, order } => {
                let inv_order = invert_permutation(order);
                let new_src: Vec<usize> = order.iter().map(|&i| src_radices[i]).collect();
                Ok(Transform::Permute { src_radices: new_src, order: inv_order })
            }

            Transform::Project { .. } => Err(LayoutError::TransformIncompatibility(
                "Project is not invertible".to_string(),
            )),

            Transform::Embed { .. } => Err(LayoutError::TransformIncompatibility(
                "Embed is not directly invertible".to_string(),
            )),

            Transform::BlockAffine { radices, positions, matrix, offset } => {
                let radix = radices[positions[0]];
                let m_inv = mat_inv_mod(matrix, radix)?;
                let neg_offset: Vec<usize> =
                    offset.iter().map(|&c| (radix - c % radix) % radix).collect();
                let offset_inv = mat_vec_mod(&m_inv, &neg_offset, radix);
                Ok(Transform::BlockAffine {
                    radices: radices.clone(),
                    positions: positions.clone(),
                    matrix: m_inv,
                    offset: offset_inv,
                })
            }

            Transform::Compose(pieces) => {
                let inv_pieces: Result<Vec<_>, _> =
                    pieces.iter().rev().map(|p| p.invert()).collect();
                Ok(Transform::Compose(inv_pieces?))
            }
        }
    }

    /// Collapses adjacent fusible transforms and removes identities.
    pub fn simplify(self) -> Transform {
        let pieces = match self {
            Transform::Compose(ps) => ps,
            other => return other,
        };

        // Flatten nested Compose, simplify leaves.
        let flat: Vec<Transform> = pieces
            .into_iter()
            .flat_map(|p| match p.simplify() {
                Transform::Compose(inner) => inner,
                other => vec![other],
            })
            .collect();

        // Preserve original src_radices so a fully-cancelled chain returns the
        // correct identity rather than an empty one.
        let orig_src_radices = flat.first().map(|p| p.src_radices()).unwrap_or_default();

        let mut out: Vec<Transform> = Vec::new();
        for p in flat {
            // Skip identities.
            if matches!(&p, Transform::Identity { radices } if radices.is_empty()
                || true /* any identity is a no-op */)
            {
                if let Transform::Identity { .. } = &p {
                    continue;
                }
            }

            match out.last_mut() {
                // Fuse Refactor + Refactor.
                Some(Transform::Refactor { dst_radices: prev_dst, .. })
                    if matches!(&p, Transform::Refactor { src_radices, .. }
                        if src_radices == prev_dst) =>
                {
                    let new_dst = match &p {
                        Transform::Refactor { dst_radices, .. } => dst_radices.clone(),
                        _ => unreachable!(),
                    };
                    if let Some(Transform::Refactor { dst_radices, .. }) = out.last_mut() {
                        *dst_radices = new_dst;
                    }
                }

                // Fuse Permute + Permute: composed[i] = prev_order[next_order[i]].
                // src_radices stays as the first permute's src — only order changes.
                Some(Transform::Permute { order: prev_order, .. })
                    if matches!(&p, Transform::Permute { .. }) =>
                {
                    let next_order = match &p {
                        Transform::Permute { order, .. } => order.clone(),
                        _ => unreachable!(),
                    };
                    let composed: Vec<usize> =
                        next_order.iter().map(|&i| prev_order[i]).collect();
                    if let Some(Transform::Permute { order, .. }) = out.last_mut() {
                        *order = composed;
                    }
                }

                // Try to cancel inverse pairs.
                Some(prev) => {
                    let cancel = prev
                        .invert()
                        .ok()
                        .map(|inv| inv.ext_equal(&p).unwrap_or(false))
                        .unwrap_or(false);
                    if cancel {
                        out.pop();
                    } else {
                        out.push(p);
                    }
                }

                None => {
                    out.push(p);
                }
            }
        }

        match out.len() {
            0 => Transform::Identity { radices: orig_src_radices },
            1 => out.remove(0),
            _ => Transform::Compose(out),
        }
    }

    /// Sequential composition: `self` then `other`.
    pub fn then(self, other: Transform) -> Result<Transform, LayoutError> {
        if self.dst_radices() != other.src_radices() {
            return Err(LayoutError::LengthMismatch(
                "Transform chain radices mismatch".to_string(),
            ));
        }
        Ok(Transform::Compose(vec![self, other]).simplify())
    }

    /// Number of source elements that map to each destination element (fiber size).
    pub fn kernel_size(&self) -> usize {
        match self {
            Transform::Identity { .. }
            | Transform::Refactor { .. }
            | Transform::Permute { .. }
            | Transform::Embed { .. }
            | Transform::BlockAffine { .. } => 1,

            Transform::Project { src_radices, keep } => {
                let keep_set: std::collections::HashSet<usize> =
                    keep.iter().copied().collect();
                src_radices
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| !keep_set.contains(i))
                    .map(|(_, &r)| r)
                    .product()
            }

            Transform::Compose(pieces) => pieces.iter().map(|p| p.kernel_size()).product(),
        }
    }

    /// Extensional equality: same inputs produce same outputs for all digit vectors.
    pub fn ext_equal(&self, other: &Transform) -> Result<bool, LayoutError> {
        if self.src_radices() != other.src_radices()
            || self.dst_radices() != other.dst_radices()
        {
            return Ok(false);
        }
        let src = self.src_radices();
        let total: usize = if src.is_empty() { 1 } else { src.iter().product() };
        for x in 0..total {
            let d = if src.is_empty() {
                vec![]
            } else {
                int_to_digits(x, &src)?
            };
            if self.apply(&d)? != other.apply(&d)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_roundtrip() {
        let t = Transform::identity(vec![2, 3]).unwrap();
        assert_eq!(t.apply(&[1, 2]).unwrap(), vec![1, 2]);
        assert_eq!(t.src_radices(), vec![2, 3]);
        assert_eq!(t.dst_radices(), vec![2, 3]);
    }

    #[test]
    fn test_refactor() {
        // Flatten [2,3] -> [6]: same total cardinality
        let t = Transform::refactor(vec![2, 3], vec![6]).unwrap();
        // digit (1,2) = 1 + 2*2 = 5 -> digit (5,) in base 6
        assert_eq!(t.apply(&[1, 2]).unwrap(), vec![5]);
        let inv = t.invert().unwrap();
        assert_eq!(inv.apply(&[5]).unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_permute() {
        let t = Transform::permute(vec![2, 3, 5], vec![2, 0, 1]).unwrap();
        // output[0] = input[2], output[1] = input[0], output[2] = input[1]
        assert_eq!(t.apply(&[0, 1, 4]).unwrap(), vec![4, 0, 1]);
        let inv = t.invert().unwrap();
        assert!(t.then(inv).unwrap().ext_equal(&Transform::identity(vec![2, 3, 5]).unwrap()).unwrap());
    }

    #[test]
    fn test_project_kernel_size() {
        let t = Transform::project(vec![2, 3, 5], vec![0, 2]).unwrap();
        assert_eq!(t.apply(&[1, 2, 4]).unwrap(), vec![1, 4]);
        assert_eq!(t.kernel_size(), 3);
    }

    #[test]
    fn test_embed() {
        let t = Transform::embed(vec![2, 3], vec![2, 5, 3], vec![0, 2], None).unwrap();
        // src[0] -> dst[0], src[1] -> dst[2], dst[1] gets fill=0
        assert_eq!(t.apply(&[1, 2]).unwrap(), vec![1, 0, 2]);
    }

    #[test]
    fn test_block_affine_invert() {
        // XOR-swap on two bits: M=[[0,1],[1,0]] mod 2
        let t = Transform::block_affine(
            vec![2, 2],
            vec![0, 1],
            vec![vec![0, 1], vec![1, 0]],
            None,
        )
        .unwrap();
        assert_eq!(t.apply(&[1, 0]).unwrap(), vec![0, 1]);
        let inv = t.invert().unwrap();
        assert!(t.then(inv).unwrap().ext_equal(&Transform::identity(vec![2, 2]).unwrap()).unwrap());
    }

    #[test]
    fn test_compose_simplify_fuse_refactor() {
        let a = Transform::refactor(vec![2, 3], vec![6]).unwrap();
        let b = Transform::refactor(vec![6], vec![2, 3]).unwrap();
        let composed = a.then(b).unwrap();
        // Should simplify back to identity-like (Refactor [2,3]->[2,3])
        match &composed {
            Transform::Refactor { src_radices, dst_radices } => {
                assert_eq!(src_radices, dst_radices);
            }
            Transform::Identity { .. } => {}
            _ => panic!("Expected Refactor or Identity after fusion, got {:?}", composed),
        }
    }

    #[test]
    fn test_compose_cancel_inverse() {
        let t = Transform::permute(vec![2, 3], vec![1, 0]).unwrap();
        let inv = t.invert().unwrap();
        let composed = t.then(inv).unwrap();
        // t maps [2,3]->[3,2], inv maps [3,2]->[2,3], so composed is identity on [2,3].
        assert!(composed
            .ext_equal(&Transform::identity(vec![2, 3]).unwrap())
            .unwrap());
    }
}
