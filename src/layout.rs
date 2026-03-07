use crate::error::LayoutError;
use crate::math_utils::{canonical_factorization, digits_to_int, int_to_digits, iterate_coords};
use crate::spaces::Space;
use crate::transforms::Transform;
use std::ops::Range;

// ── DigitSpec ─────────────────────────────────────────────────────────────────
//
// Private: describes how coordinates of a Space convert to/from a flat digit stream.
// A Space is just a coordinate domain; DigitSpec is the representation choice.

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct DigitSpec {
    pub(crate) factorizations: Vec<Vec<usize>>, // per-axis prime factors; empty for size-1 axes
    pub(crate) axis_order: Vec<usize>,          // which axis's digits come first in the stream
}

impl DigitSpec {
    /// Canonical spec: binary-first factorization, last axis leads the stream (right-major).
    pub(crate) fn canonical(shape: &[usize]) -> Result<Self, LayoutError> {
        let ndim = shape.len();
        let factorizations = shape
            .iter()
            .map(|&n| if n == 1 { Ok(vec![]) } else { canonical_factorization(n, true) })
            .collect::<Result<Vec<_>, _>>()?;
        let axis_order = (0..ndim).rev().collect();
        Ok(Self { factorizations, axis_order })
    }

    /// Like canonical but with an explicit axis ordering (e.g. for left-major).
    pub(crate) fn canonical_with_order(shape: &[usize], axis_order: Vec<usize>) -> Result<Self, LayoutError> {
        let factorizations = shape
            .iter()
            .map(|&n| if n == 1 { Ok(vec![]) } else { canonical_factorization(n, true) })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { factorizations, axis_order })
    }

    pub(crate) fn digit_radices(&self) -> Vec<usize> {
        self.axis_order
            .iter()
            .flat_map(|&a| self.factorizations[a].iter().copied())
            .collect()
    }

    pub(crate) fn num_digits(&self) -> usize {
        self.factorizations.iter().map(|f| f.len()).sum()
    }

    /// Convert a coordinate to a flat digit stream.
    pub(crate) fn rank(&self, coord: &[usize], shape: &[usize]) -> Result<Vec<usize>, LayoutError> {
        if coord.len() != shape.len() {
            return Err(LayoutError::LengthMismatch("coord/ndim".to_string()));
        }
        let mut out = Vec::with_capacity(self.num_digits());
        for &a in &self.axis_order {
            let x = coord[a];
            if x >= shape[a] {
                return Err(LayoutError::OutOfBounds(format!(
                    "axis {}: coord {} out of range [0,{})",
                    a, x, shape[a]
                )));
            }
            if !self.factorizations[a].is_empty() {
                out.extend(int_to_digits(x, &self.factorizations[a])?);
            }
        }
        Ok(out)
    }

    /// Convert a flat digit stream back to a coordinate.
    pub(crate) fn unrank(&self, digits: &[usize], shape: &[usize]) -> Result<Vec<usize>, LayoutError> {
        let mut coord = vec![0usize; shape.len()];
        let mut pos = 0;
        for &a in &self.axis_order {
            let n = self.factorizations[a].len();
            if n > 0 {
                coord[a] = digits_to_int(&digits[pos..pos + n], &self.factorizations[a])?;
                pos += n;
            }
            // size-1 axis stays at 0
        }
        Ok(coord)
    }

    /// For each axis, the range of digit positions it occupies in the flat stream.
    pub(crate) fn axis_digit_slices(&self, ndim: usize) -> Vec<Range<usize>> {
        let mut slices = vec![0..0; ndim];
        let mut pos = 0;
        for &a in &self.axis_order {
            let n = self.factorizations[a].len();
            slices[a] = pos..pos + n;
            pos += n;
        }
        slices
    }
}

// ── RankedDigitLayout ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RankedDigitLayout {
    pub src: Space,
    src_spec: DigitSpec,
    pub transform: Transform,
    dst_spec: DigitSpec,
    pub dst: Space,
    pub name: String,
}

impl RankedDigitLayout {
    /// Public constructor: derives canonical (right-major, binary-first) digit specs
    /// from the shapes of `src` and `dst`.
    pub fn new(
        src: Space,
        transform: Transform,
        dst: Space,
        name: impl Into<String>,
    ) -> Result<Self, LayoutError> {
        let src_spec = DigitSpec::canonical(&src.shape)?;
        let dst_spec = DigitSpec::canonical(&dst.shape)?;
        Self::with_specs(src, src_spec, transform, dst_spec, dst, name)
    }

    /// Internal constructor with explicit digit specs.
    pub(crate) fn with_specs(
        src: Space,
        src_spec: DigitSpec,
        transform: Transform,
        dst_spec: DigitSpec,
        dst: Space,
        name: impl Into<String>,
    ) -> Result<Self, LayoutError> {
        if src_spec.digit_radices() != transform.src_radices() {
            return Err(LayoutError::TransformIncompatibility(format!(
                "src digit radices {:?} do not match transform src radices {:?}",
                src_spec.digit_radices(),
                transform.src_radices(),
            )));
        }
        if dst_spec.digit_radices() != transform.dst_radices() {
            return Err(LayoutError::TransformIncompatibility(format!(
                "dst digit radices {:?} do not match transform dst radices {:?}",
                dst_spec.digit_radices(),
                transform.dst_radices(),
            )));
        }
        Ok(Self { src, src_spec, transform, dst_spec, dst, name: name.into() })
    }

    /// Map a source coordinate to a destination coordinate.
    pub fn map_coord(&self, coord: &[usize]) -> Result<Vec<usize>, LayoutError> {
        let digits = self.src_spec.rank(coord, &self.src.shape)?;
        let out_digits = self.transform.apply(&digits)?;
        self.dst_spec.unrank(&out_digits, &self.dst.shape)
    }

    /// Flatten the destination coordinate to a single integer.
    ///
    /// Most useful when `dst` is a 1-D flat space.
    pub fn flat_index(&self, coord: &[usize]) -> Result<usize, LayoutError> {
        let dst_coord = self.map_coord(coord)?;
        let digits = self.dst_spec.rank(&dst_coord, &self.dst.shape)?;
        digits_to_int(&digits, &self.dst_spec.digit_radices())
    }

    pub fn invert(&self) -> Result<RankedDigitLayout, LayoutError> {
        let name = if self.name.is_empty() {
            String::new()
        } else {
            format!("{}^-1", self.name)
        };
        RankedDigitLayout::with_specs(
            self.dst.clone(),
            self.dst_spec.clone(),
            self.transform.invert()?,
            self.src_spec.clone(),
            self.src.clone(),
            name,
        )
    }

    /// Sequential composition: `self` then `other`.
    pub fn then(
        &self,
        other: &RankedDigitLayout,
        name: impl Into<String>,
    ) -> Result<RankedDigitLayout, LayoutError> {
        if self.dst_spec.digit_radices() != other.src_spec.digit_radices() {
            return Err(LayoutError::TransformIncompatibility(
                "Layout composition requires compatible intermediate digit radices".to_string(),
            ));
        }
        RankedDigitLayout::with_specs(
            self.src.clone(),
            self.src_spec.clone(),
            self.transform.clone().then(other.transform.clone())?,
            other.dst_spec.clone(),
            other.dst.clone(),
            name,
        )
    }

    pub fn simplify(&self) -> RankedDigitLayout {
        RankedDigitLayout {
            src: self.src.clone(),
            src_spec: self.src_spec.clone(),
            transform: self.transform.clone().simplify(),
            dst_spec: self.dst_spec.clone(),
            dst: self.dst.clone(),
            name: self.name.clone(),
        }
    }

    /// Extensional equality: both layouts map every src coordinate identically.
    pub fn ext_equal(&self, other: &RankedDigitLayout, max_size: usize) -> Result<bool, LayoutError> {
        if self.src != other.src || self.dst != other.dst {
            return Ok(false);
        }
        if self.src.total_size() > max_size {
            return Err(LayoutError::MathError(format!(
                "Space too large for extensional equality: total={}",
                self.src.total_size()
            )));
        }
        for coord in iterate_coords(&self.src.shape) {
            if self.map_coord(&coord)? != other.map_coord(&coord)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Number of source elements that map to each destination element.
    pub fn analytical_kernel_size(&self) -> usize {
        self.transform.kernel_size()
    }

    /// Maximum contiguous vectorisation width along logical `axis`.
    pub fn analytical_contiguity(&self, axis: usize) -> Result<usize, LayoutError> {
        let ndim = self.src.ndim();
        if axis >= ndim {
            return Err(LayoutError::OutOfBounds(format!(
                "axis {} out of range [0,{})",
                axis, ndim
            )));
        }

        let src_radices = self.src_spec.digit_radices();
        let dst_radices = self.dst_spec.digit_radices();
        let slices = self.src_spec.axis_digit_slices(ndim);
        let axis_digits: Vec<usize> = (slices[axis].start..slices[axis].end).collect();

        if axis_digits.is_empty() {
            return Ok(1); // size-1 axis: trivially contiguous
        }

        // Evaluate the flat dst address at the all-zeros digit vector.
        let z0 = vec![0usize; src_radices.len()];
        let dst_z0 = self.transform.apply(&z0)?;
        let flat_z0 = digits_to_int(&dst_z0, &dst_radices)? as isize;

        let mut expected_stride: isize = 1;
        let mut width: usize = 1;

        for &d_idx in &axis_digits {
            let r = src_radices[d_idx];

            // Stride at the first step (d_idx = 1, rest = 0).
            let mut z1 = z0.clone();
            z1[d_idx] = 1;
            let flat_z1 = digits_to_int(&self.transform.apply(&z1)?, &dst_radices)? as isize;
            if flat_z1 - flat_z0 != expected_stride {
                break;
            }

            // Verify linearity across the full radix range.
            let mut z_max = z0.clone();
            z_max[d_idx] = r - 1;
            let flat_zmax =
                digits_to_int(&self.transform.apply(&z_max)?, &dst_radices)? as isize;
            if flat_zmax - flat_z0 != expected_stride * (r as isize - 1) {
                break;
            }

            width *= r;
            expected_stride *= r as isize;
        }

        Ok(width)
    }

    /// Alias for `analytical_contiguity`.
    pub fn get_contiguous_elements(&self, axis: usize) -> Result<usize, LayoutError> {
        self.analytical_contiguity(axis)
    }

    /// Logical axes that do not affect the physical address (size-1 axes or constant axes).
    pub fn get_broadcasted_dims(&self) -> Result<Vec<usize>, LayoutError> {
        let ndim = self.src.ndim();
        let src_radices = self.src_spec.digit_radices();
        let z0 = vec![0usize; src_radices.len()];
        let dst_z0 = self.transform.apply(&z0)?;
        let slices = self.src_spec.axis_digit_slices(ndim);

        let mut broadcasted = Vec::new();
        for axis in 0..ndim {
            let sl = &slices[axis];
            if sl.is_empty() {
                broadcasted.push(axis);
                continue;
            }
            let mut is_bcast = true;
            'outer: for d_idx in sl.clone() {
                let r = src_radices[d_idx];
                for v in 1..r {
                    let mut z1 = z0.clone();
                    z1[d_idx] = v;
                    if self.transform.apply(&z1)? != dst_z0 {
                        is_bcast = false;
                        break 'outer;
                    }
                }
            }
            if is_bcast {
                broadcasted.push(axis);
            }
        }
        Ok(broadcasted)
    }

    /// `self = Rest ∘ hw` — recover `Rest`.
    pub fn algebraic_divide(
        &self,
        hw: &RankedDigitLayout,
        name: impl Into<String>,
    ) -> Result<RankedDigitLayout, LayoutError> {
        hw.invert()?.then(self, name)
    }

    pub fn left_divide(
        &self,
        m1: &RankedDigitLayout,
        name: impl Into<String>,
    ) -> Result<RankedDigitLayout, LayoutError> {
        self.algebraic_divide(m1, name)
    }

    pub fn dump_table(&self, limit: usize) -> String {
        let mut rows = Vec::new();
        let mut count = 0;
        for coord in iterate_coords(&self.src.shape) {
            match self.map_coord(&coord) {
                Ok(dst) => rows.push(format!("{:?} -> {:?}", coord, dst)),
                Err(e) => rows.push(format!("{:?} -> ERROR: {}", coord, e)),
            }
            count += 1;
            if count >= limit {
                if self.src.total_size() > limit {
                    rows.push("...".to_string());
                }
                break;
            }
        }
        rows.join("\n")
    }

    pub fn summary(&self) -> String {
        let title = if self.name.is_empty() {
            "Layout".to_string()
        } else {
            format!("Layout<{}>", self.name)
        };
        format!(
            "{}\nsrc={:?} radices={:?}\ndst={:?} radices={:?}\ntransform={:?}",
            title,
            self.src.shape,
            self.src_spec.digit_radices(),
            self.dst.shape,
            self.dst_spec.digit_radices(),
            self.transform,
        )
    }
}

// ── Convenience constructors ──────────────────────────────────────────────────

impl RankedDigitLayout {
    /// Right-major layout: last axis varies fastest (C order).
    pub fn right_major(shape: &[usize]) -> Result<Self, LayoutError> {
        if shape.is_empty() {
            return Err(LayoutError::MathError("shape must be non-empty".to_string()));
        }
        let n: usize = shape.iter().product();
        let src = Space::of(shape)?;
        let dst = Space::of(&[n])?;
        let src_spec = DigitSpec::canonical(shape)?;
        let dst_spec = DigitSpec::canonical(&[n])?;
        let t = Transform::refactor(src_spec.digit_radices(), dst_spec.digit_radices())?;
        RankedDigitLayout::with_specs(src, src_spec, t, dst_spec, dst, "right_major")
    }

    /// Left-major layout: first axis varies fastest (Fortran order).
    pub fn left_major(shape: &[usize]) -> Result<Self, LayoutError> {
        if shape.is_empty() {
            return Err(LayoutError::MathError("shape must be non-empty".to_string()));
        }
        let n: usize = shape.iter().product();
        let src = Space::of(shape)?;
        let dst = Space::of(&[n])?;
        let axis_order = (0..shape.len()).collect();
        let src_spec = DigitSpec::canonical_with_order(shape, axis_order)?;
        let dst_spec = DigitSpec::canonical(&[n])?;
        let t = Transform::refactor(src_spec.digit_radices(), dst_spec.digit_radices())?;
        RankedDigitLayout::with_specs(src, src_spec, t, dst_spec, dst, "left_major")
    }
}

// ── Movement operations ───────────────────────────────────────────────────────

impl RankedDigitLayout {
    /// Reshape the logical view — element count must be preserved.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, LayoutError> {
        let new_n: usize = new_shape.iter().product();
        if new_n != self.src.total_size() {
            return Err(LayoutError::MathError(format!(
                "reshape: element count mismatch — got {new_n}, expected {}",
                self.src.total_size()
            )));
        }
        let new_src = Space::of(new_shape)?;
        let new_spec = DigitSpec::canonical(new_shape)?;
        let t = Transform::refactor(new_spec.digit_radices(), self.src_spec.digit_radices())?;
        let step = RankedDigitLayout::with_specs(
            new_src, new_spec, t, self.src_spec.clone(), self.src.clone(), "reshape",
        )?;
        step.then(self, "reshaped")
    }

    /// Permute the logical axes. `None` reverses all axes.
    pub fn transpose(&self, perm: Option<&[usize]>) -> Result<Self, LayoutError> {
        let ndim = self.src.ndim();
        let rev: Vec<usize>;
        let perm: &[usize] = match perm {
            Some(p) => p,
            None => {
                rev = (0..ndim).rev().collect();
                &rev
            }
        };

        let mut sorted = perm.to_vec();
        sorted.sort_unstable();
        if sorted != (0..ndim).collect::<Vec<_>>() {
            return Err(LayoutError::InvalidAxisOrder);
        }

        // Build the dst space (permuted shape and factorizations).
        let src = &self.src;
        let src_spec = &self.src_spec;
        let dst_shape: Vec<usize> = perm.iter().map(|&i| src.shape[i]).collect();
        let dst_facs: Vec<Vec<usize>> = perm.iter().map(|&i| src_spec.factorizations[i].clone()).collect();
        let dst_labels: Vec<String> = if src.labels.is_empty() {
            vec![]
        } else {
            perm.iter().map(|&i| src.labels[i].clone()).collect()
        };
        let dst = Space::new(dst_shape, dst_labels)?;
        // dst_spec: default right-major axis_order over the new axes.
        let dst_spec = DigitSpec {
            factorizations: dst_facs,
            axis_order: (0..ndim).rev().collect(),
        };

        // Derive the digit-level permutation order.
        let src_slices = src_spec.axis_digit_slices(ndim);
        let dst_slices = dst_spec.axis_digit_slices(ndim);
        let total_digits = src_spec.digit_radices().len();
        let mut order = vec![0usize; total_digits];
        for (dst_axis, &src_axis) in perm.iter().enumerate() {
            let src_range = src_slices[src_axis].clone();
            let dst_range = dst_slices[dst_axis].clone();
            for (d, s) in dst_range.zip(src_range) {
                order[d] = s;
            }
        }

        let t = Transform::permute(src_spec.digit_radices(), order)?;
        let step = RankedDigitLayout::with_specs(
            src.clone(), src_spec.clone(), t, dst_spec, dst, "transpose",
        )?;
        step.then(self, "transposed")
    }

    /// Apply a block-affine (e.g. XOR) swizzle over selected digits.
    pub fn swizzle(
        &self,
        positions: &[usize],
        matrix: Vec<Vec<usize>>,
        offset: Option<Vec<usize>>,
    ) -> Result<Self, LayoutError> {
        let t = Transform::block_affine(
            self.src_spec.digit_radices(),
            positions.to_vec(),
            matrix,
            offset,
        )?;
        let step = RankedDigitLayout::with_specs(
            self.src.clone(),
            self.src_spec.clone(),
            t,
            self.src_spec.clone(),
            self.src.clone(),
            "swizzle",
        )?;
        step.then(self, "swizzled")
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn space(shape: &[usize]) -> Space {
        Space::of(shape).unwrap()
    }

    #[test]
    fn test_map_coord_identity() {
        let s = space(&[2, 3]);
        let t = Transform::identity(DigitSpec::canonical(&[2, 3]).unwrap().digit_radices()).unwrap();
        let l = RankedDigitLayout::new(s.clone(), t, s.clone(), "id").unwrap();
        assert_eq!(l.map_coord(&[1, 2]).unwrap(), vec![1, 2]);
    }

    #[test]
    fn test_invert_roundtrip() {
        let s = space(&[4]);
        let d = space(&[2, 2]);
        let l = RankedDigitLayout::new(s.clone(), Transform::refactor(vec![2, 2], vec![2, 2]).unwrap(), d.clone(), "refactor").unwrap();
        let inv = l.invert().unwrap();
        let composed = l.then(&inv, "rt").unwrap();
        let id = RankedDigitLayout::new(s.clone(), Transform::identity(vec![2, 2]).unwrap(), s.clone(), "").unwrap();
        assert!(composed.ext_equal(&id, 1000).unwrap());
    }

    #[test]
    fn test_right_major_4x4() {
        let l = RankedDigitLayout::right_major(&[4, 4]).unwrap();
        assert_eq!(l.flat_index(&[0, 0]).unwrap(), 0);
        assert_eq!(l.flat_index(&[0, 3]).unwrap(), 3);
        assert_eq!(l.flat_index(&[1, 0]).unwrap(), 4);
        assert_eq!(l.flat_index(&[3, 3]).unwrap(), 15);
    }

    #[test]
    fn test_left_major_4x4() {
        let l = RankedDigitLayout::left_major(&[4, 4]).unwrap();
        assert_eq!(l.flat_index(&[0, 0]).unwrap(), 0);
        assert_eq!(l.flat_index(&[3, 0]).unwrap(), 3);
        assert_eq!(l.flat_index(&[0, 1]).unwrap(), 4);
        assert_eq!(l.flat_index(&[3, 3]).unwrap(), 15);
    }

    #[test]
    fn test_analytical_contiguity_right_major() {
        let l = RankedDigitLayout::right_major(&[4, 4]).unwrap();
        assert_eq!(l.analytical_contiguity(1).unwrap(), 4);
        assert_eq!(l.analytical_contiguity(0).unwrap(), 1);
    }

    #[test]
    fn test_transpose_method() {
        let l = RankedDigitLayout::right_major(&[4, 4]).unwrap();
        let t = l.transpose(None).unwrap();
        assert_eq!(t.flat_index(&[0, 2]).unwrap(), 8);
        assert_eq!(t.flat_index(&[3, 1]).unwrap(), 7);
    }

    #[test]
    fn test_reshape_method() {
        let l = RankedDigitLayout::right_major(&[4, 4]).unwrap();
        let r = l.reshape(&[2, 8]).unwrap();
        assert_eq!(r.src.shape, vec![2, 8]);
        let mut seen = vec![false; 16];
        for row in 0..2usize {
            for col in 0..8usize {
                seen[r.flat_index(&[row, col]).unwrap()] = true;
            }
        }
        assert!(seen.iter().all(|&x| x));
    }

    #[test]
    fn test_reshape_after_transpose() {
        let l = RankedDigitLayout::right_major(&[4, 4]).unwrap();
        let r = l.transpose(None).unwrap().reshape(&[2, 8]).unwrap();
        let mut seen = vec![false; 16];
        for row in 0..2usize {
            for col in 0..8usize {
                seen[r.flat_index(&[row, col]).unwrap()] = true;
            }
        }
        assert!(seen.iter().all(|&x| x), "bijection broken after transpose+reshape");
    }

    #[test]
    fn test_swizzle_bijection() {
        let l = RankedDigitLayout::right_major(&[8, 8]).unwrap();
        let s = l.swizzle(&[0, 3], vec![vec![1, 1], vec![0, 1]], None).unwrap();
        let mut seen = vec![false; 64];
        for r in 0..8usize {
            for c in 0..8usize {
                let addr = s.flat_index(&[r, c]).unwrap();
                assert!(!seen[addr], "double-covered at ({r},{c})");
                seen[addr] = true;
            }
        }
        assert!(seen.iter().all(|&x| x));
    }

    #[test]
    fn test_broadcasted_dims() {
        let l = RankedDigitLayout::right_major(&[1, 4]).unwrap();
        let bc = l.get_broadcasted_dims().unwrap();
        assert!(bc.contains(&0), "axis 0 (size 1) should be broadcasted");
    }
}
