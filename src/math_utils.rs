use crate::error::LayoutError;
use std::cmp::Ordering;

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

pub fn check_radices(radices: &[usize]) -> Result<(), LayoutError> {
    if radices.iter().any(|&r| r < 2) {
        return Err(LayoutError::InvalidRadices(radices.to_vec()));
    }
    Ok(())
}

pub fn prime_factorization(mut n: usize) -> Result<Vec<usize>, LayoutError> {
    if n < 1 {
        return Err(LayoutError::InvalidSize);
    }
    if n == 1 {
        return Ok(vec![]);
    }
    let mut out = Vec::new();
    let mut d = 2;
    while d * d <= n {
        while n % d == 0 {
            out.push(d);
            n /= d;
        }
        d += 1;
    }
    if n > 1 {
        out.push(n);
    }
    Ok(out)
}

pub fn canonical_factorization(n: usize, binary_first: bool) -> Result<Vec<usize>, LayoutError> {
    let mut fac = prime_factorization(n)?;
    if binary_first {
        fac.sort_by(|a, b| {
            if *a == 2 && *b != 2 {
                Ordering::Less
            } else if *a != 2 && *b == 2 {
                Ordering::Greater
            } else {
                a.cmp(b)
            }
        });
    } else {
        fac.sort();
    }
    Ok(fac)
}

/// Canonical digit radices for `shape` — binary-first factorization, last axis first.
///
/// This is the digit representation assumed by default when constructing layouts.
/// The result is the flat digit stream radices in the order the digits appear.
pub fn canonical_radices(shape: &[usize]) -> Result<Vec<usize>, LayoutError> {
    let mut out = Vec::new();
    for &n in shape.iter().rev() {
        if n == 1 {
            // size-1 axis contributes no digits
        } else {
            out.extend(canonical_factorization(n, true)?);
        }
    }
    Ok(out)
}

/// Mixed-radix decode: least-significant digit first.
pub fn digits_to_int(digits: &[usize], radices: &[usize]) -> Result<usize, LayoutError> {
    check_radices(radices)?;
    if digits.len() != radices.len() {
        return Err(LayoutError::LengthMismatch("digits and radices".to_string()));
    }
    let mut x = 0;
    let mut mul = 1;
    for (&d, &r) in digits.iter().zip(radices.iter()) {
        if d >= r {
            return Err(LayoutError::OutOfBounds(format!(
                "Digit {} out of range for radix {}",
                d, r
            )));
        }
        x += d * mul;
        mul *= r;
    }
    Ok(x)
}

/// Mixed-radix encode: least-significant digit first. Inverse of `digits_to_int`.
pub fn int_to_digits(mut x: usize, radices: &[usize]) -> Result<Vec<usize>, LayoutError> {
    check_radices(radices)?;
    let total: usize = if radices.is_empty() { 1 } else { radices.iter().product() };
    if x >= total {
        return Err(LayoutError::OutOfBounds(format!(
            "x={} out of range [0,{})",
            x, total
        )));
    }
    let mut out = Vec::with_capacity(radices.len());
    for &r in radices {
        out.push(x % r);
        x /= r;
    }
    Ok(out)
}

pub fn inv_mod(a: isize, m: isize) -> Result<isize, LayoutError> {
    let a_mod = a.rem_euclid(m);
    let mut t = 0isize;
    let mut newt = 1isize;
    let mut r = m;
    let mut newr = a_mod;

    while newr != 0 {
        let q = r / newr;
        let tmp_t = t - q * newt;
        t = newt;
        newt = tmp_t;
        let tmp_r = r - q * newr;
        r = newr;
        newr = tmp_r;
    }

    if r > 1 {
        return Err(LayoutError::MathError(format!(
            "{} has no inverse modulo {}",
            a, m
        )));
    }
    if t < 0 {
        t += m;
    }
    Ok(t)
}

pub fn mat_vec_mod(m_mat: &[Vec<usize>], x: &[usize], modulus: usize) -> Vec<usize> {
    m_mat
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(&a, &b)| a * b).sum::<usize>() % modulus)
        .collect()
}

pub fn mat_inv_mod(m_mat: &[Vec<usize>], modulus: usize) -> Result<Vec<Vec<usize>>, LayoutError> {
    let n = m_mat.len();
    if m_mat.iter().any(|row| row.len() != n) {
        return Err(LayoutError::MathError("Matrix must be square".to_string()));
    }
    let mod_i = modulus as isize;
    let mut a: Vec<Vec<isize>> = vec![vec![0; 2 * n]; n];
    for i in 0..n {
        for j in 0..n {
            a[i][j] = (m_mat[i][j] % modulus) as isize;
        }
        a[i][n + i] = 1;
    }

    for col in 0..n {
        let pivot = (col..n)
            .find(|&row| gcd(a[row][col].unsigned_abs(), modulus) == 1)
            .ok_or_else(|| {
                LayoutError::MathError(format!("Matrix is not invertible modulo {}", modulus))
            })?;

        if pivot != col {
            a.swap(col, pivot);
        }

        let inv_p = inv_mod(a[col][col], mod_i)?;
        for j in 0..2 * n {
            a[col][j] = (inv_p * a[col][j]).rem_euclid(mod_i);
        }

        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = a[row][col].rem_euclid(mod_i);
            if factor != 0 {
                for j in 0..2 * n {
                    a[row][j] = (a[row][j] - factor * a[col][j]).rem_euclid(mod_i);
                }
            }
        }
    }

    let mut res = vec![vec![0usize; n]; n];
    for i in 0..n {
        for j in 0..n {
            res[i][j] = a[i][n + j] as usize;
        }
    }
    Ok(res)
}

pub fn invert_permutation(order: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; order.len()];
    for (i, &j) in order.iter().enumerate() {
        inv[j] = i;
    }
    inv
}

/// Enumerate all coordinates in row-major order (last axis changes fastest).
pub fn iterate_coords(shape: &[usize]) -> Vec<Vec<usize>> {
    if shape.is_empty() {
        return vec![vec![]];
    }
    let total: usize = shape.iter().product();
    let mut result = Vec::with_capacity(total);
    let mut coord = vec![0usize; shape.len()];
    for _ in 0..total {
        result.push(coord.clone());
        for i in (0..shape.len()).rev() {
            coord[i] += 1;
            if coord[i] < shape[i] {
                break;
            }
            coord[i] = 0;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_digits_roundtrip() {
        let radices = &[2, 3, 5];
        for x in 0..30usize {
            let d = int_to_digits(x, radices).unwrap();
            assert_eq!(digits_to_int(&d, radices).unwrap(), x);
        }
    }

    #[test]
    fn test_canonical_factorization() {
        assert_eq!(canonical_factorization(12, true).unwrap(), vec![2, 2, 3]);
        assert_eq!(canonical_factorization(1, true).unwrap(), vec![]);
        assert_eq!(canonical_factorization(7, false).unwrap(), vec![7]);
    }

    #[test]
    fn test_iterate_coords_shape_2x3() {
        let coords = iterate_coords(&[2, 3]);
        assert_eq!(coords.len(), 6);
        assert_eq!(coords[0], vec![0, 0]);
        assert_eq!(coords[1], vec![0, 1]);
        assert_eq!(coords[5], vec![1, 2]);
    }

    #[test]
    fn test_iterate_coords_empty() {
        assert_eq!(iterate_coords(&[]), vec![vec![]]);
    }

    #[test]
    fn test_mat_inv_mod() {
        let m = vec![vec![1usize, 1], vec![0, 1]];
        let inv = mat_inv_mod(&m, 2).unwrap();
        // [[1,1],[0,1]] is self-inverse mod 2
        assert_eq!(inv, vec![vec![1, 1], vec![0, 1]]);
    }

    #[test]
    fn test_invert_permutation() {
        assert_eq!(invert_permutation(&[2, 0, 1]), vec![1, 2, 0]);
    }
}
