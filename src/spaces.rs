use crate::error::LayoutError;

/// A coordinate space: a product of finite cyclic groups Z_{n₀} × … × Z_{nₖ}.
///
/// A `Space` is purely a coordinate domain — a shape and optional axis labels.
/// It carries no information about how coordinates decompose into digits; that
/// is the responsibility of the `RankedDigitLayout` that maps from this space.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Space {
    pub shape: Vec<usize>,
    pub labels: Vec<String>,
}

impl Space {
    pub fn new(shape: Vec<usize>, labels: Vec<String>) -> Result<Self, LayoutError> {
        if !labels.is_empty() && labels.len() != shape.len() {
            return Err(LayoutError::LengthMismatch("labels/shape".to_string()));
        }
        if shape.iter().any(|&n| n == 0) {
            return Err(LayoutError::MathError("axis size must be ≥ 1".to_string()));
        }
        Ok(Self { shape, labels })
    }

    /// Labelled space.
    pub fn named(shape: &[usize], labels: &[&str]) -> Result<Self, LayoutError> {
        Self::new(shape.to_vec(), labels.iter().map(|s| s.to_string()).collect())
    }

    /// Anonymous space (no labels).
    pub fn of(shape: &[usize]) -> Result<Self, LayoutError> {
        Self::new(shape.to_vec(), vec![])
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn total_size(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_space_basic() {
        let s = Space::named(&[4, 8], &["row", "col"]).unwrap();
        assert_eq!(s.ndim(), 2);
        assert_eq!(s.total_size(), 32);
        assert_eq!(s.labels, vec!["row", "col"]);
    }

    #[test]
    fn test_space_label_mismatch() {
        assert!(Space::named(&[4, 8], &["row"]).is_err());
    }

    #[test]
    fn test_space_zero_size() {
        assert!(Space::of(&[0, 4]).is_err());
    }
}
