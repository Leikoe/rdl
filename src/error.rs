use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq, Eq)]
pub enum LayoutError {
    #[error("All radices must be >= 2, got {0:?}")]
    InvalidRadices(Vec<usize>),
    #[error("Length mismatch: {0}")]
    LengthMismatch(String),
    #[error("Out of bounds: {0}")]
    OutOfBounds(String),
    #[error("Value must be >= 1")]
    InvalidSize,
    #[error("Factorization {0:?} does not multiply to {1}")]
    FactorizationMismatch(Vec<usize>, usize),
    #[error("Axis order must be a permutation")]
    InvalidAxisOrder,
    #[error("Math error: {0}")]
    MathError(String),
    #[error("Transform incompatibility: {0}")]
    TransformIncompatibility(String),
}
