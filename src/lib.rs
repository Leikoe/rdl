pub mod analysis;
pub mod builders;
pub mod render;
pub mod error;
pub mod layout;
pub mod math_utils;
pub mod spaces;
pub mod transforms;

pub use analysis::{contiguous_run_along_axis, factor_through_refactor_prefix};
pub use builders::{
    block_affine_layout, embed_layout, identity_layout, project_layout, refactor_layout,
    reshape_layout, transpose_layout,
};
pub use error::LayoutError;
pub use layout::RankedDigitLayout;
pub use math_utils::canonical_radices;
pub use spaces::Space;
pub use transforms::Transform;
