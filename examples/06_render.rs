//! # Example 6: SVG Rendering
//!
//! Two renderers visualise a layout as an SVG grid.
//!
//! ## `render_svg`  — forward (offset) view
//!
//! Grid = source space.  Each cell is the destination flat offset for that
//! source coordinate.  Colour sweeps blue→red by value; stride patterns jump
//! out visually.
//!
//! ## `render_distribution_svg`  — inverse (distribution) view
//!
//! Grid = destination space.  Each cell shows which source coordinate maps
//! there, rendered in three layers that match Figure 1 of the Linear Layouts
//! paper:
//!
//!   - **Warp**: large semi-transparent watermark centred over each warp region.
//!   - **Lane**: bold label centred in each lane's 2×2 super-cell.  Background
//!     colour cycles through a 4-colour pastel palette by warp index.
//!   - **Register** (reg0–reg3): small label in each individual cell, row-major
//!     within the 2×2 sub-tile: reg0=top-left, reg1=top-right,
//!     reg2=bottom-left, reg3=bottom-right.
//!
//! Source space: (warp=2, lane=32, reg=4) where reg = trow*2 + tcol
//! (row-major register index within each thread's 2×2 sub-tile).
//!
//! The digit stream for [2, 32, 4] is identical to [2, 32, 2, 2]:
//!   reg_b0 = tcol_b0, reg_b1 = trow_b0 — so the same bit permutations apply.
//!
//! This example reproduces the two layouts from Figure 1:
//!
//!   Layout A — warp 0/1 on top/bottom half; lanes laid out row-wise.
//!   Layout B — warp 0/1 on left/right half; lanes laid out column-wise.
//!
//! Output SVG files are written to /tmp/.
//!
//! Run with: `cargo run --example 06_render`

use std::fs;

use ranked_digit_layouts::{canonical_radices, RankedDigitLayout, Space, Transform};

fn save(path: &str, svg: &str) {
    fs::write(path, svg).expect("failed to write SVG");
    println!("  wrote {path}  ({} bytes)", svg.len());
}

// ─── shared spaces ────────────────────────────────────────────────────────────

/// Source space: (warp=2, lane=32, reg=4) — 256 elements total.
///
/// `reg` is a row-major register index within each thread's 2×2 sub-tile:
///   reg = trow * 2 + tcol,  where trow,tcol ∈ {0,1}
///
/// The digit stream is the same as [2, 32, 2, 2]:
///   [reg_b0, reg_b1, lane_b0..b4, warp_b0]
fn hw_space() -> Space {
    Space::named(&[2, 32, 4], &["warp", "lane", "reg"]).unwrap()
}

/// Destination: 16×16 tensor tile.
fn tile_space() -> Space {
    Space::named(&[16, 16], &["dim0", "dim1"]).unwrap()
}

// ─── layout helpers ───────────────────────────────────────────────────────────

/// Layout A from Figure 1:
///   row = warp * 8  +  (lane % 4) * 2  +  (reg / 2)
///   col =             (lane / 4) * 2  +  (reg % 2)
///
/// Digit stream for src [2, 32, 4], right-major:
///   reg_b0=0, reg_b1=1, lane_b0=2, lane_b1=3, lane_b2=4, lane_b3=5, lane_b4=6, warp_b0=7
///
/// Dst [16, 16] right-major:
///   col_b0=0, col_b1=1, col_b2=2, col_b3=3, row_b0=4, row_b1=5, row_b2=6, row_b3=7
///
/// Permutation: dst[i] = src[order[i]]
///   col_b0=reg_b0→0, col_b1=lane_b2→4, col_b2=lane_b3→5, col_b3=lane_b4→6
///   row_b0=reg_b1→1, row_b1=lane_b0→2, row_b2=lane_b1→3, row_b3=warp_b0→7
fn layout_a() -> RankedDigitLayout {
    let src = hw_space();
    let dst = tile_space();
    let radices = canonical_radices(&src.shape).unwrap();
    let t = Transform::permute(radices, vec![0, 4, 5, 6, 1, 2, 3, 7]).unwrap();
    RankedDigitLayout::new(src, t, dst, "layout_a").unwrap()
}

/// Layout B from Figure 1:
///   row =             (lane % 8) * 2  +  (reg / 2)
///   col = warp * 8  +  (lane / 8) * 2  +  (reg % 2)
///
/// Permutation:
///   col_b0=reg_b0→0, col_b1=lane_b3→5, col_b2=lane_b4→6, col_b3=warp_b0→7
///   row_b0=reg_b1→1, row_b1=lane_b0→2, row_b2=lane_b1→3, row_b3=lane_b2→4
fn layout_b() -> RankedDigitLayout {
    let src = hw_space();
    let dst = tile_space();
    let radices = canonical_radices(&src.shape).unwrap();
    let t = Transform::permute(radices, vec![0, 5, 6, 7, 1, 2, 3, 4]).unwrap();
    RankedDigitLayout::new(src, t, dst, "layout_b").unwrap()
}

// ─── main ─────────────────────────────────────────────────────────────────────

fn main() {
    // ── 1. Forward (offset) view ───────────────────────────────────────────────
    println!("=== 1. Forward (offset) view ===\n");

    let flat = RankedDigitLayout::right_major(&[4, 4]).unwrap();
    save("/tmp/right_major.svg", &flat.render_svg(48).unwrap());
    println!("  4×4 right-major: blue (low offset) → red (high offset).\n");

    let transposed = flat.transpose(Some(&[1, 0])).unwrap();
    save("/tmp/transposed.svg", &transposed.render_svg(48).unwrap());
    println!("  Same layout, axes swapped: strides now run down columns.\n");

    // ── 2. Distribution view — Layout A (Figure 1a) ────────────────────────────
    //
    // Warp 0 on TOP  half (rows 0-7), warp 1 on BOTTOM half (rows 8-15).
    // Lanes 0-31 laid out row-wise within each warp half.
    // Each lane owns 4 registers in a 2×2 sub-tile (reg0..reg3, row-major).
    println!("=== 2. Distribution view — Layout A (warp on rows) ===\n");

    let la = layout_a();
    // Spot-check: map_coord(&[warp, lane, reg]) → [row, col]
    let check_a = |lane: usize, reg: usize| {
        let trow = reg / 2;
        let tcol = reg % 2;
        let got = la.map_coord(&[0, lane, reg]).unwrap();
        let exp_row = (lane % 4) * 2 + trow;
        let exp_col = (lane / 4) * 2 + tcol;
        assert_eq!(
            got,
            vec![exp_row, exp_col],
            "layout A mismatch: lane={lane} reg={reg}"
        );
    };
    check_a(0, 0); check_a(0, 3); check_a(7, 1); check_a(31, 3);
    println!("  Spot checks passed.");

    let svg = la.render_distribution_svg(32).unwrap();
    save("/tmp/layout_a.svg", &svg);
    println!("  Grid=16×16.  Warp watermark.  Lane labels.  reg1/reg2/reg3 per cell.\n");

    // ── 3. Distribution view — Layout B (Figure 1b) ────────────────────────────
    //
    // Warp 0 on LEFT half (cols 0-7), warp 1 on RIGHT half (cols 8-15).
    // Lanes 0-31 laid out column-wise within each warp half.
    println!("=== 3. Distribution view — Layout B (warp on cols) ===\n");

    let lb = layout_b();
    let check_b = |lane: usize, reg: usize| {
        let trow = reg / 2;
        let tcol = reg % 2;
        let got = lb.map_coord(&[0, lane, reg]).unwrap();
        let exp_row = (lane % 8) * 2 + trow;
        let exp_col = (lane / 8) * 2 + tcol;
        assert_eq!(
            got,
            vec![exp_row, exp_col],
            "layout B mismatch: lane={lane} reg={reg}"
        );
    };
    check_b(0, 0); check_b(7, 3); check_b(8, 0); check_b(31, 3);
    println!("  Spot checks passed.");

    let svg = lb.render_distribution_svg(32).unwrap();
    save("/tmp/layout_b.svg", &svg);
    println!("  Grid=16×16.  Warp watermark.  Lane labels.  reg1/reg2/reg3 per cell.\n");

    // ── 4. Summary ─────────────────────────────────────────────────────────────
    println!("Files written:");
    println!("  /tmp/right_major.svg");
    println!("  /tmp/transposed.svg");
    println!("  /tmp/layout_a.svg   (Figure 1a)");
    println!("  /tmp/layout_b.svg   (Figure 1b)");
    println!("\nOpen in a browser or SVG viewer to compare against Figure 1.");
}
