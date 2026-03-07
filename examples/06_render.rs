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
//!   - **Warp** (`watermark_axis`): large semi-transparent "w0"/"w1" label
//!     centred in each warp region.
//!   - **Thread** (`label_axis`): one bold label ("t0", "t5", …) centred in
//!     each thread's 2×2 super-cell.  Background colour cycles through a
//!     4-colour pastel palette with `lane / color_stride % 4`.
//!   - **Register** (remaining axes): small gray "r0"–"r3" in each cell corner.
//!
//! This example reproduces the two layouts from Figure 1:
//!
//!   Layout A — warp 0/1 on top/bottom half; threads laid out row-wise.
//!   Layout B — warp 0/1 on left/right half; threads laid out column-wise.
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

/// Source space: (warp=2, lane=32, trow=2, tcol=2) — 256 elements total.
fn hw_space() -> Space {
    Space::named(&[2, 32, 2, 2], &["warp", "lane", "trow", "tcol"]).unwrap()
}

/// Destination: 16×16 tensor tile.
fn tile_space() -> Space {
    Space::named(&[16, 16], &["dim0", "dim1"]).unwrap()
}

// ─── layout helpers ───────────────────────────────────────────────────────────

/// Layout A from Figure 1:
///   row = warp * 8  +  (lane % 4) * 2  +  trow
///   col =             (lane / 4) * 2  +  tcol
///
/// Bit derivation (canonical right-major digit stream for src [2,32,2,2]):
///   src pos: tcol_b0=0, trow_b0=1, lane_b0=2, lane_b1=3, lane_b2=4,
///            lane_b3=5, lane_b4=6, warp_b0=7
/// Dst [16,16] right-major:
///   dst pos: col_b0=0, col_b1=1, col_b2=2, col_b3=3,
///            row_b0=4, row_b1=5, row_b2=6, row_b3=7
/// Permutation order: dst[i] = src[order[i]]
///   col_b0=tcol_b0→0, col_b1=lane_b2→4, col_b2=lane_b3→5, col_b3=lane_b4→6,
///   row_b0=trow_b0→1, row_b1=lane_b0→2, row_b2=lane_b1→3, row_b3=warp_b0→7
fn layout_a() -> RankedDigitLayout {
    let src = hw_space();
    let dst = tile_space();
    let radices = canonical_radices(&src.shape).unwrap();
    let t = Transform::permute(radices, vec![0, 4, 5, 6, 1, 2, 3, 7]).unwrap();
    RankedDigitLayout::new(src, t, dst, "layout_a").unwrap()
}

/// Layout B from Figure 1:
///   row =             (lane % 8) * 2  +  trow
///   col = warp * 8  +  (lane / 8) * 2  +  tcol
///
/// Permutation order:
///   col_b0=tcol_b0→0, col_b1=lane_b3→5, col_b2=lane_b4→6, col_b3=warp_b0→7,
///   row_b0=trow_b0→1, row_b1=lane_b0→2, row_b2=lane_b1→3, row_b3=lane_b2→4
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
    // Threads t0-t31 laid out row-wise (8 threads per tile-row, 4 tile-rows per warp).
    //
    // color_stride=4: lanes 0-3 share color 0 (same col group), 4-7 share color 1, …
    // → 4-colour repeating vertical-stripe pattern as in the paper.
    println!("=== 2. Distribution view — Layout A (warp on rows) ===\n");

    let la = layout_a();
    // Spot-check a few coordinates.
    let check = |lane: usize, trow: usize, tcol: usize| {
        let got = la.map_coord(&[0, lane, trow, tcol]).unwrap();
        let exp_row = (lane % 4) * 2 + trow;
        let exp_col = (lane / 4) * 2 + tcol;
        assert_eq!(got, vec![exp_row, exp_col],
            "layout A mismatch for lane={lane} trow={trow} tcol={tcol}");
    };
    check(0, 0, 0); check(0, 1, 1); check(7, 0, 1); check(31, 1, 1);
    println!("  Spot checks passed.");

    //   watermark_axis=0 (warp), label_axis=1 (lane), color_stride=4
    let svg = la.render_distribution_svg(20, 0, 1, 4).unwrap();
    save("/tmp/layout_a.svg", &svg);
    println!("  Grid=16×16.  Warp watermarks w0/w1.  Thread labels t0-t31.");
    println!("  4-colour stripe: each 2-column group shares one colour.\n");

    // ── 3. Distribution view — Layout B (Figure 1b) ────────────────────────────
    //
    // Warp 0 on LEFT half (cols 0-7), warp 1 on RIGHT half (cols 8-15).
    // Threads t0-t31 laid out column-wise (8 threads per tile-col, 4 tile-cols per warp).
    //
    // color_stride=8: lanes 0-7 share color 0 (cols 0-1), 8-15 share color 1, …
    println!("=== 3. Distribution view — Layout B (warp on cols) ===\n");

    let lb = layout_b();
    let check_b = |lane: usize, trow: usize, tcol: usize| {
        let got = lb.map_coord(&[0, lane, trow, tcol]).unwrap();
        let exp_row = (lane % 8) * 2 + trow;
        let exp_col = (lane / 8) * 2 + tcol;
        assert_eq!(got, vec![exp_row, exp_col],
            "layout B mismatch for lane={lane} trow={trow} tcol={tcol}");
    };
    check_b(0, 0, 0); check_b(7, 1, 1); check_b(8, 0, 0); check_b(31, 1, 1);
    println!("  Spot checks passed.");

    //   watermark_axis=0 (warp), label_axis=1 (lane), color_stride=8
    let svg = lb.render_distribution_svg(20, 0, 1, 8).unwrap();
    save("/tmp/layout_b.svg", &svg);
    println!("  Grid=16×16.  Warp watermarks w0/w1.  Thread labels t0-t31.");
    println!("  4-colour stripe: each pair of warp-cols shares one colour.\n");

    // ── 4. Summary ─────────────────────────────────────────────────────────────
    println!("Files written:");
    println!("  /tmp/right_major.svg");
    println!("  /tmp/transposed.svg");
    println!("  /tmp/layout_a.svg   (Figure 1a)");
    println!("  /tmp/layout_b.svg   (Figure 1b)");
    println!("\nOpen in a browser or SVG viewer to compare against Figure 1.");
}
