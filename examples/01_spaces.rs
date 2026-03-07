//! # Example 1: Coordinate Spaces and Digit Representations
//!
//! A `Space` is a coordinate domain — a shape and optional axis labels.
//! It carries no memory-layout information.
//!
//! The digit representation (how coordinates decompose into prime-factor digits)
//! lives in the `RankedDigitLayout` that maps from the space, not in the space itself.
//!
//! This example shows what those digits look like and why the separation matters.
//!
//! Run with: `cargo run --example 01_spaces`

use ranked_digit_layouts::{canonical_radices, RankedDigitLayout, Space};

fn main() {
    // ── 1. A space is just a shape ────────────────────────────────────────────

    let flat = Space::named(&[8], &["i"]).unwrap();
    println!("=== 1. Space: flat array of 8 elements ===");
    println!("Shape:  {:?}", flat.shape);
    println!("Labels: {:?}", flat.labels);
    println!("(No digit information — that belongs to the layout.)\n");

    // ── 2. The digit representation lives in the layout ───────────────────────
    //
    // canonical_radices(shape) computes the default digit stream for a shape:
    // binary-first factorization, last axis leads the stream (right-major).

    let mat = Space::named(&[4, 8], &["row", "col"]).unwrap();
    println!("=== 2. Matrix [4, 8]: canonical digit radices ===");
    let radices = canonical_radices(&mat.shape).unwrap();
    println!("canonical_radices([4, 8]) = {:?}", radices);
    println!("(col bits come first — 3 bits for col=8, then 2 bits for row=4)\n");

    // ── 3. Two layouts, one space ─────────────────────────────────────────────
    //
    // The same space can be mapped to flat memory in different ways.
    // right_major: last axis fastest  (C order)
    // left_major:  first axis fastest (Fortran order)
    // The space [4, 4] is identical in both cases — only the layout differs.

    let s = Space::of(&[4, 4]).unwrap();
    println!("=== 3. Same space, different layouts ===");
    println!("Space: {:?}", s.shape);

    let rm = RankedDigitLayout::right_major(&[4, 4]).unwrap();
    let lm = RankedDigitLayout::left_major(&[4, 4]).unwrap();

    println!("\n  right_major [r,c] → flat addr (last axis fastest):");
    for r in 0..4usize {
        print!("    row {r}:");
        for c in 0..4usize { print!(" {:>2}", rm.flat_index(&[r, c]).unwrap()); }
        println!();
    }

    println!("\n  left_major [r,c] → flat addr (first axis fastest):");
    for r in 0..4usize {
        print!("    row {r}:");
        for c in 0..4usize { print!(" {:>2}", lm.flat_index(&[r, c]).unwrap()); }
        println!();
    }

    // ── 4. Non-power-of-2 sizes ───────────────────────────────────────────────
    //
    // canonical_radices handles mixed radices automatically.
    // 12 = 2 × 2 × 3 → three digits in [Z₂, Z₂, Z₃].

    println!("\n=== 4. Non-power-of-2: shape [12] = 2×2×3 ===");
    let r12 = canonical_radices(&[12]).unwrap();
    println!("canonical_radices([12]) = {:?}  (Z₂, Z₂, Z₃)\n", r12);

    let l12 = RankedDigitLayout::right_major(&[12]).unwrap();
    println!("  coord  → flat addr");
    println!("  {}", "-".repeat(24));
    for i in 0..12usize {
        println!("  [{i:>2}]    {}", l12.flat_index(&[i]).unwrap());
    }

    // ── 5. Axis labels are just metadata ─────────────────────────────────────

    let hw = Space::named(&[32, 2, 2], &["lane", "trow", "tcol"]).unwrap();
    println!("\n=== 5. Labelled space ===");
    println!("Shape:  {:?}", hw.shape);
    println!("Labels: {:?}", hw.labels);
    println!("ndim:   {}", hw.ndim());
    println!("total:  {}", hw.total_size());
    println!("(Labels are metadata only — no effect on digit representation.)");
}
