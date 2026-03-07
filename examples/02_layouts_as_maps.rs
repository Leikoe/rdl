//! # Example 2: Layouts as Coordinate Maps
//!
//! A `RankedDigitLayout` is a function: logical coordinate → flat address.
//! This example shows three layouts over the same 4×4 logical tensor and
//! compares where each one places each element.
//!
//! The critical insight: **the data never moves — only the map changes**.
//!
//! Run with: `cargo run --example 02_layouts_as_maps`

use ranked_digit_layouts::RankedDigitLayout;

fn print_matrix(label: &str, l: &RankedDigitLayout) {
    println!("  {label}");
    print!("    ");
    for c in 0..4 { print!("col{c}  "); }
    println!();
    for r in 0..4 {
        print!("  row{r}  ");
        for c in 0..4 {
            print!("{:>4}   ", l.flat_index(&[r, c]).unwrap());
        }
        println!();
    }
    println!();
}

fn main() {
    // ── 1. Right-major layout ────────────────────────────────────────────────
    //
    // Last axis varies fastest: offset = col + 4·row.

    let right = RankedDigitLayout::right_major(&[4, 4]).unwrap();

    println!("=== 1. Right-major: offset = col + 4·row ===");
    print_matrix("physical offset", &right);

    // ── 2. Left-major layout ─────────────────────────────────────────────────
    //
    // First axis varies fastest: offset = row + 4·col.

    let left = RankedDigitLayout::left_major(&[4, 4]).unwrap();

    println!("=== 2. Left-major: offset = row + 4·col ===");
    print_matrix("physical offset", &left);

    // ── 3. Transposed layout (zero-copy) ─────────────────────────────────────
    //
    // Transposing a right-major layout does NOT rearrange memory.
    // `.transpose()` composes a new map — the buffer is unchanged.

    let transposed = right.transpose(None).unwrap();

    println!("=== 3. Transposed right-major (same buffer, different map) ===");
    println!("  src shape: {:?}", transposed.src.shape);
    println!("  Logical [r,c] of the transposed view → same flat address as [c,r] in original.\n");

    for r in 0..4usize {
        for c in 0..4usize {
            assert_eq!(
                transposed.flat_index(&[r, c]).unwrap(),
                right.flat_index(&[c, r]).unwrap(),
            );
        }
    }
    println!("  Verified: transposed[r,c] == right[c,r] for all 16 entries. ✓\n");

    // ── 4. Analytical contiguity ─────────────────────────────────────────────
    //
    // How many consecutive logical elements along an axis land at consecutive
    // physical addresses? That's the vectorisation width a compiler can exploit.

    println!("=== 4. Analytical vectorisation width ===");

    println!("  Right-major 4×4:");
    println!("    col axis (axis 1) contiguity = {}  ← full vector load", right.analytical_contiguity(1).unwrap());
    println!("    row axis (axis 0) contiguity = {}  ← strided, not vectorisable", right.analytical_contiguity(0).unwrap());

    println!("  Transposed 4×4:");
    println!("    col axis (axis 1) contiguity = {}", transposed.analytical_contiguity(1).unwrap());
    println!("    row axis (axis 0) contiguity = {}", transposed.analytical_contiguity(0).unwrap());

    // ── 5. Broadcasted axes ───────────────────────────────────────────────────
    //
    // A size-1 axis contributes no digits to the stream — it is free to
    // broadcast over any shape. The algebra makes this explicit.

    let bias = RankedDigitLayout::right_major(&[1, 4]).unwrap();
    let bcast = bias.get_broadcasted_dims().unwrap();

    println!("\n=== 5. Broadcasted dimensions ===");
    println!("  Layout src shape {:?}:", bias.src.shape);
    println!("  Broadcasted axes: {:?}  (axis 0 has size 1 — no digits — free to broadcast)", bcast);
}
