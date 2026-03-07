//! # Example 3: Zero-Copy Reshape and Transpose Composition
//!
//! PyTorch's `.contiguous()` exists because strides break when you chain
//! transpose with reshape. This library has no such limitation.
//!
//! Here we show:
//!   1. View a flat buffer as [4, 4].
//!   2. Transpose it — no copy.
//!   3. Reshape the transposed view to [2, 8] — **still no copy**.
//!      In PyTorch this would require `.contiguous()` first.
//!
//! All three steps are method calls on a single `RankedDigitLayout`.
//! The composed layout is a closed-form map — no intermediate buffers.
//!
//! Run with: `cargo run --example 03_zero_copy_reshape`

use ranked_digit_layouts::RankedDigitLayout;

fn print_flat(label: &str, l: &RankedDigitLayout) {
    let [rows, cols] = [l.src.shape[0], l.src.shape[1]];
    println!("{label}");
    for r in 0..rows {
        print!("  row {r}: ");
        for c in 0..cols {
            print!("{:>2} ", l.flat_index(&[r, c]).unwrap());
        }
        println!();
    }
    println!();
}

fn main() {
    println!("=== Buffer: 16 floats in memory ===");
    println!("    [{}]\n", (0..16).map(|i| format!("{i:>2}")).collect::<Vec<_>>().join(", "));

    // ── Step 1: View as [4, 4] right-major ───────────────────────────────────

    let view = RankedDigitLayout::right_major(&[4, 4]).unwrap();

    println!("=== Step 1: View as [4, 4] (right-major) ===");
    print_flat("", &view);

    // ── Step 2: Transpose → axes swapped, same buffer ────────────────────────

    let transposed = view.transpose(None).unwrap();

    println!("=== Step 2: Transpose (same buffer, columns are now rows) ===");
    print_flat("", &transposed);

    // ── Step 3: Reshape transposed [4,4] → [2, 8] ───────────────────────────
    //
    // PyTorch: `tensor.transpose(0,1).reshape(2,8)` → RuntimeError unless
    // you call `.contiguous()` first, allocating 16 new floats.
    //
    // Here: one more method call. The algebra handles it.

    let reshaped = transposed.reshape(&[2, 8]).unwrap();

    println!("=== Step 3: Reshape transposed [4,4] → [2, 8] (NO COPY) ===");
    for r in 0..2usize {
        print!("  row {r}: ");
        for c in 0..8usize {
            print!("{:>2} ", reshaped.flat_index(&[r, c]).unwrap());
        }
        println!();
    }
    println!();

    // ── Verification ─────────────────────────────────────────────────────────
    //
    // The final layout is a single composed transform.
    // Every buffer slot 0..16 must appear exactly once.

    println!("=== Composition proof ===");
    println!("  Composed transform: {:?}\n", reshaped.transform);

    let mut all_offsets: Vec<usize> = (0..2)
        .flat_map(|r| (0..8).map(move |c| (r, c)))
        .map(|(r, c)| reshaped.flat_index(&[r, c]).unwrap())
        .collect();
    all_offsets.sort_unstable();

    assert_eq!(all_offsets, (0..16).collect::<Vec<_>>(), "layout must be a bijection");
    println!("  All 16 buffer slots appear exactly once — layout is bijective. ✓");
    println!("  No allocation was needed. ✓");

    // ── Contiguity ───────────────────────────────────────────────────────────

    println!("\n  Column axis (axis 1) contiguity in final view: {}", reshaped.analytical_contiguity(1).unwrap());
    println!("  Row    axis (axis 0) contiguity in final view: {}", reshaped.analytical_contiguity(0).unwrap());
    println!("\n  A vectorising compiler reads this table analytically — no runtime scan needed.");
}
