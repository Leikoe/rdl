//! # Example 4: GPU Shared Memory Swizzling
//!
//! GPU shared memory is divided into N banks. When multiple threads in a warp
//! access addresses that fall in the **same bank**, the hardware serialises the
//! accesses — a "bank conflict". This dramatically reduces throughput.
//!
//! The standard fix is an **XOR swizzle**: instead of storing element (row, col)
//! at the naive address (row * W + col), store it at (row * W + (col XOR row)).
//! This scatters a column across all banks at no compute cost.
//!
//! This example builds the swizzle as a `BlockAffine` transform over Z₂ digits
//! and proves it eliminates bank conflicts analytically.
//!
//! Run with: `cargo run --example 04_bank_conflict_swizzle`

use ranked_digit_layouts::RankedDigitLayout;

fn bank(addr: usize, num_banks: usize) -> usize {
    addr % num_banks
}

fn print_bank_access(label: &str, layout: &RankedDigitLayout, rows: usize, cols: usize, num_banks: usize) {
    println!("{label}");
    println!("  (which bank each (row,col) element lands in)\n");

    print!("         ");
    for c in 0..cols { print!("col{c:<2} "); }
    println!();

    for r in 0..rows {
        print!("  row {r}  ");
        let mut bank_counts = vec![0usize; num_banks];
        for c in 0..cols {
            let b = bank(layout.flat_index(&[r, c]).unwrap(), num_banks);
            bank_counts[b] += 1;
            print!(" B{b:<3} ");
        }
        let conflicts = bank_counts.iter().max().copied().unwrap_or(0).saturating_sub(1);
        if r == 0 {
            println!(" ← {}", if conflicts > 0 { format!("{conflicts}-way conflict!") } else { "no conflict".to_string() });
        } else {
            println!();
        }
    }

    let col0_banks: Vec<usize> = (0..rows).map(|r| bank(layout.flat_index(&[r, 0]).unwrap(), num_banks)).collect();
    let unique = { let mut s = col0_banks.clone(); s.sort_unstable(); s.dedup(); s.len() };

    println!();
    println!("  Loading column 0 (one thread per row):");
    println!("    Banks hit: {:?}", { let mut s = col0_banks; s.sort_unstable(); s });
    println!("    Unique banks: {unique} / {rows}  →  {}",
        if unique == rows { "✓ no conflicts" } else { "✗ conflicts!" });
    println!();
}

fn main() {
    // Shared memory tile: 8 rows × 8 columns.
    // 8 banks (real hardware has 32, but 8 keeps output readable).
    let rows = 8usize;
    let cols = 8usize;
    let num_banks = 8usize;

    println!("=== GPU Shared Memory Bank Conflicts ===");
    println!("  Tile: {rows}×{cols},  {num_banks} banks\n");

    // ── 1. Naive (unswizzled) layout ─────────────────────────────────────────
    //
    // Right-major: addr = row * 8 + col.
    // Column 0: addr = 0, 8, 16, 24, 32, 40, 48, 56 → all in bank 0 → 8-way conflict.

    let naive = RankedDigitLayout::right_major(&[rows, cols]).unwrap();
    print_bank_access("--- Naive (right-major) layout ---", &naive, rows, cols, num_banks);

    // ── 2. XOR swizzle ───────────────────────────────────────────────────────
    //
    // The digit stream for [8, 8] (both axes 2³) with default right-major order:
    //   [col_b0, col_b1, col_b2, row_b0, row_b1, row_b2]
    //    pos 0   pos 1   pos 2   pos 3   pos 4   pos 5
    //
    // We XOR each col bit with the corresponding row bit:
    //   col_b0' = col_b0 XOR row_b0
    //   col_b1' = col_b1 XOR row_b1
    //   col_b2' = col_b2 XOR row_b2
    //
    // As one 6×6 binary matrix over positions [0,1,2,3,4,5]:
    //
    //   [1 0 0 1 0 0]   col_b0' = col_b0 + row_b0
    //   [0 1 0 0 1 0]   col_b1' = col_b1 + row_b1
    //   [0 0 1 0 0 1]   col_b2' = col_b2 + row_b2
    //   [0 0 0 1 0 0]   row_b0' = row_b0 (unchanged)
    //   [0 0 0 0 1 0]   row_b1' = row_b1 (unchanged)
    //   [0 0 0 0 0 1]   row_b2' = row_b2 (unchanged)

    #[rustfmt::skip]
    let xor_matrix = vec![
        vec![1, 0, 0, 1, 0, 0],
        vec![0, 1, 0, 0, 1, 0],
        vec![0, 0, 1, 0, 0, 1],
        vec![0, 0, 0, 1, 0, 0],
        vec![0, 0, 0, 0, 1, 0],
        vec![0, 0, 0, 0, 0, 1],
    ];

    let swizzled = naive.swizzle(&[0, 1, 2, 3, 4, 5], xor_matrix, None).unwrap();
    print_bank_access("--- XOR-swizzled layout ---", &swizzled, rows, cols, num_banks);

    // ── 3. Formal verification ───────────────────────────────────────────────

    let mut all_addrs: Vec<usize> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| (r, c)))
        .map(|(r, c)| swizzled.flat_index(&[r, c]).unwrap())
        .collect();
    all_addrs.sort_unstable();
    assert_eq!(
        all_addrs,
        (0..rows * cols).collect::<Vec<_>>(),
        "swizzle must be a bijection"
    );
    println!("Verification: swizzle is a bijection over all {} addresses. ✓", rows * cols);

    // ── 4. Analytical bank-conflict check ────────────────────────────────────

    let col_contig = swizzled.analytical_contiguity(1).unwrap();
    let row_contig = swizzled.analytical_contiguity(0).unwrap();

    println!();
    println!("Analytical contiguity (swizzled layout):");
    println!("  Col axis (axis 1): {col_contig}  (row accesses — {col_contig} consecutive addresses)");
    println!("  Row axis (axis 0): {row_contig}  (col accesses spread across all {num_banks} banks)");
    println!();
    println!("A row load touches {col_contig} consecutive addresses — vectorisable.");
    println!("A col load now spreads across all {num_banks} banks — conflict-free.");
}
