//! # Example 5: Hardware Lowering via Algebraic Division
//!
//! A compiler maps a logical tensor layout onto a hardware layout (a warp tile,
//! a tensor-core fragment, etc.) and derives what each thread must do.
//!
//! The core operation is:
//!
//!     Rest = Logical ÷ Hardware   means   Logical = Rest ∘ Hardware
//!
//! `Rest` is the per-thread code the compiler emits. It answers:
//! "given my hardware iteration state (lane, trow, tcol), what flat memory address?"
//!
//! This example works through a concrete warp tile assignment for a [16×8] output.
//! The key insight: the hw→tile mapping is a **digit permutation** — pure algebra,
//! no conditional logic.
//!
//! Run with: `cargo run --example 05_hardware_lowering`

use ranked_digit_layouts::{
    builders::{refactor_layout, transpose_layout},
    canonical_radices, RankedDigitLayout, Space, Transform,
};

fn main() {
    // ── Problem ───────────────────────────────────────────────────────────────
    //
    // Tile: [16 rows × 8 cols] = 128 elements.
    // Warp: 32 threads, each owns a [2 × 2] sub-tile (4 elements).
    //   Thread grid: 8 row-groups × 4 col-groups = 32 threads.
    //   row = (lane % 8) * 2 + trow
    //   col = (lane / 8) * 2 + tcol

    println!("=== Hardware Lowering: Warp Tile ===\n");
    println!("  Tile:       [16 × 8] = 128 elements");
    println!("  Warp:       32 threads");
    println!("  Per-thread: 2×2 = 4 elements\n");

    // ── 1. Build the spaces ───────────────────────────────────────────────────

    let hw_space   = Space::named(&[32, 2, 2], &["lane", "trow", "tcol"]).unwrap();
    let tile_space = Space::named(&[16, 8],    &["row", "col"]).unwrap();
    let flat_space = Space::of(&[128]).unwrap();

    println!("hw_space   canonical digit radices: {:?}", canonical_radices(&hw_space.shape).unwrap());
    println!("tile_space canonical digit radices: {:?}", canonical_radices(&tile_space.shape).unwrap());
    println!("(Both are 7 binary digits — the hardware mapping is a bit permutation.)\n");

    // ── 2. Derive the hw→tile digit permutation analytically ─────────────────
    //
    // canonical digit stream for hw_space [32,2,2], right-major (last axis first):
    //   [tcol_b0, trow_b0, lane_b0, lane_b1, lane_b2, lane_b3, lane_b4]
    //    pos 0     pos 1    pos 2    pos 3    pos 4    pos 5    pos 6
    //
    // canonical digit stream for tile_space [16,8], right-major:
    //   [col_b0, col_b1, col_b2, row_b0, row_b1, row_b2, row_b3]
    //    pos 0   pos 1   pos 2   pos 3   pos 4   pos 5   pos 6
    //
    // Formula:
    //   col = (lane/8)*2 + tcol → col_b0=tcol_b0, col_b1=lane_b3, col_b2=lane_b4
    //   row = (lane%8)*2 + trow → row_b0=trow_b0, row_b1=lane_b0, row_b2=lane_b1, row_b3=lane_b2

    let hw_to_tile_order = vec![0usize, 5, 6, 1, 2, 3, 4];

    println!("=== 2. HW → Tile digit permutation ===");
    println!("  order: {:?}", hw_to_tile_order);
    println!("  (output[i] = hw_digit[order[i]])\n");

    let hw_radices   = canonical_radices(&hw_space.shape).unwrap();
    let hw_to_tile_t = Transform::permute(hw_radices, hw_to_tile_order).unwrap();
    let hw_layout    = RankedDigitLayout::new(hw_space.clone(), hw_to_tile_t, tile_space.clone(), "hw_to_tile").unwrap();

    fn expected(lane: usize, trow: usize, tcol: usize) -> [usize; 2] {
        [(lane % 8) * 2 + trow, (lane / 8) * 2 + tcol]
    }

    println!("  Spot checks (lane, trow, tcol) → tile [row, col]:");
    for (lane, trow, tcol) in [(0,0,0),(1,0,0),(7,1,1),(8,0,0),(31,1,1)] {
        let got = hw_layout.map_coord(&[lane, trow, tcol]).unwrap();
        let exp = expected(lane, trow, tcol);
        let mark = if got == exp.to_vec() { "✓" } else { "✗" };
        println!("    lane={lane:>2}, trow={trow}, tcol={tcol}  → got {:?}  exp {:?}  {mark}", got, exp);
    }

    let mut seen = vec![false; 128];
    for lane in 0..32usize {
        for trow in 0..2usize {
            for tcol in 0..2usize {
                let [r, c] = expected(lane, trow, tcol);
                assert!(!seen[r * 8 + c], "double-covered at ({r},{c})");
                seen[r * 8 + c] = true;
            }
        }
    }
    assert!(seen.iter().all(|&x| x));
    println!("  Bijection verified: all 128 tile elements covered exactly once. ✓\n");

    // ── 3. Logical layout: tile → flat (right-major) ──────────────────────────

    let logical_layout = refactor_layout(tile_space.clone(), flat_space.clone(), "logical").unwrap();

    // ── 4. Lowering: hw → flat ────────────────────────────────────────────────

    let lowered = hw_layout.then(&logical_layout, "lowered").unwrap();

    println!("=== 3. Lowered layout: (lane, trow, tcol) → flat address ===\n");
    println!("  lane  trow tcol  flat_addr  (tile[row,col])");
    println!("  {}", "-".repeat(48));
    for lane in [0, 1, 7, 8, 31] {
        for trow in 0..2usize {
            for tcol in 0..2usize {
                let addr = lowered.flat_index(&[lane, trow, tcol]).unwrap();
                let [r, c] = expected(lane, trow, tcol);
                println!("  {lane:>4}   {trow}    {tcol}    {addr:>4}       (tile[{r:>2},{c:>1}])");
            }
        }
        println!();
    }

    // ── 5. Algebraic division ─────────────────────────────────────────────────

    println!("=== 4. Algebraic division: thread view ===\n");
    println!("  hw_layout kernel size: {}", hw_layout.analytical_kernel_size());
    println!("  → Each (lane, trow, tcol) triple maps to exactly 1 tile element.\n");

    println!("  Lane 0 owns tile coords and flat addresses:");
    for trow in 0..2usize {
        for tcol in 0..2usize {
            let tile_coord = hw_layout.map_coord(&[0, trow, tcol]).unwrap();
            let flat_addr  = logical_layout.flat_index(&tile_coord).unwrap();
            println!("    thread[{trow},{tcol}] → tile{:?} → flat[{flat_addr}]", tile_coord);
        }
    }

    // ── 6. Vectorisation analysis ─────────────────────────────────────────────

    println!("\n=== 5. Vectorisation analysis ===\n");
    let tcol_contig = lowered.analytical_contiguity(2).unwrap();
    let trow_contig = lowered.analytical_contiguity(1).unwrap();
    let lane_contig = lowered.analytical_contiguity(0).unwrap();

    println!("  Contiguity along tcol (axis 2): {tcol_contig}");
    println!("    → A thread can load both column elements in one vector op.");
    println!("  Contiguity along trow (axis 1): {trow_contig}");
    println!("  Contiguity along lane (axis 0): {lane_contig}");
    println!("\n  A compiler reads this table analytically — no runtime scan needed.");

    // ── 7. Transposed tile ────────────────────────────────────────────────────

    println!("\n=== 6. Transposed tile: same warp, different memory pattern ===\n");

    let transposed_tile = transpose_layout(tile_space.clone(), &[1, 0], "tile_T").unwrap();
    let t_flat = Space::of(&[128]).unwrap();
    let t_logical = refactor_layout(transposed_tile.dst.clone(), t_flat, "logical_T").unwrap();
    let t_lowered = hw_layout
        .then(&transposed_tile, "t_step").unwrap()
        .then(&t_logical, "t_lowered").unwrap();

    println!("  Lane 0 with transposed tile:");
    for trow in 0..2usize {
        for tcol in 0..2usize {
            let addr = t_lowered.flat_index(&[0, trow, tcol]).unwrap();
            println!("    thread[{trow},{tcol}] → flat[{addr}]");
        }
    }
    let t_tcol_contig = t_lowered.analytical_contiguity(2).unwrap();
    println!("\n  tcol contiguity after transpose: {t_tcol_contig}");
    println!("  (Changed from {tcol_contig} — the compiler adjusts vector load width automatically.)");
}
