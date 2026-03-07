use crate::error::LayoutError;
use crate::layout::RankedDigitLayout;
use crate::math_utils::iterate_coords;

/// Map a value in [0, max] to an HSL colour string.
/// Hue sweeps from blue (low) to red (high) — periodic patterns pop visually.
fn value_to_hsl(value: usize, max: usize) -> String {
    if max == 0 {
        return "hsl(0,0%,88%)".to_string();
    }
    // 240 = blue, 0 = red; sweep 240 → 0 as value rises
    let hue = 240.0 - (value as f64 / max as f64) * 240.0;
    format!("hsl({:.1},65%,68%)", hue)
}

/// Choose text colour (black or white) for legibility on the background.
fn text_colour(value: usize, max: usize) -> &'static str {
    if max == 0 {
        return "black";
    }
    // Dark text on light backgrounds (low and high ends), light text in the middle.
    let t = value as f64 / max as f64;
    if t < 0.25 || t > 0.75 { "black" } else { "white" }
}

/// Four-colour soft-pastel palette, matching the style of Figure 1
/// in the Linear Layouts paper (yellow, green, blue, purple).
const PALETTE: &[&str] = &[
    "hsl(50,70%,82%)",   // soft yellow
    "hsl(130,45%,80%)",  // soft green
    "hsl(210,55%,82%)",  // soft blue
    "hsl(280,45%,80%)",  // soft purple
];

fn palette_color(group: usize) -> &'static str {
    PALETTE[group % PALETTE.len()]
}

/// Convert a destination coordinate to (row, col) in the rendering grid.
fn dst_to_rowcol(dst_coord: &[usize], dst_shape: &[usize]) -> (usize, usize) {
    match dst_shape.len() {
        1 => (0, dst_coord[0]),
        2 => (dst_coord[0], dst_coord[1]),
        _ => {
            let col = dst_coord[1..].iter().zip(dst_shape[1..].iter())
                .fold(0usize, |acc, (&v, &s)| acc * s + v);
            (dst_coord[0], col)
        }
    }
}

/// Register-level label: all source axes except `skip1` and `skip2`,
/// using the first character of each axis label as prefix (e.g. "r0", "r2").
fn format_register_label(src: &[usize], skip1: usize, skip2: usize, labels: &[String]) -> String {
    src.iter().enumerate()
        .filter(|&(i, _)| i != skip1 && i != skip2)
        .map(|(i, &v)| {
            if i < labels.len() && !labels[i].is_empty() {
                format!("{}{v}", labels[i].chars().next().unwrap())
            } else {
                format!("{v}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

impl RankedDigitLayout {
    /// Render the layout as an SVG grid.
    ///
    /// Each cell corresponds to one source coordinate.  Its fill colour encodes the
    /// flat destination offset; the numeric value is printed inside.
    ///
    /// For 1-D sources: a single row.
    /// For 2-D sources: rows × cols grid directly.
    /// For N-D sources: first axis as rows, remaining axes flattened as columns.
    ///
    /// `cell_px` controls the side length of each cell in pixels.
    pub fn render_svg(&self, cell_px: u32) -> Result<String, LayoutError> {
        let shape = &self.src.shape;

        if shape.is_empty() {
            return Err(LayoutError::MathError("cannot render a 0-D space".to_string()));
        }

        let (nrows, ncols) = match shape.len() {
            1 => (1usize, shape[0]),
            _ => (shape[0], shape[1..].iter().product()),
        };

        // Evaluate layout at every source coordinate.
        let coords = iterate_coords(shape);
        let offsets: Vec<usize> = coords
            .iter()
            .map(|c| self.flat_index(c))
            .collect::<Result<_, _>>()?;

        let max_offset = *offsets.iter().max().unwrap_or(&0);

        // Pixel geometry.
        let border = 1u32;
        let pad = 12u32;
        let font_size = (cell_px as f64 * 0.32).max(8.0) as u32;
        let svg_w = ncols as u32 * cell_px + 2 * pad;
        let svg_h = nrows as u32 * cell_px + 2 * pad;

        let mut out = String::with_capacity(offsets.len() * 120);

        // Header.
        out.push_str(&format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">"#
        ));

        // Background.
        out.push_str(&format!(
            r##"<rect width="{svg_w}" height="{svg_h}" fill="#f8f8f8"/>"##
        ));

        // Cells.
        for (i, (&offset, _coord)) in offsets.iter().zip(coords.iter()).enumerate() {
            let row = i / ncols;
            let col = i % ncols;
            let x = pad + col as u32 * cell_px;
            let y = pad + row as u32 * cell_px;

            let fill = value_to_hsl(offset, max_offset);
            let fg   = text_colour(offset, max_offset);

            // Cell background.
            out.push_str(&format!(
                r##"<rect x="{x}" y="{y}" width="{cell_px}" height="{cell_px}" fill="{fill}" stroke="#fff" stroke-width="{border}"/>"##,
            ));

            // Value label, centred in the cell.
            let tx = x + cell_px / 2;
            let ty = y + cell_px / 2 + font_size / 3;
            out.push_str(&format!(
                r#"<text x="{tx}" y="{ty}" font-family="monospace" font-size="{font_size}" text-anchor="middle" fill="{fg}">{offset}</text>"#,
            ));
        }

        // Axis labels, if the space has them.
        if !self.src.labels.is_empty() && shape.len() >= 2 {
            let label_size = font_size.saturating_sub(2).max(7);
            // Row axis label (left side).
            let rx = pad / 2;
            let ry = pad + (nrows as u32 * cell_px) / 2;
            out.push_str(&format!(
                r##"<text x="{rx}" y="{ry}" font-family="monospace" font-size="{label_size}" text-anchor="middle" fill="#666" transform="rotate(-90,{rx},{ry})">{}</text>"##,
                self.src.labels[0]
            ));
            // Column axis label (top).
            let col_label = self.src.labels[1..].join(",");
            let cx = pad + (ncols as u32 * cell_px) / 2;
            let cy = pad / 2 + label_size;
            out.push_str(&format!(
                r##"<text x="{cx}" y="{cy}" font-family="monospace" font-size="{label_size}" text-anchor="middle" fill="#666">{col_label}</text>"##,
            ));
        }

        out.push_str("</svg>");
        Ok(out)
    }

    /// Render a "distribution" view matching the style of Figure 1 in the
    /// Linear Layouts paper.
    ///
    /// The **destination** space forms the grid. Three levels of hierarchy:
    ///
    /// - **Warp** (`watermark_axis`): large semi-transparent label ("w0", "w1")
    ///   centred in each group's region.
    ///
    /// - **Thread** (`label_axis`): one bold label ("t0", "t5", …) centred in
    ///   the bounding box of all cells belonging to that thread. Background color
    ///   cycles through a 4-color pastel palette using
    ///   `src[label_axis] / color_stride % 4` — set `color_stride` so that
    ///   threads sharing a column group get the same color (e.g. 4 or 8).
    ///
    /// - **Register** (remaining axes): small gray label ("r0", "r2", …) in the
    ///   top-left corner of each individual cell.
    ///
    /// Works best when the layout is injective (one source per destination cell).
    pub fn render_distribution_svg(
        &self,
        cell_px: u32,
        watermark_axis: usize,
        label_axis: usize,
        color_stride: usize,
    ) -> Result<String, LayoutError> {
        let src_shape = &self.src.shape;
        let dst_shape = &self.dst.shape;

        if dst_shape.is_empty() {
            return Err(LayoutError::MathError(
                "cannot render a 0-D destination space".to_string(),
            ));
        }
        if watermark_axis >= src_shape.len() {
            return Err(LayoutError::MathError(format!(
                "watermark_axis {watermark_axis} out of range"
            )));
        }
        if label_axis >= src_shape.len() {
            return Err(LayoutError::MathError(format!(
                "label_axis {label_axis} out of range"
            )));
        }

        let (nrows, ncols) = match dst_shape.len() {
            1 => (1usize, dst_shape[0]),
            _ => (dst_shape[0], dst_shape[1..].iter().product()),
        };

        // grid[row * ncols + col] = src coord that maps to that dst cell.
        let mut grid: Vec<Option<Vec<usize>>> = vec![None; nrows * ncols];
        for coord in iterate_coords(src_shape) {
            let dst_coord = self.map_coord(&coord)?;
            let (row, col) = dst_to_rowcol(&dst_coord, dst_shape);
            let idx = row * ncols + col;
            if idx < grid.len() {
                grid[idx] = Some(coord);
            }
        }

        let border = 1u32;
        let pad = 20u32;
        let font_size = (cell_px as f64 * 0.32).max(8.0) as u32;
        let small_font = (cell_px as f64 * 0.22).max(6.0) as u32;
        let svg_w = ncols as u32 * cell_px + 2 * pad;
        let svg_h = nrows as u32 * cell_px + 2 * pad;
        let stride = color_stride.max(1);

        let mut out = String::with_capacity(grid.len() * 220);
        out.push_str(&format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">"#
        ));
        out.push_str(&format!(
            r##"<rect width="{svg_w}" height="{svg_h}" fill="#f8f8f8"/>"##
        ));

        // ── Pass 1: cell background rects ─────────────────────────────────────
        for row in 0..nrows {
            for col in 0..ncols {
                let x = pad + col as u32 * cell_px;
                let y = pad + row as u32 * cell_px;
                let fill = match &grid[row * ncols + col] {
                    None => "hsl(0,0%,88%)",
                    Some(src) => palette_color(src[label_axis] / stride),
                };
                out.push_str(&format!(
                    r##"<rect x="{x}" y="{y}" width="{cell_px}" height="{cell_px}" fill="{fill}" stroke="#ccc" stroke-width="{border}"/>"##,
                ));
            }
        }

        // ── Pass 2: small register labels (top-left corner of each cell) ──────
        for row in 0..nrows {
            for col in 0..ncols {
                if let Some(src) = &grid[row * ncols + col] {
                    let reg = format_register_label(
                        src, watermark_axis, label_axis, &self.src.labels,
                    );
                    if !reg.is_empty() {
                        let x = pad + col as u32 * cell_px + 2;
                        let y = pad + row as u32 * cell_px + small_font + 1;
                        out.push_str(&format!(
                            r##"<text x="{x}" y="{y}" font-family="monospace" font-size="{small_font}" fill="#888">{reg}</text>"##,
                        ));
                    }
                }
            }
        }

        // ── Pass 3: super-cell labels (one per label_axis value) ──────────────
        {
            let label_range = src_shape[label_axis];
            let label_prefix = self.src.labels
                .get(label_axis)
                .and_then(|s| s.chars().next())
                .unwrap_or('t');

            let mut bounds: Vec<Option<[usize; 4]>> = vec![None; label_range];
            for row in 0..nrows {
                for col in 0..ncols {
                    if let Some(src) = &grid[row * ncols + col] {
                        let g = src[label_axis];
                        bounds[g] = Some(match bounds[g] {
                            None => [row, row, col, col],
                            Some([r0, r1, c0, c1]) => [
                                r0.min(row), r1.max(row),
                                c0.min(col), c1.max(col),
                            ],
                        });
                    }
                }
            }

            for g in 0..label_range {
                if let Some([r0, r1, c0, c1]) = bounds[g] {
                    let cx = pad + c0 as u32 * cell_px
                        + (c1 - c0 + 1) as u32 * cell_px / 2;
                    let cy = pad + r0 as u32 * cell_px
                        + (r1 - r0 + 1) as u32 * cell_px / 2
                        + font_size / 3;
                    let lbl = format!("{label_prefix}{g}");
                    out.push_str(&format!(
                        r##"<text x="{cx}" y="{cy}" font-family="monospace" font-size="{font_size}" font-weight="bold" text-anchor="middle" fill="#222">{lbl}</text>"##,
                    ));
                }
            }
        }

        // ── Pass 4: warp watermarks (one per watermark_axis value) ────────────
        {
            let wm_range = src_shape[watermark_axis];
            let wm_prefix = self.src.labels
                .get(watermark_axis)
                .and_then(|s| s.chars().next())
                .unwrap_or('w');
            let wm_font = ((nrows as u32 * cell_px / wm_range.max(1) as u32) / 2)
                .min(60)
                .max(10);

            let mut wm_rows: Vec<Vec<u32>> = vec![Vec::new(); wm_range];
            let mut wm_cols: Vec<Vec<u32>> = vec![Vec::new(); wm_range];
            for row in 0..nrows {
                for col in 0..ncols {
                    if let Some(src) = &grid[row * ncols + col] {
                        let g = src[watermark_axis];
                        wm_rows[g].push(row as u32);
                        wm_cols[g].push(col as u32);
                    }
                }
            }
            for g in 0..wm_range {
                if wm_rows[g].is_empty() { continue; }
                let med_r = { let mut v = wm_rows[g].clone(); v.sort_unstable(); v[v.len() / 2] };
                let med_c = { let mut v = wm_cols[g].clone(); v.sort_unstable(); v[v.len() / 2] };
                let wx = pad + med_c * cell_px + cell_px / 2;
                let wy = pad + med_r * cell_px + cell_px / 2 + wm_font / 3;
                let wlabel = format!("{wm_prefix}{g}");
                out.push_str(&format!(
                    r##"<text x="{wx}" y="{wy}" font-family="sans-serif" font-size="{wm_font}" font-weight="bold" text-anchor="middle" fill="#000" fill-opacity="0.12">{wlabel}</text>"##,
                ));
            }
        }

        // ── Dst axis labels ────────────────────────────────────────────────────
        if !self.dst.labels.is_empty() && dst_shape.len() >= 2 {
            let label_size = font_size.saturating_sub(2).max(7);
            let rx = pad / 2;
            let ry = pad + (nrows as u32 * cell_px) / 2;
            out.push_str(&format!(
                r##"<text x="{rx}" y="{ry}" font-family="monospace" font-size="{label_size}" text-anchor="middle" fill="#555" transform="rotate(-90,{rx},{ry})">{}</text>"##,
                self.dst.labels[0]
            ));
            let col_label = self.dst.labels[1..].join(",");
            let cx = pad + (ncols as u32 * cell_px) / 2;
            let cy = pad / 2 + label_size;
            out.push_str(&format!(
                r##"<text x="{cx}" y="{cy}" font-family="monospace" font-size="{label_size}" text-anchor="middle" fill="#555">{col_label}</text>"##,
            ));
        }

        out.push_str("</svg>");
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_render_produces_svg() {
        let l = RankedDigitLayout::right_major(&[4, 4]).unwrap();
        let svg = l.render_svg(40).unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        // All 16 offsets 0..=15 should appear.
        for i in 0..16usize {
            assert!(svg.contains(&format!(">{i}<")), "missing offset {i}");
        }
    }

    #[test]
    fn test_render_1d() {
        let l = RankedDigitLayout::right_major(&[8]).unwrap();
        let svg = l.render_svg(40).unwrap();
        assert!(svg.starts_with("<svg"));
    }

    #[test]
    fn test_render_distribution_hw_tile() {
        use crate::{Space, Transform, canonical_radices};

        // Figure-1 style: (warp=2, lane=32, trow=2, tcol=2) → tile [16×16]
        let hw_space   = Space::named(&[2, 32, 2, 2], &["warp", "lane", "trow", "tcol"]).unwrap();
        let tile_space = Space::named(&[16, 16], &["row", "col"]).unwrap();
        let hw_radices = canonical_radices(&hw_space.shape).unwrap();
        // Layout B permutation: order = [0, 5, 6, 7, 1, 2, 3, 4]
        let hw_to_tile = Transform::permute(hw_radices, vec![0, 5, 6, 7, 1, 2, 3, 4]).unwrap();
        let hw_layout  = RankedDigitLayout::new(hw_space, hw_to_tile, tile_space, "hw").unwrap();

        // watermark_axis=0 (warp), label_axis=1 (lane), color_stride=8
        let svg = hw_layout.render_distribution_svg(20, 0, 1, 8).unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        // Warp watermarks w0/w1 and thread labels l0..l31 should appear.
        assert!(svg.contains("w0"));
        assert!(svg.contains("w1"));
        assert!(svg.contains("l0"));
    }
}
