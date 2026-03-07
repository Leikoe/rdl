use std::collections::HashMap;

use crate::error::LayoutError;
use crate::layout::RankedDigitLayout;
use crate::math_utils::iterate_coords;

/// Map a value in [0, max] to an HSL colour string.
/// Hue sweeps from blue (low) to red (high) — periodic patterns pop visually.
fn value_to_hsl(value: usize, max: usize) -> String {
    if max == 0 {
        return "hsl(0,0%,88%)".to_string();
    }
    let hue = 240.0 - (value as f64 / max as f64) * 240.0;
    format!("hsl({:.1},65%,68%)", hue)
}

/// Choose text colour (black or white) for legibility on the background.
fn text_colour(value: usize, max: usize) -> &'static str {
    if max == 0 {
        return "black";
    }
    let t = value as f64 / max as f64;
    if t < 0.25 || t > 0.75 {
        "black"
    } else {
        "white"
    }
}

/// Four-colour soft-pastel palette matching Figure 1 of the Linear Layouts paper.
const PALETTE: &[&str] = &[
    "hsl(50,70%,82%)",  // soft yellow
    "hsl(130,45%,80%)", // soft green
    "hsl(210,55%,82%)", // soft blue
    "hsl(280,45%,80%)", // soft purple
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
            let col = dst_coord[1..]
                .iter()
                .zip(dst_shape[1..].iter())
                .fold(0usize, |acc, (&v, &s)| acc * s + v);
            (dst_coord[0], col)
        }
    }
}

impl RankedDigitLayout {
    /// Render the layout as an SVG grid.
    ///
    /// Each cell corresponds to one source coordinate.  Its fill colour encodes
    /// the flat destination offset; the numeric value is printed inside.
    ///
    /// For 1-D sources: a single row.
    /// For 2-D sources: rows × cols grid directly.
    /// For N-D sources: first axis as rows, remaining axes flattened as columns.
    pub fn render_svg(&self, cell_px: u32) -> Result<String, LayoutError> {
        let shape = &self.src.shape;

        if shape.is_empty() {
            return Err(LayoutError::MathError(
                "cannot render a 0-D space".to_string(),
            ));
        }

        let (nrows, ncols) = match shape.len() {
            1 => (1usize, shape[0]),
            _ => (shape[0], shape[1..].iter().product()),
        };

        let coords = iterate_coords(shape);
        let offsets: Vec<usize> = coords
            .iter()
            .map(|c| self.flat_index(c))
            .collect::<Result<_, _>>()?;

        let max_offset = *offsets.iter().max().unwrap_or(&0);

        let border = 1u32;
        let pad = 12u32;
        let font_size = (cell_px as f64 * 0.32).max(8.0) as u32;
        let svg_w = ncols as u32 * cell_px + 2 * pad;
        let svg_h = nrows as u32 * cell_px + 2 * pad;

        let mut out = String::with_capacity(offsets.len() * 120);

        out.push_str(&format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">"#
        ));
        out.push_str(&format!(
            r##"<rect width="{svg_w}" height="{svg_h}" fill="#f8f8f8"/>"##
        ));

        for (i, (&offset, _coord)) in offsets.iter().zip(coords.iter()).enumerate() {
            let row = i / ncols;
            let col = i % ncols;
            let x = pad + col as u32 * cell_px;
            let y = pad + row as u32 * cell_px;

            let fill = value_to_hsl(offset, max_offset);
            let fg = text_colour(offset, max_offset);

            out.push_str(&format!(
                r##"<rect x="{x}" y="{y}" width="{cell_px}" height="{cell_px}" fill="{fill}" stroke="#fff" stroke-width="{border}"/>"##,
            ));

            let tx = x + cell_px / 2;
            let ty = y + cell_px / 2 + font_size / 3;
            out.push_str(&format!(
                r#"<text x="{tx}" y="{ty}" font-family="monospace" font-size="{font_size}" text-anchor="middle" fill="{fg}">{offset}</text>"#,
            ));
        }

        if !self.src.labels.is_empty() && shape.len() >= 2 {
            let label_size = font_size.saturating_sub(2).max(7);
            let rx = pad / 2;
            let ry = pad + (nrows as u32 * cell_px) / 2;
            out.push_str(&format!(
                r##"<text x="{rx}" y="{ry}" font-family="monospace" font-size="{label_size}" text-anchor="middle" fill="#666" transform="rotate(-90,{rx},{ry})">{}</text>"##,
                self.src.labels[0]
            ));
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
    /// Linear Layouts paper.  Completely axis-agnostic — no parameters beyond
    /// cell size.
    ///
    /// The **destination** space is the grid.  Source hierarchy is inferred
    /// automatically:
    ///
    /// - **First element of each dimension** (value 0) shows the inner
    ///   composition by leaving it blank — the surrounding non-first elements
    ///   are labelled, making the structure visible.
    ///
    /// - **Every other element** (value ≥ 1) shows `axisname + index` (e.g.
    ///   `"warp1"`, `"lane5"`) centred in the bounding box of all destination
    ///   cells that share the same source prefix up to that dimension.
    ///
    /// Background colour cycles through a 4-colour pastel palette based on
    /// the first (coarsest) source axis value.
    ///
    /// Works best when the layout is injective (one source per destination cell).
    pub fn render_distribution_svg(&self, cell_px: u32) -> Result<String, LayoutError> {
        let src_shape = &self.src.shape;
        let dst_shape = &self.dst.shape;

        if dst_shape.is_empty() {
            return Err(LayoutError::MathError(
                "cannot render a 0-D destination space".to_string(),
            ));
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
        let svg_w = ncols as u32 * cell_px + 2 * pad;
        let svg_h = nrows as u32 * cell_px + 2 * pad;
        let ndim = src_shape.len();

        let mut out = String::with_capacity(grid.len() * 180);
        out.push_str(&format!(
            r#"<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}">"#
        ));
        out.push_str(&format!(
            r##"<rect width="{svg_w}" height="{svg_h}" fill="#f8f8f8"/>"##
        ));

        // ── Pass 1: cell backgrounds — colour by first (coarsest) axis ────────
        for row in 0..nrows {
            for col in 0..ncols {
                let x = pad + col as u32 * cell_px;
                let y = pad + row as u32 * cell_px;
                let fill = match &grid[row * ncols + col] {
                    None => "hsl(0,0%,88%)",
                    Some(s) => palette_color(s.first().copied().unwrap_or(0)),
                };
                out.push_str(&format!(
                    r##"<rect x="{x}" y="{y}" width="{cell_px}" height="{cell_px}" fill="{fill}" stroke="#ccc" stroke-width="{border}"/>"##,
                ));
            }
        }

        // ── Pass 2: hierarchical labels ───────────────────────────────────────
        //
        // For each dimension d and value v:
        //   Region = cells where s[0..d-1] are ALL zero AND s[d] == v.
        //   Label  = "{axis_name}{v}" centred in that region's bounding box.
        //
        // The all-zeros prefix rule means:
        //   - All values of d=0 (warp) are labelled across the full grid.
        //   - Lane labels only appear inside warp=0 (not inside warp=1, etc.).
        //   - Register labels only appear inside warp=0,lane=0.
        //
        // Font size and opacity are depth-based:
        //   d=0 (coarsest): large semi-transparent watermark.
        //   d=1:            medium bold, fully opaque.
        //   d≥2 (fine):     small, fully opaque.
        for d in 0..ndim {
            // Accumulate bounding boxes keyed by v; prefix is implicitly all-zeros.
            let mut bounds: HashMap<usize, [usize; 4]> = HashMap::new();

            for row in 0..nrows {
                for col in 0..ncols {
                    if let Some(s) = &grid[row * ncols + col] {
                        // Only render labels inside the all-zeros prefix path.
                        if !s[..d].iter().all(|&x| x == 0) {
                            continue;
                        }
                        let v = s[d];
                        let bb = bounds.entry(v).or_insert([row, row, col, col]);
                        bb[0] = bb[0].min(row);
                        bb[1] = bb[1].max(row);
                        bb[2] = bb[2].min(col);
                        bb[3] = bb[3].max(col);
                    }
                }
            }

            let axis_name = self
                .src
                .labels
                .get(d)
                .filter(|s| !s.is_empty())
                .map(|s| s.as_str())
                .unwrap_or("dim");

            let (font, opacity) = match d {
                0 => (((cell_px as f64 * 0.75) as u32).min(48).max(8), "0.22"),
                1 => (((cell_px as f64 * 0.42) as u32).min(20).max(7), "1"),
                _ => (((cell_px as f64 * 0.28) as u32).min(14).max(6), "1"),
            };

            for (v, [r0, r1, c0, c1]) in &bounds {
                let lbl = format!("{axis_name}{v}");
                let cx = pad + *c0 as u32 * cell_px + (c1 - c0 + 1) as u32 * cell_px / 2;
                let cy = pad + *r0 as u32 * cell_px + (r1 - r0 + 1) as u32 * cell_px / 2 + font / 3;
                out.push_str(&format!(
                    r##"<text x="{cx}" y="{cy}" font-family="monospace" font-size="{font}" font-weight="bold" text-anchor="middle" fill="#222" fill-opacity="{opacity}">{lbl}</text>"##,
                ));
            }
        }

        // ── Dst axis labels ────────────────────────────────────────────────────
        if !self.dst.labels.is_empty() && dst_shape.len() >= 2 {
            let label_size = 10u32;
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
        use crate::{canonical_radices, Space, Transform};

        // Figure-1 style: (warp=2, lane=32, trow=2, tcol=2) → tile [16×16]
        let hw_space = Space::named(&[2, 32, 2, 2], &["warp", "lane", "trow", "tcol"]).unwrap();
        let tile_space = Space::named(&[16, 16], &["row", "col"]).unwrap();
        let hw_radices = canonical_radices(&hw_space.shape).unwrap();
        // Layout B permutation
        let hw_to_tile = Transform::permute(hw_radices, vec![0, 5, 6, 7, 1, 2, 3, 4]).unwrap();
        let hw_layout = RankedDigitLayout::new(hw_space, hw_to_tile, tile_space, "hw").unwrap();

        let svg = hw_layout.render_distribution_svg(20).unwrap();
        assert!(svg.starts_with("<svg"));
        assert!(svg.ends_with("</svg>"));
        // warp1 and lane1 labels must appear.
        assert!(svg.contains("warp1"));
        assert!(svg.contains("lane1"));
    }
}
