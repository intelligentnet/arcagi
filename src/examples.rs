use std::collections::{HashMap, BTreeMap, BTreeSet};
use crate::cats::*;
use crate::data::*;
use crate::grid::*;
use crate::shape::*;

//#[derive(Debug, Clone, PartialEq)]
//pub enum BoolOp { And, Or, Xor }

#[derive(Debug, Clone, PartialEq)]
pub struct Example {
    pub input: QualifiedGrid,
    pub output: QualifiedGrid,
    pub cat: BTreeSet<GridCategory>,
    pub pairs: Vec<(Shape, Shape, bool)>,
    pub coloured_pairs: Vec<(Shape, Shape, bool)>,
    //categorise_io_edges(in_shapes: &mut Shapes, out_shapes: &Shapes) {
    //edges: BTreeMap<Shape, BTreeSet<ShapeEdgeCategory>>,
    //pub io_edges: BTreeSet<ShapeEdgeCategory>,
}

impl Example {
    pub fn new(data: &IO) -> Self {
        let input = QualifiedGrid::new(&data.input);
        let output = match &data.output {
            Some(output) => QualifiedGrid::new(output),
            None => QualifiedGrid::trivial(),
        };
        let cat = Example::categorise_grid(&input, &output);
        let pairs = Vec::new();
        let coloured_pairs = Vec::new();

        Example { input, output, cat, pairs, coloured_pairs }
    }

    /*
    #[allow(clippy::nonminimal_bool)]
    pub fn bool_op(&self, op: BoolOp) -> Option<Grid> {
        let g1 = &self.input;
        let g2 = &self.output;

        if g1.x != g2.x || g1.y != g2.y {
            return None;
        }

        let mut interesting = false;
        let mut newg = Grid::blank(g1.x, g1.y);

        for (c1, c2) in g1.cells.iter().zip(&g2.cells) {
            if (op == BoolOp::Xor && c1.colour != Colour::Black && c1.colour != Colour::Black && c1.colour != c2.colour) || 
                (op == BoolOp::And && c1.colour != Colour::Black && c1.colour != Colour::Black && c1.colour == c2.colour) ||
                (op == BoolOp::Or && (c1.colour != Colour::Black || c2.colour != Colour::Black))
            {
                newg.cells[(c1.x, c1.y)].colour = Colour::NoColour;
            } else {
                interesting = true;
            }
        }

        if interesting { Some(newg) } else { None }
    }
    */

    pub fn is_equal(&self) -> bool {
        self.input.grid.cells.columns == self.output.grid.cells.columns && self.input.grid.cells.rows == self.output.grid.cells.rows
    }

    pub fn is_bigger(&self) -> bool {
        self.input.grid.cells.columns * self.input.grid.cells.rows > self.output.grid.cells.columns * self.output.grid.cells.rows
    }

    pub fn is_smaller(&self) -> bool {
        self.input.grid.cells.columns * self.input.grid.cells.rows < self.output.grid.cells.columns * self.output.grid.cells.rows
    }

    pub fn diff(&self) -> Option<Grid> {
        self.input.grid.diff(&self.output.grid)
    }

    pub fn categorise_grid(input: &QualifiedGrid, output: &QualifiedGrid ) -> BTreeSet<GridCategory> {
        let mut cats: BTreeSet<GridCategory> = BTreeSet::new();

        if output.grid.size() == 0 {
            cats.insert(GridCategory::EmptyOutput);

            //return cats;
        }

        let in_dim = input.grid.dimensions();
        let out_dim = output.grid.dimensions();

        if input.grid.is_empty() {
            cats.insert(GridCategory::InEmpty);
        }
        if in_dim.0 > 1 && in_dim.0 == in_dim.1 && out_dim.0 == out_dim.1 && in_dim == out_dim {
            cats.insert(GridCategory::InOutSquareSameSize);
            //cats.insert(GridCategory::InOutSameSize);
            if in_dim.0 % 2 == 0 {
                cats.insert(GridCategory::InOutSquareSameSizeEven);
            } else {
                cats.insert(GridCategory::InOutSquareSameSizeOdd);
            }

            if input.grid.rotated_90(1).to_json() == output.grid.to_json() {
                cats.insert(GridCategory::Rot90);
            }
            if input.grid.rotated_90(2).to_json() == output.grid.to_json() {
                cats.insert(GridCategory::Rot180);
            }
            if input.grid.rotated_270(1).to_json() == output.grid.to_json() {
                cats.insert(GridCategory::Rot270);
            }
            if input.grid.transposed().to_json() == output.grid.to_json() {
                cats.insert(GridCategory::Transpose);
            }
            /* There are none?
            if input.grid.inv_transposed().to_json() == output.grid.to_json() {
                cats.insert(GridCategory::InvTranspose);
            }
            */
            if input.grid.mirrored_x().to_json() == output.grid.to_json() {
                cats.insert(GridCategory::MirroredX);
            }
            if input.grid.mirrored_y().to_json() == output.grid.to_json() {
                cats.insert(GridCategory::MirroredY);
            }
        } else {
            //if (in_dim.0 > 1 || in_dim.1 > 1) && in_dim == out_dim {
            if in_dim.0 == out_dim.0 && in_dim.1 == out_dim.1 {
                cats.insert(GridCategory::InOutSameSize);
            }
            if in_dim.0 > 1 && out_dim.0 > 1 && in_dim.0 == in_dim.1 && out_dim.0 == out_dim.1 {
                cats.insert(GridCategory::InOutSquare);
            } else if in_dim.0 > 1 && in_dim.0 == in_dim.1 {
                cats.insert(GridCategory::InSquare);
            } else if out_dim.0 > 1 && out_dim.0 == out_dim.1 {
                cats.insert(GridCategory::OutSquare);
            }
        }
        if out_dim.0 == 1 && out_dim.1 == 1 {
            cats.insert(GridCategory::SinglePixelOut);
        }
        if in_dim.0 >= out_dim.0 && in_dim.1 > out_dim.1 || in_dim.0 > out_dim.0 && in_dim.1 >= out_dim.1 {
            cats.insert(GridCategory::OutLessThanIn);
        } else if in_dim.0 <= out_dim.0 && in_dim.1 < out_dim.1 || in_dim.0 < out_dim.0 && in_dim.1 <= out_dim.1 {
            cats.insert(GridCategory::InLessThanOut);
        }
        if input.grid.is_symmetric() {
            cats.insert(GridCategory::SymmetricIn);
        }
        if output.grid.is_symmetric() {
            cats.insert(GridCategory::SymmetricOut);
        }

        let in_is_mirror_x = input.grid.is_mirror_x();
        let in_is_mirror_y = input.grid.is_mirror_y();
        let out_is_mirror_x = output.grid.is_mirror_x();
        let out_is_mirror_y = output.grid.is_mirror_y();
        if in_is_mirror_x {
            cats.insert(GridCategory::MirrorXIn);
        }
        if in_is_mirror_y {
            cats.insert(GridCategory::MirrorYIn);
        }
        if out_is_mirror_x {
            cats.insert(GridCategory::MirrorXOut);
        }
        if out_is_mirror_y {
            cats.insert(GridCategory::MirrorYOut);
        }
        /*
        if input.grid_likelyhood() > 0.5 {
            cats.insert(GridCategory::GridLikelyhood);
        }
        if input.is_mirror_offset_x(-1) {
            cats.insert(GridCategory::MirrorXInSkewR);  // FIX
        }
        if input.is_mirror_offset_x(1) {
            cats.insert(GridCategory::MirrorXInSkewL);
        }
        if output.is_mirror_offset_x(-1) {
            cats.insert(GridCategory::MirrorXOutSkewR);
        }
        if output.is_mirror_offset_x(1) {
            cats.insert(GridCategory::MirrorXOutSkewL);
        }
        if input.is_mirror_offset_y(-1) {
            cats.insert(GridCategory::MirrorYInSkewR);
        }
        if input.is_mirror_offset_y(1) {
            cats.insert(GridCategory::MirrorYInSkewL);
        }
        if output.is_mirror_offset_y(-1) {
            cats.insert(GridCategory::MirrorYOutSkewR);
        }
        if output.is_mirror_offset_y(1) {
            cats.insert(GridCategory::MirrorYOutSkewL);
        }
        */
        if input.grid.has_bg_grid() != Colour::NoColour {
            cats.insert(GridCategory::BGGridInBlack);
        }
        if output.grid.has_bg_grid() != Colour::NoColour {
            cats.insert(GridCategory::BGGridOutBlack);
        }
        if input.grid.has_bg_grid_coloured() != Colour::NoColour {
            cats.insert(GridCategory::BGGridInColoured);
        }
        if output.grid.has_bg_grid_coloured() != Colour::NoColour {
            cats.insert(GridCategory::BGGridOutColoured);
        }
        if input.grid.is_panelled_x() {
            cats.insert(GridCategory::IsPanelledXIn);
        }
        if output.grid.is_panelled_x() {
            cats.insert(GridCategory::IsPanelledXOut);
        }
        if input.grid.is_panelled_y() {
            cats.insert(GridCategory::IsPanelledYIn);
        }
        if output.grid.is_panelled_y() {
            cats.insert(GridCategory::IsPanelledYOut);
        }
        let in_no_colours = input.grid.no_colours();
        let out_no_colours = output.grid.no_colours();
        if in_no_colours == 0 {
            cats.insert(GridCategory::BlankIn);
        }
        if out_no_colours == 0 {
            cats.insert(GridCategory::BlankOut);
        }
        if in_no_colours == 1 {
            cats.insert(GridCategory::SingleColourIn);
        }
        if out_no_colours == 1 {
            cats.insert(GridCategory::SingleColourOut);
        }
        if input.grid.colour == output.grid.colour && input.grid.colour != Colour::Mixed {
            cats.insert(GridCategory::SameColour);
        }
        if input.shapes.len() == 1 {
            cats.insert(GridCategory::SingleShapeIn);
        } else if input.coloured_shapes.len() == 1 {
            cats.insert(GridCategory::SingleColouredShapeIn);
        }
        if output.shapes.len() == 1 {
            cats.insert(GridCategory::SingleShapeOut);
        } else if output.coloured_shapes.len() == 1 {
            cats.insert(GridCategory::SingleColouredShapeOut);
        }
        if input.shapes.len() > 1 && input.shapes.len() == output.shapes.len() {
            cats.insert(GridCategory::InSameCountOut);
        } else if input.shapes.len() > 1 && input.coloured_shapes.len() == output.coloured_shapes.len() {
            cats.insert(GridCategory::InSameCountOutColoured);
        } else if input.shapes.len() < output.shapes.len() {
            cats.insert(GridCategory::InLessCountOut);
        } else if input.coloured_shapes.len() < output.coloured_shapes.len() {
            cats.insert(GridCategory::InLessCountOutColoured);
        } else if input.shapes.len() < output.shapes.len() {
            cats.insert(GridCategory::OutLessCountIn);
        } else if input.coloured_shapes.len() > output.coloured_shapes.len() {
            cats.insert(GridCategory::OutLessCountInColoured);
        }
        let in_border_top = input.grid.border_top();
        let in_border_bottom = input.grid.border_bottom();
        let in_border_left = input.grid.border_left();
        let in_border_right = input.grid.border_right();
        if in_border_top {
            cats.insert(GridCategory::BorderTopIn);
        }
        if in_border_bottom {
            cats.insert(GridCategory::BorderBottomIn);
        }
        if in_border_left {
            cats.insert(GridCategory::BorderLeftIn);
        }
        if in_border_right {
            cats.insert(GridCategory::BorderRightIn);
        }
        if output.grid.size() > 0 {
            let out_border_top = output.grid.border_top();
            let out_border_bottom = output.grid.border_bottom();
            let out_border_left = output.grid.border_left();
            let out_border_right = output.grid.border_right();
            if out_border_top {
                cats.insert(GridCategory::BorderTopOut);
            }
            if out_border_bottom {
                cats.insert(GridCategory::BorderBottomOut);
            }
            if out_border_left {
                cats.insert(GridCategory::BorderLeftOut);
            }
            if out_border_right {
                cats.insert(GridCategory::BorderRightOut);
            }
        }
        if input.grid.even_rows() {
            cats.insert(GridCategory::EvenRowsIn);
        }
        if output.grid.even_rows() {
            cats.insert(GridCategory::EvenRowsOut);
        }
        if input.grid.is_full() {
            cats.insert(GridCategory::FullyPopulatedIn);
        }
        if output.grid.is_full() {
            cats.insert(GridCategory::FullyPopulatedOut);
        }
        if !input.grid.has_gravity_down() && output.grid.has_gravity_down() {
            cats.insert(GridCategory::GravityDown);
        } else if !input.grid.has_gravity_up() && output.grid.has_gravity_up() {
            cats.insert(GridCategory::GravityUp);
        } else if !input.grid.has_gravity_left() && output.grid.has_gravity_left() {
            cats.insert(GridCategory::GravityLeft);
        } else if !input.grid.has_gravity_right() && output.grid.has_gravity_right() {
            cats.insert(GridCategory::GravityRight);
        }
        if input.grid.is_3x3() {
            cats.insert(GridCategory::Is3x3In);
        }
        if output.grid.is_3x3() {
            cats.insert(GridCategory::Is3x3Out);
        }
        if input.grid.div9() {
            cats.insert(GridCategory::Div9In);
        }
        if output.grid.div9() {
            cats.insert(GridCategory::Div9Out);
        }
        if in_dim.0 * 2 == out_dim.0 && in_dim.1 * 2 == out_dim.1 {
            cats.insert(GridCategory::Double);
        }
        if in_dim.0 == 3 && in_dim.1 == 7 {
            cats.insert(GridCategory::In3x7);
        }
        if in_dim.0 == 7 && in_dim.1 == 3 {
            cats.insert(GridCategory::In7x3);
        }
        if input.shapes.shapes.len() == output.shapes.shapes.len() {
            cats.insert(GridCategory::InOutShapeCount);
        }
        if input.shapes.coloured_shapes.len() == output.coloured_shapes.shapes.len() {
            cats.insert(GridCategory::InOutShapeCountColoured);
        }
        if !input.black.shapes.is_empty() {
            cats.insert(GridCategory::BlackPatches);
        }
        if input.has_bg_shape() && output.has_bg_shape() {
            cats.insert(GridCategory::HasBGShape);
        }
        if input.has_bg_coloured_shape() && output.has_bg_coloured_shape() {
            cats.insert(GridCategory::HasBGShapeColoured);
        }
        let hin = input.grid.cell_colour_cnt_map();
        let hout = output.grid.cell_colour_cnt_map();
        if hin == hout {
            cats.insert(GridCategory::IdenticalColours);
        } else if hin.len() == hout.len() {
            cats.insert(GridCategory::IdenticalNoColours);
        } else {
            let inp: usize = hin.values().sum();
            let outp: usize = hout.values().sum();

            if inp == outp {
                cats.insert(GridCategory::IdenticalNoPixels);
            }
        }

        cats
    }

    pub fn single_shape_in(&self) -> usize {
        self.input.shapes.len()
    }

    pub fn single_shape_out(&self) -> usize {
        self.output.shapes.len()
    }

    pub fn single_coloured_shape_in(&self) -> usize {
        self.input.coloured_shapes.len()
    }

    pub fn single_coloured_shape_out(&self) -> usize {
        self.output.coloured_shapes.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Examples {
    pub examples: Vec<Example>,
    pub tests: Vec<Example>,
    pub cat: BTreeSet<GridCategory>,
}

impl Examples {
    pub fn new(data: &Data) -> Self {
        let mut examples: Vec<Example> = data.train.iter()
            .map(Example::new)
            .collect();

        let tests: Vec<Example> = data.test.iter()
            .map(Example::new)
            .collect();

        let cat = Self::categorise_grids(&mut examples);

        Examples { examples, tests, cat }
    }

    pub fn match_shapes(&self) -> BTreeMap<Shape, Shape> {
        let mut mapping: BTreeMap<Shape, Shape> = BTreeMap::new();

        for shapes in &self.examples {
            for (si, so) in shapes.input.coloured_shapes.shapes.iter().zip(shapes.output.coloured_shapes.shapes.iter()) {
                if so.is_contained(si) {
                    let (si, so) = so.normalise(si);

                    mapping.insert(si.clone(), so.clone());
                }
            }
        }

        mapping
    }

    pub fn categorise_grids(examples: &mut [Example]) -> BTreeSet<GridCategory> {
        let mut cats: BTreeSet<GridCategory> = BTreeSet::new();

        for ex in examples.iter_mut() {
            let mut cat = Example::categorise_grid(&ex.input, &ex.output);

            if cat.contains(&GridCategory::InOutSquareSameSize) {
                cat.insert(GridCategory::InOutSameSize);
            }
            ex.pairs = ex.input.shapes.pair_shapes(&ex.output.shapes, true);
            if !ex.pairs.is_empty() {
                cat.insert(GridCategory::InOutSameShapes);
            }
            ex.coloured_pairs = ex.input.coloured_shapes.pair_shapes(&ex.output.coloured_shapes, true);
            if !ex.coloured_pairs.is_empty() {
                cat.insert(GridCategory::InOutSameShapesColoured);
            }

            if cats.is_empty() {
                cats = cat;
            } else {
                cats = cats.intersection(&cat).cloned().collect();
            }
        }

        if cats.contains(&GridCategory::InOutSquareSameSize) && cats.contains(&GridCategory::InOutSameSize) {
            cats.remove(&GridCategory::InOutSameSize);
        }

        cats
    }

    pub fn find_input_colours(&self) -> Vec<Colour> {
        let mut common = Colour::all_colours();

        for ex in self.examples.iter() {
            let h = ex.input.grid.cell_colour_cnt_map();
            let v: BTreeSet<Colour> = h.keys().map(|c| *c).collect();

            common = common.intersection(&v).map(|c| *c).collect();
        }

        Vec::from_iter(common)
    }

    pub fn find_output_colours(&self) -> Vec<Colour> {
        let mut common = Colour::all_colours();

        for ex in self.examples.iter() {
            let h = ex.output.grid.cell_colour_cnt_map();
            let v: BTreeSet<Colour> = h.keys().map(|c| *c).collect();

            common = common.intersection(&v).map(|c| *c).collect();
        }

        Vec::from_iter(common)
    }

    pub fn find_hollow_cnt_colour_map(&self) -> BTreeMap<usize, Colour> {
        let mut ccm: BTreeMap<usize, Colour> = BTreeMap::new();

        for ex in self.examples.iter() {
            let h = ex.output.shapes.hollow_cnt_colour_map();

            for (k, v) in &h {
                ccm.insert(*k, *v);
            }
        }

        ccm
    }

    pub fn find_colour_io_map(&self) -> BTreeMap<Colour, Colour> {
        let mut h: BTreeMap<Colour, Colour> = BTreeMap::new();

        for ex in self.examples.iter() {
            if ex.input.shapes.shapes.len() != ex.output.shapes.shapes.len() {
                return h;
            }

            for (si, so) in ex.input.shapes.shapes.iter().zip(ex.output.shapes.shapes.iter()) {
                h.insert(si.colour, so.colour);
            }
        }

        h
    }

    pub fn largest_shape_colour(&self) -> Colour {
        let mut colour = Colour::NoColour;

        for ex in self.examples.iter() {
            let s = ex.output.shapes.largest();

            if colour == Colour::NoColour {
                colour = s.colour;
            } else if colour != s.colour {
                return Colour::NoColour;    // No common large colour
            }
        }

        colour
    }

    pub fn bleached_io_map(&self) -> HashMap<String, Grid> {
        let mut h: HashMap<String, Grid> = HashMap::new();

        for ex in self.examples.iter() {
            h.insert(ex.input.grid.bleach().to_json(), ex.output.grid.clone());
        }

        h
    }
}

