use std::collections::{HashMap, BTreeMap, BTreeSet};
use crate::cats::*;
use crate::cats::Colour::*;
use crate::cats::GridCategory::*;
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

    pub fn new_cons(data: &IO) -> Self {
        let input = QualifiedGrid::new_cons(&data.input);
        let output = match &data.output {
            Some(output) => QualifiedGrid::new_cons(output),
            None => QualifiedGrid::trivial(),
        };
        let cat = Example::categorise_grid(&input, &output);
        let pairs = Vec::new();
        let coloured_pairs = Vec::new();

        Example { input, output, cat, pairs, coloured_pairs }
    }

    pub fn transform(&self, trans: Transformation, input: bool) -> Self {
        let mut example = self.clone();

        example.transform_mut(trans, input);

        example
    }

    pub fn transform_mut(&mut self, trans: Transformation, input: bool) {
        let qgrid = if input { &mut self.input } else { &mut self.output };

        qgrid.grid = qgrid.grid.transform(trans);

        qgrid.shapes = if qgrid.bg == NoColour {
            qgrid.grid.to_shapes()
        } else {
            qgrid.grid.to_shapes_bg(qgrid.bg)
        };
        qgrid.coloured_shapes = if qgrid.bg == NoColour {
            qgrid.grid.to_shapes_coloured()
        } else {
            qgrid.grid.to_shapes_coloured_bg(qgrid.bg)
        };
        if !qgrid.black.is_empty() {
            qgrid.black = qgrid.grid.find_black_patches();
        }
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
            if (op == BoolOp::Xor && c1.colour != Black && c1.colour != Black && c1.colour != c2.colour) || 
                (op == BoolOp::And && c1.colour != Black && c1.colour != Black && c1.colour == c2.colour) ||
                (op == BoolOp::Or && (c1.colour != Black || c2.colour != Black))
            {
                newg.cells[(c1.x, c1.y)].colour = NoColour;
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
            cats.insert(EmptyOutput);
            //return cats;
        }

        let in_dim = input.grid.dimensions();
        let out_dim = output.grid.dimensions();

        if input.grid.is_empty() {
            cats.insert(InEmpty);
        }
        if in_dim.0 > 1 && in_dim.0 == in_dim.1 && out_dim.0 == out_dim.1 && in_dim == out_dim {
            cats.insert(InOutSquareSameSize);
            //cats.insert(InOutSameSize);
            if in_dim.0 % 2 == 0 {
                cats.insert(InOutSquareSameSizeEven);
            } else {
                cats.insert(InOutSquareSameSizeOdd);
            }

            if input.grid.rotated_90(1).to_json() == output.grid.to_json() {
                cats.insert(Rot90);
            }
            if input.grid.rotated_90(2).to_json() == output.grid.to_json() {
                cats.insert(Rot180);
            }
            if input.grid.rotated_270(1).to_json() == output.grid.to_json() {
                cats.insert(Rot270);
            }
            if input.grid.transposed().to_json() == output.grid.to_json() {
                cats.insert(Transpose);
            }
            /* There are none?
            if input.grid.inv_transposed().to_json() == output.grid.to_json() {
                cats.insert(InvTranspose);
            }
            */
            if input.grid.mirrored_rows().to_json() == output.grid.to_json() {
                cats.insert(MirroredX);
            }
            if input.grid.mirrored_cols().to_json() == output.grid.to_json() {
                cats.insert(MirroredY);
            }
        } else {
            //if (in_dim.0 > 1 || in_dim.1 > 1) && in_dim == out_dim {
            if in_dim.0 == out_dim.0 && in_dim.1 == out_dim.1 {
                cats.insert(InOutSameSize);
            }
            if in_dim.0 > 1 && out_dim.0 > 1 && in_dim.0 == in_dim.1 && out_dim.0 == out_dim.1 {
                cats.insert(InOutSquare);
            } else if in_dim.0 > 1 && in_dim.0 == in_dim.1 {
                cats.insert(InSquare);
            } else if out_dim.0 > 1 && out_dim.0 == out_dim.1 {
                cats.insert(OutSquare);
            }
        }
        if out_dim.0 == 1 && out_dim.1 == 1 {
            cats.insert(SinglePixelOut);
        }
        if in_dim.0 >= out_dim.0 && in_dim.1 > out_dim.1 || in_dim.0 > out_dim.0 && in_dim.1 >= out_dim.1 {
            cats.insert(OutLessThanIn);
        } else if in_dim.0 <= out_dim.0 && in_dim.1 < out_dim.1 || in_dim.0 < out_dim.0 && in_dim.1 <= out_dim.1 {
            cats.insert(InLessThanOut);
        }
        let is_mirror_rows_in = input.grid.is_mirror_rows();
        let is_mirror_cols_in = input.grid.is_mirror_cols();
        let is_mirror_rows_out = output.grid.is_mirror_rows();
        let is_mirror_cols_out = output.grid.is_mirror_cols();
        if is_mirror_rows_in && is_mirror_cols_in {
            cats.insert(SymmetricIn);
        } else if is_mirror_rows_in {
            cats.insert(SymmetricInUD);
        } else if is_mirror_cols_in {
            cats.insert(SymmetricInLR);
        }
        if is_mirror_rows_out && is_mirror_cols_out {
            cats.insert(SymmetricOut);
        } else if is_mirror_rows_out {
            cats.insert(SymmetricOutUD);
        } else if is_mirror_cols_out {
            cats.insert(SymmetricOutLR);
        }

        let in_is_mirror_x = input.grid.is_mirror_rows();
        let in_is_mirror_y = input.grid.is_mirror_cols();
        let out_is_mirror_x = output.grid.is_mirror_rows();
        let out_is_mirror_y = output.grid.is_mirror_cols();
        if in_is_mirror_x {
            cats.insert(MirrorXIn);
        }
        if in_is_mirror_y {
            cats.insert(MirrorYIn);
        }
        if out_is_mirror_x {
            cats.insert(MirrorXOut);
        }
        if out_is_mirror_y {
            cats.insert(MirrorYOut);
        }
        /*
        if input.grid_likelyhood() > 0.5 {
            cats.insert(GridLikelyhood);
        }
        if input.is_mirror_offset_x(-1) {
            cats.insert(MirrorXInSkewR);  // FIX
        }
        if input.is_mirror_offset_x(1) {
            cats.insert(MirrorXInSkewL);
        }
        if output.is_mirror_offset_x(-1) {
            cats.insert(MirrorXOutSkewR);
        }
        if output.is_mirror_offset_x(1) {
            cats.insert(MirrorXOutSkewL);
        }
        if input.is_mirror_offset_y(-1) {
            cats.insert(MirrorYInSkewR);
        }
        if input.is_mirror_offset_y(1) {
            cats.insert(MirrorYInSkewL);
        }
        if output.is_mirror_offset_y(-1) {
            cats.insert(MirrorYOutSkewR);
        }
        if output.is_mirror_offset_y(1) {
            cats.insert(MirrorYOutSkewL);
        }
        */
        if input.grid.has_bg_grid() != NoColour {
            cats.insert(BGGridInBlack);
        }
        if output.grid.has_bg_grid() != NoColour {
            cats.insert(BGGridOutBlack);
        }
        if input.grid.has_bg_grid_coloured() != NoColour {
            cats.insert(BGGridInColoured);
        }
        if output.grid.has_bg_grid_coloured() != NoColour {
            cats.insert(BGGridOutColoured);
        }
        if input.grid.is_panelled_rows() {
            cats.insert(IsPanelledXIn);
        }
        if output.grid.is_panelled_rows() {
            cats.insert(IsPanelledXOut);
        }
        if input.grid.is_panelled_cols() {
            cats.insert(IsPanelledYIn);
        }
        if output.grid.is_panelled_cols() {
            cats.insert(IsPanelledYOut);
        }
        let in_no_colours = input.grid.no_colours();
        let out_no_colours = output.grid.no_colours();
        if in_no_colours == 0 {
            cats.insert(BlankIn);
        }
        if out_no_colours == 0 {
            cats.insert(BlankOut);
        }
        if in_no_colours == 1 {
            cats.insert(SingleColourIn);
        }
        if out_no_colours == 1 {
            cats.insert(SingleColourOut);
        }
        if input.grid.colour == output.grid.colour && input.grid.colour != Mixed {
            cats.insert(SameColour);
        }
/*
//eprintln!("{}", output.shapes.len());
//output.shapes.show();
        if output.shapes.len() > 0 && output.shapes.len() % 5 == 0 {
            let h = output.shapes.colour_cnt();
eprintln!("{h:?} {}", h.iter().filter(|(_,&v)| v == 1 || v == 4).count());
            if h.len() == 2 && h.iter().filter(|(_,&v)| v == 1 || v == 4).count() == 2 {
eprintln!("here");
                cats.insert(SurroundOut);
            }; 
        }
*/
        if input.shapes.len() == 1 {
            cats.insert(SingleShapeIn);
        } else if input.coloured_shapes.len() == 1 {
            cats.insert(SingleColouredShapeIn);
        }
        if output.shapes.len() == 1 {
            cats.insert(SingleShapeOut);
        } else if output.coloured_shapes.len() == 1 {
            cats.insert(SingleColouredShapeOut);
        }
        if input.shapes.len() > 1 && input.shapes.len() == output.shapes.len() {
            cats.insert(InSameCountOut);
        } else if input.shapes.len() > 1 && input.coloured_shapes.len() == output.coloured_shapes.len() {
            cats.insert(InSameCountOutColoured);
        } else if input.shapes.len() < output.shapes.len() {
            cats.insert(InLessCountOut);
        } else if input.coloured_shapes.len() < output.coloured_shapes.len() {
            cats.insert(InLessCountOutColoured);
        } else if input.shapes.len() < output.shapes.len() {
            cats.insert(OutLessCountIn);
        } else if input.coloured_shapes.len() > output.coloured_shapes.len() {
            cats.insert(OutLessCountInColoured);
        }
        let in_border_top = input.grid.border_top();
        let in_border_bottom = input.grid.border_bottom();
        let in_border_left = input.grid.border_left();
        let in_border_right = input.grid.border_right();
        if in_border_top {
            cats.insert(BorderTopIn);
        }
        if in_border_bottom {
            cats.insert(BorderBottomIn);
        }
        if in_border_left {
            cats.insert(BorderLeftIn);
        }
        if in_border_right {
            cats.insert(BorderRightIn);
        }
        if output.grid.size() > 0 {
            let out_border_top = output.grid.border_top();
            let out_border_bottom = output.grid.border_bottom();
            let out_border_left = output.grid.border_left();
            let out_border_right = output.grid.border_right();
            if out_border_top {
                cats.insert(BorderTopOut);
            }
            if out_border_bottom {
                cats.insert(BorderBottomOut);
            }
            if out_border_left {
                cats.insert(BorderLeftOut);
            }
            if out_border_right {
                cats.insert(BorderRightOut);
            }
        }
        if input.grid.even_rows() {
            cats.insert(EvenRowsIn);
        }
        if output.grid.even_rows() {
            cats.insert(EvenRowsOut);
        }
        if input.grid.is_full() {
            cats.insert(FullyPopulatedIn);
        }
        if output.grid.is_full() {
            cats.insert(FullyPopulatedOut);
        }
        if !input.grid.has_gravity_down() && output.grid.has_gravity_down() {
            cats.insert(GravityDown);
        } else if !input.grid.has_gravity_up() && output.grid.has_gravity_up() {
            cats.insert(GravityUp);
        } else if !input.grid.has_gravity_left() && output.grid.has_gravity_left() {
            cats.insert(GravityLeft);
        } else if !input.grid.has_gravity_right() && output.grid.has_gravity_right() {
            cats.insert(GravityRight);
        }
        if input.grid.is_3x3() {
            cats.insert(Is3x3In);
        }
        if output.grid.is_3x3() {
            cats.insert(Is3x3Out);
        }
        if input.grid.div9() {
            cats.insert(Div9In);
        }
        if output.grid.div9() {
            cats.insert(Div9Out);
        }
        if in_dim.0 * 2 == out_dim.0 && in_dim.1 * 2 == out_dim.1 {
            cats.insert(Double);
        }
        if input.shapes.shapes.len() == output.shapes.shapes.len() {
            cats.insert(InOutShapeCount);
        }
        if input.shapes.coloured_shapes.len() == output.coloured_shapes.shapes.len() {
            cats.insert(InOutShapeCountColoured);
        }
        if !input.black.shapes.is_empty() {
            cats.insert(BlackPatches);
        }
        if input.has_bg_shape() && output.has_bg_shape() {
            cats.insert(HasBGShape);
        }
        if input.has_bg_coloured_shape() && output.has_bg_coloured_shape() {
            cats.insert(HasBGShapeColoured);
        }
        let hin = input.grid.cell_colour_cnt_map();
        let hout = output.grid.cell_colour_cnt_map();
        if hin == hout {
            cats.insert(IdenticalColours);
        } else if hin.len() == hout.len() {
            cats.insert(IdenticalNoColours);
        } else {
            let inp: usize = hin.values().sum();
            let outp: usize = hout.values().sum();

            if inp == outp {
                cats.insert(IdenticalNoPixels);
            }
        }
        let hin_colours: usize = hin.values().sum();
        let hout_colours: usize = hout.values().sum();
        if hin.len() == 1 {
            cats.insert(SingleColourCountIn(hin_colours));
        }
        if hout.len() == 1 {
            cats.insert(SingleColourCountOut(hout_colours));
        }
        if hin_colours == hout_colours * 2 {
            cats.insert(SingleColourIn2xOut);
        }
        if hin_colours == hout_colours * 4 {
            cats.insert(SingleColourIn2xOut);
        }
        if hin_colours * 2 == hout_colours {
            cats.insert(SingleColourOut2xIn);
        }
        if hin_colours * 4 == hout_colours {
            cats.insert(SingleColourOut2xIn);
        }
        if output.grid.is_diag_origin() {
            cats.insert(DiagonalOutOrigin);
        } else if output.grid.is_diag_not_origin() {
            cats.insert(DiagonalOutNotOrigin);
        }
        if input.grid.colour == Mixed {
            cats.insert(NoColouredShapesIn(input.coloured_shapes.len()));
        }
        if output.grid.colour == Mixed {
            cats.insert(NoColouredShapesOut(output.coloured_shapes.len()));
        }
        if input.shapes.overlay_shapes_same_colour() {
            cats.insert(OverlayInSame);
        }
        if output.shapes.overlay_shapes_same_colour() {
            cats.insert(OverlayOutSame);
        }
        if input.shapes.overlay_shapes_diff_colour() {
            cats.insert(OverlayInDiff);
        }
        if output.shapes.overlay_shapes_diff_colour() {
            cats.insert(OverlayOutDiff);
        }
        if input.shapes.len() == 1 && input.shapes.shapes[0].is_line() {
            cats.insert(InLine);
        }
        if output.shapes.len() == 1 && output.shapes.shapes[0].is_line() {
            cats.insert(OutLine);
        }
        cats.insert(NoShapesIn(input.shapes.len()));
        cats.insert(NoShapesOut(output.shapes.len()));
        if input.shapes.is_square_same() {
            cats.insert(SquareShapeSide(input.shapes.shapes[0].cells.rows));
            cats.insert(SquareShapeSize(input.shapes.shapes[0].size()));
        }
        let mut cc = input.shapes.colour_cnt();
        if cc.len() == 2 {
            let first = if let Some(first) = cc.pop_first() {
                first.1
            } else {
                0
            };
            let second = if let Some(second) = cc.pop_first() {
                second.1
            } else {
                0
            };

            cats.insert(ShapeMinCntIn(first.min(second)));
            cats.insert(ShapeMaxCntIn(first.max(second)));
        }
        let mut cc = output.shapes.colour_cnt();
        if cc.len() == 2 {
            let first = if let Some(first) = cc.pop_first() {
                first.1
            } else {
                0
            };
            let second = if let Some(second) = cc.pop_first() {
                second.1
            } else {
                0
            };

            cats.insert(ShapeMinCntOut(first.min(second)));
            cats.insert(ShapeMaxCntOut(first.max(second)));
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

    pub fn new_cons(data: &Data) -> Self {
        let mut examples: Vec<Example> = data.train.iter()
            .map(Example::new_cons)
            .collect();

        let tests: Vec<Example> = data.test.iter()
            .map(Example::new_cons)
            .collect();

        let cat = Self::categorise_grids(&mut examples);

        Examples { examples, tests, cat }
    }

    pub fn transformation(&self, trans: Transformation) -> Self {
        let mut examples = self.clone();

        examples.transformation_mut(trans);

        examples
    }

    pub fn transformation_mut(&mut self, trans: Transformation) {
        self.examples.iter_mut().for_each(|ex| ex.transform_mut(trans, true));
        //self.tests.iter_mut().for_each(|ex| ex.transform_mut(trans, false));
        //self.tests.iter_mut().for_each(|ex| Example::categorise_grid(&ex.input, &ex.output));
    }

    pub fn inverse_transformation(&self, trans: Transformation) -> Self {
        let mut examples = self.clone();

        examples.inverse_transformation_mut(trans);

        examples
    }

    pub fn inverse_transformation_mut(&mut self, trans: Transformation) {
        let trans = Transformation::inverse(&trans);

        self.transformation(trans);
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
        let mut extra: BTreeSet<GridCategory> = BTreeSet::new();

        for ex in examples.iter_mut() {
            let cat = Example::categorise_grid(&ex.input, &ex.output);

            if cat.contains(&InOutSquareSameSize) {
                extra.insert(InOutSameSize);
            }
            if cat.contains(&OverlayInSame) {
                extra.insert(OverlayInSame);
            }
            if cat.contains(&OverlayOutSame) {
                extra.insert(OverlayOutSame);
            }
            if cat.contains(&OverlayInDiff) {
                extra.insert(OverlayInDiff);
            }
            if cat.contains(&OverlayOutDiff) {
                extra.insert(OverlayOutDiff);
            }
            ex.pairs = ex.input.shapes.pair_shapes(&ex.output.shapes, true);
            if !ex.pairs.is_empty() {
                extra.insert(InOutSameShapes);
            }
            ex.coloured_pairs = ex.input.coloured_shapes.pair_shapes(&ex.output.coloured_shapes, true);
            if !ex.coloured_pairs.is_empty() {
                extra.insert(InOutSameShapesColoured);
            }

            if cats.is_empty() {
                cats = cat;
            } else {
                cats = cats.intersection(&cat).cloned().collect();
            }
        }

        if cats.contains(&InOutSquareSameSize) && cats.contains(&InOutSameSize) {
            cats.remove(&InOutSameSize);
        }

        cats = cats.union(&extra).cloned().collect();

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
        let mut colour = NoColour;

        for ex in self.examples.iter() {
            let s = ex.output.shapes.largest();

            if colour == NoColour {
                colour = s.colour;
            } else if colour != s.colour {
                return NoColour;    // No common large colour
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

