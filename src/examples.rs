use std::collections::{BTreeMap, BTreeSet}; // Don't use Hash, need ordering
use array_tool::vec::{Uniq, Union};
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

            let output_grid_json = output.grid.to_json();

            if input.grid.rotated_90(1).to_json() == output_grid_json {
                cats.insert(Rot90);
            }
            if input.grid.rotated_90(2).to_json() == output_grid_json {
                cats.insert(Rot180);
            }
            if input.grid.rotated_270(1).to_json() == output_grid_json {
                cats.insert(Rot270);
            }
            if input.grid.transposed().to_json() == output_grid_json {
                cats.insert(Transpose);
            }
            /* There are none?
            if input.grid.inv_transposed().to_json() == output_grid_json {
                cats.insert(InvTranspose);
            }
            */
            if input.grid.mirrored_rows().to_json() == output_grid_json {
                cats.insert(MirroredR);
            }
            if input.grid.mirrored_cols().to_json() == output_grid_json {
                cats.insert(MirroredC);
            }
        } else {
            if in_dim.0 == out_dim.0 && in_dim.1 == out_dim.1 {
                cats.insert(InOutSameSize);
            }
            if in_dim.0 > 1 && out_dim.0 > 1 && in_dim.0 == in_dim.1 && out_dim.0 == out_dim.1 {
                cats.insert(InOutSquare);
                cats.insert(NxNIn(in_dim.0));
                cats.insert(NxNOut(out_dim.0));
                if in_dim.0 * in_dim.0 == out_dim.0 {
                    cats.insert(InToSquaredOut);
                }
            } else if in_dim.0 > 1 && in_dim.0 == in_dim.1 {
                cats.insert(InSquare);
                cats.insert(NxNIn(in_dim.0));
            } else if out_dim.0 > 1 && out_dim.0 == out_dim.1 {
                cats.insert(OutSquare);
                cats.insert(NxNOut(out_dim.0));
            }
        }
        if out_dim.0 > 0 {
            if in_dim.0 % out_dim.0 == 0 && in_dim.0 != out_dim.0 {
                cats.insert(OutRInWidth(in_dim.0 / out_dim.0));
            } else if in_dim.1 % out_dim.1 == 0 && in_dim.1 != out_dim.1 {
                cats.insert(OutRInHeight(in_dim.1 / out_dim.1));
            }
            if out_dim.0 % in_dim.0 == 0 && out_dim.0 != in_dim.0 {
                cats.insert(InROutWidth(out_dim.0 / in_dim.0));
            } else if out_dim.1 % in_dim.1 == 0 && out_dim.1 != in_dim.1 {
                cats.insert(InROutHeight(out_dim.1 / in_dim.1));
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
            cats.insert(MirrorRIn);
        }
        if in_is_mirror_y {
            cats.insert(MirrorCIn);
        }
        if out_is_mirror_x {
            cats.insert(MirrorROut);
        }
        if out_is_mirror_y {
            cats.insert(MirrorCOut);
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
        if input.grid.has_bg_grid() == Black {
            cats.insert(BGGridInBlack);
        }
        if output.grid.has_bg_grid() == Black {
            cats.insert(BGGridOutBlack);
        }
        if input.grid.has_bg_grid_coloured() != NoColour {
            cats.insert(BGGridInColoured);
        }
        if output.grid.has_bg_grid_coloured() != NoColour {
            cats.insert(BGGridOutColoured);
        }
        if input.grid.is_panelled_rows() {
            cats.insert(IsPanelledRIn);
        }
        if output.grid.is_panelled_rows() {
            cats.insert(IsPanelledROut);
        }
        if input.grid.is_panelled_cols() {
            cats.insert(IsPanelledCIn);
        }
        if output.grid.is_panelled_cols() {
            cats.insert(IsPanelledCOut);
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
        } else if input.shapes.shapes.len() == 1 && input.shapes.shapes[0].bare_corners() {
            cats.insert(BareCornersIn);
        }
        if output.grid.is_full() {
            cats.insert(FullyPopulatedOut);
        } else if output.shapes.shapes.len() == 1 && output.shapes.shapes[0].bare_corners() {
            cats.insert(BareCornersOut);
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
        if input.coloured_shapes.shapes.len() == output.coloured_shapes.shapes.len() {
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
        if input.shapes.len() == input.coloured_shapes.len() {
            cats.insert(NoNetColouredShapesIn);
        }
        if output.shapes.len() == output.coloured_shapes.len() {
            cats.insert(NoNetColouredShapesOut);
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

    pub fn io_colour_diff(&self) -> Colour {
        let in_colours = self.input.grid.cell_colour_cnt_map();
        let out_colours = self.output.grid.cell_colour_cnt_map();

        let remainder: Vec<_> = if out_colours.len() > in_colours.len() {
            out_colours.keys().filter(|k| !in_colours.contains_key(k)).collect()
        } else {
            in_colours.keys().filter(|k| !out_colours.contains_key(k)).collect()
        };

        if remainder.len() == 1 {
            *remainder[0]
        } else {
            NoColour
        }
    }

    pub fn colour_shape_map(&self, out: bool) -> BTreeMap<Colour, Shape> {
        let mut bt: BTreeMap<Colour, Shape> = BTreeMap::new();
        let io = if out {
            &self.output.shapes.shapes
        } else {
            &self.input.shapes.shapes
        };

        for s in io.iter() {
            bt.insert(s.colour, s.clone());
        }

        bt
    }

    pub fn colour_attachment_map(&self, out: bool) -> BTreeMap<Colour, bool> {
        let mut bt: BTreeMap<Colour, bool> = BTreeMap::new();
        let io = if out {
            &self.output.shapes.shapes
        } else {
            &self.input.shapes.shapes
        };

        let mut prev = &Shape::trivial();

        for s in io.iter() {
            if *prev != Shape::trivial() {
                bt.insert(s.colour, s.ocol == prev.ocol || s.cells.columns == 1);
            }
            prev = &s;
        }

        bt
    }

    pub fn shape_pixels_to_colour(&self) -> BTreeMap<usize, Colour> {
        let mut spc: BTreeMap<usize, Colour> = BTreeMap::new();

        for s in self.output.shapes.shapes.iter() {
            spc.insert(s.pixels(), s.colour);
        }

        spc
    }
    
    pub fn shape_adjacency_map(&self) -> BTreeMap<Shape, Colour> {
        let in_shapes = &self.input.shapes;
        let out_shapes = &self.output.shapes;
        let mut sam: BTreeMap<Shape, Colour> = BTreeMap::new();
        let ind_colour = in_shapes.smallest().colour;

        for si in in_shapes.shapes.iter() {
            for so in out_shapes.shapes.iter() {
                if si.colour != ind_colour && si.equal_shape(&so) {
                    sam.insert(so.clone(), NoColour);
                } else if si.colour == ind_colour && si.touching(&so) {
                    sam.insert(si.clone(), so.colour);
                }
            }
        }

        let mut map: BTreeMap<Shape, Colour> = BTreeMap::new();

        for (s1, colour1) in sam.iter() {
            if *colour1 != NoColour {
                for (s2, colour2) in sam.iter() {
                    if *colour2 == NoColour && s1.touching(&s2) {
                        map.insert(s1.to_origin(), s2.colour);
                    }
                }
            }
        }
//map.iter().for_each(|(s1,s2)| {s1.show_summary(); s2.show_summary();});

        map
    }

    pub fn some(&self, isout: bool, f: &dyn Fn(&Shapes) -> Shape) -> Shape {
        let s = if isout {
            &self.output.shapes
        } else {
            &self.input.shapes
        };

        f(&s)
    }

    pub fn all(&self, isout: bool) -> Vec<Shape> {
        let s = if isout {
            &self.output.shapes
        } else {
            &self.input.shapes
        };

        s.shapes.clone()
    }

    pub fn some_coloured(&self, isout: bool, f: &dyn Fn(&Shapes) -> Shape) -> Shape {
        let s = if isout {
            &self.output.coloured_shapes
        } else {
            &self.input.coloured_shapes
        };

        f(&s)
    }

    pub fn all_coloured(&self, isout: bool) -> Vec<Shape> {
        let s = if isout {
            &self.output.coloured_shapes
        } else {
            &self.input.coloured_shapes
        };

        s.shapes.clone()
    }

    pub fn map_coloured_shapes_to_shape(&self, _shapes: Vec<Shape>) -> Vec<Shapes> {
        // TODO 626c0bcc
        Vec::new()
    }

    pub fn colour_cnt_diff(&self, inc: bool) -> Colour {
        let incc = self.input.grid.cell_colour_cnt_map();
        let outcc = self.output.grid.cell_colour_cnt_map();

        if incc.len() != outcc.len() {
            return NoColour;
        }

        for ((icol, icnt), (ocol, ocnt)) in incc.iter().zip(outcc.iter()) {
            if icol != ocol {
                return NoColour;
            }
            if inc && icnt < ocnt || !inc && icnt > ocnt {
                return *icol;
            }
        }

        NoColour
    }

    pub fn colour_cnt_inc(&self) -> Colour {
        self.colour_cnt_diff(true)
    }

    pub fn colour_cnt_dec(&self) -> Colour {
        self.colour_cnt_diff(false)
    }

    pub fn split_n_map_horizontal(&self, n: usize) -> BTreeMap<Grid, Grid> {
        let ins: Vec<Grid> = self.input.grid.split_n_horizontal(n);
        let outs: Vec<Grid> = self.output.grid.split_n_horizontal(n);
        let mut bt: BTreeMap<Grid, Grid> = BTreeMap::new();

        if ins.len() != outs.len() {
            return bt;
        }

        for (is, os) in ins.iter().zip(outs.iter()) {
            bt.insert(is.to_origin(), os.to_origin());
        }

        bt
    }

    pub fn split_n_map_vertical(&self, n: usize) -> BTreeMap<Grid, Grid> {
        let ins: Vec<Grid> = self.input.grid.split_n_vertical(n);
        let outs: Vec<Grid> = self.output.grid.split_n_vertical(n);
        let mut bt: BTreeMap<Grid, Grid> = BTreeMap::new();

        if ins.len() != outs.len() {
            return bt;
        }

        for (is, os) in ins.iter().zip(outs.iter()) {
            bt.insert(is.to_origin(), os.to_origin());
        }

        bt
    }

    pub fn majority_dimensions(&self) -> (usize, usize){
        if self.input.shapes.shapes.is_empty() {
            return (0, 0);
        }

        let mut sc: BTreeMap<(usize, usize), usize> = BTreeMap::new();

        for s in self.input.shapes.shapes.iter() {
            *sc.entry(s.dimensions()).or_insert(0) += 1;
        }

        if let Some((_, dim)) = sc.iter().map(|(k, v)| (v, k)).max() {
            *dim
        } else {
            (0, 0)
        }
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

    pub fn full_shapes(&self, input: bool, sq: bool) -> Shapes {
        let mut shapes: Vec<Shape> = Vec::new();

        for ex in self.examples.iter() {
            let ss = if input {
                &ex.input.shapes
            } else {
                &ex.output.shapes
            };

            let it = if sq {
                ss.full_shapes()
            } else {
                ss.full_shapes_sq()
            };

            for s in it.iter() {
                let s = s.to_origin();

                if !shapes.contains(&s) {
                    shapes.push(s);
                }
            }
        }

        Shapes::new_shapes(&shapes)
    }

    pub fn full_shapes_in(&self) -> Shapes {
        self.full_shapes(true, false)
    }

    pub fn full_shapes_out(&self) -> Shapes {
        self.full_shapes(false, false)
    }

    pub fn full_shapes_in_sq(&self) -> Shapes {
        self.full_shapes(true, true)
    }

    pub fn full_shapes_out_sq(&self) -> Shapes {
        self.full_shapes(false, true)
    }

    pub fn common(&self, input: bool) -> Grid {
        let mut grid = Grid::trivial();

        for ex in self.examples.iter() {
            let g = if input {
                &ex.input.grid
            } else {
                &ex.output.grid
            };

            if grid == Grid::trivial() {
                grid = g.clone();
            } else if grid.dimensions() != g.dimensions() {
                return Grid::trivial();
            } else {
                for (c1, c2) in g.cells.values().zip(grid.cells.values_mut()) {
                    c2.colour = c1.colour.and(&c2.colour);
                }
            }
        }

        grid
    }

    pub fn all_shapes(&self, input: bool, sq: bool) -> Shapes {
        let mut shapes: Vec<Shape> = Vec::new();

        for ex in self.examples.iter() {
            let ss = if input {
                &ex.input.shapes
            } else {
                &ex.output.shapes
            };

            let it = if sq {
                ss.all_shapes()
            } else {
                ss.all_shapes_sq()
            };

            for s in it.iter() {
                let s = s.to_origin();

                shapes.push(s);
            }
        }

        Shapes::new_shapes(&shapes)
    }

    pub fn all_shapes_in(&self) -> Shapes {
        self.all_shapes(true, false)
    }

    pub fn all_shapes_out(&self) -> Shapes {
        self.all_shapes(false, false)
    }

    pub fn all_shapes_in_sq(&self) -> Shapes {
        self.all_shapes(true, true)
    }

    pub fn all_shapes_out_sq(&self) -> Shapes {
        self.all_shapes(false, true)
    }

    pub fn some(&self, isout: bool, f: &dyn Fn(&Shapes) -> Shape) -> Vec<Shape> {
        let mut s: Vec::<Shape> = Vec::new();

        for ex in self.examples.iter() {
            s.push(ex.some(isout, f));
        }

        s
    }

    pub fn all(&self, isout: bool) -> Vec<Shape> {
        let mut s: Vec::<Shape> = Vec::new();

        for ex in self.examples.iter() {
            let vs = ex.all(isout);

            for ex2 in vs.iter() {
                s.push(ex2.clone());
            }
        }

        s
    }

    pub fn some_coloured(&self, isout: bool, f: &dyn Fn(&Shapes) -> Shape) -> Vec<Shape> {
        let mut s: Vec::<Shape> = Vec::new();

        for ex in self.examples.iter() {
            s.push(ex.some_coloured(isout, f));
        }

        s
    }

    pub fn all_coloured(&self, isout: bool) -> Vec<Shape> {
        let mut s: Vec::<Shape> = Vec::new();

        for ex in self.examples.iter() {
            let vs = ex.all_coloured(isout);

            for ex2 in vs.iter() {
                s.push(ex2.clone());
            }
        }

        s
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
            let v: BTreeSet<Colour> = h.keys().copied().collect();

            common = common.intersection(&v).copied().collect();
        }

        Vec::from_iter(common)
    }

    pub fn find_output_colours(&self) -> Vec<Colour> {
        let mut common = Colour::all_colours();

        for ex in self.examples.iter() {
            let h = ex.output.grid.cell_colour_cnt_map();
            let v: BTreeSet<Colour> = h.keys().copied().collect();

            common = common.intersection(&v).copied().collect();
        }

        Vec::from_iter(common)
    }

    pub fn find_all_output_colours(&self) -> Vec<Colour> {
        let mut common = Vec::new();

        for ex in self.examples.iter() {
            let h = ex.output.grid.cell_colour_cnt_map();
            let v: Vec<Colour> = h.keys().copied().collect();

            common = Union::union(&common, v);
        }

        Vec::from_iter(common)
    }

    pub fn find_all_input_colours(&self) -> Vec<Colour> {
        let mut common = Vec::new();

        for ex in self.examples.iter() {
            let h = ex.input.grid.cell_colour_cnt_map();
            let v: Vec<Colour> = h.keys().copied().collect();

            common = Union::union(&common, v);
        }

        Vec::from_iter(common)
    }

    pub fn io_colour_diff(&self) -> Vec<Colour> {
        let in_colours = self.find_all_input_colours();
        let out_colours = self.find_all_output_colours();

        Uniq::uniq(&out_colours, in_colours)
    }

    pub fn io_all_colour_diff(&self) -> Vec<Colour> {
        let in_colours = self.find_all_input_colours();
        let out_colours = self.find_all_output_colours();

        if out_colours.len() > in_colours.len() {
            Uniq::uniq(&out_colours, in_colours)
        } else {
            Uniq::uniq(&in_colours, out_colours)
        }
    }

    pub fn io_colour_common(&self) -> Vec<Colour> {
        let in_colours = self.find_input_colours();
        let out_colours = self.find_output_colours();

        Union::union(&out_colours, in_colours)
    }

    pub fn io_common_row_colour(&self) -> Colour {
        let mut colour = NoColour;

        for ex in self.examples.iter() {
            for (i, o) in ex.input.shapes.shapes.iter().zip(ex.input.shapes.shapes.iter()) {
                if colour == NoColour && i.orow == o.orow && i.colour == o.colour {
                    colour = i.colour;

                    break;
                } else if i.orow == o.orow && i.colour != o.colour {
                    return NoColour;
                }
            }
        }

        colour
    }

    pub fn io_common_col_colour(&self) -> Colour {
        let mut colour = NoColour;

        for ex in self.examples.iter() {
            for (i, o) in ex.input.shapes.shapes.iter().zip(ex.input.shapes.shapes.iter()) {
                if colour == NoColour && i.ocol == o.ocol && i.colour == o.colour {
                    colour = i.colour;

                    break;
                } else if i.ocol == o.ocol && i.colour != o.colour {
                    return NoColour;
                }
            }
        }

        colour
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

    pub fn bleached_io_map(&self) -> BTreeMap<String, Grid> {
        let mut h: BTreeMap<String, Grid> = BTreeMap::new();

        for ex in self.examples.iter() {
            h.insert(ex.input.grid.bleach().to_json(), ex.output.grid.clone());
        }

        h
    }

    pub fn in_max_size(&self) -> (usize, usize) {
        let mut rs = 0;
        let mut cs = 0;

        for ex in self.examples.iter() {
            let (r, c) = ex.input.grid.dimensions();

            (rs, cs) = (rs, cs).max((r, c));
        }

        (rs, cs)
    }

    pub fn derive_missing_rule(&self) -> Grid {
        let mut i_rs = 0;
        let mut i_cs = 0;
        let mut in_grid = Grid::trivial();
        let mut out_grid = Grid::trivial();

        // Find largest with one pixel
        for ex in self.examples.iter() {
            let grid = &ex.input.grid;

            if grid.is_square() && grid.pixels() == 1 && grid.cells[(grid.cells.rows / 2, grid.cells.columns / 2)].colour != Black {
                (i_rs, i_cs) = (i_rs, i_cs).max((grid.cells.rows, grid.cells.columns));

                in_grid = ex.input.grid.clone();
                out_grid = ex.output.grid.clone();
            }
        }

        in_grid.derive_missing_rule(&out_grid)
    }

    pub fn shape_pixels_to_colour(&self) -> BTreeMap<usize, Colour> {
        let mut spc: BTreeMap<usize, Colour> = BTreeMap::new();

        for ex in self.examples.iter() {
            spc.extend(ex.shape_pixels_to_colour());
        }

        spc
    }

    pub fn shape_adjacency_map(&self) -> BTreeMap<Shape, Colour> {
        let mut sam: BTreeMap<Shape, Colour> = BTreeMap::new();

        for ex in self.examples.iter() {
            sam.extend(ex.shape_adjacency_map());
        }

        sam
    }

    pub fn colour_shape_map(&self, out: bool) -> BTreeMap<Colour, Shape> {
        let mut bt: BTreeMap<Colour, Shape> = BTreeMap::new();

        for ex in self.examples.iter() {
            bt.extend(ex.colour_shape_map(out));
        }

        bt
    }

    pub fn colour_attachment_map(&self, out: bool) -> BTreeMap<Colour, bool> {
        let mut bt: BTreeMap<Colour, bool> = BTreeMap::new();

        for ex in self.examples.iter() {
            bt.extend(ex.colour_attachment_map(out));
        }

        bt
    }

    pub fn colour_cnt_diff(&self, inc: bool) -> Colour {
        let mut colour = NoColour;

        for ex in self.examples.iter() {
            let new_col = ex.colour_cnt_diff(inc);

            if colour == NoColour {
                colour = new_col;
            } else if colour != new_col {
                return NoColour;
            }
        }

        colour
    }

    pub fn colour_cnt_inc(&self) -> Colour {
        self.colour_cnt_diff(true)
    }

    pub fn colour_cnt_dec(&self) -> Colour {
        self.colour_cnt_diff(false)
    }

    pub fn colour_diffs(&self, inc: bool) -> Vec<Colour> {
        let mut cc: Vec<Colour> = Vec::new();

        for ex in self.examples.iter() {
            let new_col = ex.colour_cnt_diff(inc);

            if new_col != NoColour {
                cc.push(new_col);
            }
        }

        cc
    }

    pub fn colour_incs(&self) -> Vec<Colour> {
        self.colour_diffs(true)
    }

    pub fn colour_decs(&self) -> Vec<Colour> {
        self.colour_diffs(false)
    }

    pub fn split_n_map_horizontal(&self, n: usize) -> BTreeMap<Grid, Grid> {
        let mut bt: BTreeMap<Grid, Grid> = BTreeMap::new();

        for ex in self.examples.iter() {
            bt.extend(ex.split_n_map_horizontal(n));
        }

        bt
    }

    pub fn split_n_map_vertical(&self, n: usize) -> BTreeMap<Grid, Grid> {
        let mut bt: BTreeMap<Grid, Grid> = BTreeMap::new();

        for ex in self.examples.iter() {
            bt.extend(ex.split_n_map_vertical(n));
        }

        bt
    }
}
