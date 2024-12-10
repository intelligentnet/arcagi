use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use pathfinding::prelude::Matrix;
use crate::cats::Colour::*;
use crate::cats::Direction::*;
use crate::cats::*;
use crate::cell::*;
use crate::grid::*;
use crc::Crc;

#[derive(Debug, Clone, Eq)]
pub struct Shape {
    pub orow: usize,
    pub ocol: usize,
    pub colour: Colour,
//    pub sid: u32,
    pub cells: Matrix<Cell>,
    pub cats: BTreeSet<ShapeCategory>,
    pub io_edges: BTreeSet<ShapeEdgeCategory>,
}

/*
impl Hash for Shape {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        fn calculate_hash(t: &str) -> u64 {
            let mut s = DefaultHasher::new();

            s.write(t.as_bytes());
            s.finish()
        }

        state.write_usize(self.ox);
        state.write_usize(self.oy);
        state.write_u64(calculate_hash(&self.to_json()));
        state.finish();
    }
}
*/

impl PartialEq for Shape {
    fn eq(&self, other: &Shape) -> bool {
        self.orow == other.orow && self.ocol == other.ocol && self.cells.rows == other.cells.rows && self.cells.columns == other.cells.columns
    }
}
 
        //let sdist = ((self.ox * self.ox + self.oy * self.oy) as f64).sqrt();
 
impl Ord for Shape {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.orow, self.ocol, &self.to_json(), &self.colour).cmp(&(other.orow, other.ocol, &other.to_json(), &other.colour))
        //(&self.to_json(), self.ox, &self.oy).cmp(&(&other.to_json(), other.ox, &other.oy))
    }
}

impl PartialOrd for Shape {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Default for Shapes {
    fn default() -> Self {
        Self::new()
    }
}


impl Shape {
    pub fn new(ox: usize, oy: usize, cells: &Matrix<Cell>) -> Self {
        let (colour, _) = Self::cell_colour_cnt(cells, true);
//        let sid = Self::sid(cells, false);
//println!("=== {:?} {} {}", col, cnt, cells.len());
        //let colour = if cnt == cells.len() { col } else { Mixed };
        let mut new_cells = cells.clone();

        for (x, y) in cells.keys() {
            new_cells[(x, y)].row = x + ox;
            new_cells[(x, y)].col = y + oy;
        }

        let cats: BTreeSet<ShapeCategory> = BTreeSet::new();
        let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        let new_cells = Self::cell_category(&new_cells);

        let mut res = Self { orow: ox, ocol: oy, colour, cells: new_cells, cats, io_edges };

        res.categorise_shape();

        res
    }

    pub fn new_cells(cells: &Matrix<Cell>) -> Self {
        let (colour, _) = Self::cell_colour_cnt(cells, true);
//        let sid = Self::sid(cells, false);
//println!("=== {:?} {} {}", col, cnt, cells.len());
        //let colour = if cnt == cells.len() { col } else { Mixed };
        let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        let cats: BTreeSet<ShapeCategory> = BTreeSet::new();
        let cells = Self::cell_category(cells);

        let orow = cells[(0,0)].row;
        let ocol = cells[(0,0)].col;
        let mut res = Self { orow, ocol, colour, cells, cats, io_edges };

        res.categorise_shape();

        res
    }

    pub fn new_sized_coloured(rows: usize, cols: usize, colour: Colour) -> Self {
        let cells: Matrix<Cell> = Matrix::from_fn(rows, cols, |(rows, cols)| Cell::new_colour(rows, cols, colour));
        let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        let cats: BTreeSet<ShapeCategory> = BTreeSet::new();

        Self { orow: 0, ocol: 0, colour, cells, cats, io_edges }
    }

    pub fn new_sized_coloured_position(orow: usize, ocol: usize, row: usize, col: usize, colour: Colour) -> Self {
        let cells: Matrix<Cell> = Matrix::from_fn(row, col, |(r, c)| Cell::new_colour(r + orow, c + ocol, colour));

        let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        let cats: BTreeSet<ShapeCategory> = BTreeSet::new();

        Self { orow, ocol, colour, cells, cats, io_edges }
    }

    pub fn new_empty() -> Self {
        let cells = Matrix::new(0, 0, Cell::new(0, 0, 0));
        let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        let cats: BTreeSet<ShapeCategory> = BTreeSet::new();

        Self { orow: 0, ocol: 0, colour: NoColour, cells, cats, io_edges }
    }

/*
    pub fn new9(&self, corners: bool, colour: Colour) -> Self {
        let mut s = if self.orow == 0 && self.ocol == 0 {
            let mut s = Self::new_sized_coloured(2, 2, Transparent);
            s.cells[(0,0)].colour = self.colour;

            if corners {
                s.cells[(1,1)].colour = colour;
            } else {
                s.cells[(0,1)].colour = colour;
                s.cells[(1,0)].colour = colour;
            }

            s
        } else if self.orow > 0 && self.ocol == 0 {
            let mut s = Self::new_sized_coloured(3, 2, Transparent);

            s.cells[(1,0)].colour = self.colour;

            if corners {
                s.cells[(0,1)].colour = colour;
                s.cells[(2,1)].colour = colour;
            } else {
                s.cells[(2,0)].colour = colour;
                s.cells[(1,1)].colour = colour;
            }

            s
        } else if self.orow == 0 && self.ocol > 0 {
            let mut s = Self::new_sized_coloured(2, 3, Transparent);

            s.cells[(0,1)].colour = self.colour;

            if corners {
                s.cells[(1,0)].colour = colour;
                s.cells[(1,2)].colour = colour;
            } else {
                s.cells[(0,2)].colour = colour;
                s.cells[(1,1)].colour = colour;
            }

            s
        } else {
            let mut s = Self::new_sized_coloured(3, 3, Transparent);

            s.cells[(1,1)].colour = self.colour;

            if corners {
                s.cells[(0,0)].colour = colour;
                s.cells[(0,2)].colour = colour;
                s.cells[(2,0)].colour = colour;
                s.cells[(2,2)].colour = colour;
            } else {
                s.cells[(0,1)].colour = colour;
                s.cells[(1,0)].colour = colour;
                s.cells[(2,1)].colour = colour;
                s.cells[(1,2)].colour = colour;
            }

            s
        };

        s.orow = if self.orow > 0 { self.orow - 1 } else { self.orow };
        s.ocol = if self.ocol > 0 { self.ocol - 1 } else { self.ocol };

        s.colour = Mixed;

        for x in 0 .. s.cells.rows {
            for y in 0 .. s.cells.columns {
                s.cells[(x,y)].row = s.orow + x;
                s.cells[(x,y)].col = s.ocol + y;
            }
        }

        s
    }
*/

    pub fn trivial() -> Self {
        Self::new_empty()
    }

    /*
    pub fn shape_from_reachable(&self, reachable: BTreeSet<(usize, usize)>) -> Self {
        reachable.iter().for_each(|&i| self.cells[i].colour = new_colour)
    }
    */

    pub fn is_full(&self) -> bool {
        for c in self.cells.values() {
            if c.colour == Black {
                return false;
            }
        }

        true
    }

    pub fn corners(&self) -> (usize, usize, usize, usize) {
        (self.orow, self.ocol, self.orow + self.cells.rows - 1, self.ocol + self.cells.columns - 1)
    }

    pub fn same_patch(&self, other: &Self) -> bool {
        if self.size() != other.size() || self.cells.rows != other.cells.rows {
            return false;
        }

        for (xy, c) in self.cells.items() {
            if c.colour != Black && c.colour != other.cells[xy].colour {
                return false;
            }
        }

        true
    }

    pub fn fill(&self, other: &Self) -> Self {
        if self.size() != other.size() || !self.is_square() {
            return self.clone();
        }

        let mut shape = self.clone();

        for xy in self.cells.keys() {
            if self.cells[xy].colour == Black {
                shape.cells[xy].colour = other.cells[xy].colour;
            }
        }
        
        shape
    }

    pub fn make_symmetric(&self) -> Self {
        let shape = self.clone();
        let shape = shape.fill(&shape.rotated_90());
        let shape = shape.fill(&shape.rotated_180());
        let shape = shape.fill(&shape.rotated_270());

        shape
    }

    // must be odd size
    pub fn new_square(x: usize, y: usize, size: usize, colour: Colour) -> Self {
        let mut square = Self::new_sized_coloured(size, size, Black);

        square.colour = colour;

        for (x, y) in square.cells.keys() {
            if x == 0 || y == 0 || x == square.cells.rows - 1 || y == square.cells.columns - 1 {
                square.cells[(x, y)].colour = colour;
            }
        }

        square.translate_absolute(x, y)
    }

    pub fn is_trivial(&self) -> bool {
        self.cells.rows == 0 && self.cells.columns == 0 && self.colour == Black
    }

    pub fn remove(&mut self, c: &Cell) {
        let mut c = c.clone();

        c.colour = Black;

        self.cells[(c.row, c.col)] = c.clone();
    }

    pub fn to_shapes(&self) -> Shapes {
        let mut inner_shapes = self.to_grid().to_shapes();

        for s in inner_shapes.shapes.iter_mut() {
            s.orow += self.orow;
            s.ocol += self.ocol;

            for x in 0 .. s.cells.rows {
                for y in 0 .. s.cells.columns {
                    s.cells[(x,y)].row += self.orow;
                    s.cells[(x,y)].col += self.ocol;
                }
            }
        }

        inner_shapes
    }

    pub fn shrink_any(&self, coloured: bool) -> Self {
        let mut s = if coloured {
            self.to_grid().to_shapes_coloured().shapes[0].clone()
        } else {
            self.to_grid().to_shapes().shapes[0].clone()
        };

        s.orow = self.orow;
        s.ocol = self.ocol;

        for x in 0 .. s.cells.rows {
            for y in 0 .. s.cells.columns {
                s.cells[(x,y)].row = s.orow + x;
                s.cells[(x,y)].col = s.ocol + y;
            }
        }

        s
    }

    pub fn shrink(&self) -> Self {
        self.shrink_any(false)
    }

    pub fn shrink_coloured(&self) -> Self {
        self.shrink_any(true)
    }

    pub fn euclidian(&self) -> f64 {
        ((self.orow * self.orow + self.ocol * self.ocol) as f64).sqrt()
    }

    // Must have at least one other!
    pub fn nearest(&self, other: &Shapes) -> Self {
        let dself = self.euclidian();
        let mut min: f64 = f64::MAX;
        let mut pos: &Self = &other.shapes[0];

        for s in &other.shapes {
            if self.orow > s.orow + 1 && self.ocol > s.ocol + 1 {
                let dist = dself - s.euclidian();

                if dist < min {
                    min = dist;
                    pos = s;
                }
            }
        }

        pos.clone()
    }

    /*
    pub fn touching(&self, other: &Shape) -> bool {
        let l_tlx = self.ox;
        let l_lty = self.oy;
        let l_brx = self.ox + self.cells.rows;
        let l_bry = self.oy + self.cells.columns;
        let r_tlx = other.ox;
        let r_lty = other.oy;
        let r_brx = other.ox + other.cells.rows;
        let r_bry = other.oy + other.cells.columns;
        // TODO complete

        false
    }
    */

    /* TODO
    pub fn add_outer_layer(&self, colour: Colour, thickness: usize) -> Self {
        //let mut m: Matrix<Cell> = self.cells.clone();
        //Self::new(self.ox, self.oy, &m)

        self.clone()
    }

    pub fn add_inner_layer(&self, colour: Colour, thickness: usize) -> Self {
        self.clone()
    }
    */

    pub fn is_pixel(&self) -> bool {
        self.cells.rows == 1 && self.cells.columns == 1
    }

    pub fn fill_boundary_colour(&self) -> Self {
        if self.colour == Mixed {
            return self.clone();
        }

        let mut shape = self.clone();

        for ((x, y), c) in self.cells.items() {
            if c.colour == Black && (x == 0 || y == 0 || x == self.cells.rows - 1 || y == self.cells.columns - 1) {
                shape.flood_fill_mut(x, y, NoColour, self.colour);
            }
        }

        shape
    }

    pub fn flood_fill(&self, x: usize, y: usize, ignore_colour: Colour, new_colour: Colour) -> Self {
        let mut shape = self.clone();

        shape.flood_fill_mut(x, y, ignore_colour, new_colour);

        shape
    }

    pub fn flood_fill_mut(&mut self, x: usize, y: usize, ignore_colour: Colour, new_colour: Colour) {
        let reachable = self.cells.bfs_reachable((x, y), false, |i| self.cells[i].colour == Black || self.cells[i].colour == ignore_colour);

        reachable.iter().for_each(|&i| self.cells[i].colour = new_colour);
    }

    pub fn sid(m: &Matrix<Cell>, coloured: bool) -> u32 {
        let crc = Crc::<u32>::new(&crc::CRC_32_ISCSI);
        let mut digest = crc.digest();

        for ((x, y), c) in m.items() {
            let colour = if c.colour == Black {
                Black
            } else if coloured {
                c.colour
            } else {
                Mixed 
            };

            digest.update(&x.to_ne_bytes());
            digest.update(&y.to_ne_bytes());
            digest.update(&Colour::to_usize(colour).to_ne_bytes());
        }

        digest.finalize()
    }

    pub fn has_bg_grid(&self) -> Colour {
        self.to_grid().has_bg_grid()
    }

    pub fn has_bg_grid_coloured(&self) -> Colour {
        self.to_grid().has_bg_grid()
    }

    pub fn has_bg_grid_not_sq(&self) -> Colour {
        self.to_grid().has_bg_grid_not_sq()
    }

    pub fn has_bg_grid_coloured_not_sq(&self) -> Colour {
        self.to_grid().has_bg_grid_not_sq()
    }

    // Same footprint + colours
    pub fn equals(&self, other: &Self) -> Colour {
        if !self.equal_footprint(other) {
            return DiffShape;
        }

        for (c1, c2) in self.cells.values().zip(other.cells.values()) {
            if c1.colour != c2.colour {
                return DiffBlack + c2.colour;
            }
        }

        Same
    }

    // Same position
    pub fn equal_position(&self, other: &Self) -> bool {
        self.orow == other.orow && self.ocol == other.ocol
    }

    // Same footprint
    pub fn equal_footprint(&self, other: &Self) -> bool {
        self.cells.columns == other.cells.columns && self.cells.rows == other.cells.rows
    }

    // Same shape
    pub fn equal_shape(&self, other: &Self) -> bool {
        if !self.equal_footprint(other) {
            return false;
        }

        for ((sx, sy), (ox, oy)) in self.cells.keys().zip(other.cells.keys()) {
            if sx != ox || sy != oy {
                return false;
            }
        }

        true
    }

    pub fn show_summary(&self) {
        println!("{}/{}: {}/{} {:?} {:?}", self.orow, self.ocol, self.cells.rows, self.cells.columns, self.colour, self.cats);
    }

    fn show_any(&self, diff: bool, io: bool) {
        println!("--------Shape-------");
        Grid::show_matrix(&self.cells, diff, io);
        println!();
    }

    pub fn show(&self) {
        self.show_any(true, false);
    }

    pub fn show_full(&self) {
        self.show_any(false, true);
    }

    pub fn show_out(&self) {
        self.show_any(false, false);
    }

    pub fn show_io(&self) {
        self.show_any(true, true);
    }

    pub fn is_empty(&self) -> bool {
        for c in self.cells.values() {
            if c.colour != Black {
                return false
            }
        }

        true
    }

    pub fn subshape_remain(&self, tlx: usize, sx: usize, tly: usize, sy: usize) -> Self {
        let mut s = self.subshape(tlx, sx, tly, sy);

        s.orow = self.orow + tlx;
        s.ocol = self.ocol + tly;

        for x in 0 .. s.cells.rows {
            for y in 0 .. s.cells.columns {
                s.cells[(x,y)].row = s.orow + x;
                s.cells[(x,y)].col = s.ocol + y;
            }
        }

        s
    }

    pub fn subshape(&self, tlx: usize, sx: usize, tly: usize, sy: usize) -> Self {
        self.to_grid().subgrid(tlx, sx, tly, sy).as_shape()
    }

    pub fn subshape2(&self, tlx: usize, sx: usize, tly: usize, sy: usize) -> Self {
        self.to_grid().subgrid2(tlx, sx, tly, sy).as_shape()
    }

    pub fn id(&self) -> String {
        format!("{}/{}", self.orow, self.ocol)
    }

    pub fn above(&self, other: &Self) -> bool {
        let (sx, sy) = self.centre_of_exact();
        let (ox, oy) = other.centre_of_exact();
//println!("{:?} {:?}", self.centre_of(), other.centre_of());
        
        sx < ox && (sx - ox).abs() > (sy - oy).abs()
    }

    pub fn below(&self, other: &Self) -> bool {
        let (sx, sy) = self.centre_of_exact();
        let (ox, oy) = other.centre_of_exact();
        
        sx > ox && (sx - ox).abs() > (sy - oy).abs()
    }

    pub fn right(&self, other: &Self) -> bool {
        let (sx, sy) = self.centre_of_exact();
        let (ox, oy) = other.centre_of_exact();
        
        sy < oy && (sx - ox).abs() < (sy - oy).abs()
    }

    pub fn left(&self, other: &Self) -> bool {
        let (sx, sy) = self.centre_of_exact();
        let (ox, oy) = other.centre_of_exact();
        
        sy > oy && (sx - ox).abs() < (sy - oy).abs()
    }

    pub fn diag(&self, other: &Self) -> bool {
        let (sx, sy) = self.centre_of();
        let (ox, oy) = other.centre_of();
        
        let dx = if sx > ox { sx - ox } else { ox - sx };
        let dy = if sy > oy { sy - oy } else { oy - sy };

        dx == dy
    }

    pub fn orientation_horizontal(&self) -> Option<bool> {
        let width = self.width();
        let height = self.height();

        if width == 1 && height > 1 {
            Some(true)
        } else if width > 1 && height == 1 {
            Some(false)
        } else {
            None
        }
    }

    pub fn size(&self) -> usize {
        self.cells.columns * self.cells.rows
    }

    pub fn width(&self) -> usize {
        self.cells.columns
    }

    pub fn height(&self) -> usize {
        self.cells.rows
    }

    pub fn origin(&self) -> (usize, usize) {
        (self.orow, self.ocol)
    }

    pub fn pixels(&self) -> usize {
        self.cells.values()
            .filter(|c| c.colour != Black).
            count()
    }

    pub fn same_size(&self, other: &Self) -> bool {
        self.size() == other.size()
    }

    pub fn same_shape(&self, other: &Self) -> bool {
        self.cells.columns == other.cells.columns && self.cells.rows == other.cells.rows
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.cells.columns, self.cells.rows)
    }

    pub fn density(&self) -> f32 {
        self.size() as f32 / self.cells.len() as f32
    }

    pub fn distinct_colour_cnt(&self) -> usize {
        let mut s: BTreeSet<Colour> = BTreeSet::new();

        for c in self.cells.values() {
            if c.colour != Black {
                s.insert(c.colour);
            }
        }

        s.len()
    }

    pub fn cell_colour_cnt_map(&self) -> BTreeMap<Colour, usize>  {
        let mut h: BTreeMap<Colour, usize> = BTreeMap::new();

        for c in self.cells.values() {
            if c.colour != Black {
                *h.entry(c.colour).or_insert(0) += 1;
            }
        }

        h
    }

    pub fn cell_colour_cnt(cells: &Matrix<Cell>, max: bool) -> (Colour, usize) {
        let mut h: HashMap<usize, usize> = HashMap::new();

        for c in cells.values() {
            if c.colour != Black {
                *h.entry(Colour::to_usize(c.colour)).or_insert(0) += 1;
            }
        }
        let mm = if max {
            h.iter().max_by(|col, c| col.1.cmp(c.1))
        } else {
            h.iter().min_by(|col, c| col.1.cmp(c.1))
        };
        let pair: Option<(Colour, usize)> = mm
            .map(|(col, cnt)| (Colour::from_usize(*col), *cnt));

        match pair {
            None => (NoColour, 0),
            Some((colour, cnt)) => (colour, cnt)
        }
    }

    pub fn colour_cnt(&self, max: bool) -> (Colour, usize) {
        Self::cell_colour_cnt(&self.cells, max)
    }

    pub fn majority_colour(&self) -> Colour {
        let cc = self.colour_cnt(true);

        cc.0
    }

    pub fn minority_colour(&self) -> Colour {
        let cc = self.colour_cnt(false);

        cc.0
    }
    
    pub fn colour_position(&self, colour: Colour) -> Vec<(usize, usize)> {
        let mut ans: Vec<(usize, usize)> = Vec::new();

        for ((x, y), c) in self.cells.items() {
            if c.colour == colour {
                ans.push((x, y));
            }
        }

        ans
    }

    /*
    pub fn make_square(&self) -> Self {
        // Figure must also be a square
        let brr_tlr = brr - tlr;
        let brc_tlc = brc - tlc;
        let mut tb = true;  // top/bottom equal extent?
        if brr_tlr > brc_tlc {
            let tl = tlr.min(tlc);

            if tl == tlr {
                tlc = tl;
            } else {
                tlr = tl;
            }
        } else if brr_tlr < brc_tlc {
            let br = brr.max(tlc);

            if br == brr {
                brc = br;
            } else {
                brr = br;
            }
        }
    }
    */

    pub fn origin_centre(&self) -> (usize, usize, usize, usize) { 
        let (tlr, tlc, _, _) = self.corners();
        let side = self.cells.rows.max(self.cells.columns);

        (tlr.min(tlc), tlr.min(tlc), side, if side % 2 == 1 { 1 } else { 4 })
    }

    pub fn larger(&self, other: &Self) -> Self {
        if self.size() >= other.size() {
            self.clone()
        } else {
            other.clone()
        }
    }

    pub fn smaller(&self, other: &Self) -> Self {
        if self.size() <= other.size() {
            self.clone()
        } else {
            other.clone()
        }
    }

    pub fn to_square(&self) -> Self {
        let (or, oc, side, _) = self.origin_centre();
        let shape = Shape::new_sized_coloured_position(or, oc, side, side, Black);

        Shapes::new_shapes(&[shape, self.clone()]).to_shape()
    }

    pub fn to_square_grid(&self) -> Grid {
        let (or, oc, side, _) = self.origin_centre();
        let shape = Shape::new_sized_coloured_position(or, oc, side, side, Black);

        Shapes::new_shapes(&[shape, self.clone()]).to_grid()
    }

    /* not necessary
    // must be square and assme 3 out of four shapes + centre already
    pub fn symmetric_slice(&self) -> (Self, Self) {
        let mut shape = self.clone();
        let mut centre = self.clone();
        let mut arm = self.clone();

        let (mut tlr, mut tlc, mut brr, mut brc) = shape.corners();
shape.show();

        // Rotate to minimise origin
        for i in 1 .. 4 {
            let rs = shape.rot_90(i);
rs.show();
            let (a, b, c, d) = rs.corners();

            if a < tlr || b < tlc || c < brr || d < brc {
                tlr = a;
                tlc = b;
                brr = c;
                brc = d;

                shape = rs;
            }
        }
println!("-- {tlr}, {tlc}, {brr}, {brc}");

        let (mut tlr, mut tlc, mut brr, mut brc) = shape.corners();

println!(">> {tlr}, {tlc}, {brr}, {brc}");
        // Figure must also be a square
        let brr_tlr = brr - tlr;
        let brc_tlc = brc - tlc;
        let mut tb = true;  // top/bottom equal extent?
        if brr_tlr > brc_tlc {
            let tl = tlr.min(tlc);

            if tl == tlr {
                tlc = tl;
            } else {
                tlr = tl;
            }
        } else if brr_tlr < brc_tlc {
            let br = brr.max(tlc);

            if br == brr {
                brc = br;
            } else {
                brr = br;
            }
        }

        // if square side is odd then centre is pixel else size 4
        let side = brr_tlr.max(brc_tlc) + 1;

        if side % 2 == 1 {
            let r = side / 2;
            let c = side / 2;

            centre = self.subshape(r - tlr, 1, c - tlc, 1);
        } else {
            let r = side / 2 - 1;
            let c = side / 2 - 1;

            centre = self.subshape(r - tlr, 2, c - tlc, 2);
        }
println!("<< {tlr}, {tlc}, {brr}, {brc} : {side}");

        (centre, arm)
    }
    */

    pub fn distance_x(&self, other: &Self) -> f32 {
        let tl_dist = self.orow.max(other.orow) - other.orow.min(self.orow);
        let br_dist = self.cells.columns.max(other.cells.columns) - other.cells.columns.min(self.cells.columns);

        ((tl_dist * tl_dist + br_dist * br_dist) as f32).sqrt() / 2.0
    }

    pub fn distance_y(&self, other: &Self) -> f32 {
        let tl_dist = self.ocol.max(other.ocol) - other.ocol.min(self.ocol);
        let br_dist = self.cells.rows.max(other.cells.rows) - other.cells.rows.min(self.cells.rows);

        ((tl_dist * tl_dist + br_dist * br_dist) as f32).sqrt() / 2.0
    }

    pub fn distance(&self, other: &Self) -> f32 {
        (((self.orow * self.orow + other.ocol * other.ocol) as f32).sqrt() +
         ((self.cells.columns * self.cells.columns + other.cells.rows * other.cells.rows) as f32).sqrt()) / 2.0
    }

    fn is_mirrored(&self, other: &Self, lr: bool) -> bool {
        if self.width() != other.width() || self.height() != other.height() {
            return false;
        }

        let mut flip: Matrix<Cell> = self.cells.clone();

        if lr {
            flip.flip_lr();
        } else {
            flip.flip_ud();
        }

        for (c1, c2) in flip.values().zip(other.cells.values()) {
            if c1.colour != c2.colour {
                return false;
            }
        }

        true
    }

    pub fn is_mirrored_x(&self, other: &Self) -> bool {
        self.is_mirrored(other, false)
    }

    pub fn is_mirrored_y(&self, other: &Self) -> bool {
        self.is_mirrored(other, true)
    }

    pub fn is_mirror_x(&self) -> bool {
        if self.cells.rows == 1 {
            return false;
        }
        let inc = if self.cells.rows % 2 == 0 { 0 } else { 1 };
        let s1 = self.subshape(0, self.cells.rows / 2, 0, self.cells.columns);
        let s2 = self.subshape(self.cells.rows / 2 + inc, self.cells.rows / 2, 0, self.cells.columns);

        s1.is_mirrored_x(&s2)
    }

    pub fn is_mirror_y(&self) -> bool {
        if self.cells.columns == 1 {
            return false;
        }
        let inc = if self.cells.columns % 2 == 0 { 0 } else { 1 };
        let s1 = self.subshape(0, self.cells.rows, 0, self.cells.columns / 2);
        let s2 = self.subshape(0, self.cells.rows, self.cells.columns / 2 + inc, self.cells.columns / 2);

        s1.is_mirrored_y(&s2)
    }

    fn mirrored(&self, lr: bool) -> Self {
        let mut m: Matrix<Cell> = self.cells.clone();

        if lr {
            m.flip_lr();
        } else {
            m.flip_ud();
        }

        for (x, y) in self.cells.keys() {
            m[(x, y)].row = x + self.orow;
            m[(x, y)].col = y + self.ocol;
        }
        
        Self::new(self.orow, self.ocol, &m)
    }

    pub fn mirrored_x(&self) -> Self {
        self.mirrored(false)
    }

    pub fn mirrored_y(&self) -> Self {
        self.mirrored(true)
    }

    pub fn transposed(&self) -> Self {
        let mut m: Matrix<Cell> = self.cells.clone();

        m.transpose();

        for (x, y) in self.cells.keys() {
            m[(x, y)].row = x + self.orow;
            m[(x, y)].col = y + self.ocol;
        }
        
        Self::new(self.orow, self.ocol, &m)
    }

    pub fn is_transposed(&self, other: &Self) -> bool {
        if self.width() != other.width() || self.height() != other.height() {
            return false;
        }

        let mut flip: Matrix<Cell> = self.cells.clone();

        flip.transpose();

        for (c1, c2) in flip.values().zip(other.cells.values()) {
            if c1 != c2 {
                return false;
            }
        }

        true
    }

    fn rot_90(&self, n: usize) -> Self {
        if !self.is_square() {
            return self.clone();
        }
        let mut m = if n < 3 {
            self.cells.rotated_cw(n)
        } else {
            self.cells.rotated_ccw(1)
        };

        for (x, y) in self.cells.keys() {
            m[(x, y)].row = x + self.orow;
            m[(x, y)].col = y + self.ocol;
        }
        
        Self::new(self.orow, self.ocol, &m)
    }

    pub fn rotated_90(&self) -> Self {
        self.rot_90(1)
    }

    pub fn rotated_180(&self) -> Self {
        self.rot_90(2)
    }

    pub fn rotated_270(&self) -> Self {
        self.rot_90(3)
    }

    pub fn is_rotated_90(&self, other: &Self) -> bool {
        let rot = self.cells.rotated_cw(1);

        for (c1, c2) in rot.values().zip(other.cells.values()) {
            if c1 != c2 {
                return false;
            }
        }

        true
    }

    pub fn is_rotated_180(&self, other: &Self) -> bool {
        let rot = self.cells.rotated_cw(2);

        for (c1, c2) in rot.values().zip(other.cells.values()) {
            if c1 != c2 {
                return false;
            }
        }

        true
    }

    pub fn is_rotated_270(&self, other: &Self) -> bool {
        let rot = self.cells.rotated_ccw(2);

        for (c1, c2) in rot.values().zip(other.cells.values()) {
            if c1 != c2 {
                return false;
            }
        }

        true
    }

    /*
    pub fn transposed(&self) -> Self {
        Self::new_cells(&self.cells.transposed())
    }
    */

    pub fn to_grid(&self) -> Grid {
        Grid::new_from_matrix(&self.cells)
    }

    pub fn to_json(&self) -> String {
        let mut grid: Vec<Vec<usize>> = vec![vec![0; self.cells.columns]; self.cells.rows];

        for x in 0 .. self.cells.rows {
            for y in 0 .. self.cells.columns {
                let colour: usize = self.cells[(x, y)].colour.to_usize();

                grid[x][y] = colour;
            }
        }

        serde_json::to_string(&grid).unwrap()
    }

    pub fn corner_idx(&self) -> (Self, Direction) {
        let (grid, dir) = self.to_grid().corner_idx();

        (grid.as_shape(), dir)
    }

    pub fn corner_body(&self, dir: Direction) -> Self {
        self.to_grid().corner_body(dir).as_shape()
    }

    pub fn split_4(&self) -> Vec<Self> {
        self.to_grid().split_4().iter().map(|s| s.as_shape()).collect()
    }

    pub fn diff(&self, other: &Self) -> Option<Self> {
        self.diff_impl(other, true)
    }

    pub fn diff_orig(&self, other: &Self) -> Option<Self> {
        self.diff_impl(other, false)
    }

    pub fn diff_impl(&self, other: &Self, diff: bool) -> Option<Self> {
        let grid = self.to_grid().diff_impl(&other.to_grid(), diff);

        if let Some(diff) = grid {
            Some(diff.as_shape())
        } else {
            None
        }
    }

    pub fn diff_only_transparent(&self) -> Self {
        let mut s = self.clone();

        for c in s.cells.values_mut() {
            if c.colour != Black {
                c.colour = Colour::to_base(c.colour);
            }
        }

        s
    }

    pub fn recolour(&self, from: Colour, to: Colour) -> Self {
        let mut shape = self.clone();

        shape.colour = to;

        for c in shape.cells.values_mut() {
            if c.colour == from || from == NoColour {
                c.colour = to;
            }
        }

        shape
    }

    pub fn recolour_mut(&mut self, from: Colour, to: Colour) {
        self.colour = to;

        for c in self.cells.values_mut() {
            if c.colour == from || from == NoColour {
                c.colour = to;
            }
        }
    }

    pub fn force_recolour(&self, to: Colour) -> Self {
        let mut shape = self.clone();

        shape.colour = to;

        for c in shape.cells.values_mut() {
            c.colour = to;
        }

        shape
    }

    pub fn force_recolour_mut(&mut self, to: Colour) {
        self.colour = to;

        for c in self.cells.values_mut() {
            c.colour = to;
        }
    }

    pub fn uncolour(&self) -> Self {
        let mut shape = self.clone();

        for c in shape.cells.values_mut() {
            c.colour = c.colour.to_base();
        }

        shape
    }

    pub fn uncolour_mut(&mut self) {
        for c in self.cells.values_mut() {
            c.colour = c.colour.to_base();
        }
    }

    pub fn is_line(&self) -> bool {
        self.cells.rows == 1 && self.cells.columns > 0 || self.cells.rows > 1 && self.cells.columns == 1 
    }

    pub fn is_square(&self) -> bool {
        self.cells.rows == self.cells.columns
    }

    pub fn make_square(&self) -> Self {
        let sz = self.cells.rows.max(self.cells.columns);
        let mut cells = Matrix::new(sz, sz, Cell::new(0, 0, 0));

        for (i, c) in self.cells.items() {
            cells[i].colour = c.colour;
        }
        for (x, y) in cells.keys() {
            cells[(x,y)].row = self.orow + x;
            cells[(x,y)].col = self.ocol + y;
        }

        Shape::new(self.orow, self.ocol, &cells)
    }

    pub fn mut_recolour(&mut self, from: Colour, to: Colour) {
        self.colour = to;

        for c in self.cells.values_mut() {
            if c.colour == from || from == NoColour {
                c.colour = to;
            }
        }
    }

    pub fn mut_force_recolour(&mut self, to: Colour) {
        self.colour = to;

        for c in self.cells.values_mut() {
            c.colour = to;
        }
    }

    pub fn fill_missing(&self, to: Colour) -> Self {
        let mut shape = self.clone();

        for (x, y) in self.cells.keys() {
            if self.cells[(x, y)].colour == Black {
                shape.cells[(x, y)].colour = to;
            }
        }

        shape
    }

    pub fn scale_up(&self, factor: usize) -> Shape {
        let mut cells = Matrix::new(self.cells.rows * factor, self.cells.columns * factor, Cell::new(0, 0, 0));

        for y in 0 .. cells.columns {
            for x in 0 .. cells.rows {
                let xf = x / factor;
                let yf = y / factor;

                cells[(x, y)].row = x;
                cells[(x, y)].col = y;
                cells[(x, y)].colour = self.cells[(xf, yf)].colour;
            }
        }

        Shape::new(0, 0, &cells)
    }

    /*
    pub fn scale_up_divider(&self, factor: usize, border: bool, divider: Colour) -> Shape {
        let resize = divider != NoColour;
        let size = if resize { factor + 1 } else { factor };
        let div = if !resize { 0 } else { factor - 1 };
        let mut cells = Matrix::new(self.cells.rows * factor + div, self.cells.columns * factor + div, Cell::new(0, 0, 0));

        let mut dx = 0;
        let mut dy = 0;

        for y in 0 .. cells.columns {
            if resize {
                dy = y % size;
            }

            for x in 0 .. cells.rows {
                if resize {
                    dx = x % size;
                }

                if x != 0 && y != 0 && resize && (x % size == 0 || y % size == 0) {
                } else {
                    let xf = (x + dx) % factor;
                    let yf = (y + dy) % factor;
println!("{x}/{y} {}/{} {} {}/{}", xf, yf, factor, dx, dy);

                    cells[(x + dx, y + dy)].x = x;
                    cells[(x + dx, y + dy)].y = y;
                    cells[(x + dx, y + dy)].colour = self.cells[(xf, yf)].colour;
                }
            }
        }

        Shape::new(0, 0, &cells)
    }

    pub fn scale_up_divider(&self, factor: usize, border: bool, divider: Colour) -> Shape {
        let bord = if border { 2 } else { 0 };
        let resize = divider != NoColour;
        let size = if resize { factor + 1 } else { factor };
        let div = if !resize { 0 } else { factor - 1 };
        let mut cells = Matrix::new(self.cells.rows * factor + div + bord, self.cells.columns * factor + div + bord, Cell::new(0, 0, 0));

        for b in 0 .. factor {
            for x in 0 .. self.cells.rows {
                for y in 0 .. self.cells.columns {
//println!("{x}/{y} {}/{} {} {}/{}", xf, yf, factor, dx, dy);

                    let nx = x + self.cells.rows * b;
                    let ny = y + self.cells.columns * b;
//println!("{}/{} {}/{}", nx, ny, xf, yf);

                    cells[(nx, ny)].x = nx;
                    cells[(nx, ny)].y = ny;
                    cells[(nx, ny)].colour = self.cells[(nx % (b + 1), ny % (b + 1))].colour;
                }
            }
        }

        Shape::new(0, 0, &cells)
    }
    */

    pub fn has_border_hole(&self, hole: bool, fc: &dyn Fn(usize, usize, usize, usize) -> bool) -> bool {
        let (x, y) = self.dimensions();
        if x < 3 || y < 3 {
            return false;
        }
        for ((x, y), c) in self.cells.items() {
            let pred = fc(x, y, self.cells.rows, self.cells.columns);

            if c.colour == Black && pred || hole && c.colour != Black && !pred {
                return false;
            }
        }

        true
    }

    pub fn has_border(&self) -> bool {
        self.has_border_hole(false, &|x, y, rows, cols| x == 0 || y == 0 || x == rows - 1 || y == cols - 1)
    }

    pub fn has_hole(&self) -> bool {
        self.has_border_hole(true, &|x, y, rows, cols| x == 0 || y == 0 || x == rows - 1 || y == cols - 1)
    }

    pub fn has_open_border_top(&self) -> bool {
        self.has_border_hole(true, &|x, y, rows, cols| y == 0 || x == rows - 1 || y == cols - 1)
    }

    pub fn has_open_hole_top(&self) -> bool {
        self.has_border_hole(false, &|x, y, rows, cols| y == 0 || x == rows - 1 || y == cols - 1)
    }

    pub fn has_open_border_bottom(&self) -> bool {
        self.has_border_hole(true, &|x, y, _rows, cols| x == 0 || y == 0 || y == cols - 1)
    }

    pub fn has_open_hole_bottom(&self) -> bool {
        self.has_border_hole(false, &|x, y, _rows, cols| x == 0 || y == 0 || y == cols - 1)
    }

    pub fn has_open_border_left(&self) -> bool {
        self.has_border_hole(true, &|x, y, rows, cols| x == 0 || x == rows - 1 || y == cols - 1)
    }

    pub fn has_open_hole_left(&self) -> bool {
        self.has_border_hole(false, &|x, y, rows, cols| x == 0 || x == rows - 1 || y == cols - 1)
    }

    pub fn has_open_border_right(&self) -> bool {
        self.has_border_hole(true, &|x, y, rows, _cols| x == 0 || y == 0 || x == rows - 1)
    }

    pub fn has_open_hole_right(&self) -> bool {
        self.has_border_hole(false, &|x, y, rows, _cols| x == 0 || y == 0 || x == rows - 1)
    }

    pub fn has_border_break(&self) -> (Direction, usize, usize) {
        for ((r, c), cell) in self.cells.items() {
            let grow = self.orow + r;
            let gcol = self.ocol + c;

            if cell.colour == Black && r == 0 {
                return (Up, grow, gcol);
            }
            if cell.colour == Black && c == 0  {
                return (Left, grow, gcol);
            }
            if cell.colour == Black && r == self.cells.rows - 1 {
                return (Down, grow, gcol);
            }
            if cell.colour == Black && c == self.cells.columns - 1 {
                return (Right, grow, gcol);
            }
        }

        (Other, 0, 0)
    }

    // TODO - remove old other from Shapes?
    pub fn centre_in(&self, other: &Self) -> Self {
        let new_other = self.move_in(other);
        let mut shape = self.clone();

        for ((x, y), c) in new_other.cells.items() {
            shape.cells[(x, y)] = c.clone();
        }

        shape
    }

    /*
    pub fn mv(&self, xx: isize, yy: isize) -> Self {
        let mut shape = self.clone();

        shape.ox = (shape.ox as isize + xx) as usize;
        shape.oy = (shape.oy as isize + yy) as usize;

        for (x, y) in self.cells.keys() {
            shape.cells[(x, y)].x = (x as isize + xx) as usize;
            shape.cells[(x, y)].y = (y as isize + yy) as usize;
        }

        shape
    }
    */

    // TODO - remove old other from Shapes?
    pub fn put_all_in(&mut self, other: &Self) {
        for ((x, y), c) in other.cells.items() {
            self.cells[(x, y)] = c.clone();
        }
    }

    pub fn centre_of(&self) -> (usize, usize) {
        (self.orow + self.cells.rows / 2, self.ocol + self.cells.columns / 2)
    }

    pub fn centre_of_exact(&self) -> (f32, f32) {
        (self.orow as f32 + self.cells.rows as f32  / 2.0, self.ocol as f32 + self.cells.columns as f32  / 2.0)
    }

    pub fn pixel_coords(&self, colour: Colour) -> Option<(usize, usize)> {
        for c in self.cells.values() {
            if c.colour == colour {
                return Some((c.row, c.col));
            }
        }

        None
    }

    pub fn move_in(&self, other: &Self) -> Self {
        let (x, y) = self.centre_of();
        let (ox, oy) = other.centre_of();
        let dx = (x - ox) as isize;
        let dy = (y - oy) as isize;

        other.translate(dx, dy)
    }

    fn container(&self, other: &Self) -> bool {
//println!("{} <= {} && {} <= {} && {} > {} && {} > {}",
//        self.ox, other.ox, self.oy, other.oy, self.cells.rows + self.ox, other.cells.rows + other.ox, self.cells.columns + self.oy, other.cells.columns + other.oy);
        self.orow <= other.orow && self.ocol <= other.ocol && self.cells.rows + self.orow >= other.cells.rows + other.orow && self.cells.columns + self.ocol >= other.cells.columns + other.ocol
    }

    pub fn can_contain(&self, other: &Self) -> bool {
        let (s, o) = if self.size() > other.size() { (self, other) } else { (other, self) };
        if s.width() < 3 || s.height() < 3 || s.width() < o.width() + 2 || s.height() < o.height() + 2 { return false };
        for (x, y) in other.cells.keys() {
            if x != 0 && y != 0 && x < s.cells.rows && x < s.cells.columns && s.cells[(x, y)].colour != Black {
                return false;
            }
        }

        s.container(other)
    }

    pub fn contained_by(&self, other: &Self) -> bool {
        other.can_contain(self)
    }

    pub fn is_contained(&self, other: &Self) -> bool {
        let (s, o) = if self.size() > other.size() { (self, other) } else { (other, self) };
        if s.width() < 3 || s.height() < 3 || s.width() < o.width() + 2 || s.height() < o.height() + 2 { return false };
        for (x, y) in o.cells.keys() {
            if x != 0 && y != 0 && x < s.cells.rows && x < s.cells.columns && s.cells[(x, y)].colour != o.cells[(x - 1, y - 1)].colour {
                return false;
            }
        }

        s.container(other)
    }

    pub fn contained_in(&self, other: &Self) -> bool {
        other.is_contained(self)
    }

    pub fn adjacent(&self, other: &Self) -> bool {
        self.orow + self.cells.columns + 1 == other.orow ||
        self.ocol + self.cells.rows + 1 == other.ocol ||
        other.orow + other.cells.columns + 1 == self.orow ||
        other.ocol + other.cells.rows + 1 == self.ocol
    }

    pub fn single_colour(&self) -> Option<Colour> {
        let colour = self.cells[(0, 0)].colour;

        for c in self.cells.values() {
            if c.colour != colour { return None; };
        }

        Some(colour)
    }

    pub fn is_single_colour(&self) -> bool {
        let mut first = true;
        let mut colour = NoColour;

        for c in self.cells.values() {
            if first {
                colour = c.colour;
                first = false;
            } else if c.colour != colour {
                return false;
            }
        }
        
        true
    }

    pub fn translate_x(&self, x: isize) -> Self {
        Self::translate(self, x, 0)
    }

    pub fn translate_y(&self, y: isize) -> Self {
        Self::translate(self, 0, y)
    }

    pub fn translate(&self, x: isize, y: isize) -> Self {
        let mut shape = self.clone();

        if self.orow as isize + x < 0 || self.ocol as isize + y < 0 {
            return shape;
        }


        shape.orow = (shape.orow as isize + x) as usize;
        shape.ocol = (shape.ocol as isize + y) as usize;

        shape.cells.iter_mut()
            .for_each(|c| {
                c.row = (c.row as isize + x) as usize;
                c.col = (c.col as isize + y) as usize;
            });

        shape
    }

    pub fn translate_absolute_x(&self, x: usize) -> Self {
        Self::translate_absolute(self, x, 0)
    }

    pub fn translate_absolute_y(&self, y: usize) -> Self {
        Self::translate_absolute(self, 0, y)
    }

    pub fn translate_absolute(&self, x: usize, y: usize) -> Self {
        let mut shape = self.normalise_key();

        shape.orow = x;
        shape.ocol = y;
//println!("{} {}", x, y);

        shape.cells.iter_mut()
            .for_each(|c| {
                c.row += x;
                c.col += y;
            });

        shape
    }

    // TODO!
    /*
    pub fn translate_absolute_clip(&self, x: isize, y: isize) -> Self {
        let offx = self.ox as isize + x;
        let offy = self.oy as isize + y;
        let rows = if offx < 0 { (self.cells.rows as isize + offx) as usize } else { self.cells.rows };
        let cols = if offy < 0 { (self.cells.columns as isize + offy) as isize as usize } else { self.cells.columns };
//println!("{offx}/{offy} {}/{}", rows, cols);

        let mut shape = Shape::new_sized_coloured(rows, cols, Black);

        shape.ox = if x < 0 { 0 } else { x as usize };
        shape.oy = if y < 0 { 0 } else { y as usize };

        let x = if x < 0 { self.cells.rows - rows } else { x as usize };
        let y = if y < 0 { self.cells.columns - cols } else { y as usize };

        shape.colour = self.colour;

println!("{rows}/{cols} {}/{} {x}/{y} {}/{}", self.cells.rows, self.cells.columns, shape.ox, shape.oy);
        for xi in 0 .. rows {
            for yi in 0 .. cols {
                shape.cells[(xi, yi)].x = xi + shape.ox;
                shape.cells[(xi, yi)].y = yi + shape.oy;
                shape.cells[(xi, yi)].colour = self.cells[(xi + shape.ox, yi + shape.oy)].colour;
            }
        }

        shape
    }
    */

    /*
    fn quartered(&self, ox : usize, oy: usize) -> Option<Self> {
        if self.cells.rows % 2 == 1 || self.cells.columns % 2 == 1 {
            return None
        }

        let xs = self.cells.rows / 2;
        let ys = self.cells.columns / 2;

        let mut cells = Matrix::new(xs, ys, Cell::new(0, 0, 0));

        for y in 0 .. ys {
            for x in 0 .. xs {
                cells[(x, y)].x = x;
                cells[(x, y)].y = y;
                cells[(x, y)].colour = self.cells[(ox + x, oy + y)].colour;
            }
        }

        Some(Self::new_cells(&cells))
    }

    pub fn quarter_tl(&self) -> Option<Self> {
        self.quartered(0, 0)
    }

    pub fn quarter_tr(&self) -> Option<Self> {
        self.quartered(self.cells.columns / 2, 0)
    }

    pub fn quarter_bl(&self) -> Option<Self> {
        self.quartered(0, self.cells.rows / 2)
    }

    pub fn quarter_br(&self) -> Option<Self> {
        self.quartered(self.cells.columns / 2, self.cells.rows / 2)
    }
    */

    /*
    #[allow(clippy::comparison_chain)]
    pub fn extend_x(&self, x: isize) -> Self {
        if x == 0 {
            return self.clone();
        }

        let mut cells = self.cells.clone();

        self.cells.iter()
            .for_each(|c| {
                if x < 0 {
                    for i in 0 .. -x {
                        let mut nc = c.clone();

                        nc.x -= i + 1;

                        if !cells.contains(&nc) { cells.push(nc); }
                    }
                } else if x > 0 {
                    for i in 0 .. x {
                        let mut nc = c.clone();

                        nc.x += i + 1;

                        if !cells.contains(&nc) { cells.push(nc); }
                    }
                }
            });

        Self::new(&cells)
    }

    #[allow(clippy::comparison_chain)]
    pub fn extend_y(&self, y: isize) -> Self {
        if y == 0 {
            return self.clone();
        }

        let mut cells = self.cells.clone();

        self.cells.iter()
            .for_each(|c| {
                if y < 0 {
                    for i in 0 .. -y {
                        let mut nc = c.clone();

                        nc.y -= i + 1;

                        if !cells.contains(&nc) { cells.push(nc); }
                    }
                } else if y > 0 {
                    for i in 0 .. y {
                        let mut nc = c.clone();

                        nc.y += i + 1;

                        if !cells.contains(&nc) { cells.push(nc); }
                    }
                }
            });

        Self::new(&cells)
    }
    */

    /*
    // Assumes they have overlapping x or y
    pub fn connect(&self, other: &Self, colour: Colour) -> Self {
        let (sx, sy) = self.centre_of();
        let (ox, oy) = other.centre_of();
        let (x, y) = if self.size() < other.size() { (sx, sy) } else { (ox, oy) };
        let cells = 
            if self.above(other) {
                let mut cells: Matrix<Cell> = Matrix::from_fn(1, oy - sy , |(_, _)| Cell::new_empty());

                for i in sy .. oy {
                    let nc = Cell::new(i, y, colour.to_usize());

                    if !self.cells.contains(&nc) && !other.cells.contains(&nc) {
                        cells[(0, i - sy)] = nc;
                    }
                }

                cells
            } else if self.below(other) {
                let mut cells: Matrix<Cell> = Matrix::from_fn(1, sy - oy , |(_, _)| Cell::new_empty());

                for i in oy .. sy {
                    let nc = Cell::new(i, y, colour.to_usize());

                    if !self.cells.contains(&nc) && !other.cells.contains(&nc) {
                        cells[(0, i - oy)] = nc;
                    }
                }

                cells
            } else if self.left(other) {
                let mut cells: Matrix<Cell> = Matrix::from_fn(sx - ox, 1, |(_, _)| Cell::new_empty());

                for i in sx .. ox {
                    let nc = Cell::new(x, i, colour.to_usize());

                    if !self.cells.contains(&nc) && !other.cells.contains(&nc) {
                        cells[(i - ox, 0)] = nc;
                    }
                }

                cells
            } else /* right */ {
                let mut cells: Matrix<Cell> = Matrix::from_fn(ox - sx, 1, |(_, _)| Cell::new_empty());

                for i in ox .. sx {
                    let nc = Cell::new(x, i, colour.to_usize());

                    if !self.cells.contains(&nc) && !other.cells.contains(&nc) {
                        cells[(i - sx, 0)] = nc;
                    }
                }

                cells
            };

        Self::new_cells(&cells)
    }

    // TODO fix plus x N
    pub fn connect_angle(&self, other: &Self, colour: Colour, horizontal: bool) -> Vec<Self> {
        let (sx, sy) = self.centre_of();
        let (ox, oy) = other.centre_of();

        let cell = if horizontal {
            Cell::new(sx, oy, colour.to_usize())
        } else {
            Cell::new(ox, sy, colour.to_usize())
        };

        let cells: Matrix<Cell> = Matrix::from_fn(cell.x, cell.y , |(_, _)| cell.clone());
        let joint = Self::new_cells(&cells);
        let arm_1 = self.connect(&joint, colour);
        let arm_2 = other.connect(&joint, colour);

        vec![Self::new_cells(&arm_1.cells), Self::new_cells(&arm_2.cells)]
    }
    */

    pub fn have_common_pixel_colour(&self, other: &Self) -> bool {
        let (colour_l, cnt_l) = self.colour_cnt(false);
        let (colour_r, cnt_r) = other.colour_cnt(false);

        colour_l == colour_r && cnt_l == 1 && cnt_r == 1
    }

    pub fn merge_on_common_pixel(&self, other: &Self) -> Option<Self> {
        let (colour_l, cnt_l) = self.colour_cnt(false);
        let (colour_r, cnt_r) = other.colour_cnt(false);

        if colour_l != colour_r || cnt_l != 1 || cnt_r != 1 {
            return None;
        }

        if let Some((x, y)) = self.pixel_coords(colour_l) {
            if let Some((ox, oy)) = other.pixel_coords(colour_r) {
                let dx = (x - ox) as isize;
                let dy = (y - oy) as isize;

                Some(other.translate(dx, dy))
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn normalise(&self, si: &Self) -> (Self, Self) {
        let mut o = self.clone();
        let mut i = si.clone();

        i.orow -= self.orow;
        i.ocol -= self.ocol;

        for c in i.cells.values_mut() {
            c.row -= self.orow;
            c.col -= self.ocol;
        }

        o.orow -= self.orow;
        o.ocol -= self.ocol;

        for c in o.cells.values_mut() {
            c.row -= self.orow;
            c.col -= self.ocol;
        }

        (i, o)
    }

    pub fn normalise_key(&self) -> Self {
        let mut i = self.clone();
        let ox = if self.orow == 0 { self.orow } else { self.orow - 1 };
        let oy = if self.ocol == 0 { self.ocol } else { self.ocol - 1 };

        i.orow -= ox;
        i.ocol -= oy;
        //i.ox = 0;
        //i.oy = 0;

        for c in i.cells.values_mut() {
            c.row -= ox;
            c.col -= oy;
        }

        i
    }

    pub fn to_origin(&self) -> Self {
        self.to_position(0, 0)
    }

    pub fn to_origin_mut(&mut self) {
        self.to_position_mut(0, 0)
    }

    // relative so call to_origin first if absolute
    pub fn to_position(&self, r: usize, c: usize) -> Self {
        let mut i = self.clone();

        i.to_position_mut(r, c);

        i
    }

    pub fn to_position_mut(&mut self, r: usize, c: usize) {
        for cell in self.cells.values_mut() {
            cell.row = cell.row - self.orow + r;
            cell.col = cell.col - self.ocol + c;
        }

        self.orow = r;
        self.ocol = c;
    }

    pub fn compare(&self, shape: &Self) -> bool {
        if self.cells.rows != shape.cells.rows || self.cells.columns != shape.cells.columns {
            return false;
        }
        let diffor = self.orow - shape.orow;
        let diffoc = self.ocol - shape.ocol;

        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                let col1 = self.cells[(r,c)].colour;
                let col2 = shape.cells[(r,c)].colour;
                let diffr = self.cells[(r,c)].row - shape.cells[(r,c)].row;
                let diffc = self.cells[(r,c)].col - shape.cells[(r,c)].col;

                if col1 != col2 || diffor != diffr || diffoc != diffc {
                    return false;
                }
            }
        }

        true
    }

    // Shapesmust be the sample size
    pub fn copy_into(&self, shape: &Self) -> Self {
        let mut s = shape.clone();

        if self.width() != shape.width() || self.height() != shape.height() {
            return s;
        }

        for ((r, c), cell) in self.cells.items() {
            s.cells[(r, c)].colour = cell.colour;
        }

        s
    }

    pub fn position(&self, ci: &Self) -> Self {
        let mut i = self.clone();
        let ox = if ci.orow == 0 { ci.orow } else { ci.orow - 1 };
        let oy = if ci.ocol == 0 { ci.ocol } else { ci.ocol - 1 };

        i.orow = ox;
        i.ocol = oy;

        for c in i.cells.values_mut() {
            c.row += ox;
            c.col += oy;
        }

        i
    }

    /*
    pub fn shrink(&self) -> Self {
        let mut sx = 0;
        let mut ex = self.cells.rows;
        let mut sy = 0;
        let mut ey = self.cells.columns;

        for x in 0 .. self.cells.rows {
            if (x .. self.cells.columns).filter(|&c| c != 0).count() == 0 {
                sx += 1;
            } else {
                break;
            }
        }
        for x in (0 .. self.cells.rows).rev() {
            if (x .. self.cells.columns).filter(|&c| c != 0).count() == 0 {
                ex -= 1;
            } else {
                break;
            }
        }
        if sx > ex { return self.clone(); }
        for y in 0 .. self.cells.columns {
            if (y .. self.cells.rows).filter(|&c| c != 0).count() == 0 {
                sy += 1;
            } else {
                break;
            }
        }
        for y in (0 .. self.cells.columns).rev() {
            if (y .. self.cells.rows).filter(|&c| c != 0).count() == 0 {
                ey -= 1;
            } else {
                break;
            }
        }
        if sy > ey { return self.clone(); }
println!("{sx} -> {ex}, {sy} -> {ey}");
        let mut m = Matrix::new(ex - sx, ey - sy, Cell::new(0, 0, 0));

        for x in sx .. ex {
            for y in sy .. ey {
                m[(x - sx, y - sy)] = self.cells[(x, y)].clone();
            }
        }
println!("{sx} -> {ex}, {sy} -> {ey}");

        Self::new_cells(&m)
    }
    */

    pub fn get_arm(&self) -> (Direction, usize){
        for pat in &self.cats {
            match pat {
                ShapeCategory::ArmTop(n) =>
                    return (Up, *n),
                ShapeCategory::ArmBottom(n) => 
                    return (Down, *n),
                ShapeCategory::ArmLeft(n) => 
                    return (Left, *n),
                ShapeCategory::ArmRight(n) => 
                    return (Right, *n),
                _ => ()
            }
        }

        (Other, 0)
    }

    pub fn trim_arm(&self) -> Self {
        let (dir, distance) = self.get_arm();
//self.show_summary();
//println!("--- {:?} {}", dir, distance);

        match dir {
            Up =>
                self.subshape2(self.orow, self.cells.rows - distance, self.ocol, self.cells.columns),
            Down => 
                self.subshape2(self.orow + distance, self.cells.rows - distance, self.ocol, self.cells.columns),
            Left => 
                self.subshape2(self.orow, self.cells.rows, self.ocol, self.cells.columns - distance),
            Right => 
                self.subshape2(self.orow + distance, self.cells.rows - distance, self.ocol + distance, self.cells.columns - distance),
            _ => self.clone(),
        }
    }

    pub fn has_arm(&self) -> ShapeCategory {
        for ((x, y), c) in self.cells.items() {
            match c.cat {
                CellCategory::PointT => {
                    if self.cells.rows > 1 && self.cells.columns == 1 {
                        return ShapeCategory::VerticalLine;
                    }
                    let mut i = 1;
                    while x + i < self.cells.rows && self.cells[(x+i,y)].cat == CellCategory::StemTB {
                        i += 1;
                    }
                    if i >= 2 {
                        return ShapeCategory::ArmTop(i)
                    }
                },
                CellCategory::PointB => {
                    if self.cells.rows > 1 && self.cells.columns == 1 {
                        return ShapeCategory::VerticalLine;
                    }
                    let mut i = 1;
                    while x < i && self.cells[(x-i,y)].cat == CellCategory::StemTB {
                        i += 1;
                    }
                    if i >= 2 {
                        return ShapeCategory::ArmBottom(i)
                    }
                },
                CellCategory::PointL => {
                    if self.cells.rows == 1 && self.cells.columns > 1 {
                        return ShapeCategory::HorizontalLine;
                    }
                    let mut i = 1;
                    while y + i < self.cells.columns && self.cells[(x,y+i)].cat == CellCategory::StemLR {
                        i += 1;
                    }
                    if i >= 2 {
                        return ShapeCategory::ArmLeft(i)
                    }
                },
                CellCategory::PointR => {
                    if self.cells.rows == 1 && self.cells.columns > 1 {
                        return ShapeCategory::HorizontalLine;
                    }
                    let mut i = 1;
                    while y > i && self.cells[(x,y-i)].cat == CellCategory::StemLR {
                        i += 1;
                    }
                    if i >= 2 {
                        return ShapeCategory::ArmRight(i)
                    }
                },
                _ => continue,
            }
        }

        ShapeCategory::Other
    }

    /*
    pub fn repeat(&self, x: isize, y: isize) -> Self {
    }
    */

    // TODO
    // pub fn contract?

    /*
    pub fn horizontal_stretch(&self, cells: Vec<Cell>) -> Self {
        self.clone()
    }

    pub fn vertical_stretch(&self, cells: Vec<Cell>) -> Self {
        self.clone()
    }

    // bool is true == X axis, false = Y axis
    pub fn striped_x(&self) -> Colour {
        NoColour
    }

    pub fn striped_y(&self) -> Colour {
        NoColour
    }

    pub fn strip_colours_x(&self) -> Vec<Colour> {
        vec![]
    }

    pub fn strip_colours_y(&self) -> Vec<Colour> {
        vec![]
    }

    pub fn flip_colours(&self) -> Self {
        self.clone()
    }

    pub fn box_it(&self) -> Self {
        self.clone()
    }

    pub fn four_borders(&self, colour: Colour) -> Self {
        self.clone()
    }

    pub fn four_diagonals(&self, colour: Colour) -> Self {
        self.clone()
    }
    */

    pub fn bordered_shape(&self) -> bool {
        let colour = self.cells[(0,0)].colour;

        if self.cells.rows < 3 || self.cells.columns < 3 || colour == Black {
            return false;
        }
        
        for ((x, y), c) in self.cells.items() {
            if x == 0 && c.colour != colour {
                return false;
            }
            if y == 0 && c.colour != colour {
                return false;
            }
            if x == self.cells.rows - 1 && c.colour != colour {
                return false;
            }
            if y == self.cells.columns - 1 && c.colour != colour {
                return false;
            }
        }

        true
    }

    // Expensive!
    pub fn hollow(&self) -> bool {
        if self.cells.rows < 3 || self.cells.columns < 3 || self.cells[(1,1)].colour != Black && self.cells.rows == 3 && self.cells.columns == 3 && !self.is_full(){
            return false;
        }
//self.show();

        'outer:
        for ((x, y), c) in self.cells.items() {
            // Interior cells only interesting
            if x == 0 || y == 0 || x == self.cells.rows - 1 || y == self.cells.columns - 1 || c.colour != Black {
                continue;
            }

            let reachable = self.cells.bfs_reachable((x, y), true, |i| self.cells[i].colour != self.colour);

            for (x,y) in &reachable {
                if *x == 0 || *y == 0 || *x == self.cells.rows - 1 || *y == self.cells.columns - 1 {
                    continue 'outer;
                }
            }

            return true;
        }
//println!("{hollow}");

        false
    }

    /*
    pub fn find_hollows(&self) -> Shapes {
        if self.cells.rows < 3 || self.cells.columns < 3 || self.cells[(1,1)].colour != Black && self.cells.rows == 3 && self.cells.columns == 3 && !self.is_full(){
            return Shapes::new();
        }
//self.show();
        let mut shapes = Shapes::new_from_shape(self);

        'outer:
        for ((x, y), c) in self.cells.items() {
            // Interior cells only interesting
            if x == 0 || y == 0 || x == self.cells.rows - 1 || y == self.cells.columns - 1 || c.colour != Black {
                continue;
            }

            let reachable = self.cells.bfs_reachable((x, y), true, |i| self.cells[i].colour != self.colour);

            for (x,y) in &reachable {
                if *x == 0 || *y == 0 || *x == self.cells.rows - 1 || *y == self.cells.columns - 1 {
                    continue 'outer;
                }
            }

            let s = Shape::shape_from_reachable(self, &reachable);

            shapes.shapes.push(s);
        }
//println!("{hollow}");

        shapes
    }
    */

    pub fn hollow_colour_count(&self) -> (Colour, usize) {
        let ss = self.fill_boundary_colour().to_grid().find_colour_patches(Black);

        (self.colour, ss.shapes.len())
    }

    pub fn full_shape(&self) -> bool {
        let colour = self.cells[(0,0)].colour;

        for c in self.cells.values() {
            if c.colour != colour {
                return false;
            }
        }

        true
    }

    pub fn toddle_colour(&self, bg: Colour, fg: Colour) -> Self {
        let s = self.recolour(bg, ToBlack + bg).recolour(fg, bg);

        s.recolour(ToBlack + bg, fg)
    }

    pub fn extend_top(&self, n: usize) -> Self {
        let mut cells = Matrix::new(self.cells.rows + n, self.cells.columns, Cell::new(0, 0, 0));
        for y in 0 .. cells.columns {
            for x in 0 .. n {
                cells[(x, y)].row = x + self.orow;
                cells[(x, y)].col = y + self.ocol;
                cells[(x, y)].colour = self.cells[(0, y)].colour;
            }
        }
        for y in 0 .. cells.columns {
            for x in n .. cells.rows {
                cells[(x, y)].row = x + self.orow;
                cells[(x, y)].col = y + self.ocol;
                cells[(x, y)].colour = self.cells[(x - n, y)].colour;
            }
        }

        Shape::new(self.orow, self.ocol, &cells)
    }

    pub fn extend_bottom(&self, n: usize) -> Self {
        self.mirrored_x().extend_top(n).mirrored_x()
    }

    pub fn extend_left(&self, n: usize) -> Self {
        let mut cells = Matrix::new(self.cells.rows, self.cells.columns + n, Cell::new(0, 0, 0));

        for x in 0 .. cells.rows {
             for y in 0 .. n {
                cells[(x, y)].row = x + self.orow;
                cells[(x, y)].col = y + self.ocol;
                cells[(x, y)].colour = self.cells[(x, 0)].colour;
            }
        }
        for x in 0 .. cells.rows {
            for y in n .. cells.columns {
                cells[(x, y)].row = x + self.orow;
                cells[(x, y)].col = y + self.ocol;
                cells[(x, y)].colour = self.cells[(x, y - n)].colour;
            }
        }

        Shape::new(self.orow, self.ocol, &cells)
    }

    pub fn extend_right(&self, n: usize) -> Self {
        self.mirrored_y().extend_left(n).mirrored_y()
    }

    pub fn dense(&self) -> bool {
        for c in self.cells.values() {
            if c.colour == Black {
                return false;
            }
        }

        true
    }

    pub fn split_shapes(&self) -> Shapes {
        if self.cells.rows < 4 || self.cells.rows % 2 != 0 || self.cells.rows != self.cells.columns {
            return Shapes::new_from_shape(&self);
        }

        Shapes::new_from_shape(&self)
    }

    // Single colour shapes only
    pub fn de_subshape(&self, shapes: &mut Shapes) {
        if self.size() > 9 {
            let ss = self.to_grid().to_shapes_sq();

            if ss.len() > 1
            {
                for ns in &ss.shapes {
                    shapes.shapes.push(ns.clone());
                }
            } else {
                shapes.add(self);
            }
        } else {
            shapes.add(self);
        }
    }

    pub fn fill_lines(&self, colour: Colour) -> Self {
        let mut s = self.clone();
        let mut rows: BTreeMap<usize, usize> = BTreeMap::new();
        let mut cols: BTreeMap<usize, usize> = BTreeMap::new();

        for ((row, col), c) in self.cells.items() {
            if c.colour != Black {
                *rows.entry(row).or_insert(0) += 1;
                *cols.entry(col).or_insert(0) += 1;
            }
        }

        for ((row, col), cell) in s.cells.items_mut() {
            if let Some(r) = rows.get(&row) {
                if *r > 3 && cell.colour == Black {
                    cell.colour = colour;
                }
            }
            if let Some(c) = cols.get(&col) {
                if *c > 3 && cell.colour == Black {
                    cell.colour = colour;
                }
            }
        }

        s
    }

    pub fn find_first_blacks(&self) -> Vec<(usize,usize)> {
        let mut fb: Vec<(usize,usize)> = Vec::new();
        let mut black = false;
        let mut nr = 0;

        for ((r, c), cell) in self.cells.items() {
            if cell.colour == Black {
                if !black && nr == r {
                    fb.push((r,c));
                }
            } else if c == 0 {
                nr = r + 1;
            }
            black = cell.colour == Black;
        }

        fb
    }

    // Result may need trimming later
    pub fn surround(&self, thickness: usize, colour: Colour, all: bool, corners: bool) -> Self {
        if self.orow < thickness || self.ocol < thickness {
            return self.clone();
        }

        let height = self.cells.rows + thickness * 2;
        let width = self.cells.columns + thickness * 2;
        let mut shape = Shape::new_sized_coloured(height, width, Transparent);

        //let this = self.translate_absolute(100, 100);
//println!("{this:?}");
        let this = self.clone();

        shape.colour = colour;
        shape.orow = this.orow - thickness;
        shape.ocol = this.ocol - thickness;

        for x in 0 .. shape.cells.rows {
            for y in 0 .. shape.cells.columns {
                shape.cells[(x,y)].row = this.orow + x - thickness;
                shape.cells[(x,y)].col = this.ocol + y - thickness;

                let bounds = x < thickness && y < thickness || x < thickness && y >= this.cells.columns + thickness || x >= this.cells.rows + thickness && y < thickness || x >= this.cells.rows + thickness && y >= this.cells.columns + thickness;

                if !all && (!corners && bounds || corners && !bounds) {
                    continue;
                }
                if x < thickness || x >= this.cells.rows + thickness || y < thickness || y >= this.cells.columns + thickness {
                    shape.cells[(x,y)].colour = colour;
                }
            }
        }

//println!("{:?}", shape.translate_absolute(self.ox, self.oy));
        //shape.translate_absolute(self.ox, self.oy)
        shape
    }

    pub fn categorise_shape(&mut self) {
        let has_border = self.has_border();

        /*
        if has_border {
            self.cats.insert(ShapeCategory::SingleCell);
        }
        */
        if has_border && self.has_hole() {
            self.cats.insert(ShapeCategory::HasHole);
        }
        if !has_border && self.cells.rows == self.cells.columns {
            if self.cells.rows == 1 {
                self.cats.insert(ShapeCategory::Pixel);
                //self.cats.insert(ShapeCategory::HorizontalLine);
                //self.cats.insert(ShapeCategory::VerticalLine);
            } else {
                self.cats.insert(ShapeCategory::Square);
            }
        }
        if !has_border {
            self.cats.insert(ShapeCategory::HasBorder);
        }
        if !has_border && self.has_open_border_top() {
            self.cats.insert(ShapeCategory::OpenTop);
        }
        if !has_border && self.has_open_border_bottom() {
            self.cats.insert(ShapeCategory::OpenBottom);
        }
        if !has_border && self.has_open_border_left() {
            self.cats.insert(ShapeCategory::OpenLeft);
        }
        if !has_border && self.has_open_border_right() {
            self.cats.insert(ShapeCategory::OpenRight);
        }
        if !has_border && self.has_open_hole_top() {
            self.cats.insert(ShapeCategory::OpenTopHole);
        }
        if !has_border && self.has_open_hole_bottom() {
            self.cats.insert(ShapeCategory::OpenBottomHole);
        }
        if !has_border && self.has_open_hole_left() {
            self.cats.insert(ShapeCategory::OpenLeftHole);
        }
        if !has_border && self.has_open_hole_right() {
            self.cats.insert(ShapeCategory::OpenRightHole);
        }
        //if !has_border && self.is_full() {
        if self.is_full() {
            self.cats.insert(ShapeCategory::Full);
        }
        /*
        // Only sensible when called tactically
        } else if !has_border {
            if self.is_mirror_x() {
                self.cats.insert(ShapeCategory::MirrorX);
            }
            if self.is_mirror_y() {
                self.cats.insert(ShapeCategory::MirrorY);
            }
        }
        else if !has_border && self.hollow() {
//println!("hhhh");
            // Too expensive
            self.cats.insert(ShapeCategory::Hollow);
        }
        */

//        for _ in 0 .. 2 {
            let arm = self.has_arm();
            if !has_border && arm != ShapeCategory::Other {
//println!("-- {:?}", arm);
                self.cats.insert(arm);
            }
//        }
//println!("{:?}", self.cats);
    }

    pub fn cell_category(cells: &Matrix<Cell>) -> Matrix<Cell> {
        let mut m = cells.clone();

        for ((x, y), c) in m.items_mut() {
            if c.colour == Black { continue; }

            let cat = 
                if x == 0 {
                    if (y == 0 || cells[(x,y-1)].colour == Black) && (y == cells.columns - 1 || cells[(x,y+1)].colour == Black) {
                        CellCategory::PointT
                    } else if y == 0 {
                        if cells.rows == 1 || cells[(x+1,y)].colour == Black {
                            CellCategory::PointL
                        } else {
                            CellCategory::CornerTL
                        }
                    } else if y == cells.columns - 1 {
                        CellCategory::CornerTR
                    } else if y < cells.columns - 1 && cells[(x,y+1)].colour == Black {
                        CellCategory::InternalCornerTR
                    } else if y > 0 && cells[(x,y-1)].colour == Black {
                        CellCategory::InternalCornerTL
                    } else if cells.rows == 1 || cells[(x+1,y)].colour == Black {
                        CellCategory::StemLR
                    } else {
                        CellCategory::EdgeT
                    }
                } else if x == cells.rows - 1 {
                    if (y == 0 || cells[(x,y-1)].colour == Black) && (y == cells.columns - 1 || cells[(x,y+1)].colour == Black) {
                        CellCategory::PointB
                    } else if y == 0 {
                        CellCategory::CornerBL
                    } else if y == cells.columns - 1 {
                        if cells[(x-1,y)].colour == Black {
                            CellCategory::PointR
                        } else {
                            CellCategory::CornerBR
                        }
                    } else if y < cells.columns - 1 && cells[(x,y+1)].colour == Black {
                        CellCategory::InternalCornerBR
                    } else if y > 0 && cells[(x,y-1)].colour == Black {
                        CellCategory::InternalCornerBL
                    } else if cells.rows == 1 || cells[(x-1,y)].colour == Black {
                        CellCategory::StemLR
                    } else {
                        CellCategory::EdgeB
                    }
                } else if y == 0 {
//println!("{x}/{y}: {:?} {:?}", cells[(x-1,y)].colour, cells[(x+1,y)].colour);
                    if (x == 0 || cells[(x-1,y)].colour == Black) && (x == cells.rows - 1 || cells[(x+1,y)].colour == Black) {
                        CellCategory::PointL
                    } else if x > 0 && cells[(x-1,y)].colour == Black {
                        CellCategory::InternalCornerTL
                    } else if cells.columns == 1 || cells[(x,y+1)].colour == Black {
                        CellCategory::StemTB
                    } else {
                        CellCategory::EdgeL
                    }
                } else if y == cells.columns - 1 {
//println!("{x}/{y}: {:?} {:?}", cells[(x-1,y)].colour, cells[(x+1,y)].colour);
                    if (x == 0 || cells[(x-1,y)].colour == Black) && (x == cells.rows - 1 || cells[(x+1,y)].colour == Black) {
                        CellCategory::PointR
                    } else if x > 0 && cells[(x-1,y)].colour == Black {
                        CellCategory::InternalCornerTR
                    } else if cells.columns == 1 || cells[(x,y-1)].colour == Black {
                        CellCategory::StemTB
                    } else {
                        CellCategory::EdgeR
                    }
                } else if cells[(x-1,y)].colour == Black && cells[(x+1,y)].colour == Black {
                    CellCategory::StemLR
                } else if cells[(x,y-1)].colour == Black && cells[(x,y+1)].colour == Black {
                    CellCategory::StemTB
                } else {
                    CellCategory::Middle
                };

            c.cat = cat;
        }

        m
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shapes {
    pub nx: usize,
    pub ny: usize,
    pub colour: Colour,
    pub shapes: Vec<Shape>,
    pub coloured_shapes: Vec<Shape>,
    pub cats: BTreeSet<ShapeCategory>,
    pub coloured_cats: BTreeSet<ShapeCategory>,
}

/*
impl Default for Shapes {
    fn default() -> Self {
         Self::new()
    }
}
*/

impl Shapes {
    pub fn new() -> Self {
        Self { nx: 0, ny: 0, colour: NoColour, shapes: Vec::new(), coloured_shapes: Vec::new(), cats: BTreeSet::new(), coloured_cats: BTreeSet::new() }
    }

    pub fn new_sized(nx: usize, ny: usize) -> Self {
        Self { nx, ny, colour: NoColour, shapes: Vec::new(), coloured_shapes: Vec::new(), cats: BTreeSet::new(), coloured_cats: BTreeSet::new() }
    }

    pub fn new_given(nx: usize, ny: usize, shapes: &Vec<Shape>) -> Self {
        Self { nx, ny, colour: Self::find_colour(shapes), shapes: shapes.to_vec(), coloured_shapes: Vec::new(), cats: BTreeSet::new(), coloured_cats: BTreeSet::new() }
    }

    pub fn new_from_shape(shape: &Shape) -> Self {
        Shapes::new_shapes(&[shape.clone()])
    }

    // May be same size as source grid
    pub fn new_shapes(shapes: &[Shape]) -> Self {
        let mut new_shapes = Self::new();
        let mut colour = NoColour;

        for s in shapes.iter() {
            new_shapes.add(s);
            if colour == NoColour {
                colour = s.colour;
            } else if colour != s.colour {
                colour = Mixed;
            }
        }
        new_shapes.colour = colour;

        new_shapes
    }

    pub fn new_shapes_sized(nx: usize, ny: usize, shapes: &[Shape]) -> Self {
        let mut new_shapes = Self::new_sized(nx, ny);
        let mut colour = NoColour;

        for s in shapes.iter() {
            new_shapes.add(s);
            if colour == NoColour {
                colour = s.colour;
            } else if colour != s.colour {
                colour = Mixed;
            }
        }
        new_shapes.colour = colour;

        new_shapes
    }

    pub fn merge_mut(&mut self, other: &Self) {
        for s in &other.shapes {
            self.shapes.push(s.clone());
        }
    }

    pub fn joined_by(&self) -> Option<Vec<(Shape, Shape, Shape)>> {
        /*
        for s in self.shapes.iter() {
        }
        */
        // TODO complete

        None
    }

    pub fn is_line(&self) -> bool {
        if self.shapes.is_empty() {
            return false;
        }
        for s in self.shapes.iter() {
            if !s.is_line() {
                return false;
            }
        }

        true
    }

    pub fn is_square(&self) -> bool {
        if self.shapes.is_empty() {
            return false;
        }
        for s in self.shapes.iter() {
            if !s.is_square() {
                return false;
            }
        }

        true
    }

    pub fn is_square_same(&self) -> bool {
        if self.shapes.is_empty() || !self.shapes[0].is_square() || self.shapes[0].size() == 1 {
            return false;
        }
        let size = self.shapes[0].size();

        for s in self.shapes.iter() {
            if !s.is_square() || s.size() != size {
                return false;
            }
        }

        true
    }

    pub fn anomalous_colour(&self) -> Option<Colour> {
        let h = self.colour_cnt();

        if h.len() <= 1 {
            return None;
        }

        if h.iter().filter(|(_,&v)| v == 1).count() == 1 {
            for (k, v) in &h {
                if *v == 1 {
                    return Some(*k);
                }
            }
        }

        None
    }

    pub fn embellish(&self, colour: Colour) ->  Shapes {
        let mut shapes = self.clone();

        for s in self.shapes.iter() {
            if s.is_pixel() {
                let ns = if s.orow == 0 && s.ocol == 0 {
                    let mut ns = Shape::new_sized_coloured_position(s.orow, s.ocol, 2, 2, Black);
                    ns.cells[(1,1)].colour = colour;

                    ns
                } else if s.orow == 0 {
                    let mut ns = Shape::new_sized_coloured_position(s.orow, s.ocol - 1, 2, 3, Black);
                    ns.cells[(1,0)].colour = colour;
                    ns.cells[(1,2)].colour = colour;

                    ns
                } else if s.ocol == 0 {
                    let mut ns = Shape::new_sized_coloured_position(s.orow - 1, s.ocol, 3, 2, Black);
                    ns.cells[(0,1)].colour = colour;
                    ns.cells[(2,1)].colour = colour;

                    ns
                } else {
                    let mut ns = Shape::new_sized_coloured_position(s.orow - 1, s.ocol - 1, 3, 3, Black);
                    ns.cells[(0,0)].colour = colour;
                    ns.cells[(0,2)].colour = colour;
                    ns.cells[(2,0)].colour = colour;
                    ns.cells[(2,2)].colour = colour;

                    ns
                };

                shapes.shapes.push(ns);
//            } else {
//                shapes.shapes.push(s.clone());
            }
        }

        shapes
    }

    pub fn colours(&self) -> Vec<Colour> {
        self.shapes.iter().map(|s| s.colour).collect()
    }

    // TODO: share code
    pub fn anomalous_size(&self) -> Option<usize> {
        let h = self.size_cnt();

        if h.len() <= 1 {
            return None;
        }

        if h.iter().filter(|(_,&v)| v == 1).count() == 1 {
            for (k, v) in &h {
                if *v == 1 {
                    return Some(*k);
                }
            }
        }

        None
    }

    pub fn colour_groups_to_shapes(&self, bg: Colour) -> Shapes {
        let mut h: BTreeMap<Colour, Vec<Shape>> = BTreeMap::new();

        for shape in self.shapes.iter() {
            if shape.is_pixel() {
                h.entry(shape.colour).or_insert(Vec::new()).push(shape.clone());
            }
        }

        let mut max_r = 0;
        let mut max_c = 0;

        for (_, v) in h.iter() {
            let mut smin_r = usize::MAX;
            let mut smax_r = 0;
            let mut smin_c = usize::MAX;
            let mut smax_c = 0;

            for s in v.iter() {
                smin_r = smin_r.min(s.orow);
                smax_r = smax_r.max(s.orow + s.cells.rows - 1);
                smin_c = smin_c.min(s.ocol);
                smax_c = smax_c.max(s.ocol + s.cells.columns - 1);
            }
            max_r = max_r.max(smax_r - smin_r);
            max_c = max_c.max(smax_c - smin_c);
        }

        let mut ss = Self::new_sized(max_r + 1, max_c + 1);
//println!("{} {}", max_r + 1, max_c + 1);

        ss.shapes.push(Shape::new_sized_coloured(max_r + 1, max_c + 1, bg));

        for (_, v) in h.iter() {
//println!("--- {k:?}");
            let mut orow = usize::MAX;
            let mut ocol = usize::MAX;
            let mut smax_r = 0;
            let mut smax_c = 0;

            for s in v.iter() {
                orow = orow.min(s.orow);
                ocol = ocol.min(s.ocol);
                smax_r = smax_r.max(s.orow);
                smax_c = smax_c.max(s.ocol);
            }

            for s in v.iter() {
                let r = (max_r - (smax_r - orow)) / 2;
                let c = (max_c - (smax_c - ocol)) / 2;
                let s = s.to_position(s.orow - orow + r, s.ocol - ocol + c);

                ss.shapes.push(s)
            }
        }

        ss
    }

    pub fn colour_cnt(&self) -> BTreeMap<Colour, usize> {
        let mut h: BTreeMap<Colour, usize> = BTreeMap::new();

        for s in self.shapes.iter() {
            *h.entry(s.colour).or_insert(0) += 1;
        }

        h
    }

    pub fn size_cnt(&self) -> BTreeMap<usize, usize> {
        let mut h: BTreeMap<usize, usize> = BTreeMap::new();

        for s in self.shapes.iter() {
            *h.entry(s.size()).or_insert(0) += 1;
        }

        h
    }

    pub fn corners(&self) -> (usize, usize, usize, usize) {
        let mut min_x = usize::MAX;
        let mut min_y = usize::MAX;
        let mut max_x = 0;
        let mut max_y = 0;

        for s in self.shapes.iter() {
            if min_x > s.orow {
                min_x = s.orow;
            }
            if min_y > s.ocol {
                min_y = s.ocol;
            }
            if max_x < s.orow + s.cells.rows {
                max_x = s.orow + s.cells.rows;
            }
            if max_y < s.ocol + s.cells.columns {
                max_y = s.ocol + s.cells.columns;
            }
        }

        if min_x == usize::MAX {
            min_x = 0;
        }
        if min_y == usize::MAX {
            min_y = 0;
        }

        (min_x, min_y, max_x, max_y)
    }

    pub fn to_shape(&self) -> Shape {
        let (min_x, min_y, max_x, max_y) = self.corners();

        let mut shape = Shape::new_sized_coloured_position(min_x, min_y, max_x - min_x, max_y - min_y, Black);

        shape.colour = Mixed;
        shape.orow = min_x;
        shape.ocol = min_y;

        for s in self.shapes.iter() {
            for c in s.cells.values() {
                //shape.cells[(x - min_x, y - min_y)] = c.clone();
                shape.cells[(c.row - min_x, c.col - min_y)].row = c.row;
                shape.cells[(c.row - min_x, c.col - min_y)].col = c.col;
                shape.cells[(c.row - min_x, c.col - min_y)].colour = c.colour;
            }
        }

        shape
    }

    pub fn border_only(&self) -> Shape {
        let mut shape = Shape::trivial();

        for s in self.shapes.iter() {
            if s.has_border() {
                shape = s.clone();

                break;
            }
        }

        shape
    }

    pub fn smallest(&self) -> Shape {
        let mut shape = Shape::trivial();

        for s in self.shapes.iter() {
            if shape.size() == 0 || s.size() <= shape.size() {
                shape = s.clone();
            }
        }

        shape
    }

    pub fn largest(&self) -> Shape {
        let mut shape = Shape::trivial();

        for s in self.shapes.iter() {
            if s.size() >= shape.size() {
                shape = s.clone();
            }
        }

        shape
    }

    pub fn hollow_cnt_colour_map(&self) -> BTreeMap<usize, Colour> {
        let mut h: BTreeMap<usize, Colour> = BTreeMap::new();

        for s in &self.shapes {
            let (colour, n) = s.hollow_colour_count();

            h.insert(n, colour);
        }

        h
    }

    pub fn hollow_cnt_max(&self) -> Shape {
        let mut shape = Shape::trivial();
        let mut n = 0;

        for s in &self.shapes {
            let (_, cnt) = s.hollow_colour_count();

            if cnt > n {
                n = cnt;
                shape = s.clone();
            }
        }

        shape
    }

    pub fn hollow_cnt_min(&self) -> Shape {
        let mut shape = Shape::trivial();
        let mut n = usize::MAX;

        for s in &self.shapes {
            let (_, cnt) = s.hollow_colour_count();

            if cnt < n {
                n = cnt;
                shape = s.clone();
            }
        }

        shape
    }

    pub fn hollow_cnt_map(&self) -> BTreeMap<usize, Vec<Shape>> {
        let mut h: BTreeMap<usize, Vec<Shape>> = BTreeMap::new();

        for s in self.shapes.iter() {
            let (_, cnt) = s.hollow_colour_count();

            h.entry(cnt).or_insert(Vec::new()).push(s.clone());
        }

        h
    }

    pub fn hollow_cnt_unique(&self) -> Shape {
        let mut shape = Shape::trivial();
        let h = self.hollow_cnt_map();

        for (_, sv) in &h {
            if sv.len() == 1 {
                shape = sv[0].clone();
                break;
            }
        }

        shape
    }

    pub fn first(&self) -> Shape {
        if self.shapes.is_empty() {
            return Shape::trivial();
        }

        self.shapes[0].clone()
    }

    pub fn last(&self) -> Shape {
        if self.shapes.is_empty() {
            return Shape::trivial();
        }

        self.shapes[self.shapes.len() - 1].clone()
    }

    pub fn shape_colour_cnt_map(&self) -> BTreeMap<Colour, Vec<Shape>>  {
        let mut h: BTreeMap<Colour, Vec<Shape>> = BTreeMap::new();

        for s in self.shapes.iter() {
            h.entry(s.colour).or_insert(Vec::new()).push(s.clone());
        }

        h
    }

    /*
    pub fn shape_colour_cnt_min(&self) -> Colour {
        let scm = self.shape_colour_cnt_map();

        let m = scm.iter().min_by_key(|(_, v)| v.len());
println!("{m:?}");

        NoColour
    }
    */

    pub fn pixels_in_shapes(&self) -> bool {
        let pixels: Vec<&Shape> = self.shapes.iter()
            .filter(|s| s.is_pixel())
            .collect();
            
        if pixels.is_empty() {
            return false;
        }

        for pix in pixels.iter() {
            for s in self.shapes.iter() {
                if !s.is_pixel() && pix.contained_by(&s) {
                    return true;
                }
            }
        }

        false
    }

    fn overlay_shapes(&self, same: bool) -> bool {
        for so in self.shapes.iter() {
            if so.size() <= 4 {
                continue;
            }
            for si in self.shapes.iter() {
                if si.size() >= so.size() {
                    continue;
                }
                if so.can_contain(si) && si.cells.rows < so.cells.rows && si.cells.columns < so.cells.columns && (same && si.colour == so.colour || !same && si.colour != so.colour) {
                    return true;
                }
            }
        }

        false
    }

    pub fn overlay_shapes_same_colour(&self) -> bool {
        self.overlay_shapes(true)
    }

    pub fn overlay_shapes_diff_colour(&self) -> bool {
        self.overlay_shapes(false)
    }

    pub fn overlay_shapes_coloured(&self) -> bool {
        for so in self.coloured_shapes.iter() {
            if so.size() <= 4 {
                continue;
            }
            for si in self.coloured_shapes.iter() {
                if si.size() >= so.size() {
                    continue;
                }
                if so.can_contain(si) && si.cells.rows < so.cells.rows && si.cells.columns < so.cells.columns {
                    return true;
                }
            }
        }

        false
    }

    // Alternative to patch_shapes
    pub fn consolidate_shapes(&self) -> Self {
        let mut shapes = self.clone();
        let mut removals: Vec<Shape> = Vec::new();

        for so in shapes.shapes.iter_mut() {
            if so.size() <= 4 {
                continue;
            }
            for si in self.shapes.iter() {
                if so.can_contain(si) && si.cells.rows < so.cells.rows && si.cells.columns < so.cells.columns && si.colour == so.colour {
                    for (xy, c) in si.cells.items() {
                        let nx = si.cells[xy].row;
                        let ny = si.cells[xy].col;

                        so.cells[(nx - so.orow, ny - so.ocol)].colour = c.colour;
                    }
                    removals.push(si.clone());
                }
            }
        }

        // Now get rid of the small fry
        for s in removals.iter() {
            shapes.remove(s);
        }
//shapes.show();

        shapes
    }

    pub fn consolidate_shapes_coloured(&self) -> Self {
        let mut shapes = self.clone();
        let mut removals: Vec<Shape> = Vec::new();

        for so in shapes.coloured_shapes.iter_mut() {
            if so.size() <= 4 {
                continue;
            }
            for si in self.coloured_shapes.iter() {
                if so.can_contain(si) && si.cells.rows < so.cells.rows && si.cells.columns < so.cells.columns {
                    for (xy, c) in si.cells.items() {
                        let nx = si.cells[xy].row;
                        let ny = si.cells[xy].col;

                        so.cells[(nx - so.orow, ny - so.ocol)].colour = c.colour;
                    }
                    removals.push(si.clone());
                }
            }
        }

        // Now get rid of the small fry
        for s in removals.iter() {
            shapes.remove(s);
        }
//shapes.show();

        shapes
    }

    pub fn find_pixels(&self) -> Self {
        let mut pixels = Self::new_sized(self.nx, self.ny);
        let mut colour = NoColour;

        for s in self.shapes.iter() {
            if s.is_pixel() {
                pixels.shapes.push(s.clone());

                if colour == NoColour {
                    colour = s.colour;
                } else if colour != s.colour {
                    colour = Mixed;
                }
            }
        }

        pixels.colour = colour;

        pixels
    }

    pub fn find_shapes(&self) -> Self {
        let mut shapes = Self::new_sized(self.nx, self.ny);
        let mut colour = NoColour;

        for s in self.shapes.iter() {
            if !s.is_pixel() {
                shapes.shapes.push(s.clone());

                if colour == NoColour {
                    colour = s.colour;
                } else if colour != s.colour {
                    colour = Mixed;
                }
            }
        }

        shapes.colour = colour;

        shapes
    }

    pub fn size(&self) -> usize {
        self.nx * self.ny
    }

    pub fn width(&self) -> usize {
        self.ny
    }

    pub fn height(&self) -> usize {
        self.nx
    }

    fn find_colour(shapes: &Vec<Shape>) -> Colour {
        let mut colour = NoColour;

        for s in shapes {
            if s.colour != Black {
                if colour == NoColour {
                    colour = s.colour;
                } else if colour != s.colour {
                    colour = Mixed;
                    break;
                }
            }
        }

        colour
    }

    fn find_min_max(&self, min: bool) -> Shape {
        let mut ho: BTreeMap<Shape, usize> = BTreeMap::new();
        let mut hs: BTreeMap<Shape, Shape> = BTreeMap::new();

        for s in &self.shapes {
            let os = s.to_origin();
            *ho.entry(os.clone()).or_insert(0) += 1;
            hs.insert(s.clone(), os);
        }

        let mut shape = Shape::trivial();
        let mut n = if min { usize::MAX } else { 0 };

        for (s, os) in hs {
            let c = if let Some(c) = ho.get(&os) {
                c
            } else {
                &if min { usize::MAX } else { 0 }
            };
            // Poor assumption that last is best?
            if (min && *c <= n) || (!min && *c >= n) {
                n = *c;
                shape = s;
            }
        }

        shape
    }

    pub fn find_min(&self) -> Shape {
        self.find_min_max(true)
    }

    pub fn find_max(&self) -> Shape {
        self.find_min_max(false)
    }

    fn find_sub_min_max(&self, min: bool) -> Shape {
        let mut h: BTreeMap<Shape, usize> = BTreeMap::new();

        for s in &self.shapes {
            let ss = s.to_grid().to_shapes_sq();

            *h.entry(s.clone()).or_insert(0) += ss.shapes.len();
        }

        let mut n = if min { usize::MAX } else { 0 };
        let mut shape = Shape::trivial();

        for (k, v) in h.iter() {
            if (min && *v <= n) || (!min && *v >= n) {
                n = *v;
                shape = k.clone();
            }
        }

        shape
    }

    pub fn find_sub_min(&self) -> Shape {
        self.find_sub_min_max(true)
    }

    pub fn find_sub_max(&self) -> Shape {
        self.find_sub_min_max(false)
    }

    pub fn find_pixels_min(&self) -> Shape {
        let mut shape = Shape::trivial();
        let mut pixels = usize::MAX;

        for s in self.shapes.iter() {
            if s.pixels() < pixels {
                pixels = s.pixels();
                shape = s.clone();
            }
        }

        shape
    }

    pub fn find_pixels_max(&self) -> Shape {
        let mut shape = Shape::trivial();
        let mut pixels = 0;

        for s in self.shapes.iter() {
            if s.pixels() > pixels {
                pixels = s.pixels();
                shape = s.clone();
            }
        }

        shape
    }

    pub fn find_max_colour_count(&self) -> Shape {
        let mut choice = &Shape::trivial();
        let mut n = 0;
        
        for s in &self.shapes {
            let cnt = s.distinct_colour_cnt();

            if cnt > n {
                choice = s;
                n = cnt;
            }
        }

        choice.clone()
    }

    pub fn find_sub_largest_count(&self) -> Shape {
        let mut choice = &Shape::trivial();
        let mut biggest = 0;
        
        for s in &self.shapes {
            if s.size() > 1 {
                let ss = s.to_grid().to_shapes_sq();
                let mut sh: BTreeMap<Shape, usize> = BTreeMap::new();

                for i in &ss.shapes  {
                    if i.size() > 1 && i.size() != s.size() {
                        let ni = i.to_origin();
                        *sh.entry(ni).or_insert(0) += 1;
                    }
                }

                if let Some(mx) = sh.values().max() {
                    if *mx > biggest {
                        biggest = *mx;
                        choice = s;
                    }
                }
            }
        }

        choice.clone()
    }

    pub fn position_pixels(&self) -> Option<(Self, Self)> {
        let mut xp = usize::MAX;
        let mut yp = usize::MAX;
        let mut cp = NoColour;
        let mut xgap = 0;
        let mut ygap = 0;
        let mut pos: Vec<Shape> = Vec::new();
        let mut shapes: Vec<Shape> = Vec::new();

        for s in &self.shapes {
            if s.size() == 1 {
                if xp == usize::MAX {
                    xp = s.orow;
                    yp = s.ocol;
                    cp = s.colour;

                    let cell = Cell::new_colour(s.orow, s.ocol, cp);
                    let cells = Matrix::new(1, 1, cell);
                    pos.push(Shape::new(s.orow, s.ocol, &cells));
                } else if s.colour == cp {
                    if yp == s.ocol && s.orow > xp {
                        if xgap == 0 {
                            xgap = s.orow - xp;
                        } else if (s.orow - xp) % xgap != 0 {
                            return None;
                        }
                    }
                    if xp == s.orow && s.ocol > yp {
                        if ygap == 0 {
                            ygap = s.ocol - yp;
                        } else if (s.ocol - yp) % ygap != 0 {
                            return None;
                        }
                    }

                    // needs to be a square, so equal gaps
                    if xgap > 0 && xgap != ygap {
                        return None;
                    }

                    let cell = Cell::new_colour(s.orow, s.ocol, cp);
                    let cells = Matrix::new(1, 1, cell);
                    pos.push(Shape::new(s.orow, s.ocol, &cells));

                    // Add extra to right?
                    /*
                    if s.oy + ygap > s.cells.columns {
                        let cell = Cell::new_colour(s.ox, s.oy + ygap, cp);
                        let cells = Matrix::new(1, 1, cell);
                        pos.push(Shape::new(s.ox, s.oy + ygap, &cells));
                    }
                    */
                }
            } else {
                shapes.push(s.clone());
            }
        }

        Some((Shapes::new_shapes_sized(self.nx, self.ny, &pos),
             Shapes::new_shapes_sized(self.nx, self.ny, &shapes)))
    }

    pub fn position_centres(&self, positions: &Self) -> Self {
        if positions.shapes.is_empty() || self.shapes[0].cells.rows <= 1 || (self.shapes[0].cells.rows > 1 && self.shapes[0].cells.rows != self.shapes[0].cells.columns) {
            return Self::new();
        }
        let gap = positions.shapes[1].ocol as isize - positions.shapes[0].ocol as isize;
        if gap <= self.shapes[0].cells.rows as isize {
            return Self::new();
        }
        let offset = if gap % 2 == 0 {
            gap as usize - self.shapes[0].cells.rows
        } else {
            gap as usize - self.shapes[0].cells.rows - 1
        };
        let mut nps = self.clone();

        for s in nps.shapes.iter_mut() {
            let p = s.nearest(positions);
            let xoffset = if p.orow >= s.orow { offset } else { offset - 1 };
            let yoffset = if p.ocol >= s.ocol { offset } else { offset - 1 };

            *s = s.translate_absolute(p.orow + xoffset, p.ocol + yoffset);
        }
        for s in positions.shapes.iter() {
            nps.shapes.push(s.clone());
        }

//nps.to_grid().show();
        nps
    }

    /*
    pub fn categorise {
        self.categorise_shapes();
        categorise_io_edges(in_shapes: &mut Shapes, out_shapes: &Shapes) {
        self.categorise_shape_edges();
    }
    */

    pub fn contained_pairs(&self) -> BTreeMap<Shape, Shape> {
        let mut pairs: BTreeMap<Shape, Shape> = BTreeMap::new();

        for s1 in self.shapes.iter() {
            for s2 in self.shapes.iter() {
                if s1.contained_by(&s2) {
                    pairs.insert(s2.clone(), s1.clone());

                    break;
                }
            }
        }

        pairs
    }

    pub fn pair_shapes(&self, other: &Shapes, match_colour: bool) -> Vec<(Shape, Shape, bool)> {
        let mut pairs: Vec<(Shape, Shape, bool)> = Vec::new();

        if self.shapes.len() != other.shapes.len() {
            return pairs;
        }

//println!(">>> {:?} {:?}", self.cats, other.cats);
        let mut si = self.clone();
        if match_colour {
            si.shapes.sort_by(|a, b| (a.colour, a.orow, a.ocol, &a.to_json()).cmp(&(b.colour, b.orow, b.ocol, &b.to_json())));
        } else {
            si.shapes.sort();
        }
        let mut so = other.clone();
        if match_colour {
            //so.shapes.sort_by(|a, b| a.colour.cmp(&b.colour));
            so.shapes.sort_by(|a, b| (a.colour, a.orow, a.ocol, &a.to_json()).cmp(&(b.colour, b.orow, b.ocol, &b.to_json())));
        } else {
            so.shapes.sort();
        }
//so.categorise_shapes();
//println!("<<< {:?}", so.cats);

        //for (si, so) in self.shapes.iter().zip(other.shapes.iter()) {
        for (si, so) in si.shapes.iter().zip(so.shapes.iter()) {
            let si_sid = Shape::sid(&si.cells, match_colour);
            let so_sid = Shape::sid(&so.cells, match_colour);
//println!(">>> {:?} {:?}", si.cats, so.cats);

            pairs.push((si.clone(), so.clone(), si_sid == so_sid));
        }

        pairs
    }

    pub fn toddle_colours(&self) -> Self {
        let mut shapes = self.clone();

        for s in shapes.shapes.iter_mut() {
            let mut h = s.cell_colour_cnt_map();

            if h.len() != 2 {
                return Self::new();
            }

            // Should never panic as preconditions satisfied
            let Some((l, _)) = h.pop_first() else { todo!() };
            let Some((r, _)) = h.pop_first() else { todo!() };
//println!("{l:?}, {r:?}");

            for ((x, y), c) in s.clone().cells.items() {
                if c.colour == l {
                    s.cells[(x,y)].colour = r;
                } else {
                    s.cells[(x,y)].colour = l;
                }
            }
        }

        shapes
    }

    pub fn ignore_pixels(&self) -> Self {
        let mut ss = self.clone();

        ss.shapes = Vec::new();

        for s in &self.shapes {
            if s.size() > 1 {
                ss.shapes.push(s.clone());
            }
        }

        ss
    }

    pub fn de_subshape(&self) -> Shapes {
        let mut ss = self.clone();

        ss.shapes = Vec::new();

        for s in &self.shapes {
            s.de_subshape(&mut ss);
        }

        ss
    }

    pub fn diff(&self, other: &Self) -> Option<Vec<Option<Shape>>> {
        if self.nx != other.nx || self.ny != other.ny || self.shapes.len() != other.shapes.len() {
            return None;
        }

        let mut diffs: Vec<Option<Shape>> = Vec::new();

        for (s1, s2) in self.shapes.iter().zip(other.shapes.iter()) {
            let gdiff = s1.diff(s2);
            if let Some(diff) = gdiff {
//diff.show_full();
                diffs.push(Some(diff));
            } else {
                diffs.push(None);
            }
        }

        Some(diffs)
    }

    pub fn len(&self) -> usize {
        self.shapes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.shapes.is_empty()
    }

    // resizing only works for no-overallping shapes
    pub fn add(&mut self, shape: &Shape) {
        if shape.orow + shape.cells.rows > self.nx {
            self.nx = shape.orow + shape.cells.rows;
        }
        if shape.ocol + shape.cells.columns > self.ny {
            self.ny = shape.ocol + shape.cells.columns;
        }
        /*
        */
        if self.colour == NoColour {
            self.colour = shape.colour;
        } else if self.colour != shape.colour && self.colour != Black {
            self.colour = Mixed;
        }

        self.shapes.push(shape.clone());

/*
        // Resize if necessary
        for s in self.shapes.iter_mut() {
            //let ss = s.to_origin();
            for c in s.cells.values() {
                /*
                if c.x - s.ox >= self.nx {
                    self.nx = c.x - s.ox + s.cells.rows;
                }
                if c.y - s.oy >= self.ny {
                    self.ny = c.y - s.oy + s.cells.columns;
                }
                */
                if c.x >= self.nx {
                    self.nx = c.x;
                }
                if c.y >= self.ny {
                    self.ny = c.y;
                }
            }
        }
println!("{}/{} {}/{}", shape.ox + shape.cells.rows, shape.oy + shape.cells.columns, self.nx, self.ny);
*/
    }

    // Must be supplied with single colour shapes
    pub fn noise_colour(&self) -> Colour {
        let mut size = self.shapes.len();

        if size < 3 || self.colour != Mixed {
            return Black;
        }

        let mut h: BTreeMap<usize, usize> = BTreeMap::new();

        for c in &self.shapes {
//c.show();
            //if c.colour != Mixed {
            //    return Black;
            //} else {
                *h.entry(Colour::to_usize(c.colour)).or_insert(0) += 1;
            //}
        }

        //if h.len() < 8 {
            //return Black;
        //}

        size -= h.len();

        for (c, cnt) in h {
            if cnt > size {
                return Colour::from_usize(c);
            }
        }

        Black
    }

    // Must be supplied with single colour shapes
    pub fn important_shapes(&self) -> Vec<Shape> {
        let shapes: Vec<Shape> = Vec::new();
        let size = self.shapes.len();

        if size < 3 || self.colour != Mixed {
            return shapes;
        }

        let mut h: BTreeMap<Shape, usize> = BTreeMap::new();

        for s in &self.shapes {
            *h.entry(s.clone()).or_insert(0) += 1;
        }

        h.iter().filter(|(_, &cnt)| cnt == 1).map(|(s, _)| s.clone()).collect()
    }

    pub fn remove(&mut self, shape: &Shape) {
        let index = self.shapes.iter().position(|x| *x == *shape);

        if let Some(index) = index {
            self.shapes.remove(index);
        }
    }

    pub fn hollow_shapes(&self) -> Shapes {
        let mut new_shapes = Self::new_sized(self.nx, self.ny);

        for s in self.shapes.iter() {
            if s.size() != self.size() && s.hollow() {
                let ss = s.recolour(Blue, Teal);

                new_shapes.add(&ss);
            }
        }

        new_shapes
    }

    pub fn merge_replace_shapes(&self, other: &Self) -> Self {
        let mut new_shapes = Self::new_sized(self.nx, self.ny);

        for s in self.shapes.iter() {
            if !other.shapes.contains(s) {
                new_shapes.add(s);
            }
        }
        for s in other.shapes.iter() {
            new_shapes.add(s);
        }

        new_shapes
    }

    pub fn show_summary(&self) {
        for s in &self.shapes {
            s.show_summary();
            println!();
        }
        println!("--------");
    }

    pub fn show(&self) {
        for s in &self.shapes {
            s.show();
            println!();
        }
        println!("--------");
    }

    pub fn show_full(&self) {
        for s in &self.shapes {
            s.show_full();
            println!();
        }
        println!("--------");
    }

    /*
    pub fn add_in(&mut self) -> Self {
        let mut holes = Self::new();

        for s in &self.shapes {
            if s.has_hole() {
                holes.shapes.push(s.clone());
            }
        }

        for h in &mut holes.shapes {
            for s in &mut self.shapes {
                if h.can_contain(s) {
                    h.put_all_in(s);
                }
            }
        }

        holes
    }
    */

    pub fn trim_grid(&self) -> Self {
        let mut trimmed = self.clone();

        trimmed.shapes = Vec::new();

        for s in self.shapes.iter() {
            if s.orow + s.cells.rows > self.nx || s.ocol + s.cells.columns > self.ny {
                let x = self.nx.min(s.orow + s.cells.rows);
                let y = self.ny.min(s.ocol + s.cells.columns);

                if let Ok(mat) = s.cells.slice(0 .. x - s.orow, 0 .. y - s.ocol) {
                    trimmed.shapes.push(Shape::new_cells(&mat));
                } 
//            } else {
//                trimmed.shapes.push(s.clone());
            }
        }

        trimmed
    }
        
    pub fn trim_to_grid(&self) -> Grid {
        let trimmed = self.trim_grid();

        trimmed.to_grid_impl(Black, false)
    }

    pub fn trim_to_grid_transparent(&self) -> Grid {
        let trimmed = self.trim_grid();

        trimmed.to_grid_impl(Black, true)
    }

    pub fn to_grid(&self) -> Grid {
        self.to_grid_impl(Black, false)
    }

    pub fn to_grid_transparent(&self) -> Grid {
        self.to_grid_impl(Black, true)
    }

    pub fn to_grid_colour(&self, colour: Colour) -> Grid {
        self.to_grid_impl(colour, false)
    }

    pub fn to_grid_colour_transparent(&self, colour: Colour) -> Grid {
        self.to_grid_impl(colour, true)
    }

    pub fn to_grid_impl(&self, colour: Colour, transparent: bool) -> Grid {
        if self.nx == 0 || self.ny == 0 {
            return Grid::trivial();
        }
        let mut grid = Grid::new(self.nx, self.ny, colour);

        if self.nx > 1000 || self.ny > 1000 {
            return grid;
        }

        grid.colour = self.colour;

        for shape in &self.shapes {
            for c in shape.cells.values() {
                if  grid.cells.rows <= c.row || grid.cells.columns <= c.col {
                    break;
                }
                if c.colour == Transparent {
                    continue;
                }
                if !transparent || c.colour != Black {
                    grid.cells[(c.row, c.col)].colour = c.colour;
                }
            }
        }

        grid
    }

    pub fn to_json(&self) -> String {
        let mut grid: Vec<Vec<usize>> = vec![vec![0; self.nx]; self.ny];

        for shape in &self.shapes {
            for ((x, y), c) in shape.cells.items() {
                grid[x][y] = c.colour.to_usize();
            }
        }

        serde_json::to_string(&grid).unwrap()
    }

    pub fn to_json_coloured(&self) -> String {
        let mut grid: Vec<Vec<usize>> = vec![vec![0; self.nx]; self.ny];

        for shape in &self.coloured_shapes {
            for ((x, y), c) in shape.cells.items() {
                grid[x][y] = c.colour.to_usize();
            }
        }

        serde_json::to_string(&grid).unwrap()
    }

    /*
    pub fn cells(&self) -> Vec<Cell> {
        let mut cells: Matrix<Cell> = Matrix::from_fn(self.nx, self.ny, |(_, _)| Cell::new_empty());

        for ((x, y), c) in &self.shapes.items() {
            for c in &s.cells {
                cells.push(c.clone());
            }
        }

        cells
    }
    */

    // TODO
    pub fn shape_counts(&self) -> Vec<(Shape, usize)> {
        vec![]
    }

    pub fn holes_sizes(&self) -> Vec<(Shape, usize)> {
        vec![]
    }

    pub fn have_common_pixel(&self) -> (Colour,Vec<Self>) {
        (NoColour, vec![])
    }

    /*
    pub fn pack_common_centre(&self) -> Shape {
        Shape::new_empty()
    }
    */

    // bool is true == X axis, false = Y axis
    pub fn striped_x(&self) -> Colour {
        NoColour
    }

    pub fn striped_y(&self) -> Colour {
        NoColour
    }

    /*
    pub fn stretch_x(&self, cells: Vec<Cell>) -> Self {
        self.clone()
    }

    pub fn stretch_y(&self, cells: Vec<Cell>) -> Self {
        self.clone()
    }
    */

    /*
    */
    pub fn shrink(&self) -> Self {
         let mut shapes: Vec<Shape> = Vec::new();

         for s in self.shapes.iter() {
             shapes.push(s.shrink());
         }

         Self::new_shapes(&shapes)
    }

    pub fn fill_missing(&self, to: Colour) -> Self {
//println!("{:?}", self);
        let mut shapes = Shapes::new_sized(self.nx, self.ny);

        for shape in self.shapes.iter() {
            shapes.add(&shape.fill_missing(to));
        }

        shapes
    }

    pub fn categorise_shapes(&mut self, same_colour: bool) {
        let the_shapes = if same_colour {
            &mut self.shapes
        } else {
            &mut self.coloured_shapes
        };

        if the_shapes.is_empty() {
            return;
        }
        if same_colour {
            if the_shapes.len() == 1 {
                self.cats.insert(ShapeCategory::SingleShape);
            }
            if the_shapes.len() > 1 {
                self.cats.insert(ShapeCategory::ManyShapes);
            }
        } else {
            if the_shapes.len() == 1 {
                self.coloured_cats.insert(ShapeCategory::SingleShape);
            }
            if the_shapes.len() > 1 {
                self.coloured_cats.insert(ShapeCategory::ManyShapes);
            }
        }

        for shape in the_shapes.iter_mut() {
            if same_colour {
                if self.cats.is_empty() {
                    self.cats = shape.cats.clone();
                } else {
                    self.cats = self.cats.union(&shape.cats).cloned().collect();
                }
            } else {
                if self.coloured_cats.is_empty() {
                    self.coloured_cats = shape.cats.clone();
                } else {
                    self.coloured_cats = self.cats.union(&shape.cats).cloned().collect();
                }
            }
        }
    }

    pub fn biggest_shape(&self) -> Shape {
        if self.shapes.is_empty() {
            return Shape::trivial();
        }

        let mut biggest_size = 0;
        let mut biggest_idx = 0;

        for (i, s) in self.shapes.iter().enumerate() {
            if s.size() > biggest_size {
                biggest_size = s.size();
                biggest_idx = i;
            }
        }

        self.shapes[biggest_idx].clone()
    }

    pub fn has_mirror_x(&self) -> Shape {

        for s in &self.shapes  {
            if s.is_mirror_x() {
                return s.clone();
            }
        }

        Shape::trivial()
    }

    pub fn has_mirror_y(&self) -> Shape {
        for s in &self.shapes  {
            if s.is_mirror_y() {
                return s.clone();
            }
        }

        Shape::trivial()
    }

    pub fn cell_colour_cnts(&self, colour: Colour) -> Shape {
        let mut count = 0;
        let mut shape = Shape::trivial();

        for s in &self.shapes {
            let h = s.cell_colour_cnt_map();
//println!("{h:?}");
            //let mut ordinal: Vec<(usize, Colour)> = h.iter().map(|(k, v)| (*v, *k)).collect();

            //ordinal.sort();
//println!("{:?}", h.get(&colour));

            //let (cnt, _colour) = ordinal[position];
            if let Some(cnt) = h.get(&colour) {
                if *cnt > count {
                    count = *cnt;
                    shape = s.clone();
                }
            }
        }

        shape
    }

    pub fn get_by_colour(&self, colour: Colour) -> Vec<Shape> {
        let mut shapes: Vec<Shape> = Vec::new();

        for s in &self.shapes {
            if s.colour == colour {
                shapes.push(s.clone());
            }
        }

        shapes
    }

    pub fn categorise_io_edges(in_shapes: &mut Shapes, out_shapes: &Shapes) { //-> BTreeSet<ShapeEdgeCategory> {
        let shapes1 = &mut in_shapes.shapes;
        let shapes2 = &out_shapes.shapes;
//println!("{} == {}", shapes1.len(), shapes2.len());
        //let mut edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
//if shapes1.len() != shapes2.len() {
    //shapes1.show();
    //shapes2.show();
//}

        'outer:
        for shape1 in shapes1.iter_mut() {
            for shape2 in shapes2.iter() {
                //if shape1.ox == shape2.oy { // ???
                if shape1.orow == shape2.orow && shape1.ocol == shape2.ocol { // ???
//shape1.show();
//shape2.show();
                    if shape1.size() == 1 && shape2.size() == 1 {
                        if shape1.colour == shape2.colour {
                            shape1.io_edges.insert(ShapeEdgeCategory::SameSingle);
                        } else {
                            shape1.io_edges.insert(ShapeEdgeCategory::SingleDiffColour);
                        }

                        continue 'outer;
                    }

                    let same_colour = shape1.colour != Mixed && shape1.colour == shape2.colour;
                    let same_shape = shape1.same_shape(shape2);
                    let same_size = shape1.same_size(shape2);
                    let same_pixels = shape1.pixels() == shape2.pixels();

                    if same_colour && same_shape && same_size && same_pixels {
                        shape1.io_edges.insert(ShapeEdgeCategory::Same);

                        continue 'outer;
                    } else {
                        if same_colour {
                            shape1.io_edges.insert(ShapeEdgeCategory::SameColour);
                        }
                        if same_shape {
                            shape1.io_edges.insert(ShapeEdgeCategory::SameShape);
                        }
                        if same_size {
                            shape1.io_edges.insert(ShapeEdgeCategory::SameSize);
                        }
                        if same_pixels {
                            shape1.io_edges.insert(ShapeEdgeCategory::SamePixelCount);
                        }
                    }

                    /*
                    let above = shape1.above(shape2);
                    let below = shape1.below(shape2);
                    let left = shape1.left(shape2);
                    let right = shape1.right(shape2);

                    if above && left {
                        edges.insert(ShapeEdgeCategory::AboveLeft);
                    } else if above && right {
                        edges.insert(ShapeEdgeCategory::AboveRight);
                    } else if below && left {
                        edges.insert(ShapeEdgeCategory::BelowLeft);
                    } else if below && right {
                        edges.insert(ShapeEdgeCategory::BelowRight);
                    } else if left {
                        edges.insert(ShapeEdgeCategory::Left);
                    } else if right {
                        edges.insert(ShapeEdgeCategory::Right);
                    } else if above {
                        edges.insert(ShapeEdgeCategory::Above);
                    } else if below {
                        edges.insert(ShapeEdgeCategory::Below);
                    }
                    */

                    if shape1.can_contain(shape2) {
                        shape1.io_edges.insert(ShapeEdgeCategory::CanContain);
                    }
                    if shape1.adjacent(shape2) {
                        shape1.io_edges.insert(ShapeEdgeCategory::Adjacent);
                    }
                    if shape1.have_common_pixel_colour(shape2) {
                        shape1.io_edges.insert(ShapeEdgeCategory::CommonPixel);
                    }
                    if false {
                        shape1.io_edges.insert(ShapeEdgeCategory::HasArm);
                    }
                    if false {
                        shape1.io_edges.insert(ShapeEdgeCategory::Gravity);
                    }

                    /*
                    if same {
                        in_shapes.edges.insert(shape2.clone(), edges);
                    } else {
                        in_shapes.io_edges.insert(shape2.clone(), edges);
                    }
                    */

                    let mirrored_x = shape1.is_mirrored_x(shape2);
                    let mirrored_y = shape1.is_mirrored_y(shape2);

                    if mirrored_x && mirrored_y {
                        shape1.io_edges.insert(ShapeEdgeCategory::Symmetric);
                    } else {
                        if mirrored_x {
                            shape1.io_edges.insert(ShapeEdgeCategory::MirroredX);
                        }
                        if mirrored_y {
                            shape1.io_edges.insert(ShapeEdgeCategory::MirroredY);
                        }
                        if shape1.is_rotated_90(shape2) {
                            shape1.io_edges.insert(ShapeEdgeCategory::Rot90);
                        }
                        if shape1.is_rotated_180(shape2) {
                            shape1.io_edges.insert(ShapeEdgeCategory::Rot180);
                        }
                        if shape1.is_rotated_270(shape2) {
                            shape1.io_edges.insert(ShapeEdgeCategory::Rot270);
                        }
                    }
                }
            }
//println!("---- {:?}", shape1.io_edges);
        }

        //edges
    }

    pub fn categorise_shape_edges(&mut self) {
        Self::categorise_io_edges(self, &self.clone());
    }
}

