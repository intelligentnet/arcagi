use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use array_tool::vec::Uniq;
use pathfinding::prelude::Matrix;
use crate::cats::Colour::*;
use crate::cats::Direction::*;
use crate::utils::*;
//use crate::cats::CellCategory::*;
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
    //pub cats: BTreeSet<ShapeCategory>,
    //pub io_edges: BTreeSet<ShapeEdgeCategory>,
}

/*
use std::hash::Hash;
use std::hash::Hasher;
use std::hash::DefaultHasher;
impl Hash for Shape {
    fn hash<H>(&self, state: &mut H)
    where
        H: Hasher,
    {
        /*
        fn calculate_hash(t: &str) -> u64 {
            let mut s = DefaultHasher::new();

            s.write(t.as_bytes());
            s.finish()
        }
        */

        //state.write_usize(self.ox);
        //state.write_usize(self.oy);
        //state.write_u64(calculate_hash(&self.to_json()));
        for c in self.cells.values() {
            state.write_usize(c.colour.to_usize());
        }
        let _ = state.finish();
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
    pub fn new(orow: usize, ocol: usize, cells: &Matrix<Cell>) -> Self {
        let (colour, _) = Self::cell_colour_cnt_mixed(cells, true, true);
        let mut new_cells = cells.clone();

        for (r, c) in cells.keys() {
            new_cells[(r, c)].row = r + orow;
            new_cells[(r, c)].col = c + ocol;
        }

        //let cats: BTreeSet<ShapeCategory> = BTreeSet::new();
        //let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        let new_cells = Self::cell_category(&new_cells);

        let res = Self { orow, ocol, colour, cells: new_cells };

        //res.categorise_shape();

        res
    }

    pub fn new_cells(cells: &Matrix<Cell>) -> Self {
        let (colour, _) = Self::cell_colour_cnt_mixed(cells, true, true);
        //let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        //let cats: BTreeSet<ShapeCategory> = BTreeSet::new();
        let cells = Self::cell_category(cells);

        if cells.rows == 0 || cells.columns == 0 {
            return Shape::trivial();
        }

        let orow = cells[(0,0)].row;
        let ocol = cells[(0,0)].col;
        let res = Self { orow, ocol, colour, cells };

        //res.categorise_shape();

        res
    }

    pub fn new_sized_coloured(rows: usize, cols: usize, colour: Colour) -> Self {
        let cells: Matrix<Cell> = Matrix::from_fn(rows, cols, |(rows, cols)| Cell::new_colour(rows, cols, colour));
        //let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        //let cats: BTreeSet<ShapeCategory> = BTreeSet::new();

        Self { orow: 0, ocol: 0, colour, cells }
    }

    pub fn new_sized_coloured_position(orow: usize, ocol: usize, row: usize, col: usize, colour: Colour) -> Self {
        let cells: Matrix<Cell> = Matrix::from_fn(row, col, |(r, c)| Cell::new_colour(r + orow, c + ocol, colour));

        //let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        //let cats: BTreeSet<ShapeCategory> = BTreeSet::new();

        Self { orow, ocol, colour, cells }
    }

    pub fn new_empty() -> Self {
        let cells = Matrix::new(0, 0, Cell::new_empty());
        //let io_edges: BTreeSet<ShapeEdgeCategory> = BTreeSet::new();
        //let cats: BTreeSet<ShapeCategory> = BTreeSet::new();

        Self { orow: 0, ocol: 0, colour: NoColour, cells }
    }

    pub fn trivial() -> Self {
        Self::new_empty()
    }

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

        for r in 0 .. s.cells.rows {
            for c in 0 .. s.cells.columns {
                s.cells[(r,c)].row = s.orow + r;
                s.cells[(r,c)].col = s.ocol + c;
            }
        }

        s
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
//self.show();
        if self.size() != other.size() || self.cells.rows != other.cells.rows {
            return false;
        }

        for (rc, c) in self.cells.items() {
//println!("{:?} {:?}", c.colour, other.cells[rc].colour);
            if c.colour != Black && c.colour != other.cells[rc].colour {
                return false;
            }
        }
//println!("----------------");

        true
    }

    pub fn fill(&self, other: &Self) -> Self {
        if self.size() != other.size() || !self.is_square() {
            return self.clone();
        }

        let mut shape = self.clone();

        for rc in self.cells.keys() {
            if self.cells[rc].colour == Black {
                shape.cells[rc].colour = other.cells[rc].colour;
            }
        }
        
        shape
    }

    pub fn make_symmetric(&self) -> Self {
        let shape = self.clone();
        let shape = shape.fill(&shape.rotated_90());
        let shape = shape.fill(&shape.rotated_180());

        shape.fill(&shape.rotated_270())
    }

    pub fn col_symmetric_mut(&self, shapes: &mut Shapes) {
        let cols = self.cells.columns;
        let even = cols %2 == 0;
        let half = cols / 2;
        let len = half + (if even { 0 } else { 1 });
//println!("{cols} {even} {half} {len}");
let this = self.clone();
let mut shapes2 = shapes.clone_base();
let mut cnt1 = 0;
let mut cnt2 = 0;

        let s1 = self.subshape(0, self.cells.rows, 0, len);
        let s2 = self.subshape(0, self.cells.rows, half, len).mirrored_c();

        if let Some(mut diff) = s1.diff(&s2) {
            for cell in diff.cells.values_mut() {
                cell.colour = if cell.colour.is_same() {
                    cell.colour.to_base()
                } else {
cnt1 += 1;
                    Black
                };
            }

            shapes.shapes.push(diff.to_position(self.orow, self.ocol));
            shapes.shapes.push(diff.mirrored_c().to_position(self.orow, self.ocol + half));
//shapes.to_grid().show();
        }

        let s1 = this.subshape(0, self.cells.rows, 1, len);
        //let s2 = this.subshape(0, self.cells.rows, half, len).mirrored_c();

        if let Some(mut diff) = s1.diff(&s2) {
            for cell in diff.cells.values_mut() {
                cell.colour = if cell.colour.is_same() {
                    cell.colour.to_base()
                } else {
cnt2 += 1;
                    Black
                };
            }

            shapes2.shapes.push(diff.to_position(self.orow, self.ocol + 1));
            shapes2.shapes.push(diff.mirrored_c().to_position(self.orow, self.ocol + half));
//shapes2.to_grid().show();
//println!("### {cnt1} {cnt2}");
        }

        if cnt2 <= cnt1 {
            shapes.shapes.pop();
            shapes.shapes.pop();

            for s in shapes2.shapes.iter() {
                shapes.shapes.push(s.clone());
            }
        }
    }

    // must be odd size
    pub fn new_square(r: usize, c: usize, size: usize, colour: Colour) -> Self {
        let mut square = Self::new_sized_coloured(size, size, Black);

        square.colour = colour;

        for (r, c) in square.cells.keys() {
            if r == 0 || c == 0 || r == square.cells.rows - 1 || c == square.cells.columns - 1 {
                square.cells[(r, c)].colour = colour;
            }
        }

        square.translate_absolute(r, c)
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
        self.to_shapes_base(false)
    }

    pub fn to_shapes_coloured(&self) -> Shapes {
        self.to_shapes_base(true)
    }

    pub fn to_shapes_base(&self, coloured: bool) -> Shapes {
        let mut inner_shapes = if coloured {
            self.to_grid().to_shapes_coloured_cbg()
        } else {
            self.to_grid().to_shapes()
        };

        for s in inner_shapes.shapes.iter_mut() {
            s.orow += self.orow;
            s.ocol += self.ocol;

            for r in 0 .. s.cells.rows {
                for c in 0 .. s.cells.columns {
                    s.cells[(r,c)].row += self.orow;
                    s.cells[(r,c)].col += self.ocol;
                }
            }
        }

        inner_shapes
    }

    pub fn pixels_in_shape(&self) -> usize {
        let bg = self.majority_colour();

        let pixels: usize = self.cells.values()
            .filter(|c| c.colour != bg)
            .count();
            
        pixels
    }

    pub fn shrink_any(&self, coloured: bool) -> Self {
        let mut s = if coloured {
            self.to_grid().to_shapes_coloured().shapes[0].clone()
        } else {
            self.to_grid().to_shapes().shapes[0].clone()
        };

        s.orow = self.orow;
        s.ocol = self.ocol;

        for r in 0 .. s.cells.rows {
            for c in 0 .. s.cells.columns {
                s.cells[(r,c)].row = s.orow + r;
                s.cells[(r,c)].col = s.ocol + c;
            }
        }

        s
    }

    pub fn remove_border(&self) -> Self {
        let m = self.cells.slice(1 .. self.cells.rows - 2, 1 .. self.cells.columns - 2);

        if let Ok(m) = m {
            Self::new_cells(&m)
        } else {
            Self::trivial()
        }
    }

    pub fn remove_ragged_left(&self) -> Self {
        for r in 1 .. self.cells.rows - 1 {
            if self.cells[(r,0)].colour == Black {
                if let Ok(cells) = self.cells.slice(0 .. self.cells.rows, 1 .. self.cells.columns) {
                    return Shape::new_cells(&cells);
                } else {
                    break;
                }
            }
        }

        self.clone()
    }

    pub fn remove_ragged_right(&self) -> Self {
        for r in 1 .. self.cells.rows - 1 {
            if self.cells[(r,self.cells.columns-1)].colour == Black {
                if let Ok(cells) = self.cells.slice(0 .. self.cells.rows, 0 .. self.cells.columns-1) {
                    return Shape::new_cells(&cells);
                } else {
                    break;
                }
            }
        }

        self.clone()
    }

    pub fn remove_ragged_top(&self) -> Self {
        for c in 1 .. self.cells.columns - 1 {
            if self.cells[(0,c)].colour == Black {
                if let Ok(cells) = self.cells.slice(1 .. self.cells.rows, 0 .. self.cells.columns) {
                    return Shape::new_cells(&cells);
                } else {
                    break;
                }
            }
        }

        self.clone()
    }

    pub fn remove_ragged_bottom(&self) -> Self {
        for c in 1 .. self.cells.columns - 1 {
            if self.cells[(self.cells.rows-1,c)].colour == Black {
                if let Ok(cells) = self.cells.slice(0 .. self.cells.rows-1, 0 .. self.cells.columns) {
                    return Shape::new_cells(&cells);
                } else {
                    break;
                }
            }
        }

        self.clone()
    }

    pub fn has_ragged_left(&self) -> bool {
        for r in 1 .. self.cells.rows - 1 {
            if self.cells[(r,0)].colour == Black {
                return true;
            }
        }

        false
    }

    pub fn has_ragged_right(&self) -> bool {
        for r in 1 .. self.cells.rows - 1 {
            if self.cells[(r,self.cells.columns-1)].colour == Black {
                return true;
            }
        }

        false
    }

    pub fn has_ragged_top(&self) -> bool {
        for c in 1 .. self.cells.columns - 1 {
            if self.cells[(0,c)].colour == Black {
                return true;
            }
        }

        false
    }

    pub fn has_ragged_bottom(&self) -> bool {
        for c in 1 .. self.cells.columns - 1 {
            if self.cells[(self.cells.rows-1,c)].colour == Black {
                return true;
            }
        }

        false
    }

    // TODO: Fix 25094a63
    pub fn remove_ragged_border(&self) -> Self {
        let mut shape = self.clone();
        let mut done = true;

        loop {
            if shape.has_ragged_top() {
                shape = shape.remove_ragged_top();
                done = false;
            }
            if shape.has_ragged_bottom() {
                shape = shape.remove_ragged_bottom();
                done = false;
            }
            if shape.has_ragged_left() {
                shape = shape.remove_ragged_left();
                done = false;
            }
            if shape.has_ragged_right() {
                shape = shape.remove_ragged_right();
                done = false;
            }
            if done {
                break;
            } else {
                done = true;
            }
        }

        shape
    }

    /*
    pub fn shrink_ragged(&self) -> (usize, usize) {
        let mut rs = 0;
        let mut cs = 0;
        let mut re = self.cells.rows;
        let mut ce = self.cells.columns;

        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                if self.cells[(r,c)].colour != Black {
                    if r == 0 || c == 0 || self.cells[(r,c)].colour == Black 
            }
        }
    }
    */

    pub fn find_colour_pixel_coords(&self, colour: Colour) -> (usize, usize) {
        for ((r, c), cell) in self.cells.items() {
            if cell.colour == colour {
                return (r, c);
            }
        }

        (0, 0)
    }

    pub fn find_different_pixel_coords(&self) -> (usize, usize) {
        let colour = self.minority_colour();

        self.find_colour_pixel_coords(colour)
    }

    pub fn find_distinct_colours(&self, bg: Colour) -> Vec<Cell> {
        self.cells.values().filter(|c| c.colour != bg).cloned().collect()
    }

    pub fn shrink_border(&self) -> Self {
        self.shrink_border_colour_n(Black, 1)
    }

    pub fn shrink_border_n(&self, n: usize) -> Self {
        self.shrink_border_colour_n(Black, n)
    }

    pub fn shrink_border_colour(&self, bg: Colour) -> Self {
        self.shrink_border_colour_n(bg, 1)
    }

    pub fn shrink_border_colour_n(&self, bg: Colour, n: usize) -> Self {
        if self.cells.rows < n * 2 || self.cells.columns < n * 2 {
            return Self::trivial();
        }

        let mut tlr = 0;
        let mut tlc = 0;
        let mut brr = 0;
        let mut brc = 0;

        for ((r, c), cell) in self.cells.items() {
            if cell.colour == bg {
                if r == 0 {
                    tlr += 1;
                }
                if c == 0 {
                    tlc += 1;
                }
                if r == self.cells.rows - 1 {
                    brr += 1;
                }
                if c == self.cells.columns - 1 {
                    brc += 1;
                }
            }
        }

        let (tlr, tlc) = if tlr > tlc {
            (if tlr > 1 { 1 } else { 0 }, 0)
        } else {
            (0, if tlc > 1 { 1 } else { 0 })
        };
        let (brr, brc) = if brr > brc {
            (if brr > 1 { self.cells.rows - 2 } else { self.cells.rows - 1 }, self.cells.columns - 1)
        } else {
            (self.cells.rows - 1, if brc > 1 { self.cells.columns - 2 } else { self.cells.columns - 1 })
        };

        let mut this = self.subshape_trim(tlr, brr + 1 - tlr, tlc, brc + 1 - tlc);

        if n > 1 {
            this = this.shrink_border_colour_n(bg, n - 1);
        }

        this
    }

    // TODO: complete for 3f23242b
    pub fn shrink_left(&self, n: usize) -> Self {
        let mut shape = Shape::new_sized_coloured_position(0, self.ocol, self.cells.rows - n, self.cells.columns, self.colour);

        for r in n .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                shape.cells[(r - n,c)].row = self.orow - n;
                shape.cells[(r - n,c)].col = self.ocol;
                shape.cells[(r - n,c)].colour = self.cells[(r,c)].colour;
            }
        }
//self.show_summary();
//shape.show_summary();

        shape
    }

    pub fn on_edge(&self, grid: &Grid) -> bool {
        let r = self.orow;
        let c = self.ocol;
        let rs = self.cells.rows - 1;
        let cs = self.cells.columns - 1;
        let rl = grid.cells.rows - 1;
        let cl = grid.cells.columns - 1;

        r == 0 || c == 0 || r + rs == rl || c + cs == cl
    }

    // extra check for border required
    pub fn find_same_row(&self, others: &[Self]) -> Self {
        for s in others.iter() {
            if self.orow == s.orow && self.ocol < s.ocol {
                return s.clone();
            }
        }

        Self::trivial()
    }

    // extra check for border required
    pub fn find_same_col(&self, others: &[Self]) -> Self {
        for s in others.iter() {
            if self.orow < s.orow && self.ocol == s.ocol {
                return s.clone();
            }
        }

        Self::trivial()
    }

    pub fn shrink(&self) -> Self {
        self.shrink_any(false)
    }

    pub fn shrink_coloured(&self) -> Self {
        self.shrink_any(true)
    }

    pub fn euclidian(&self, other: &Self) -> f64 {
        let dr = (self.orow as isize - other.orow as isize).abs();
        let dc = (self.ocol as isize - other.ocol as isize).abs();

        ((dr * dr + dc * dc) as f64).sqrt()
    }

    pub fn manhattan(&self, other: &Self) -> usize {
        let dr = (self.orow as isize - other.orow as isize).abs();
        let dc = (self.ocol as isize - other.ocol as isize).abs();

        (dr + dc) as usize
    }

    pub fn is_diagonal(&self, other: &Self) -> bool {
        let dr = (self.orow as isize - other.orow as isize).abs();
        let dc = (self.ocol as isize - other.ocol as isize).abs();

        dr == dc
    }

    pub fn which_direction(&self, other: &Self) -> Direction {
        let dr = (self.orow as isize - other.orow as isize).abs();
        let dc = (self.ocol as isize - other.ocol as isize).abs();

        if dr == 0 && dc > 0 {
            return if self.orow < other.orow { Up } else { Down };
        } else if dr > 0 && dc == 0 {
            return if self.ocol < other.ocol { Left } else { Right };
        } else if dr == dc {
            return if self.orow < other.orow && self.ocol < other.ocol {
                DownRight
            } else if self.orow < other.orow && self.ocol > other.ocol {
                DownLeft
            } else if self.orow > other.orow && self.ocol < other.ocol {
                UpRight
            } else {
                UpLeft
            };
        }

        Other
    }

    /*
    // Must have at least one other!
    pub fn nearest(&self, other: &Shapes) -> Self {
        let dself = self.euclidian();
        let mut min: f64 = f64::MAX;
        let mut pos: &Self = &other.shapes[0];

        for s in &other.shapes {
            //if self.orow > s.orow + 1 && self.ocol > s.ocol + 1 {
                let dist = dself - s.euclidian();
println!("{:?}", dist.abs());

                if dist.abs() < min {
                    min = dist;
                    pos = s;
                }
            //}
        }

        pos.clone()
    }

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
        self.fill_boundary_colour_bg(Black)
    }

    pub fn fill_boundary_colour_bg(&self, bg: Colour) -> Self {
        if self.colour == Mixed {
            return self.clone();
        }

        let mut shape = self.clone();

        for ((r, c), cell) in self.cells.items() {
            if cell.colour == bg && (r == 0 || c == 0 || r == self.cells.rows - 1 || c == self.cells.columns - 1) {
                shape.flood_fill_mut(r, c, NoColour, self.colour);
            }
        }

        shape
    }

    /*
    pub fn fill_boundary_colour_bg(&self, bg: Colour) -> Self {
        if self.colour == Mixed {
            return self.clone();
        }

        let m = Self::fill_boundary_colour_cells(&self.cells, bg);

        Shape::new_cells(&m)
    }

    fn fill_boundary_colour_cells(cells: &Matrix<Cell>, bg: Colour) -> Matrix<Cell>{
        let mut m = cells.clone();

        for ((r, c), cell) in m.clone().items() {
            if cell.colour == bg && (r == 0 || c == 0 || r == m.rows - 1 || c == m.columns - 1) {
                let reachable = m.bfs_reachable((r, c), false, |i| m[i].colour == bg);

                reachable.iter().for_each(|&i| m[i].colour = NoColour);
            }
        }

        m
    }
    */

    pub fn flood_fill_border_mut(&mut self, ignore_colour: Colour, new_colour: Colour) {
        let rows = self.cells.rows;
        let cols = self.cells.columns;

        for r in 0 .. rows {
            for c in 0 .. cols {
                if (r == 0 || c == 0 || r == rows - 1 || c == cols - 1) && self.cells[(r, c)].colour == Black {
                    self.flood_fill_mut(r, c, ignore_colour, new_colour);
                }
            }
        }

        let (colour, _) = Self::cell_colour_cnt_mixed(&self.cells, true, true);

        self.colour = colour;
    }

    pub fn flood_fill(&self, r: usize, c: usize, ignore_colour: Colour, new_colour: Colour) -> Self {
        let mut shape = self.clone();

        shape.flood_fill_mut(r, c, ignore_colour, new_colour);

        let (colour, _) = Self::cell_colour_cnt_mixed(&shape.cells, true, true);

        shape.colour = colour;

        shape
    }

    pub fn flood_fill_mut(&mut self, r: usize, c: usize, ignore_colour: Colour, new_colour: Colour) {
        let reachable = self.cells.bfs_reachable((r, c), false, |i| self.cells[i].colour == Black || self.cells[i].colour == ignore_colour);

        reachable.iter().for_each(|&i| self.cells[i].colour = new_colour);
    }

    pub fn sid(m: &Matrix<Cell>, coloured: bool) -> u32 {
        let crc = Crc::<u32>::new(&crc::CRC_32_ISCSI);
        let mut digest = crc.digest();

        for ((r, c), cell) in m.items() {
            let colour = if cell.colour == Black {
                Black
            } else if coloured {
                cell.colour
            } else {
                Mixed 
            };

            digest.update(&r.to_ne_bytes());
            digest.update(&c.to_ne_bytes());
            digest.update(&Colour::to_usize(colour).to_ne_bytes());
        }

        digest.finalize()
    }

    pub fn contains_colour(&self, colour: Colour) -> bool {
        if self.colour == colour {
            return true;
        }

        for cell in self.cells.values() {
            if cell.colour == colour {
                return true;
            }
        }

        false
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

    pub fn gravity_down(&self) -> Self {
        self.gravity_down_colour(Black)
    }

    pub fn gravity_down_colour(&self, colour: Colour) -> Self {
        let mut values: Vec<Colour> = vec![colour; self.cells.columns];
        let mut counts: Vec<usize> = vec![0; self.cells.columns];

        for ((r, c), cell) in self.cells.items() {
            if self.cells[(r,c)].colour != colour {
                if values[c] == colour {
                    values[c] = cell.colour;
                }

                counts[c] += 1;
            }
        }

        let mut m = self.cells.clone();

        for (r, c) in self.cells.keys() {
            m[(r,c)].row = r + self.orow;
            m[(r,c)].col = c + self.ocol;

            if self.cells[(r,c)].colour == colour {
               m[(r,c)].colour = values[c];
            }
            if self.cells.rows - r > counts[c] {
               m[(r,c)].colour = colour;
            }
        }

        Self::new_cells(&m)
    }

    pub fn gravity_up(&self) -> Self {
        self.gravity_up_colour(Black)
    }

    pub fn gravity_up_colour(&self, colour: Colour) -> Self {
        self.mirrored_r().gravity_down_colour(colour).mirrored_r()
    }

    pub fn gravity_right(&self) -> Self {
        self.gravity_right_colour(Black)
    }

    pub fn gravity_right_colour(&self, colour: Colour) -> Self {
        self.rot_rect_90().gravity_down_colour(colour).rot_rect_270()
    }

    pub fn gravity_left(&self) -> Self {
        self.gravity_left_colour(Black)
    }

    pub fn gravity_left_colour(&self, colour: Colour) -> Self {
        self.rot_rect_270().gravity_down_colour(colour).rot_rect_90()
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

    pub fn find_equals(&self, shapes: &Shapes) -> Shape {
        for s in shapes.shapes.iter() {
            if s.equals(self) == Same {
                return s.clone();
            }
        }

        Shape::trivial()
    }

    // Same position
    pub fn equal_position(&self, other: &Self) -> bool {
        self.orow == other.orow && self.ocol == other.ocol
    }

    pub fn find_equal_position(&self, shapes: &Shapes) -> Shape {
        for s in shapes.shapes.iter() {
            if s.equal_position(self) {
                return s.clone();
            }
        }

        Shape::trivial()
    }

    // Same footprint
    pub fn equal_footprint(&self, other: &Self) -> bool {
        self.cells.columns == other.cells.columns && self.cells.rows == other.cells.rows
    }

    pub fn find_equal_footprint(&self, shapes: &Shapes) -> Shape {
        for s in shapes.shapes.iter() {
            if s.equal_footprint(self) {
                return s.clone();
            }
        }

        Shape::trivial()
    }

    // Same shape
    pub fn equal_shape(&self, other: &Self) -> bool {
        if !self.equal_footprint(other) {
            return false;
        }

        for (((sr, sc), c1), ((or, oc), c2)) in self.cells.items().zip(other.cells.items()) {
            if sr != or || sc != oc || (c1.colour == Black && c2.colour != Black) || (c1.colour != Black && c2.colour == Black) {
                return false;
            }
        }

        true
    }

    pub fn find_equal_shape(&self, shapes: &Shapes) -> Shape {
        for s in shapes.shapes.iter() {
            if s.equal_shape(self) {
                return s.clone();
            }
        }

        Shape::trivial()
    }

    pub fn show_summary(&self) {
        println!("{}/{}: {}/{} {:?}", self.orow, self.ocol, self.cells.rows, self.cells.columns, self.colour);
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

    pub fn subshape_remain(&self, tlr: usize, sr: usize, tlc: usize, sc: usize) -> Self {
        let mut s = self.subshape(tlr, sr, tlc, sc);

        s.orow = self.orow + tlr;
        s.ocol = self.ocol + tlc;

        for r in 0 .. s.cells.rows {
            for c in 0 .. s.cells.columns {
                s.cells[(r,c)].row = s.orow + r;
                s.cells[(r,c)].col = s.ocol + c;
            }
        }

        s
    }

    pub fn subshape(&self, tlr: usize, sr: usize, tlc: usize, sc: usize) -> Self {
        if sr == 0 || sc == 0 {
            return Self::trivial();
        }

        let mut m = Matrix::new(sr, sc, Cell::new(0, 0, 0));

        for r in 0 ..  sr {
            for c in 0 .. sc {
                m[(r,c)].row = self.cells[(r + tlr,c + tlc)].row;
                m[(r,c)].col = self.cells[(r + tlr,c + tlc)].col;
                m[(r,c)].colour = self.cells[(r + tlr,c + tlc)].colour;
            }
        }

        Self::new_cells(&m)
    }

    pub fn subshape_trim(&self, tlr: usize, sr: usize, tlc: usize, sc: usize) -> Self {
        let sr = if tlr + sr > self.cells.rows {
            self.cells.rows - tlr
        } else {
            sr
        };
        let sc = if tlc + sc > self.cells.columns {
            self.cells.columns - tlc
        } else {
            sc
        };

        self.to_grid().subgrid(tlr, sr, tlc, sc).as_shape()
    }

    pub fn id(&self) -> String {
        format!("{}/{}", self.orow, self.ocol)
    }

    pub fn above(&self, other: &Self) -> bool {
        let (sr, _) = self.centre_of_exact();
        let (or, _) = other.centre_of_exact();
        
        sr > or
    }

    pub fn below(&self, other: &Self) -> bool {
        let (sr, _) = self.centre_of_exact();
        let (or, _) = other.centre_of_exact();
        
        sr < or
    }

    pub fn right(&self, other: &Self) -> bool {
        let (_, sc) = self.centre_of_exact();
        let (_, oc) = other.centre_of_exact();
        
        sc < oc
    }

    pub fn left(&self, other: &Self) -> bool {
        let (_, sc) = self.centre_of_exact();
        let (_, oc) = other.centre_of_exact();
        
        sc > oc
    }

    pub fn diag(&self, other: &Self) -> bool {
        let (sr, sc) = self.centre_of();
        let (or, oc) = other.centre_of();
        
        let dr = if sr > or { sr - or } else { or - sr };
        let dc = if sc > oc { sc - oc } else { oc - sc };

        dr == dc
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
            .filter(|c| c.colour != Black)
            .count()
    }

    pub fn same_size(&self, other: &Self) -> bool {
        self.size() == other.size()
    }

    pub fn same_shape(&self, other: &Self) -> bool {
        self.cells.columns == other.cells.columns && self.cells.rows == other.cells.rows
    }

    pub fn same_pixel_positions(&self, other: &Self, same_colour: bool) -> bool {
        if !self.same_shape(other) {
            return false;
        }

        for (c1, c2) in self.cells.values().zip(other.cells.values()) {
            if c1.colour == Black && c2.colour != Black || c1.colour != Black && c2.colour == Black || (same_colour && c1.colour != c2.colour) {
                return false;
            }
        }

        true
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.cells.rows, self.cells.columns)
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

    pub fn find_layers(&self) -> (usize, Colour, Vec<Direction>) {
        let rows = self.cells.rows;
        let cols = self.cells.columns;
        let mut depth = 0;
        let mut colour = NoColour;
        let mut dirs: Vec<Direction> = Vec::new();

        for ((r, c), cell) in self.cells.items() {
            if cell.colour != Black {
                if colour == NoColour {
                    colour = cell.colour;
                } else if colour != NoColour && cell.colour != colour {
                    break;
                }
            }

            let mut ldepth = 0;

            if r == 0 && c == cols / 2 {
                for rr in 0 .. rows {
                    if self.cells[(rr,c)].colour == colour {
                        ldepth += 1;
                        if !dirs.contains(&Up) { dirs.push(Up); }
                    } else {
                        break;
                    }
                }
            } else if r == rows - 1 && c == cols / 2 {
                for rr in (0 .. rows).rev() {
                    if self.cells[(rr,c)].colour == colour {
                        ldepth += 1;
                        if !dirs.contains(&Down) { dirs.push(Down); }
                    } else {
                        break;
                    }
                }
            } else if c == 0 && r == rows / 2 {
                for cc in 0 .. cols {
                    if self.cells[(r,cc)].colour == colour {
                        ldepth += 1;
                        if !dirs.contains(&Left) { dirs.push(Left); }
                    } else {
                        break;
                    }
                }
            } else if c == cols - 1 && r == rows / 2 {
                for cc in (0 .. cols).rev() {
                    if self.cells[(r,cc)].colour == colour {
                        ldepth += 1;
                        if !dirs.contains(&Right) { dirs.push(Right); }
                    } else {
                        break;
                    }
                }
            }

            if ldepth > 0 {
                if depth == 0 && ldepth > 0 {
                    depth = ldepth;
                } else if ldepth != depth {
                    depth = 0;
                    dirs = Vec::new();
                    break;
                }
            }
        }

        (depth, colour, dirs)
    }

    // see 40f6cd08.todo
    pub fn paint_layers_mut(&mut self, indent: usize, depth: usize, colour: Colour, dirs: Vec<Direction>) {
        let rows = self.cells.rows;
        let cols = self.cells.columns;
        //let bg = self.colour;
        let bg = Black;

        for dir in dirs.iter() {
            match dir {
                Up => {
                    for r in indent .. indent + depth {
                        for c in indent .. cols {
                            if self.cells[(r, c)].colour == bg {
                                self.cells[(r, c)].colour = colour;
                            } else {
                                break;
                            }
                        }
                    }
                },
                Down => {
                    for r in rows - indent - depth .. rows - indent {
                        for c in indent .. cols {
                            if self.cells[(r, c)].colour == bg {
                                self.cells[(r, c)].colour = colour;
                            } else {
                                break;
                            }
                        }
                    }
                },
                Left => {
                    for r in indent .. rows  {
                        for c in indent .. indent + depth {
                            if self.cells[(r, c)].colour == bg {
                                self.cells[(r, c)].colour = colour;
                            } else {
                                break;
                            }
                        }
                    }
                },
                Right => {
println!("--- {} {}", indent, depth);
                    for r in indent .. rows {
                        //for c in 0 .. cols - indent {
                        for c in cols - indent - depth .. cols - indent {
                        //for c in indent + depth .. cols {
                        //for c in (0 .. cols  - indent - depth).rev() {
                        //for c in (indent + depth .. cols).rev() {
                        //for c in 0 .. cols - indent - depth {
                            if self.cells[(r, c)].colour == bg {
                                self.cells[(r, c)].colour = colour;
                            } else {
                                break;
                            }
                        }
                    }
                },
                _ => todo!()
            }
        }
//self.show();
    }

    pub fn cell_colours(&self) -> Vec<Colour>  {
        self.cell_colour_cnt_map().keys().map(|c| *c).collect()
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
        Self::cell_colour_cnt_mixed(cells, max, false)
    }

    pub fn cell_colour_cnt_mixed(cells: &Matrix<Cell>, max: bool, mixed: bool) -> (Colour, usize) {
        let mut h: HashMap<usize, usize> = HashMap::new();

        for c in cells.values() {
            if c.colour != Black {
                *h.entry(Colour::to_usize(c.colour)).or_insert(0) += 1;
            }
        }

        if mixed && h.len() > 1 {
            return (Mixed, 0);
        }

        let mm = if max {
            h.iter().max_by(|col, c| col.1.cmp(c.1))
        } else {
            h.iter().min_by(|col, c| col.1.cmp(c.1))
        };
        let pair: Option<(Colour, usize)> = mm
            .map(|(col, cnt)| (Colour::from_usize(*col), *cnt));

        match pair {
            None => (Black, 0),
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

        for ((r, c), cell) in self.cells.items() {
            if cell.colour == colour {
                ans.push((r, c));
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

    pub fn is_larger(&self, other: &Self) -> bool {
        self.size() >= other.size()
    }

    pub fn is_smaller(&self, other: &Self) -> bool {
        self.size() <= other.size()
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
        let shape = Self::new_sized_coloured_position(or, oc, side, side, Black);

        Shapes::new_shapes(&[shape, self.clone()]).to_shape()
    }

    pub fn to_square_grid(&self) -> Grid {
        let (or, oc, side, _) = self.origin_centre();
        let shape = Self::new_sized_coloured_position(or, oc, side, side, Black);

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

    /*
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
    */

    pub fn distance(&self, other: &Self) -> f32 {
        let (sr, sc) = self.centre_of_exact();
        let (or, oc) = other.centre_of_exact();
        let dor = sr - or;
        let doc = sc - oc;

        (dor * dor + doc * doc).sqrt()
    }

    pub fn distance_from(&self, row: usize, col: usize) -> f32 {
        let (sr, sc) = self.centre_of_exact();
        let dor = sr - row as f32;
        let doc = sc - col as f32;

        (dor * dor + doc * doc).sqrt()
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

    pub fn is_mirrored_r(&self, other: &Self) -> bool {
        self.is_mirrored(other, false)
    }

    pub fn is_mirrored_c(&self, other: &Self) -> bool {
        self.is_mirrored(other, true)
    }

    pub fn is_mirror_r(&self) -> bool {
        if self.cells.rows == 1 {
            return false;
        }
        let inc = if self.cells.rows % 2 == 0 { 0 } else { 1 };
        let s1 = self.subshape(0, self.cells.rows / 2, 0, self.cells.columns);
        let s2 = self.subshape(self.cells.rows / 2 + inc, self.cells.rows / 2, 0, self.cells.columns);

        s1.is_mirrored_r(&s2)
    }

    pub fn is_mirror_c(&self) -> bool {
        if self.cells.columns == 1 {
            return false;
        }
        let inc = if self.cells.columns % 2 == 0 { 0 } else { 1 };
        let s1 = self.subshape(0, self.cells.rows, 0, self.cells.columns / 2);
        let s2 = self.subshape(0, self.cells.rows, self.cells.columns / 2 + inc, self.cells.columns / 2);

        s1.is_mirrored_c(&s2)
    }

    fn mirrored(&self, lr: bool) -> Self {
        let mut m: Matrix<Cell> = self.cells.clone();

        if lr {
            m.flip_lr();
        } else {
            m.flip_ud();
        }

        for (r, c) in self.cells.keys() {
            m[(r, c)].row = r + self.orow;
            m[(r, c)].col = c + self.ocol;
        }
        
        Self::new(self.orow, self.ocol, &m)
    }

    pub fn mirrored_r(&self) -> Self {
        self.mirrored(false)
    }

    pub fn mirrored_c(&self) -> Self {
        self.mirrored(true)
    }

    pub fn transposed(&self) -> Self {
        let mut m: Matrix<Cell> = self.cells.clone();

        m.transpose();

        for (r, c) in self.cells.keys() {
            m[(r, c)].row = r + self.orow;
            m[(r, c)].col = c + self.ocol;
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

        for (r, c) in self.cells.keys() {
            m[(r, c)].row = r + self.orow;
            m[(r, c)].col = c + self.ocol;
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

    pub fn rot_rect_90(&self) -> Self {
        if self.cells.rows == self.cells.columns {
            self.rotated_90()
        } else {
            let mut rot = Self::new_sized_coloured_position(self.orow, self.ocol, self.cells.columns, self.cells.rows, self.colour);
            let n = self.cells.rows;
            
            for ((r, c), cell) in self.cells.items() {
                //rot.cells[(c, n - r - 1)].row = r;
                //rot.cells[(c, n - r - 1)].col = c;
                rot.cells[(c, n - r - 1)].colour = cell.colour;
            }

            rot
        }
    }

    pub fn rot_rect_180(&self) -> Self {
        if self.cells.rows == self.cells.columns {
            self.rotated_90().rotated_90()
        } else {
            self.rot_rect_90().rot_rect_90()
        }
    }

    pub fn rot_rect_270(&self) -> Self {
        if self.cells.rows == self.cells.columns {
            self.rotated_270()
        } else {
            self.rot_rect_90().rot_rect_90().rot_rect_90()
        }
    }

    pub fn extend_line(&self) -> Self {
        if self.height() > 1 && self.width() > 1 {
            return self.clone();
        }

        if self.height() == 1 {
            self.extend_right(1)
        } else {
            self.extend_bottom(1)
        }
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

    pub fn rotate_90_pos(&self, times: usize, or: usize, oc: usize) -> Self {
        let mut shape = self.to_grid().rotate_90(times).as_shape();

        shape.orow = or;
        shape.ocol = oc;

        for (r, c) in shape.clone().cells.keys() {
            shape.cells[(r,c)].row = or + r;
            shape.cells[(r,c)].col = oc + c;
        }

        shape
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

        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                let colour: usize = self.cells[(r, c)].colour.to_usize();

                grid[r][c] = colour;
            }
        }

        serde_json::to_string(&grid).unwrap()
    }

    pub fn cell_count(&self) -> usize {
        self.cell_count_colour(Black)
    }

    pub fn cell_count_colour(&self, colour: Colour) -> usize {
        self.cells.values().filter(|c| c.colour != colour).count()
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

        grid.map(|diff| diff.as_shape())
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

        for c in shape.cells.values_mut() {
            if c.colour == from || from == NoColour {
                c.colour = to;
            }
        }

        let (colour, _) = Self::cell_colour_cnt_mixed(&shape.cells, true, true);

        shape.colour = colour;

        shape
    }

    pub fn recolour_mut(&mut self, from: Colour, to: Colour) {
        self.colour = to;

        for c in self.cells.values_mut() {
            if c.colour == from || from == NoColour {
                c.colour = to;
            }
        }

        let (colour, _) = Self::cell_colour_cnt_mixed(&self.cells, true, true);

        self.colour = colour;
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

    pub fn is_horizontal_line(&self) -> bool {
        self.cells.columns > 1 && self.cells.rows == 1 
    }

    pub fn is_vertical_line(&self) -> bool {
       self.cells.rows > 1 && self.cells.columns == 1 
    }

    pub fn is_square(&self) -> bool {
        self.cells.rows == self.cells.columns
    }

    pub fn has_band(&self, rs: usize, cs: usize) -> (Direction, usize) {
        if self.colour == Mixed {
            return (Other, 0);
        }
        if self.cells.rows == 1 && self.cells.columns == cs {
            (Down, self.orow)
        } else if self.cells.rows == rs && self.cells.columns == 1 {
            (Right, self.ocol)
        } else {
            (Other, 0)
        }
    }

    pub fn make_square(&self) -> Self {
        let sz = self.cells.rows.max(self.cells.columns);
        let mut cells = Matrix::new(sz, sz, Cell::new(0, 0, 0));

        for (i, cell) in self.cells.items() {
            cells[i].colour = cell.colour;
        }
        for (r, c) in cells.keys() {
            cells[(r,c)].row = self.orow + r;
            cells[(r,c)].col = self.ocol + c;
        }

        Self::new(self.orow, self.ocol, &cells)
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

        for (r, c) in self.cells.keys() {
            if self.cells[(r, c)].colour == Black {
                shape.cells[(r, c)].colour = to;
            }
        }

        shape
    }

    pub fn scale_up(&self, factor: usize) -> Self {
        let mut cells = Matrix::new(self.cells.rows * factor, self.cells.columns * factor, Cell::new(0, 0, 0));

        for r in 0 .. cells.rows {
            for c in 0 .. cells.columns {
                let rf = r / factor;
                let cf = c / factor;

                cells[(r, c)].row = r;
                cells[(r, c)].col = c;
                cells[(r, c)].colour = self.cells[(rf, cf)].colour;
            }
        }

        Self::new(0, 0, &cells)
    }

    pub fn scale_down(&self, factor: usize) -> Self {
        let mut cells = Matrix::new(self.cells.rows / factor, self.cells.columns / factor, Cell::new(0, 0, 0));

        for r in 0 .. cells.rows {
            for c in 0 .. cells.columns {
                let rf = r * factor;
                let cf = c * factor;

                cells[(r, c)].row = r;
                cells[(r, c)].col = c;
                cells[(r, c)].colour = self.cells[(rf, cf)].colour;
            }
        }

        Self::new(0, 0, &cells)
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
        let (r, c) = self.dimensions();
        if r < 3 || c < 3 {
            return false;
        }
        for ((r, c), cell) in self.cells.items() {
            let pred = fc(r, c, self.cells.rows, self.cells.columns);

            if cell.colour == Black && pred || hole && cell.colour != Black && !pred {
                return false;
            }
        }

        true
    }

    pub fn has_border(&self) -> bool {
        self.has_border_hole(false, &|r, c, rows, cols| r == 0 || c == 0 || r == rows - 1 || c == cols - 1)
    }

    pub fn is_hollow(&self) -> bool {
        self.has_border_hole(true, &|r, c, rows, cols| r == 0 || c == 0 || r == rows - 1 || c == cols - 1)
    }

    pub fn has_open_border_top(&self) -> bool {
        self.has_border_hole(true, &|r, c, rows, cols| c == 0 || r == rows - 1 || c == cols - 1)
    }

    pub fn has_open_hole_top(&self) -> bool {
        if self.is_hollow() {
            return false;
        }
        self.has_border_hole(false, &|r, c, rows, cols| c == 0 || r == rows - 1 || c == cols - 1)
    }

    pub fn has_open_border_bottom(&self) -> bool {
        self.has_border_hole(true, &|r, c, _rows, cols| r == 0 || c == 0 || c == cols - 1)
    }

    pub fn has_open_hole_bottom(&self) -> bool {
        if self.is_hollow() {
            return false;
        }
        self.has_border_hole(false, &|r, c, _rows, cols| r == 0 || c == 0 || c == cols - 1)
    }

    pub fn has_open_border_left(&self) -> bool {
        self.has_border_hole(true, &|r, c, rows, cols| r == 0 || r == rows - 1 || c == cols - 1)
    }

    pub fn has_open_hole_left(&self) -> bool {
        if self.is_hollow() {
            return false;
        }
        self.has_border_hole(false, &|r, c, rows, cols| r == 0 || r == rows - 1 || c == cols - 1)
    }

    pub fn has_open_border_right(&self) -> bool {
        self.has_border_hole(true, &|r, c, rows, _cols| r == 0 || c == 0 || r == rows - 1)
    }

    pub fn has_open_hole_right(&self) -> bool {
        if self.is_hollow() {
            return false;
        }
        self.has_border_hole(false, &|r, c, rows, _cols| r == 0 || c == 0 || r == rows - 1)
    }

    pub fn find_a_border_break(&self) -> (Direction, usize, usize) {
        for ((r, c) , cell) in self.cells.items() {
            if r == 0 && cell.colour == Black { 
                return (Up, cell.row, cell.col);
            } else if c == self.cells.columns - 1 && cell.colour == Black {
                return (Right, cell.row, cell.col);
            } else if r == self.cells.rows - 1 && cell.colour == Black { 
                return (Down, cell.row, cell.col);
            } else if c == 0 && cell.colour == Black {
                return (Left, cell.row, cell.col);
            }
        }

        (Other, 0, 0)
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

        for ((r, c), cell) in new_other.cells.items() {
            shape.cells[(r, c)] = cell.clone();
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
        for ((r, c), cell) in other.cells.items() {
            self.cells[(r, c)] = cell.clone();
        }
    }

    pub fn centre_of(&self) -> (usize, usize) {
        (self.orow + self.cells.rows / 2, self.ocol + self.cells.columns / 2)
    }

    pub fn centre_of_exact(&self) -> (f32, f32) {
        (self.orow as f32 + self.cells.rows as f32  / 2.0, self.ocol as f32 + self.cells.columns as f32  / 2.0)
    }

    pub fn nearest_point_idx(&self, points: &Vec<(usize, usize)>) -> usize {
        let np = self.nearest_point(points);

        for (i, &rc) in points.iter().enumerate() {
            if np == rc {
                return i;
            }
        }

        usize::MAX
    }

    pub fn nearest_point(&self, points: &Vec<(usize, usize)>) -> (usize, usize) {
        let (cr, cc) = self.centre_of_exact();
        let mut dist = f32::MAX;
        let mut nr = 0.0;
        let mut nc = 0.0;

        for (r, c) in points.iter() {
            let r2_dist = ((cr - *r as f32).powi(2) + (cc - *c as f32).powi(2)).sqrt();

            if dist == f32::MAX {
                nr = *r as f32;
                nc = *c as f32;
                dist = r2_dist;
            } else if r2_dist < dist {
                nr = *r as f32;
                nc = *c as f32;
                dist = r2_dist;
            }
        }

        (nr as usize, nc as usize)
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
        let (r, c) = self.centre_of();
        let (or, oc) = other.centre_of();
        let dr = (r - or) as isize;
        let dc = (c - oc) as isize;

        other.translate(dr, dc)
    }

    pub fn container(&self, other: &Self) -> bool {
        self.orow <= other.orow && self.ocol <= other.ocol && self.cells.rows + self.orow >= other.cells.rows + other.orow && self.cells.columns + self.ocol >= other.cells.columns + other.ocol
    }

    pub fn can_contain(&self, other: &Self) -> bool {
        let (s, o) = if self.size() > other.size() { (self, other) } else { (other, self) };
        if s.width() < 2 || s.height() < 2 || s.width() < o.width() + 1 || s.height() < o.height() + 1 { return false };
        for (r, c) in other.cells.keys() {
            if r != 0 && c != 0 && r <= s.cells.rows && r <= s.cells.columns && s.cells[(r, c)].colour != Black {
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
        for (r, c) in o.cells.keys() {
            if r != 0 && c != 0 && r < s.cells.rows && r < s.cells.columns && s.cells[(r, c)].colour != o.cells[(r - 1, c - 1)].colour {
                return false;
            }
        }

        s.container(other)
    }

    pub fn contained_in(&self, other: &Self) -> bool {
        other.is_contained(self)
    }

    pub fn nest_mut(&mut self, n: usize, colour: Colour) {
        if self.cells.rows <= n || self.cells.columns <= n {
            return;
        }
        for r in n .. self.cells.rows - n {
            for c in n .. self.cells.columns - n {
                self.cells[(r,c)].colour = colour;
            }
        }
    }

    // TODO: 50f325b5
    pub fn patch_in_shape(&self, other: &Self) -> (usize, usize) {
        for ch in self.cells.chunks(self.cells.columns) {
            //w.iter().for_each(|ch| print!("{:?},", ch.colour));
            let chr = ch.iter().map(|ch| format!("{:?}", ch.colour)).collect::<Vec<_>>().join(",");
println!("{}", chr);
            let mut iter = other.cells.iter();
            let c = iter.position(|c| { println!("{:?} {:?}", c[0].colour, ch[0].colour); c[0].colour != ch[0].colour});
            let c2 = iter.next();
            let c3 = iter.next();
    //println!("{:?},{:?},{:?}", c.unwrap()[0].colour, c2.colour, c3.colour);
    //println!("{:?}", c);
println!("{:?},{:?},{:?}", c, c2, c3);
        }
        println!();
        (0, 0)
    }

    pub fn adjacent_r_or_c(&self, other: &Self) -> bool {
        self.orow + self.cells.rows == other.orow ||
        self.ocol + self.cells.columns == other.ocol ||
        other.orow + other.cells.rows == self.orow ||
        other.ocol + other.cells.columns == self.ocol
    }

    // TODO: improave? % touching...
    pub fn touching(&self, other: &Self) -> bool {
        let sr = self.orow;
        let sc = self.ocol;
        let or = other.orow;
        let oc = other.ocol;
        let srl = self.cells.rows;
        let scl = self.cells.columns;
        let orl = other.cells.rows;
        let ocl = other.cells.columns;

        sr + srl == or && oc <= sc && oc + ocl >= sc + scl ||
        sc + scl == oc && or >= sr && or + orl <= sr + srl ||
        or + orl == sr && sc <= oc && sc + scl >= oc + ocl ||
        oc + ocl == sc && sr >= or && sr + srl <= or + orl
    }

    pub fn find_touching(&self, shapes: &Shapes) -> Shape {
        for s in shapes.shapes.iter() {
            if self.touching(s) {
                return s.clone();
            }
        }

        Shape::trivial()
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

    pub fn translate_row(&self, r: isize) -> Self {
        Self::translate(self, r, 0)
    }

    pub fn translate_col(&self, c: isize) -> Self {
        Self::translate(self, 0, c)
    }

    pub fn translate(&self, r: isize, c: isize) -> Self {
        let mut shape = self.clone();

        if self.orow as isize + r < 0 || self.ocol as isize + c < 0 {
            return shape;
        }


        shape.orow = (shape.orow as isize + r) as usize;
        shape.ocol = (shape.ocol as isize + c) as usize;

        shape.cells.iter_mut()
            .for_each(|cell| {
                cell.row = (cell.row as isize + r) as usize;
                cell.col = (cell.col as isize + c) as usize;
            });

        shape
    }

    pub fn translate_absolute_r(&self, r: usize) -> Self {
        Self::translate_absolute(self, r, 0)
    }

    pub fn translate_absolute_c(&self, c: usize) -> Self {
        Self::translate_absolute(self, 0, c)
    }

    pub fn translate_absolute(&self, r: usize, c: usize) -> Self {
        let mut shape = self.normalise_key();

        shape.orow = r;
        shape.ocol = c;

        shape.cells.iter_mut()
            .for_each(|cell| {
                cell.row += r;
                cell.col += c;
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

        if let Some((r, c)) = self.pixel_coords(colour_l) {
            if let Some((or, oc)) = other.pixel_coords(colour_r) {
                let dr = (r - or) as isize;
                let dc = (c - oc) as isize;

                Some(other.translate(dr, dc))
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
        let or = if self.orow == 0 { self.orow } else { self.orow - 1 };
        let oc = if self.ocol == 0 { self.ocol } else { self.ocol - 1 };

        i.orow -= or;
        i.ocol -= oc;

        for c in i.cells.values_mut() {
            c.row -= or;
            c.col -= oc;
        }

        i
    }

    pub fn bare_corners(&self) -> bool {
        let rows = self.cells.rows;
        let cols = self.cells.columns;

        self.cells[(0,0)].colour == Black && self.cells[(rows - 1,cols - 1)].colour == Black || self.cells[(rows - 1,0)].colour == Black && self.cells[(0,cols - 1)].colour == Black
    }

    // TODO: needs backtracking
    pub fn fit_shape(&self, _shape: &Self) -> Self {
        self.clone()
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
            cell.row = cell.row + r - self.orow;
            cell.col = cell.col + c - self.ocol;
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
        let or = if ci.orow == 0 { ci.orow } else { ci.orow - 1 };
        let oc = if ci.ocol == 0 { ci.ocol } else { ci.ocol - 1 };

        i.orow = or;
        i.ocol = oc;

        for c in i.cells.values_mut() {
            c.row += or;
            c.col += oc;
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

    pub fn has_border_cell(&self, other: &Self) -> bool {
        if self.orow == 0 && self.ocol == 0 {
            return true;
        }

        for cell in self.cells.values() {
            if cell.row == other.cells.rows - 1 || cell.col == other.cells.columns - 1 {
                return true;
            }
        }

        false
    }

    pub fn is_subgrid(&self, other: &Self) -> bool {
        let (sr,sc) = self.dimensions();
        let (or,oc) = other.dimensions();

        if or > sr || oc > sc {
            return false;
        }

        for r in 0 ..= sr - or {
            for c in 0 ..= sc - oc {
                let mut is_sg = true;

                for rr in 0 .. or {
                    for cc in 0 .. oc {
//println!("{} + {}, {} + {} --- {} {}", r, rr, c, cc, sr, sc);
                        let scol = self.cells[(r + rr,c + cc)].colour;
                        let ocol = other.cells[(rr,cc)].colour;

                        if scol != ocol && ocol != Black || scol == Black {
                            is_sg = false;
                            break;
                        }
                    }
                    if !is_sg {
                        break;
                    }
                }
                if is_sg {
/*
println!(">>>>");
self.show();
other.show();
println!("<<<<");
println!("{}, {} --- {} {}", r, c, sc, oc);
*/
                    return true;
                }
            }
        }

        false
    }

    /*
    pub fn match_shapes(&self, other: &Self) -> Direction {
        fn check(s: &Shape, other: &Shape) -> bool {
            for (r, c) in other.cells.keys() {
                if s.cells[(r,c)].colour != s.colour || s.cells[(r,c)].colour == Black {
                    return false;
                }
            }

            true
        }

        let (sr, sc) = self.dimensions();
        let (or, oc) = other.dimensions();
println!("{sr}/{sc} {or}/{oc}");
self.show();

        if sr == or && sc == oc {
println!("0");
            for (c1, c2) in self.cells.values().zip(other.cells.values()) {
                if c1.colour != c2.colour && c1.colour != Black && c2.colour != Black {
                    return Other;
                }
            }

            return SameDir;
        }
        if sr == oc {
/*
println!("1");
self.show();
            let s = self.rotated_90();

            if check(&s, other) {
                return Left;
            } else {
                let s = self.rotated_270();

                if check(&s, other) {
                    return Right;
                }
            }
*/
        }
        if sc == or {
println!("2");
//self.show();
            let s = self.rotated_90();

            if check(&s, other) {
println!("2L");
                return Left;
            } else if false {
                let s = self.rotated_270();

                if check(&s, other) {
println!("2R");
                    return Right;
                }
            }
        }
        if sr == or {
println!("3");
        }
        if sc == oc {
println!("4");
        }

        Other
    }
    */

    pub fn find_black_patches_limit(&self, limit: usize) -> Shapes {
        let mut bp = self.to_grid().find_black_patches_limit(limit);

        for s in bp.shapes.iter_mut() {
            s.orow += self.orow;
            s.ocol += self.ocol;

            for cell in s.cells.values_mut() {
                cell.row += self.orow;
                cell.col += self.ocol;
            }
        }

        bp
    }

    pub fn has_arm(&self, limit: usize) -> (Direction, usize) {
        let mut bp = self.to_grid().find_black_patches_limit(limit);

        // Remove internal shapes
        if bp.shapes.len() != 2 {
            for (i, s) in bp.shapes.clone().iter().enumerate() {
                if !s.has_border_cell(self) && i < bp.len() {
                    bp.shapes.remove(i);
                }
            }
        }

        if bp.shapes.len() == 2 {
            let rows1 = bp.shapes[0].cells.rows;
            let cols1 = bp.shapes[0].cells.columns;
            let rows2 = bp.shapes[1].cells.rows;
            let cols2 = bp.shapes[1].cells.columns;

            if self.cells[(0,0)].colour == Black && self.cells[(self.cells.rows - 1,0)].colour == Black && rows1 + rows2 + 1 == self.cells.rows {
                (Direction::Left, cols1)
            } else if self.cells[(0,self.cells.columns - 1)].colour == Black && self.cells[(self.cells.rows - 1,self.cells.columns - 1)].colour == Black && rows1 + rows2 + 1 == self.cells.rows {
                (Direction::Right, cols1)
            } else if cols1 + cols2 + 1 == self.cells.columns {
                if self.cells[(0,0)].colour == Black {
                    (Direction::Up, rows1)
                } else {
                    (Direction::Down, rows1)
                }
            } else {
                (Direction::Other, 0)
            }
        } else {
            (Direction::Other, 0)
        }
    }

    /*
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
                self.subshape(self.orow, self.cells.rows - distance, self.ocol, self.cells.columns),
            Down => 
                self.subshape(self.orow + distance, self.cells.rows - distance, self.ocol, self.cells.columns),
            Left => 
                self.subshape(self.orow, self.cells.rows, self.ocol, self.cells.columns - distance),
            Right => 
                self.subshape(self.orow + distance, self.cells.rows - distance, self.ocol + distance, self.cells.columns - distance),
            _ => self.clone(),
        }
    }

    pub fn has_arm(&self) -> ShapeCategory {
        for ((x, y), c) in self.cells.items() {
            match c.cat {
                PointT => {
                    if self.cells.rows > 1 && self.cells.columns == 1 {
                        return ShapeCategory::VerticalLine;
                    }
                    let mut i = 1;
                    while x + i < self.cells.rows && self.cells[(x+i,y)].cat == InternalEdgeT {
                        i += 1;
                    }
                    if i >= 2 {
                        return ShapeCategory::ArmTop(i)
                    }
                },
                PointB => {
                    if self.cells.rows > 1 && self.cells.columns == 1 {
                        return ShapeCategory::VerticalLine;
                    }
                    let mut i = 1;
                    while x < i && self.cells[(x-i,y)].cat == InternalEdgeB {
                        i += 1;
                    }
                    if i >= 2 {
                        return ShapeCategory::ArmBottom(i)
                    }
                },
                PointL => {
                    if self.cells.rows == 1 && self.cells.columns > 1 {
                        return ShapeCategory::HorizontalLine;
                    }
                    let mut i = 1;
                    while y + i < self.cells.columns && self.cells[(x,y+i)].cat == InternalEdgeL {
                        i += 1;
                    }
                    if i >= 2 {
                        return ShapeCategory::ArmLeft(i)
                    }
                },
                PointR => {
                    if self.cells.rows == 1 && self.cells.columns > 1 {
                        return ShapeCategory::HorizontalLine;
                    }
                    let mut i = 1;
                    while y > i && self.cells[(x,y-i)].cat == InternalEdgeR {
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
    */

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

    // bool is true == Row axis, false = Col axis
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
        
        for ((r, c), cell) in self.cells.items() {
            if r == 0 && cell.colour != colour {
                return false;
            }
            if c == 0 && cell.colour != colour {
                return false;
            }
            if r == self.cells.rows - 1 && cell.colour != colour {
                return false;
            }
            if c == self.cells.columns - 1 && cell.colour != colour {
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
        for ((r, c), cell) in self.cells.items() {
            // Interior cells only interesting
            if r == 0 || c == 0 || r == self.cells.rows - 1 || c == self.cells.columns - 1 || cell.colour != Black {
                continue;
            }

            let reachable = self.cells.bfs_reachable((r, c), true, |i| self.cells[i].colour != self.colour);

            for (r,c) in &reachable {
                if *r == 0 || *c == 0 || *r == self.cells.rows - 1 || *c == self.cells.columns - 1 {
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

    pub fn add_hugging_border(&self, colour: Colour) -> Self {
        if self.orow == 0 && self.ocol == 0 {
            return self.clone();
        }

        let mut shape = self.add_border(colour);

        if self.is_full() {
            return shape;
        }

        let copy_shape = shape.clone();
        let rows = shape.cells.rows - 1;
        let cols = shape.cells.columns - 1;

        for (r, c) in copy_shape.cells.keys() {
            match (r, c) {
                (0, _) if shape.cells[(1,c)].colour == Black => {
                    for r in 1 .. rows {
                        if shape.cells[(r,c)].colour == Black {
                            shape.cells[(r,c)].colour = colour;
                        } else {
                            break;
                        }
                    }
                },
                (_, 0) if shape.cells[(r,1)].colour == Black => {
                    for c in 1 .. cols {
                        if shape.cells[(r,c)].colour == Black {
                            shape.cells[(r,c)].colour = colour;
                        } else {
                            break;
                        }
                    }
                },
                (r, _) if r == rows && shape.cells[(r-1,c)].colour == Black => {
                    for r in (1 .. rows).rev() {
                        if shape.cells[(r,c)].colour == Black {
                            shape.cells[(r,c)].colour = colour;
                        } else {
                            break;
                        }
                    }
                },
                (_, c) if c == cols && shape.cells[(r,c-1)].colour == Black => {
                    for c in (1 .. cols).rev() {
                        if shape.cells[(r,c)].colour == Black {
                            shape.cells[(r,c)].colour = colour;
                        } else {
                            break;
                        }
                    }
                },
                __ => {
                },
            }
        }

        let mut rc: Vec<(usize, usize)> = Vec::new();

        for (r, c) in shape.cells.keys() {
            let (cnt, _) = surround_cnt(&shape.cells, r, c, colour);

            if cnt == 8 {
                rc.push((r, c));
            }
        }

        for rc in rc.iter() {
            shape.cells[rc].colour = Black;
        }

        shape
    }

    pub fn add_border(&self, colour: Colour) -> Self {
        if self.orow == 0 && self.ocol == 0 {
            let mut s = Shape::new_sized_coloured_position(self.orow, self.ocol, self.cells.rows + 1, self.cells.columns + 1, colour);

            for ((r, c), cell) in self.cells.items() {
                s.cells[(r,c)] = cell.clone(); 
            }

            s
        } else if self.orow == 0 {
            let mut s = Shape::new_sized_coloured_position(self.orow, self.ocol - 1 , self.cells.rows + 1, self.cells.columns + 2, colour);

            for ((r, c), cell) in self.cells.items() {
                s.cells[(r,c+1)] = cell.clone(); 
            }

            if s.ocol > 0 { s.ocol -= 1 };

            s
        } else if self.ocol == 0 {
            let mut s = Shape::new_sized_coloured_position(self.orow - 1, self.ocol, self.cells.rows + 2, self.cells.columns + 2, colour);

            for ((r, c), cell) in self.cells.items() {
                s.cells[(r+1,c)] = cell.clone(); 
            }

            if s.orow > 0 { s.orow -= 1 };

            s
        } else {
            let mut s = Shape::new_sized_coloured_position(self.orow - 1, self.ocol - 1, self.cells.rows + 2, self.cells.columns + 2, colour);

            for ((r, c), cell) in self.cells.items() {
                s.cells[(r+1,c+1)] = cell.clone(); 
            }

            if s.orow > 0 { s.orow -= 1 };
            if s.ocol > 0 { s.ocol -= 1 };

            s
        }
    }

    pub fn toddle_colour(&self, bg: Colour, fg: Colour) -> Self {
        let s = self.recolour(bg, ToBlack + bg).recolour(fg, bg);

        s.recolour(ToBlack + bg, fg)
    }

    pub fn extend_top(&self, n: usize) -> Self {
        let mut cells = Matrix::new(self.cells.rows + n, self.cells.columns, Cell::new(0, 0, 0));
        for r in 0 .. cells.columns {
            for c in 0 .. n {
                cells[(c, r)].row = c + self.orow;
                cells[(c, r)].col = r + self.ocol;
                cells[(c, r)].colour = self.cells[(0, r)].colour;
            }
        }
        for r in n .. cells.rows {
            for c in 0 .. cells.columns {
                cells[(r, c)].row = r + self.orow;
                cells[(r, c)].col = c + self.ocol;
                cells[(r, c)].colour = self.cells[(r - n, c)].colour;
            }
        }

        Shape::new(self.orow, self.ocol, &cells)
    }

    pub fn extend_bottom(&self, n: usize) -> Self {
        self.mirrored_r().extend_top(n).mirrored_r()
    }

    pub fn extend_left(&self, n: usize) -> Self {
        let mut cells = Matrix::new(self.cells.rows, self.cells.columns + n, Cell::new(0, 0, 0));

        for r in 0 .. cells.rows {
             for c in 0 .. n {
                cells[(r, c)].row = r + self.orow;
                cells[(r, c)].col = c + self.ocol;
                cells[(r, c)].colour = self.cells[(r, 0)].colour;
            }
        }
        for r in 0 .. cells.rows {
            for c in n .. cells.columns {
                cells[(r, c)].row = r + self.orow;
                cells[(r, c)].col = c + self.ocol;
                cells[(r, c)].colour = self.cells[(r, c - n)].colour;
            }
        }

        Shape::new(self.orow, self.ocol, &cells)
    }

    pub fn extend_right(&self, n: usize) -> Self {
        self.mirrored_c().extend_left(n).mirrored_c()
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
        Shapes::new_from_shape(self)
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

    pub fn draw_line(&self, other: &Self, colour: Colour) -> Shape {
        let s1 = self.centre_of();
        let s2 = other.centre_of();

//println!("--- {s1:?} -> {s2:?}");
        let (r, c) = if s1 < s2 { s1 } else { s2 };
        let (rl, cl) = ((s1.0.max(s2.0) - s1.0.min(s2.0)).max(1), (s1.1.max(s2.1) - s1.1.min(s2.1)).max(1));

        Shape::new_sized_coloured_position(r, c, rl, cl, colour)
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
        self.find_first_colours(Black)
    }

    pub fn find_first_colours(&self, colour: Colour) -> Vec<(usize,usize)> {
        let mut fb: Vec<(usize,usize)> = Vec::new();
        let mut bg = false;
        let mut nr = 0;

        for ((r, c), cell) in self.cells.items() {
            if cell.colour == colour {
                if !bg && nr == r {
                    fb.push((r,c));
                }
            } else if c == 0 {
                nr = r + 1;
            }
            bg = cell.colour == colour;
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

        for r in 0 .. shape.cells.rows {
            for c in 0 .. shape.cells.columns {
                shape.cells[(r,c)].row = this.orow + r - thickness;
                shape.cells[(r,c)].col = this.ocol + c - thickness;

                let bounds = r < thickness && c < thickness || r < thickness && c >= this.cells.columns + thickness || r >= this.cells.rows + thickness && c < thickness || r >= this.cells.rows + thickness && c >= this.cells.columns + thickness;

                if !all && (!corners && bounds || corners && !bounds) {
                    continue;
                }
                if r < thickness || r >= this.cells.rows + thickness || c < thickness || c >= this.cells.columns + thickness {
                    shape.cells[(r,c)].colour = colour;
                }
            }
        }

        shape
    }

    /*
    pub fn categorise_shape(&mut self) {
        let has_border = self.has_border();

        /*
        if has_border {
            self.cats.insert(ShapeCategory::SingleCell);
        }
        */
        if has_border && self.is_hollow() {
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

        /*
//        for _ in 0 .. 2 {
            let arm = self.has_arm();
            if !has_border && arm != ShapeCategory::Other {
//println!("-- {:?}", arm);
                self.cats.insert(arm);
            }
//        }
        */
//println!("{:?}", self.cats);
    }
    */

    /*
    pub fn outer_cells(cells: &Matrix<Cell>) -> Vec<Cell> {
        for ((r, c), cell) in cells {
            if cell.colour 
        }
    }
    */

    pub fn bg_count(&self) -> usize {
        let mut cnt = 0;

        for cell in self.cells.values() {
            if cell.colour == Black {
                cnt += 1;
            }
        }

        cnt
    }

    pub fn split_2(&self) -> Shapes {
        self.to_grid().split_2()
    }

    pub fn cell_category(cells: &Matrix<Cell>) -> Matrix<Cell> {
        Self::cell_category_bg(cells, Black)
    }

    pub fn cell_category_bg(cells: &Matrix<Cell>, bg: Colour) -> Matrix<Cell> {
        let mut m = cells.clone();

        for ((r, c), cell) in m.items_mut() {
            if cell.colour == bg { continue; }

            let cat = 
                if r == 0 {
                    if (c == 0 || cells[(r,c-1)].colour == bg) && (c == cells.columns - 1 || cells[(r,c+1)].colour == bg) {
                        CellCategory::PointT
                    } else if c == 0 {
                        if cells.rows == 1 || cells[(r+1,c)].colour == bg {
                            CellCategory::PointL
                        } else {
                            CellCategory::CornerTL
                        }
                    } else if c == cells.columns - 1 {
                        CellCategory::CornerTR
                    } else if c < cells.columns - 1 && cells[(r,c+1)].colour == bg {
                        CellCategory::InternalCornerTR
                    } else if c > 0 && cells[(r,c-1)].colour == bg {
                        CellCategory::InternalCornerTL
                    } else if cells.rows == 1 || cells[(r+1,c)].colour == bg {
                        CellCategory::StemLR
                    } else {
                        CellCategory::EdgeT
                    }
                } else if r == cells.rows - 1 {
                    if (c == 0 || cells[(r,c-1)].colour == bg) && (c == cells.columns - 1 || cells[(r,c+1)].colour == bg) {
                        CellCategory::PointB
                    } else if c == 0 {
                        CellCategory::CornerBL
                    } else if c == cells.columns - 1 {
                        if cells[(r-1,c)].colour == bg {
                            CellCategory::PointR
                        } else {
                            CellCategory::CornerBR
                        }
                    } else if c < cells.columns - 1 && cells[(r,c+1)].colour == bg {
                        CellCategory::InternalCornerBR
                    } else if c > 0 && cells[(r,c-1)].colour == bg {
                        CellCategory::InternalCornerBL
                    } else if cells.rows == 1 || cells[(r-1,c)].colour == bg {
                        CellCategory::StemLR
                    } else {
                        CellCategory::EdgeB
                    }
                } else if c == 0 {
                    if (r == 0 || cells[(r-1,c)].colour == bg) && (r == cells.rows - 1 || cells[(r+1,c)].colour == bg) {
                        CellCategory::PointL
                    } else if r > 0 && cells[(r-1,c)].colour == bg {
                        CellCategory::InternalCornerTL
                    } else if cells.columns == 1 || cells[(r,c+1)].colour == bg {
                        CellCategory::StemTB
                    } else {
                        CellCategory::EdgeL
                    }
                } else if c == cells.columns - 1 {
                    if (r == 0 || cells[(r-1,c)].colour == bg) && (r == cells.rows - 1 || cells[(r+1,c)].colour == bg) {
                        CellCategory::PointR
                    } else if r > 0 && cells[(r-1,c)].colour == bg {
                        CellCategory::InternalCornerTR
                    } else if cells.columns == 1 || cells[(r,c-1)].colour == bg {
                        CellCategory::StemTB
                    } else {
                        CellCategory::EdgeR
                    }
                } else if cells[(r-1,c)].colour == bg && cells[(r+1,c)].colour == bg {
                    CellCategory::StemLR
                } else if cells[(r,c-1)].colour == bg && cells[(r,c+1)].colour == bg {
                    CellCategory::StemTB
                } else {
                    CellCategory::Middle
                };

            cell.cat = cat;
        }

        m
    }

    pub fn fill_corners_mut(&mut self, pixels: usize, colour: Colour) {
        if pixels > 4 {
            return;
        }
        let rows = self.cells.rows;
        let cols = self.cells.columns;

        if pixels >= 1 {
            self.cells[(0,0)].colour = colour;
        }
        if pixels >= 2 {
            self.cells[(rows - 1, 0)].colour = colour;
        }
        if pixels >= 3 {
            self.cells[(0, cols - 1)].colour = colour;
        }
        if pixels >= 4 {
            self.cells[(rows - 1, cols - 1)].colour = colour;
        }
    }

    pub fn pixel_position(&self, min_colour: Colour) -> Direction {
        if self.colour != Mixed {
            return Other;
        }
        let hr = self.cells.rows / 2;
        let hc = self.cells.columns / 2;
        let mut left = 0;
        let mut right = 0;
        let mut top = 0;
        let mut bottom = 0;

        for ((r, c), cell) in self.cells.items() {
            if cell.colour == min_colour {
                if r == hr { left += 1; right += 1; }
                else if r < hr { top += 1 }
                else { bottom += 1 }

                if c == hc { top += 1; bottom += 1; }
                else if c < hr { left += 1 }
                else { right += 1 }
            }
        }

        if top == bottom && top == left && top == right {
            Middle 
        } else if top == left && bottom == right && top > bottom {
            UpLeft
        } else if bottom == left && top == right && bottom > top {
            DownLeft
        } else if bottom == right && top < bottom && left < right {
            DownRight
        } else if top == right && top > bottom && right > left {
            UpRight
        } else if top > bottom && top > left && top > right {
            Up
        } else if right > top && right > bottom && right > left {
            Right
        } else if bottom > top && bottom > left && bottom > right {
            Down
        } else {
            Left
        }
    }

    pub fn chequer(&self, rows: usize, cols: usize, pred: &dyn Fn(usize, usize) -> bool, func: &dyn Fn(&Self) -> Self, blank: bool) -> Shapes {
        let rs = rows * self.cells.rows;
        let cs = cols * self.cells.columns;
        let mut shapes = Shapes::new_sized(rs, cs);

        for r in (0 .. rs).step_by(self.cells.rows) {
            for c in (0 .. cs).step_by(self.cells.columns) {
                let shape = if pred(r, c) {
                    &func(self)
                } else if blank {
                    &self.blank()
                } else {
                    self
                };

                let s = shape.to_position(r, c);

                shapes.shapes.push(s);
            }
        }

        shapes
    }

    pub fn combined_chequer(&self, rows: usize, cols: usize, func: &dyn Fn(&Self, usize, usize) -> Self) -> Shapes {
        let rs = rows * self.cells.rows;
        let cs = cols * self.cells.columns;
        let mut shapes = Shapes::new_sized(rs, cs);

        for r in (0 .. rs).step_by(self.cells.rows) {
            for c in (0 .. cs).step_by(self.cells.columns) {
                let shape = &func(self, r, c);

                let s = shape.to_position(r, c);

                shapes.shapes.push(s);
            }
        }

        shapes
    }

    pub fn fit_chequer(&self, rc: usize, cc: usize, rstart: usize, cstart: usize, rgap: usize, cgap: usize, rs: usize, cs: usize, func: &dyn Fn(&Self, usize, usize) -> Self) -> Shapes {
//println!("{rc} {cc} {rstart} {cstart} {rgap} {cgap}");
        let mut shapes = Shapes::new_sized(rs, cs);

        for i in 0 .. rc {
            let r = rstart + i * (rgap + self.cells.rows);

            for j in 0 .. cc {
                let c = cstart + j * (cgap + self.cells.columns);
                let shape = &func(self, r, c);

                let s = shape.to_position(r, c);

                shapes.shapes.push(s);
            }
        }

        shapes
    }

    pub fn invert_colour(&self) -> Self {
        if self.colour == Mixed {
            return Self::trivial();
        }

        let mut shape = self.clone();

        for cell in shape.cells.values_mut() {
            cell.colour = if cell.colour == Black {
                self.colour
            } else {
                Black
            };
        }

        shape
    }

    pub fn blank(&self) -> Self {
        let mut shape = self.clone();

        for cell in shape.cells.values_mut() {
            cell.colour = Black;
        }

        shape
    }

    pub fn copy(&self, other: &Self) -> Self {
        self.copy_not_colour(other, Black)
    }

    // Must be compatable
    pub fn copy_not_colour(&self, other: &Self, nc: Colour) -> Self {
        let (g, mut shape) = if self.size() <= other.size() {
            (self, other.clone())
        } else {
            (other, self.clone())
        };

        for row in 0 .. g.cells.rows {
            for col in 0 .. g.cells.columns {

                if g.cells[(row, col)].colour != nc {
                    shape.cells[(row, col)].row = g.cells[(row, col)].row;
                    shape.cells[(row, col)].col = g.cells[(row, col)].col;
                    shape.cells[(row, col)].colour = g.cells[(row, col)].colour;
                }
            }
        }

        shape
    }

    pub fn get_joined(&self) -> Shapes {
        let mut shape = self.clone();

        for ((r, c), cell) in self.cells.items() {
            // Must be internal
            if cell.colour != Black && r != 0 && c != 0 && r != self.cells.rows - 1 && c != self.cells.columns - 1 {
                if self.cells[(r+1,c)].colour == Black && self.cells[(r-1,c)].colour == Black || self.cells[(r,c+1)].colour == Black && self.cells[(r,c-1)].colour == Black {
                    shape.cells[(r,c)].colour = Black;
                };
            }
        }

        shape.to_shapes()
    }

    pub fn fill_centre_mut(&mut self, colour: Colour) {
        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                if r != 0 && c != 0 && r != self.cells.rows - 1 && c != self.cells.columns - 1 {
                    self.cells[(r,c)].colour = colour;
                }
            }
        }
    }

    pub fn mid_pixel(&self) -> (usize, usize) {
        for (r, c) in self.cells.keys() {
            if self.cells[(r,c)].colour != Black && r != 0 && c != 0 && r != self.cells.rows - 1 && c != self.cells.columns - 1 && self.cells[(r-1,c)].colour != Black && self.cells[(r,c-1)].colour != Black && self.cells[(r+1,c)].colour != Black && self.cells[(r,c+1)].colour != Black {
                return (r, c);
            }
        }

        return (9, 0)
    }

    pub fn swap_colours(&mut self, cm: &BTreeMap<Colour, Colour>) {
        for cells in self.cells.values_mut() {
            cells.colour = cells.colour + ToBlack;
        }

        for cells in self.cells.values_mut() {
            if let Some(colour) = cm.get(&cells.colour.to_base()) {
                cells.colour = *colour;
            }
        }
    }

    // fix 0e671a1a
    pub fn adjacent_to_pixel(&self, grid: &Grid) -> Vec<Direction> {
        let r = self.orow;
        let c = self.ocol;
        let mut dir: Vec<Direction> = Vec::new();

        if !self.is_pixel() || r == 0 || c == 0 || r == self.cells.rows - 1 || c == self.cells.columns - 1 {
            return dir;
        }

        if grid.cells[(r,c)].colour != Black {
            if grid.cells[(r-1,c)].colour != Black { dir.push(Up) };
            if grid.cells[(r+1,c)].colour != Black { dir.push(Down) };
            if grid.cells[(r,c-1)].colour != Black { dir.push(Left) };
            if grid.cells[(r,c+1)].colour != Black { dir.push(Right) };
        }

        dir
    }

    /* Needs backtracking
    // first shape must be largest
    pub fn align(&self, other: &Self) -> Self {
        let mut r = other.orow;
        let mut c = other.ocol;

        let mut shape = self.translate_absolute(other.orow, other.ocol);

        shape.recolour_mut(self.colour, other.colour);

        shape
    }
    */

/*
    pub fn cell_category_bg(cells: &Matrix<Cell>, bg: Colour) -> Matrix<Cell> {
        let mut m = cells.clone();

        // Special cases Single Pixel, H & V Lines
        if cells.rows == 1 && cells.columns == 1 {
            m[(0,0)].cat = Single;

            return m;
        } else if cells.rows == 1 {
            for i in 0 .. m.columns {
                m[(0,i)].cat = LineTB;
            }

            return m;
        } else if cells.columns == 1 {
            for i in 0 .. m.rows {
                m[(i,0)].cat = LineLR;
            }

            return m;
        }

        for ((rw, cl), c) in m.items_mut() {
            if c.colcheckerour == bg { continue; }

            let lr = cells.rows - 1;
            let lc = cells.columns - 1;

            let tlc = rw == 0 && cl == 0;
            let blc = rw == lr && cl == 0;
            let trc = rw == 0 && cl == lc;
            let brc = rw == lr && cl == lc;
            
            // Special case corners
            if tlc {
                c.cat = CornerTL;
                continue;
            } else if blc {
                c.cat = CornerBL;
                continue;
            } else if trc {
                c.cat = CornerTR;
                continue;
            } else if brc {
                c.cat = CornerBR;
                continue;
            }

            // Surrounding cells
            let (tl, tt, tr, tb, br, bb, bl, bt) = if rw == 0 {
                (bg, bg, bg, cells[(rw,cl+1)].colour, cells[(rw+1,cl+1)].colour, cells[(rw+1,cl)].colour, cells[(rw+1,cl-1)].colour, cells[(rw,cl-1)].colour)
            } else if cl == 0 {
                (bg, cells[(rw-1,cl)].colour, cells[(rw-1,cl+1)].colour, cells[(rw,cl+1)].colour, cells[(rw+1,cl+1)].colour, cells[(rw+1,cl)].colour, bg, bg)
            } else if rw == lr {
                (cells[(rw-1,cl-1)].colour, cells[(rw-1,cl)].colour, cells[(rw-1,cl+1)].colour, cells[(rw,cl+1)].colour, bg, bg, bg, cells[(rw,cl-1)].colour)
            } else if cl == lc {
                (cells[(rw-1,cl-1)].colour, cells[(rw-1,cl)].colour, bg, bg, bg, cells[(rw+1,cl)].colour, cells[(rw+1,cl-1)].colour, cells[(rw,cl-1)].colour)
            } else {    // Somewhere inside
                (cells[(rw-1,cl-1)].colour, cells[(rw-1,cl)].colour, cells[(rw-1,cl+1)].colour, cells[(rw,cl+1)].colour, cells[(rw+1,cl+1)].colour, cells[(rw+1,cl)].colour, cells[(rw+1,cl-1)].colour, cells[(rw,cl-1)].colour)
            };

            c.cat = if tl == bg && tt == bg && bt == bg {
                InternalCornerTL
            } else if bb == bg && bl == bg && bt == bg {
                InternalCornerBL
            } else if tt == bg && tr == bg && tb == bg {
                InternalCornerTR
            } else if tb == bg && br == bg && bb == bg {
                InternalCornerBR
                    /*
            } else if tl == bg && tt == bg && bt == bg {
                InsideCornerTL
            } else if bb == bg && bl == bg && bt == bg {
                InsideCornerBL
            } else if tt == bg && tr == bg && tb == bg {
                InsideCornerTR
            } else if tb == bg && br == bg && bb == bg {
                InsideCornerBR
                    */
            } else {
                Unknown
            };

            /*
            c.cat = 
                if rw == 0 {            // Left Edge
                    match (tl, tt, tr, tb, br, bb, bl, bt) {
                        (false, true, _, true, _, true, false, false) => EdgeL,
                        (false, true, _, true, _, false, false, false) => InternalCornerBL,
                        (false, true, _, false, _, true, false, false) => HollowEdgeL,
                        (false, true, _, false, _, false, false, false) => BottomEdgeL,
                        (false, false, _, true, _, true, false, false) => InternalCornerTL,
                        (false, false, _, true, _, false, false, false) => PointL,
                        (false, false, _, false, _, true, false, false) => TopEdgeL,
                        (false, false, _, false, _, false, false, false) => SingleL,
                        _ => todo!(),
                    }
                } else if cl == 0 {     //Top Edge
                    match (tl, tt, tr, tb, br, bb, bl, bt) {
                        (false, false, false, true, _, true, _, true) => EdgeT,
                        (false, false, false, true, _, true, _, false) => InternalCornerTL,
                        (false, false, false, true, _, false, _, true) => HollowEdgeT,
                        (false, false, false, true, _, false, _, false) => LeftEdgeT,
                        (false, false, false, false, _, true, _, true) => InternalCornerTR,
                        (false, false, false, false, _, true, _, false) => PointT,
                        (false, false, false, false, _, false, _, true) => RightEdgeT,
                        (false, false, false, false, _, false, _, false) => SingleT,
                        _ => todo!(),
                    }
                } else {
                    BG
                };
            */
        }
//println!("{m:?}");

        m
    }
*/
}

#[derive(Debug, Clone, PartialEq)]
pub struct Shapes {
    pub nrows: usize,
    pub ncols: usize,
    pub colour: Colour,
    pub shapes: Vec<Shape>,
    //pub cats: BTreeSet<ShapeCategory>,
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
        Self { nrows: 0, ncols: 0, colour: NoColour, shapes: Vec::new() }
    }

    pub const fn trivial() -> Self {
        const SHAPES: Vec<Shape> = Vec::new();

        Self { nrows: 0, ncols: 0, colour: NoColour, shapes: SHAPES }
    }

    pub fn new_sized(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, colour: NoColour, shapes: Vec::new() }
    }

    pub fn new_given(nrows: usize, ncols: usize, shapes: &Vec<Shape>) -> Self {
        Self { nrows, ncols, colour: Self::find_colour(shapes), shapes: shapes.to_vec() }
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

    pub fn new_shapes_sized(nrows: usize, ncols: usize, shapes: &[Shape]) -> Self {
        let mut new_shapes = Self::new_sized(nrows, ncols);
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

    pub fn clone_base(&self) -> Self {
        let mut this = Self::new_sized(self.nrows, self.ncols);

        this.colour = self.colour;

        this
    }

    pub fn merge_mut(&mut self, other: &Self) {
        for s in &other.shapes {
            self.shapes.push(s.clone());
        }
    }

    /*
       TODO 3490cc26
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct Pos(pub usize, pub usize);
pub struct Successor {
    pub pos: Pos,
    pub cost: usize,
}
impl PartialEq<(Pos, usize)> for Successor {
    fn eq(&self, other: &(Pos, usize)) -> bool {
        self.pos == other.0 && self.cost == other.1
    }
}

    // Straight line nearest successors
    pub fn get_sln_successors(&self, r: usize, c: usize) -> Vec<Successor> {
        Vec
    }
    */

    // Expensive : fd096ab6
    pub fn join_shapes_same_colour(&self) -> Self {
        let mut done: Vec<Shape> = Vec::new();
        let mut shapes = self.clone_base();

        for s1 in &self.shapes {
            for s2 in &self.shapes {
                if s1 == s2 {
                    continue;
                }
                if s1.equals(&s2) == Same {
                    let orow = s1.orow.min(s2.orow);
                    let ocol = s1.ocol.min(s2.ocol);
                    let rdist = s1.orow.max(s2.orow) - orow;
                    let cdist = s1.ocol.max(s2.ocol) - ocol;
                    let rmax = (s1.orow + s1.cells.rows).max(s2.orow + s2.cells.rows);
                    let cmax = (s1.ocol + s1.cells.columns).max(s2.ocol + s2.cells.columns);
                    let rlen = rmax - orow;
                    let clen = cmax - ocol;
                    let mut ns = Shape::new_sized_coloured_position(orow, ocol, rlen, clen, Black);

                    // assumes position of s1 <= s2
                    for ((r, c), cell) in s1.cells.items() {
                        if cell.colour != Black {
                            ns.cells[(r,c)].colour = cell.colour;
                        }
                    }
                    for ((r, c), cell) in s2.cells.items() {
                        if cell.colour != Black {
                            ns.cells[(r + rdist,c + cdist)].colour = cell.colour;
                        }
                    }

                    if !done.contains(&ns) {
                        shapes.shapes.push(ns.clone());
                    }
                    done.push(s1.clone());
                    done.push(ns);
                    continue;
                }
            }
            if !done.contains(s1) {
                shapes.shapes.push(s1.clone());
            }
        }

        shapes
    }

    // TODO: doesn't work yet: 60a26a3e and others
    pub fn find_straight_pairs(&self) -> BTreeMap<Shape,Shape> {
        let mut ss: BTreeMap<Shape, Shape> = BTreeMap::new();

        for s in self.shapes.iter() {
            let mut min_r = usize::MAX;
            let mut min_c = usize::MAX;
            let mut ns_r = Shape::trivial();
            let mut ns_c = Shape::trivial();

            for si in self.shapes.iter() {
                if s != si {
                   if s.orow == si.orow {
                       let p = s.ocol.max(si.ocol) - s.ocol.min(si.ocol);
println!("Row {} => {} {} {} {}", s.orow, s.ocol, si.ocol, p, min_c);

                       if min_c == usize::MAX || p < min_c {
                           if let Some(s2) = ss.get(&si) {
println!("##1##");
si.show_summary();
s2.show_summary();
println!("##1##");
                           }
                           if let Some(s2) = ss.get(&s) {
println!("##2##");
s.show_summary();
s2.show_summary();
println!("##2##");
                           }
                           min_c = p;
                           ns_r = si.clone();
                       }
                   }
                   if s.ocol == si.ocol {
println!("Col {} => {} {}", s.orow, s.ocol, si.ocol);
                       let p = s.orow.max(si.orow) - s.orow.min(si.orow);

                       if min_r == usize::MAX || p < min_r {
                           min_r = p;
                           ns_c = si.clone();
                       }
                   }
                }
            }

            if ns_r != Shape::trivial() {
                if s.orow < ns_r.orow {
                    ss.insert(s.clone(), ns_r);
                } else {
                    ss.insert(ns_r, s.clone());
                }
            }
            if ns_c != Shape::trivial() {
                if s.ocol < ns_c.ocol {
                    ss.insert(s.clone(), ns_c);
                } else {
                    ss.insert(ns_c, s.clone());
                }
            }
        }

        ss
    }

    pub fn group_containers(&self) -> BTreeMap<Shape, Vec<Shape>> {
        let mut ss: BTreeMap<Shape, Vec<Shape>> = BTreeMap::new();

        for s in self.shapes.iter() {
            for si in self.shapes.iter() {
                if s != si && s.container(si) {
                   ss.entry(s.clone()).and_modify(|v| v.push(si.clone())).or_insert(vec![si.clone()]);
                }
            }
        }

        ss
    }

    pub fn coords_to_shape(&self) -> BTreeMap<(usize, usize), Shape> {
        let mut cts: BTreeMap<(usize, usize), Shape> = BTreeMap::new();

        for s in self.shapes.iter() {
            cts.insert((s.orow, s.ocol), s.clone());
        }

        cts
    }

    pub fn centre_of(&self) -> BTreeMap<(usize, usize), Shape> {
        let mut ss: BTreeMap<(usize, usize), Shape> = BTreeMap::new();

        for s in self.shapes.iter() {
            ss.insert(s.centre_of(), s.clone());
        }

        ss
    }

    pub fn border_gravity(&self) -> BTreeMap<Colour, Direction> {
        let mut ss: BTreeMap<Colour, Direction> = BTreeMap::new();

        for s in self.shapes.iter() {
            for si in self.shapes.iter() {
                if s != si && s.container(si) {
                   if s.orow == si.orow {
                       ss.insert(s.colour, Up);
                   } else if s.ocol == si.ocol {
                       ss.insert(s.colour, Left);
                   } else if s.cells.rows == si.orow + si.cells.rows {
                       ss.insert(s.colour, Down);
                   } else if s.cells.columns == si.ocol + si.cells.columns {
                       ss.insert(s.colour, Right);
                   }
                }
            }
        }

        ss
    }

    pub fn in_range(&self, rc: usize, vertical: bool) -> bool {
        for s in self.shapes.iter() {
            if vertical && s.ocol <= rc && s.ocol + s.cells.columns > rc ||
               !vertical && s.orow <= rc && s.orow + s.cells.rows > rc {
                   return true;
            }
        }

        false
    }

    pub fn full_shapes(&self) -> Vec<Shape> {
        let mut shapes: Vec<Shape> = Vec::new();

        for s in self.shapes.iter() {
            if s.is_full() {
                shapes.push(s.clone());
            }
        }

        shapes
    }

    pub fn full_shapes_sq(&self) -> Vec<Shape> {
        let mut shapes: Vec<Shape> = Vec::new();

        for s in self.to_grid().to_shapes_sq().shapes.iter() {
            if s.is_full() {
                shapes.push(s.clone());
            }
        }

        shapes
    }

    pub fn all_shapes(&self) -> Vec<Shape> {
        let mut shapes: Vec<Shape> = Vec::new();

        for s in self.shapes.iter() {
            shapes.push(s.clone());
        }

        shapes
    }

    pub fn contains_shape(&self, shape: &Shape) -> bool {
        for s in self.shapes.iter() {
            if s.equal_shape(shape) {
                return true;
            }
        }

        false
    }

    pub fn contains_origin(shapes: &Vec<&Shape>, r: usize, c: usize) -> bool {
        for s in shapes.iter() {
            if s.orow == r && s.ocol == c {
                return true;
            }
        }

        false
    }

    pub fn all_shapes_sq(&self) -> Vec<Shape> {
        let mut shapes: Vec<Shape> = Vec::new();

        for s in self.to_grid().to_shapes_sq().shapes.iter() {
            shapes.push(s.clone());
        }

        shapes
    }

    pub fn shape_permutations(&self) -> Self {
        let mut shapes = self.clone();

        let trans: Vec<_> = vec![Shape::mirrored_r, Shape::mirrored_c, Shape::rotated_90, Shape::rotated_180, Shape::rotated_270];
        for tr in trans.iter() {
            for s in self.shapes.iter() {
                let ns = &tr(s);

                if !self.contains_shape(&ns) {
                    shapes.shapes.push(ns.clone());
                }
            }
        }

        shapes
    }

    /*
    pub fn chequer(&self, rs: usize, cs: usize) -> Shapes {
        let mut shapes = Shapes::new_sized(self.nrows * rs, self.ncols * cs);

        let mut first = true;
        let mut rp = 0;
        let mut cp = 0;

        for s in self.shapes.iter() {
            let s = s.to_position(rp, cp);

            shapes.shapes.push(s);

            rp = (rp + 1) % rs;

            if !first && rp == 0  {
                cp = (cp + 1) % cs;
            }
            first = false;
        }

        shapes
    }
    */

    pub fn find_shape_colours(&self) -> Vec<Colour> {
        let mut colours = Vec::new();

        for s in self.to_grid().to_shapes_sq().shapes.iter() {
            colours.push(s.colour);
        }

        colours
    }

    pub fn get_pixels(&self) -> Shapes {
        let mut shapes = self.clone_base();

        for s in self.shapes.iter() {
            if s.is_pixel() {
                shapes.shapes.push(s.clone());
            }
        }

        shapes
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

    pub fn has_band(&self) -> (Direction, usize) {
        for s in self.shapes.iter() {
            match s.has_band(self.nrows, self.ncols) {
                (Down, pos) => return (Down, pos),
                (Right, pos) => return (Right, pos),
                _ => (),
            }
        }

        (Other, 0)
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

        if h.iter().filter(|&(_,&v)| v == 1).count() == 1 {
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

    pub fn find_repeats(&self) -> (usize, usize) {
        let r: BTreeSet<usize> = self.shapes.iter().map(|s| s.orow).collect();
        let c: BTreeSet<usize> = self.shapes.iter().map(|s| s.ocol).collect();

        (r.len(), c.len())
    }

    pub fn find_gaps(&self) -> (usize, usize) {
        let r: BTreeSet<usize> = self.shapes.iter().map(|s| s.orow).collect();
        let c: BTreeSet<usize> = self.shapes.iter().map(|s| s.ocol).collect();

        let r: Vec<_> = r.iter().map(|i| i).collect();
        let c: Vec<_> = c.iter().map(|i| i).collect();

        if r.len() < 2 || c.len() < 2 || r[1] - r[0] < 2 || c[1] - c[0] < 2 || r[1] - r[0] < self.shapes[0].cells.rows || c[1] - c[0] < self.shapes[0].cells.columns {
            return (0, 0);
        }

        (r[1] - r[0] - self.shapes[0].cells.rows, c[1] - c[0] - self.shapes[0].cells.columns)
    }

    // TODO: share code
    pub fn anomalous_size(&self) -> Option<usize> {
        let h = self.size_cnt();

        if h.len() <= 1 {
            return None;
        }

        if h.iter().filter(|&(_,&v)| v == 1).count() == 1 {
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
                h.entry(shape.colour).or_default().push(shape.clone());
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

    pub fn cell_colour_cnt(&self) -> BTreeMap<Colour, usize> {
        let mut h: BTreeMap<Colour, usize> = BTreeMap::new();

        for s in self.shapes.iter() {
            for c in s.cells.values() {
                *h.entry(c.colour).or_insert(0) += 1;
            }
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

    pub fn nearest_shape(&self, r: usize, c: usize) -> Shape {
        let mut dist = f32::MAX;
        let mut shape = Shape::trivial();

        for s in self.shapes.iter() {
            let (cr, cc) = s.centre_of_exact();
            let r2_dist = ((cr - r as f32).powi(2) + (cc - c as f32).powi(2)).sqrt();

            if shape == Shape::trivial() {
                shape = s.clone();
                dist = r2_dist;
            } else {
                if r2_dist < dist {
                    shape = s.clone();
                    dist = r2_dist;
                }
            }
        }

        shape
    }

    pub fn full_extent(&self) -> Shape {
        for s in self.shapes.iter() {
            if s.orow == 0 && s.cells.rows == self.nrows || s.ocol == 0 && s.cells.columns == self.ncols {
//s.show();
                return s.clone();
            }
        }

        Shape::trivial()
    }

    pub fn all_corners(&self) -> Vec<(usize, usize)> {
        let (tlr, tlc, brr, brc) = self.corners();

        vec![(tlr, tlc), (tlr, brc), (brr, brc), (brr, tlc)]
    }

    pub fn vec_corners(shapes: &Vec<Shape>) -> (usize, usize, usize, usize) {
        let mut min_r = usize::MAX;
        let mut min_c = usize::MAX;
        let mut max_r = 0;
        let mut max_c = 0;

        for s in shapes.iter() {
            if min_r > s.orow {
                min_r = s.orow;
            }
            if min_c > s.ocol {
                min_c = s.ocol;
            }
            if max_r < s.orow + s.cells.rows {
                max_r = s.orow + s.cells.rows;
            }
            if max_c < s.ocol + s.cells.columns {
                max_c = s.ocol + s.cells.columns;
            }
        }

        if min_r == usize::MAX {
            min_r = 0;
        }
        if min_c == usize::MAX {
            min_c = 0;
        }

        (min_r, min_c, max_r, max_c)
    }

    pub fn corners(&self) -> (usize, usize, usize, usize) {
        Shapes::vec_corners(&self.shapes)
    }

    pub fn origins(shapes: &Vec<&Shape>) -> Vec<(usize, usize)> {
        let mut rvec: Vec<usize> = Vec::new();
        let mut cvec: Vec<usize> = Vec::new();

        for s in shapes.iter() {
            rvec.push(s.orow);
            cvec.push(s.ocol);
        }

        rvec.sort();
        rvec = rvec.unique();
        cvec.sort();
        cvec = cvec.unique();

        if rvec.len() != 2 || cvec.len() != 2 {
            return Vec::new();
        }

        // Clockwise origins
        vec![(rvec[0], cvec[0]), (rvec[0], cvec[1]), (rvec[1], cvec[1]), (rvec[1], cvec[0])]
    }

    pub fn to_shape(&self) -> Shape {
        let (min_r, min_c, max_r, max_c) = self.corners();

        let mut shape = Shape::new_sized_coloured_position(min_r, min_c, max_r - min_r, max_c - min_c, Black);

        shape.colour = Mixed;
        shape.orow = min_r;
        shape.ocol = min_c;

        for s in self.shapes.iter() {
            for c in s.cells.values() {
                shape.cells[(c.row - min_r, c.col - min_c)].row = c.row;
                shape.cells[(c.row - min_r, c.col - min_c)].col = c.col;
                shape.cells[(c.row - min_r, c.col - min_c)].colour = c.colour;
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

    pub fn all(&self) -> Vec<Shape> {
        self.shapes.clone()
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

    pub fn largest_solid(&self) -> Shape {
        let mut shape = Shape::trivial();

        for s in self.shapes.iter() {
            if s.size() >= shape.size() && s.size() == s.pixels() {
                shape = s.clone();
            }
        }

        shape
    }

    pub fn largest_solid_colour(&self, colour: Colour) -> Shape {
        let mut shape = Shape::trivial();

        for s in self.shapes.iter() {
            if s.size() >= shape.size() && s.size() == s.pixels() && s.colour == colour {
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

            h.entry(cnt).or_default().push(s.clone());
        }

        h
    }

    pub fn hollow_cnt_unique(&self) -> Shape {
        let mut shape = Shape::trivial();
        let h = self.hollow_cnt_map();

        for sv in h.values() {
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
            h.entry(s.colour).or_default().push(s.clone());
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

    pub fn pixels_in_shapes(&self, shape: &Shape) -> Vec<Shape> {
        let mut shapes: Vec<Shape> = Vec::new();
        let pixels: Vec<&Shape> = self.shapes.iter()
            .filter(|s| s.is_pixel())
            .collect();
            
        if pixels.is_empty() {
            return shapes;
        }


        for pix in pixels.into_iter() {
            if pix.contained_by(&shape) {
                shapes.push(pix.clone());
            }
        }

        shapes
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
                    for (rc, c) in si.cells.items() {
                        let nrows = si.cells[rc].row;
                        let ncols = si.cells[rc].col;

                        so.cells[(nrows - so.orow, ncols - so.ocol)].colour = c.colour;
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
        let mut pixels = Self::new_sized(self.nrows, self.ncols);
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
        let mut shapes = Self::new_sized(self.nrows, self.ncols);
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
        self.nrows * self.ncols
    }

    pub fn width(&self) -> usize {
        self.ncols
    }

    pub fn height(&self) -> usize {
        self.nrows
    }

    pub fn find_by_colour(&self, colour: Colour) -> Shape {
        for s in self.shapes.iter() {
            if s.colour == colour {
                return s.clone();
            }
        }

        Shape::trivial()
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

                if let Some(mr) = sh.values().max() {
                    if *mr > biggest {
                        biggest = *mr;
                        choice = s;
                    }
                }
            }
        }

        choice.clone()
    }

    pub fn position_pixels(&self) -> Option<(Self, Self)> {
        let mut rp = usize::MAX;
        let mut cp = usize::MAX;
        let mut cellp = NoColour;
        let mut rgap = 0;
        let mut cgap = 0;
        let mut pos: Vec<Shape> = Vec::new();
        let mut shapes: Vec<Shape> = Vec::new();

        for s in &self.shapes {
            if s.size() == 1 {
                if rp == usize::MAX {
                    rp = s.orow;
                    cp = s.ocol;
                    cellp = s.colour;

                    let cell = Cell::new_colour(s.orow, s.ocol, cellp);
                    let cells = Matrix::new(1, 1, cell);
                    pos.push(Shape::new(s.orow, s.ocol, &cells));
                } else if s.colour == cellp {
                    if cp == s.ocol && s.orow > rp {
                        if rgap == 0 {
                            rgap = s.orow - rp;
                        } else if (s.orow - rp) % rgap != 0 {
                            return None;
                        }
                    }
                    if rp == s.orow && s.ocol > cp {
                        if cgap == 0 {
                            cgap = s.ocol - cp;
                        } else if (s.ocol - cp) % cgap != 0 {
                            return None;
                        }
                    }

                    // needs to be a square, so equal gaps
                    if rgap > 0 && rgap != cgap {
                        return None;
                    }

                    let cell = Cell::new_colour(s.orow, s.ocol, cellp);
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

        Some((Shapes::new_shapes_sized(self.nrows, self.ncols, &pos),
             Shapes::new_shapes_sized(self.nrows, self.ncols, &shapes)))
    }

    /*
    pub fn position_centres(&self, positions: &Self) -> Self {
        //if positions.shapes.is_empty() || self.shapes[0].cells.rows <= 1 || (self.shapes[0].cells.rows > 1 && self.shapes[0].cells.rows != self.shapes[0].cells.columns) {
        if positions.shapes.is_empty() || self.shapes[0].cells.rows <= 1 || self.shapes[0].cells.rows != self.shapes[0].cells.columns {
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
            let roffset = if p.orow >= s.orow { offset } else { offset - 1 };
            let coffset = if p.ocol >= s.ocol { offset } else { offset - 1 };

            *s = s.translate_absolute(p.orow + roffset, p.ocol + coffset);
        }
        for s in positions.shapes.iter() {
            nps.shapes.push(s.clone());
        }

//nps.to_grid().show();
        nps
    }

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
                if s1.contained_by(s2) {
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
            let Some((lc, _)) = h.pop_first() else { todo!() };
            let Some((rc, _)) = h.pop_first() else { todo!() };
//println!("{l:?}, {r:?}");

            for ((r, c), cell) in s.clone().cells.items() {
                if cell.colour == lc {
                    s.cells[(r,c)].colour = rc;
                } else {
                    s.cells[(r,c)].colour = lc;
                }
            }
        }

        shapes
    }

    pub fn min_size(&self, sz: usize) -> (usize, usize) {
        let mut mr = usize::MAX;
        let mut mc = usize::MAX;

        for s in self.shapes.iter() {
            if s.size() > sz {
                mr = mr.min(s.height());
                mc = mc.min(s.width());
            }
        }

        (mr, mc)
    }

    pub fn split_size(&self, sz: usize) -> Self {
        let (mr, mc) = self.min_size(sz);
        let mut shapes = self.clone_base();

        for s in self.shapes.iter() {
            if s.size() > sz && s.cells.rows >= mr && s.cells.columns >= mc  {
                if s.height() > mr * 2 {
                    for i in (0 .. s.height() + 1).step_by(mr + 1) {
                        if i + mr < s.height() { 
                            let ss = s.subshape_trim(i, mr + 1, 0, mc + 1);
//ss.show();
                            shapes.shapes.push(ss);
                        }
                    }
                } else if s.width() > mc * 2 {
                    for i in (0 .. s.width() + 1).step_by(mc + 1) {
                        if i + mc < s.width() {
                            let ss = s.subshape_trim(0, mr + 1, i, mc + 1);
//ss.show();
                            shapes.shapes.push(ss);
                        }
                    }
                } else {
                    shapes.shapes.push(s.clone());
                }
            }
        }

        shapes
    }

    pub fn majority_cell(&self) -> Shape {

        if self.shapes.len() < 2 {
            return Shape::trivial();
        }

        let mut shape = self.shapes[0].to_origin();
        let mut cnts: BTreeMap<Colour, usize> = BTreeMap::new();

        for r in 0 .. shape.cells.rows {
            for c in 0 .. shape.cells.columns {
                for s in self.shapes.iter() {
                    if s.cells.rows <= r || s.cells.columns <= c {
                        return Shape::trivial();
                    }

                    *cnts.entry(s.cells[(r,c)].colour).or_insert(0) += 1;
                }

                let mx = cnts.iter().map(|(k,v)| (v, k)).max();

                if let Some((_, colour)) = mx {
                    shape.cells[(r,c)].colour = *colour;
                }

                cnts.clear();
            }
        }

        shape
    }

    pub fn ignore_pixels(&self) -> Self {
        let mut ss = self.clone_base();

        for s in &self.shapes {
            if s.size() > 1 {
                ss.shapes.push(s.clone());
            }
        }

        ss
    }

    pub fn de_subshape(&self) -> Shapes {
        let mut ss = self.clone_base();

        for s in &self.shapes {
            s.de_subshape(&mut ss);
        }

        ss
    }

    pub fn diff(&self, other: &Self) -> Option<Vec<Option<Shape>>> {
        if self.nrows != other.nrows || self.ncols != other.ncols || self.shapes.len() != other.shapes.len() {
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
        if shape.orow + shape.cells.rows > self.nrows {
            self.nrows = shape.orow + shape.cells.rows;
        }
        if shape.ocol + shape.cells.columns > self.ncols {
            self.ncols = shape.ocol + shape.cells.columns;
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

        h.iter().filter(|&(_, &cnt)| cnt == 1).map(|(s, _)| s.clone()).collect()
    }

    pub fn remove(&mut self, shape: &Shape) {
        let index = self.shapes.iter().position(|r| *r == *shape);

        if let Some(index) = index {
            self.shapes.remove(index);
        }
    }

    pub fn hollow_shapes(&self) -> Shapes {
        let mut new_shapes = Self::new_sized(self.nrows, self.ncols);

        for s in self.shapes.iter() {
            if s.size() != self.size() && s.hollow() {
                let ss = s.recolour(Blue, Teal);

                new_shapes.add(&ss);
            }
        }

        new_shapes
    }

    pub fn merge_replace_shapes(&self, other: &Self) -> Self {
        let mut new_shapes = Self::new_sized(self.nrows, self.ncols);

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
        println!("--- {} / {} ---", self.nrows, self.ncols);
    }

    pub fn show(&self) {
        for s in &self.shapes {
            s.show();
            println!();
        }
        println!("--- {} / {} ---", self.nrows, self.ncols);
    }

    pub fn show_full(&self) {
        for s in &self.shapes {
            s.show_full();
            println!();
        }
        println!("--- {} / {} ---", self.nrows, self.ncols);
    }

    /*
    pub fn add_in(&mut self) -> Self {
        let mut holes = Self::new();

        for s in &self.shapes {
            if s.is_hollow() {
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
        let mut trimmed = self.clone_base();

        for s in self.shapes.iter() {
            if s.orow + s.cells.rows > self.nrows || s.ocol + s.cells.columns > self.ncols {
                let r = self.nrows.min(s.orow + s.cells.rows);
                let c = self.ncols.min(s.ocol + s.cells.columns);

                if r < s.orow || c < s.ocol {
                    return Shapes::trivial();
                }

                if let Ok(mat) = s.cells.slice(0 .. r - s.orow, 0 .. c - s.ocol) {
                    trimmed.shapes.push(Shape::new_cells(&mat));
                } 
            } else {
                trimmed.shapes.push(s.clone());
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
        if self.nrows == 0 || self.ncols == 0 || self.nrows > 100 || self.ncols > 100 {
            return Grid::trivial();
        }

        let mut grid = Grid::new(self.nrows, self.ncols, colour);

        grid.colour = self.colour;

        for shape in &self.shapes {
            for c in shape.cells.values() {
                if  grid.cells.rows <= c.row || grid.cells.columns <= c.col {
                    break;
                }
                if c.colour == Transparent {
                    continue;
                }
                if !transparent || c.colour != colour {
                    grid.cells[(c.row, c.col)].colour = c.colour;
                }
            }
        }

        grid
    }

    pub fn to_json(&self) -> String {
        let mut grid: Vec<Vec<usize>> = vec![vec![0; self.nrows]; self.ncols];

        for shape in &self.shapes {
            for ((r, c), cell) in shape.cells.items() {
                grid[r][c] = cell.colour.to_usize();
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

    pub fn shape_counts(&self) -> BTreeMap<u32, usize> {
        let mut sc: BTreeMap<u32, usize> = BTreeMap::new();

        for s in self.shapes.iter() {
            let sid = Shape::sid(&s.cells, true);

            *sc.entry(sid).or_insert(0) += 1;
        }

        sc
    }

    // TODO
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

    // bool is true == Row axis, false = Column axis
    pub fn striped_r(&self) -> Colour {
        NoColour
    }

    pub fn striped_c(&self) -> Colour {
        NoColour
    }

    pub fn shrink(&self) -> Self {
         let mut shapes: Vec<Shape> = Vec::new();

         for s in self.shapes.iter() {
             shapes.push(s.shrink());
         }

         Self::new_shapes(&shapes)
    }

    pub fn fill_missing(&self, to: Colour) -> Self {
        let mut shapes = Shapes::new_sized(self.nrows, self.ncols);

        for shape in self.shapes.iter() {
            shapes.add(&shape.fill_missing(to));
        }

        shapes
    }

    pub fn pixel_dir(&self, grid: &Grid) -> BTreeMap<Shape, Vec<Direction>> {
        let mut cd: BTreeMap<Shape, Vec<Direction>> = BTreeMap::new();

        for s in self.shapes.iter() {
            let adj = s.adjacent_to_pixel(grid);

            if !adj.is_empty() {
                cd.insert(s.clone(), adj);
            }
        }

        cd
    }

    /*
    pub fn categorise_shapes(&mut self) {
        let the_shapes = &mut self.shapes;

        if the_shapes.is_empty() {
            return;
        }
        if the_shapes.len() == 1 {
            self.cats.insert(ShapeCategory::SingleShape);
        }
        if the_shapes.len() > 1 {
            self.cats.insert(ShapeCategory::ManyShapes);
        }

        for shape in the_shapes.iter_mut() {
            if self.cats.is_empty() {
                self.cats = shape.cats.clone();
            } else {
                self.cats = self.cats.union(&shape.cats).cloned().collect();
            }
        }
    }
    */

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

    pub fn has_mirror_r(&self) -> Shape {

        for s in &self.shapes  {
            if s.is_mirror_r() {
                return s.clone();
            }
        }

        Shape::trivial()
    }

    pub fn has_mirror_c(&self) -> Shape {
        for s in &self.shapes  {
            if s.is_mirror_c() {
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

    /*
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

                    let mirrored_r = shape1.is_mirrored_r(shape2);
                    let mirrored_c = shape1.is_mirrored_c(shape2);

                    if mirrored_r && mirrored_c {
                        shape1.io_edges.insert(ShapeEdgeCategory::Symmetric);
                    } else {
                        if mirrored_r {
                            shape1.io_edges.insert(ShapeEdgeCategory::MirroredRow);
                        }
                        if mirrored_c {
                            shape1.io_edges.insert(ShapeEdgeCategory::MirroredCol);
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
    */
}

