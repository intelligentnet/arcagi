use std::cmp::Ordering;
use std::collections::BTreeMap;
use pathfinding::prelude::Matrix;
use crate::cats::Colour::*;
use crate::cats::Direction::*;
use crate::cats::Transformation::*;
use crate::cats::Transformation;
use crate::cats::{Colour, Direction};
use crate::utils::*;
use crate::cell::*;
use crate::shape::*;

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub struct Grid {
    pub colour: Colour,
    pub cells: Matrix<Cell>,
    //pub cats: BTreeSet<ShapeCategory>,
}

impl Ord for Grid {
    fn cmp(&self, other: &Self) -> Ordering {
        (&self.to_json(), &self.colour).cmp(&(&other.to_json(), &other.colour))
        //self.colour.cmp(&other.colour)
    }
}

impl PartialOrd for Grid {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Grid {
    pub fn new(rows: usize, cols: usize, colour: Colour) -> Self {
        if cols > 100 || rows > 100 {
            return Self::trivial();
        }

        let cells: Matrix<Cell> = Matrix::from_fn(rows, cols, |(x, y)| Cell::new_colour(x, y, colour));

        Self { colour, cells }
    }

    pub fn trivial() -> Self {
        let cells: Matrix<Cell> = Matrix::new(0, 0, Cell::new_empty());

        Self { colour: Black, cells }
    }

    pub fn dummy() -> Self {
        Self::new(2, 2, Black)
    }

    pub fn is_trivial(&self) -> bool {
        self.cells.rows == 0 && self.cells.columns == 0 && self.colour == Black
    }

    pub fn is_dummy(&self) -> bool {
        self.cells.rows == 2 && self.cells.columns == 2 && self.colour == Black
    }

    pub fn new_from_matrix(cells: &Matrix<Cell>) -> Self {
        let mut colour = NoColour;

        for c in cells.values() {
            if c.colour != Black {
                if colour == NoColour {
                    colour = c.colour;
                } else if colour != c.colour {
                    colour = Mixed;
                    break;
                }
            }
        }

        Self { colour, cells: cells.clone() }
    }

    /*
    pub fn new_from_sized_matrix(x: usize, y: usize, shape: Matrix<Cell>) -> Self {
        let mut cells: Matrix<Cell> = Matrix::from_fn(x, y, |(_, _)| Cell::new_empty());

        for ((x, y), c) in shape.items() {
            cells[(x, y)].x = x;
            cells[(x, y)].y = y;
            cells[(x, y)].colour = c.colour;
        }

        Self { cells }
    }
    */

    pub fn rip(&self, colour: Colour) -> Self {
        let mut starts = Shapes::new_sized(self.cells.rows, self.cells.columns);
        let mut top = true;
        let mut left = true;
        let mut updown = true;

        for s in self.to_shapes().shapes.iter() {
            if s.orow > 0 && s.ocol > 0 {
                top =  s.orow < self.cells.rows / 2;
                left = s.ocol < self.cells.columns / 2;
            } else {
                updown = s.ocol == 0;
            }
        }

        let dir = if top && !left && updown {
            Up
        } else if !top && left && !updown {
            Left
        } else if !top && !left && !updown {
            Right
        } else {
            Down
        };

        let mut grid = match dir {
            Up => self.clone(),
            Right => self.rot_rect_270(),
            Down => self.rot_rect_180(),
            Left => self.rot_rect_90(),
            _ => todo!(),
        };

        for s in grid.to_shapes().shapes.iter() {
            if s.orow > 0 && s.ocol > 0 || s.size() < self.cells.rows && s.size() < self.cells.columns && (s.orow == 0 || s.ocol == 0) {
                starts.shapes.push(s.clone());
            }
        }

        for c in 0 .. grid.cells.columns {
            if starts.in_range(c, true) {
                let mut nc = NoColour;
                let mut nc2 = NoColour;
                let mut nc3 = NoColour;
                let mut cnt = 0;

                for r in 0 .. grid.cells.rows {
                    if starts.in_range(r, false) {
                        if grid.cells[(r,c)].colour != Black {
                            nc = grid.cells[(r,c)].colour;
                            grid.cells[(r,c)].colour = colour;
                        } else if nc != NoColour {
                            grid.cells[(r,c)].colour = nc;
                        }
                    } else if nc != NoColour && nc2 == NoColour {
                        if grid.cells[(r,c)].colour != Black {
                            nc2 = grid.cells[(r,c)].colour;
                        } else  {
                            grid.cells[(r,c)].colour = nc;
                        }
                    } else if nc2 != NoColour {
                        if grid.cells[(r,c)].colour != Black {
                            nc3 = grid.cells[(r,c)].colour;
                            cnt += 1;
                        }
                        if r >= grid.cells.rows - cnt {
                            grid.cells[(r,c)].colour = nc3;
                        } else {
                            grid.cells[(r,c)].colour = nc2;
                        }
                    }
                }
            }
        }
        
        match dir {
            Up => grid.clone(),
            Right => grid.rot_rect_90(),
            Down => grid.rot_rect_180(),
            Left => grid.rot_rect_270(),
            _ => todo!(),
        }
    }

    pub fn duplicate(grid: &[Vec<usize>]) -> Self {
        let x: usize = grid.len();
        let y: usize = grid[0].len();
        let mut colour = NoColour;
        let cells: Matrix<Cell> = Matrix::from_fn(x, y, |(x, y)| Cell::new(x, y, grid[x][y]));

        for c in cells.values() {
            if c.colour != Black {
                if colour == NoColour {
                    colour = c.colour;
                } else if colour != c.colour {
                    colour = Mixed;
                    break;
                }
            }
        }

        Self { colour, cells }
    }

    pub fn is_empty(&self) -> bool {
        for c in self.cells.values() {
            if c.colour != Black {
                return false
            }
        }

        true
    }

    pub fn to_origin(&self) -> Self {
        let mut grid = self.clone();

        grid.to_origin_mut();

        grid
    }

    pub fn to_origin_mut(&mut self) {
        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                self.cells[(r,c)].row = r;
                self.cells[(r,c)].col = c;
            }
        }
    }

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

    pub fn find_colour(&self, colour: Colour) -> Vec<Cell> {
        self.cells.values().filter(|c| c.colour == colour).cloned().collect()
    }

    pub fn in_to_squared_out(&self) -> Self {
        let rows = self.cells.rows;
        let cols = self.cells.columns;

        if rows != cols {
            return Self::trivial();
        }

        let mut g = Self::new(rows * rows, cols * cols, Black);

        for r in (0 .. rows * rows).step_by(rows) {
            for c in (0 .. cols * cols).step_by(cols) {
                g.copy_to_position_mut(self, r, c);
            }
        }

        g
    }

    pub fn find_all_colours(&self) -> BTreeMap<Colour, usize> {
        let mut c: BTreeMap<Colour, usize> = BTreeMap::new();

        for cell in self.cells.values() {
            *c.entry(cell.colour).or_insert(0) += 1;
        }

        c
    }

    pub fn find_min_colour(&self) -> Colour {
        let cols = self.find_all_colours();
        let col = cols.iter()
            .filter(|&(&col, _)| col != Black)
            .min_by(|col, c| col.1.cmp(c.1))
            .map(|(col, _)| col);

        if let Some(col) = col {
            *col
        } else {
            NoColour
        }
    }

    pub fn find_max_colour(&self) -> Colour {
        let cols = self.find_all_colours();
        let col = cols.iter()
            .filter(|&(&col, _)| col != Black)
            .max_by(|col, c| col.1.cmp(c.1))
            .map(|(col, _)| col);

        if let Some(col) = col {
            *col
        } else {
            NoColour
        }
    }

    // TODO crap improve
    pub fn stretch_down(&self) -> Self {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for y in 0 .. self.cells.columns {
            let mut colour = NoColour;

            for x in 1 .. self.cells.rows {
                if self.cells[(x - 1,y)].colour != Black {
                    if colour == NoColour {
                        colour = self.cells[(x - 1,y)].colour;
                        m[(x - 1,y)].colour = colour;
                    } else {
                        continue;
                    }
                }
                m[(x,y)].row = x;
                m[(x,y)].col = y;
                m[(x,y)].colour = if colour == NoColour {
                    self.cells[(x,y)].colour
                } else {
                    colour
                };
            }
        }

        Self::new_from_matrix(&m)
    }

    pub fn is_diag_origin(&self) -> bool {
        if !self.is_square() || self.colour == Mixed {
            return false;
        }

        for ((r, c), cell) in self.cells.items() {
            if r != c && cell.colour != Black {
                return false;
            }
        }

        true
    }

    pub fn is_diag_not_origin(&self) -> bool {
        self.rot_90().is_diag_origin()
    }

    /*
    pub fn stretch_up(&self) -> Self {
        self.mirrored_x().stretch_up().mirrored_x()
    }
    */

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
            if self.cells[(r,c)].colour == colour {
               m[(r,c)].colour = values[c];
            }
            if self.cells.rows - r > counts[c] {
               m[(r,c)].colour = colour;
            }
        }

        Self::new_from_matrix(&m)
    }

    pub fn gravity_up(&self) -> Self {
        self.mirrored_rows().gravity_down().mirrored_rows()
    }

    pub fn gravity_right(&self) -> Self {
        self.rot_rect_90().gravity_down().rot_rect_270()
    }

    pub fn gravity_left(&self) -> Self {
        self.rot_rect_270().gravity_down().rot_rect_90()
    }

    pub fn move_down(&self) -> Self {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for r in 1 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                m[(r,c)].row = r;
                m[(r,c)].col = c;
                m[(r,c)].colour = self.cells[(r - 1,c)].colour;
            }
        }

        Self::new_from_matrix(&m)
    }

    /*
     * move_up
     * stretch down
     * stretch_up
     */

    pub fn move_up(&self) -> Self {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for r in 1 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                m[(r,c)].row = r;
                m[(r,c)].col = c;
                m[(r - 1,c)].colour = self.cells[(r,c)].colour;
            }
        }

        Self::new_from_matrix(&m)
    }

    pub fn move_right(&self) -> Self {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for r in 0 .. self.cells.rows {
            for c in 1 .. self.cells.columns {
                m[(r,c)].row = r;
                m[(r,c)].col = c;
                m[(r,c)].colour = self.cells[(r,c - 1)].colour;
            }
        }

        Self::new_from_matrix(&m)
    }

    pub fn move_left(&self) -> Self {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for r in 0 .. self.cells.rows {
            for c in 1 .. self.cells.columns {
                m[(r,c)].row = r;
                m[(r,c)].col = c;
                m[(r,c - 1)].colour = self.cells[(r,c)].colour;
            }
        }

        Self::new_from_matrix(&m)
    }

    pub fn trim(&self, r: usize, c: usize) -> Self {
        if self.cells.rows <= r && self.cells.columns <= c {
            return self.clone();
        }

        let mut grid = Self::new(r, c, Black);

        for ((r, c), cell) in self.cells.items() {
            if r < grid.cells.rows && c < grid.cells.columns {
                grid.cells[(r,c)].row = cell.row;
                grid.cells[(r,c)].col = cell.col;
                grid.cells[(r,c)].colour = cell.colour;
            }
        }

        grid
    }

    pub fn free_border(&self, dir: Direction) -> bool {
        match dir {
            Up => {
                for i in 0 .. self.cells.columns {
                    if self.cells[(0, i)].colour != Black {
                        return false;
                    }
                }
                true
            },
            Down => {
                for i in 0 .. self.cells.columns {
                    if self.cells[(self.cells.rows - 1, i)].colour != Black {
                        return false;
                    }
                }
                true
            },
            Left => {
                for i in 0 .. self.cells.rows {
                    if self.cells[(i, 0)].colour != Black {
                        return false;
                    }
                }
                true
            },
            Right => {
                for i in 0 .. self.cells.rows {
                    if self.cells[(i, self.cells.columns - 1)].colour != Black {
                        return false;
                    }
                }
                true
            },
            _ => false
        }
    }

    pub fn negative_mut(&mut self, colour: Colour) {
        self.negative_dir_mut(colour, vec![]);
    }

    pub fn negative_dir_all_mut(&mut self, colour: Colour) {
        self.negative_dir_mut(colour, vec![Up, Down, Left, Right]);
    }

    pub fn negative_dir_mut(&mut self, colour: Colour, exclude: Vec<Direction>) {
        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                if r == 0 && exclude.contains(&Up) || c == 0 && exclude.contains(&Left) || r == self.cells.rows - 1 && exclude.contains(&Down) || c == self.cells.columns - 1 && exclude.contains(&Right) {
                    continue;
                }

                if self.cells[(r,c)].colour != Black {
                    self.cells[(r,c)].colour = Black;
                } else {
                    self.cells[(r,c)].colour = colour;
                }
            }
        }
    }

    pub fn background_border_mut(&mut self) {
        let (rs, cs) = self.dimensions();

        for r in 0 .. rs {
            self.cells[(r,0)].colour = Black;
            self.cells[(r,cs-1)].colour = Black;
        }

        for c in 0 .. cs {
            self.cells[(0,c)].colour = Black;
            self.cells[(rs-1,c)].colour = Black;
        }
    }

    pub fn row_dividers_mut(&mut self, rs: usize) {
        let mut r = rs + 1;

        while r < self.cells.rows {
            for c in 0 .. self.cells.columns {
                self.cells[(r,c)].colour = Black;
            }

            r += rs + 1;
        }
    }

    pub fn col_dividers_mut(&mut self, cs: usize) {
        let mut c = cs + 1;

        while c < self.cells.columns {
            for r in 0 .. self.cells.rows {
                self.cells[(r,c)].colour = Black;
            }

            c += cs + 1;
        }
    }

    pub fn recolour(&self, from: Colour, to: Colour) -> Self {
        let mut grid = self.clone();

        grid.recolour_mut(from, to);

        grid
    }

    pub fn recolour_mut(&mut self, from: Colour, to: Colour) {
        for c in self.cells.values_mut() {
            //if c.colour == from || from == NoColour {
            //    c.colour = to;
            //}
            if c.colour == from || from == NoColour {
                c.colour = to;
            } else if from == Mixed {
                if c.colour != Black {
                    c.colour = Black;
                } else {
                    c.colour = to;
                }
            }
        }
    }

    pub fn force_recolour(&self, to: Colour) -> Self {
        let mut grid = self.clone();

        grid.colour = to;

        for c in grid.cells.values_mut() {
            c.colour = to;
        }

        grid
    }

    pub fn copy_to_position(&self, grid: &Self, r: usize, c: usize) -> Self {
        let mut i = self.clone();

        i.copy_to_position_mut(grid, r, c);

        i
    }

    pub fn copy_to_position_mut(&mut self, grid: &Self, rp: usize, cp: usize) {
        if rp + grid.cells.rows > self.cells.rows || cp + grid.cells.columns > self.cells.columns {
            return;
        }

        for ((r, c), cell) in grid.cells.items() {
            self.cells[(rp + r, cp + c)] = cell.clone();
        }
    }

    pub fn copy_shape_to_grid_mut(&mut self, shape: &Shape) {
        self.copy_shape_to_grid_position_mut(shape, shape.orow, shape.ocol)
    }

    pub fn copy_shape_to_grid_position_mut(&mut self, shape: &Shape, row: usize, col: usize) {
        for (r, c) in shape.cells.keys() {
            // Clip
            if row+r >= self.cells.rows || col+c >= self.cells.columns {
                continue;
            }

            self.cells[(row+r, col+c)].colour = shape.cells[(r,c)].colour;
        }
    }

    pub fn connect_dots_pairs(&mut self) {
        self.connect_dots_dir(Other, NoColour);
    }

    pub fn connect_dots(&mut self) {
        self.connect_dots_dir(CalcDir, NoColour);
    }

    pub fn connect_dots_colour(&mut self, colour: Colour) {
        self.connect_dots_dir(CalcDir, colour);
    }

    pub fn connect_dots_colour_pairs(&mut self, colour: Colour) {
        self.connect_dots_dir(Other, colour);
    }

    // Not perfect: No start and only works for pixels not shapes
    pub fn connect_dots_dir(&mut self, dir: Direction, col: Colour) {
        let posns = self.cell_colour_posn_map();

        for (colour, vp) in posns.iter() {
            let col = if col == NoColour { *colour } else { col };

            if vp.len() == 1 {
                if dir != Other {
                    let dir = if dir == CalcDir {
                        let (r, c) = vp[0];

                        if r == 0 {
                            Down
                        } else if c == 0 {
                            Right
                        } else if r == self.cells.rows - 1 {
                            Up
                        } else {
                            Left
                        }
                    } else {
                        dir
                    };
                    self.draw_mut(dir, vp[0].0, vp[0].1, *colour);
                }
            } else {
                let mut line: BTreeMap<usize,(Direction, usize, usize, usize)> = BTreeMap::new();
                let mut done: Vec<(Direction, usize, usize, usize)> = Vec::new();

                for (r1, c1) in vp.iter() {
                    for (r2, c2) in vp.iter() {
                        if r1 != r2 && c1 == c2 {
                            let pmin = *r1.min(r2);
                            let pmax = *r1.max(r2);
                            let val = (Vertical, *c1, pmin, pmax);

                            if pmax - pmin > 1 && !done.contains(&val) {
                                line.insert(pmax - pmin, val);
                                done.push(val);
                            }
                        }
                        if r1 == r2 && c1 != c2 {
                            let pmin = *c1.min(c2);
                            let pmax = *c1.max(c2);
                            let val = (Horizontal, *r1, pmin, pmax);

                            if pmax - pmin > 1 && !done.contains(&val) {
                                line.insert(pmax - pmin, val);
                                done.push(val);
                            }
                        }
                    }
                    if let Some((_, (dir, rc, pmin, pmax))) = line.pop_first() {
                        line.clear();

                        match dir {
                            Vertical => {
                                for r in pmin .. pmax {
                                    if self.cells[(r,rc)].colour != *colour {
                                        self.cells[(r,rc)].colour = col;
                                    }
                                }
                            },
                            Horizontal => {
                                for c in pmin .. pmax {
                                    if self.cells[(rc,c)].colour != *colour {
                                        self.cells[(rc,c)].colour = col;
                                    }
                                }
                            },
                            _ => todo!()
                        }
                    }
                }
            }
        }
    }

    pub fn draw_lines_from_shapes(&mut self, shapes: &[Shape], overwrite: bool, hv: bool) {
        let v: Vec<&Shape> = shapes.iter().collect();

        self.draw_lines(&v, overwrite, hv);
    }

    pub fn draw_lines(&mut self, shapes: &[&Shape], overwrite: bool, hv: bool) {
        for (j, s) in shapes.iter().enumerate() {
            for shape in shapes.iter().skip(j) {
                if s != shape {
                    if hv {
                        self.draw_line_row(s, shape, s.colour, false, false);
                    } else {
                        self.draw_line_col(s, shape, s.colour, false, false);
                    }
                }
            }
        }
        for (j, s) in shapes.iter().enumerate() {
            for shape in shapes.iter().skip(j) {
                if s != shape {
                    if hv {
                        self.draw_line_col(s, shape, s.colour, overwrite, false);
                    } else {
                        self.draw_line_row(s, shape, s.colour, overwrite, false);
                    }
                }
            }
        }
    }

    pub fn draw_mut(&mut self, dir: Direction, r: usize, c: usize, colour: Colour) {
        self.draw_bg_mc_term_mut(dir, r, c, colour, Black, false, false);
    }

    pub fn draw_bg_mut(&mut self, dir: Direction, r: usize, c: usize, colour: Colour, bg: Colour) {
        self.draw_bg_mc_term_mut(dir, r, c, colour, bg, false, false);
    }

    pub fn draw_mc_mut(&mut self, dir: Direction, r: usize, c: usize, colour: Colour) {
        self.draw_bg_mc_term_mut(dir, r, c, colour, Black, true, false);
    }

    pub fn draw_term_mut(&mut self, dir: Direction, r: usize, c: usize, colour: Colour) {
        self.draw_bg_mc_term_mut(dir, r, c, colour, Black, false, true);
    }

    pub fn draw_bg_mc_mut(&mut self, dir: Direction, r: usize, c: usize, colour: Colour, bg: Colour) {
        self.draw_bg_mc_term_mut(dir, r, c, colour, bg, true, false)
    }

    pub fn draw_bg_term_mut(&mut self, dir: Direction, r: usize, c: usize, colour: Colour, bg: Colour) {
        self.draw_bg_mc_term_mut(dir, r, c, colour, bg, false, true)
    }

    pub fn draw_bg_mc_term_mut(&mut self, dir: Direction, r: usize, c: usize, colour: Colour, bg: Colour, mc: bool, term: bool) {
        self.draw_bg_mc_term_other_mut(dir, r, c, colour, bg, mc, term, NoColour)
    }

    pub fn draw_bg_mc_term_other_mut(&mut self, dir: Direction, r: usize, c: usize, colour: Colour, bg: Colour, mc: bool, term: bool, other_colour: Colour) {
        fn change_colour(cell_colour: &mut Colour, colour: Colour, bg: Colour, mc: bool, term: bool, other_colour: Colour) -> bool {
            //if term && *cell_colour != bg && *cell_colour != colour {
            //if term && (*cell_colour != bg || *cell_colour != colour) {
            //if term && *cell_colour != bg {
            if term && *cell_colour != bg && *cell_colour != other_colour {
                return true;
            } else if mc && *cell_colour != bg && *cell_colour != colour {
                *cell_colour = colour + ToBlack;
            } else if *cell_colour == bg || *cell_colour == colour {
                *cell_colour = colour;
            }

            false
        }

        match dir {
            Up => {
                for r in 0 ..= r {
                    if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                }
            },
            Down => {
                for r in r .. self.cells.rows {
                    if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                }
            },
            Left => {
                for c in 0 ..= c {
                    if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                }
            },
            Right => {
                for c in c .. self.cells.columns {
                    if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                }
            },
            UpRight => {
                for (r, c) in ((0 ..= r).rev()).zip(c ..= self.cells.columns) {
                    if r < self.cells.rows && c < self.cells.columns {
                        if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                    }
                }
            },
            UpLeft => {
                for (r, c) in ((0 ..= r).rev()).zip((0 ..= c).rev()) {
                    if r < self.cells.rows && c < self.cells.columns {
                        if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                    }
                }
            },
            DownRight => {
                for (r, c) in (r .. self.cells.rows).zip(c .. self.cells.columns) {
                    if r < self.cells.rows && c < self.cells.columns {
                        if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                    }
                }
            },
            DownLeft => {
                for (r, c) in (r .. self.cells.rows).zip((0 ..= c).rev()) {
                    if r < self.cells.rows && c < self.cells.columns {
                        if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                    }
                }
            },
            FromUpRight => {
                for (r, c) in (0 ..= r).rev().zip(c-1 .. self.cells.columns) {
                    if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                }
            },
            FromUpLeft => {
                for (r, c) in (0 ..= r).zip(0 ..= c) {
                    if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                }
            },
            FromDownRight => {
                for (r, c) in (r .. self.cells.rows).zip(c .. self.cells.columns) {
                    if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                }
            },
            FromDownLeft => {
                for (r, c) in (r-1 .. self.cells.rows).zip((0 ..= c).rev()) {
                    if change_colour(&mut self.cells[(r,c)].colour, colour, bg, mc,  term, other_colour) { break; }
                }
            },
            _ => {},
        }
    }

    pub fn skip_to(&self, dir: Direction, r: usize, c: usize) -> (usize, usize) {
        self.skip_to_bg(dir, r, c, Black)
    }

    pub fn skip_to_bg(&self, dir: Direction, r: usize, c: usize, bg: Colour) -> (usize, usize) {
        match dir {
            Up => {
                for r in 0 ..= r {
                    if self.cells[(r,c)].colour == bg { return (r, c); }
                }
            },
            Down => {
                for r in r .. self.cells.rows {
                    if self.cells[(r,c)].colour == bg { return (r, c); }
                }
            },
            Left => {
                for c in 0 ..= c {
                    if self.cells[(r,c)].colour == bg { return (r, c); }
                }
            },
            Right => {
                for c in c .. self.cells.columns {
                    if self.cells[(r,c)].colour == bg { return (r, c); }
                }
            },
            UpRight => {
                for (r, c) in ((0 ..= r).rev()).zip(c ..= self.cells.columns) {
                    if r < self.cells.rows && c < self.cells.columns {
                        if self.cells[(r,c)].colour == bg { return (r, c); }
                    }
                }
            },
            UpLeft => {
                for (r, c) in ((0 ..= r).rev()).zip((0 ..= c).rev()) {
                    if r < self.cells.rows && c < self.cells.columns {
                        if self.cells[(r,c)].colour == bg { return (r, c); }
                    }
                }
            },
            DownRight => {
                for (r, c) in (r .. self.cells.rows).zip(c .. self.cells.columns) {
                    if r < self.cells.rows && c < self.cells.columns {
                        if self.cells[(r,c)].colour == bg { return (r, c); }
                    }
                }
            },
            DownLeft => {
                for (r, c) in (r .. self.cells.rows).zip((0 ..= c).rev()) {
                    if r < self.cells.rows && c < self.cells.columns {
                        if self.cells[(r,c)].colour == bg { return (r, c); }
                    }
                }
            },
            FromUpRight => {
                for (r, c) in (0 ..= r).rev().zip(c-1 .. self.cells.columns) {
                    if self.cells[(r,c)].colour == bg { return (r, c); }
                }
            },
            FromUpLeft => {
                for (r, c) in (0 ..= r).zip(0 ..= c) {
                    if self.cells[(r,c)].colour == bg { return (r, c); }
                }
            },
            FromDownRight => {
                for (r, c) in (r .. self.cells.rows).zip(c .. self.cells.columns) {
                    if self.cells[(r,c)].colour == bg { return (r, c); }
                }
            },
            FromDownLeft => {
                for (r, c) in (r-1 .. self.cells.rows).zip((0 ..= c).rev()) {
                    if self.cells[(r,c)].colour == bg { return (r, c); }
                }
            },
            _ => {},
        }

        (r, c)
    }

    pub fn calc_direction(r1: usize, c1: usize, r2: usize, c2: usize) -> Direction {
        if r1 == r2 && c1 == c2 {
            return Other;
        }

        let l1 = (r1 as isize - r2 as isize).abs() as usize;
        let l2 = (c1 as isize - c2 as isize).abs() as usize;

        if l1 != 0 && l2 != 0 && l1 != l2 {
            return Other;
        }

        if l1 == 0 || l2 == 0 {
            if r1 < r2 { Down }
            else if r1 > r2 { Up }
            else if c1 < c2 { Right }
            else if c1 > c2 { Left }
            else { Other }
        } else if r1 < r2 && c1 < c2 {
            DownRight
        } else if r1 < r2 && c1 > c2 {
            DownLeft
        } else if r1 > r2 && c1 < c2 {
            UpRight
        } else if r1 > r2 && c1 > c2 {
            UpLeft
        } else {
            Other
        }
    }

    pub fn shape_in_line(&self, shape: &Shape) -> Colour {
        let r = shape.orow;
        let c = shape.ocol;

        if self.cells[(r,c)].colour == shape.colour {
            let mut cnt = 0;
            let mut colour = NoColour;
            fn func(col: Colour, colour: &mut Colour, cnt: &mut usize) {
                if *colour == NoColour || col == *colour {
                    *cnt += 1;
                    if col != NoColour {
                        *colour = col;
                    }
                }
            }

            if r == 0 {
                func(NoColour, &mut colour, &mut cnt);
            } else {
                let col = self.cells[(r-1,c)].colour;

                if col != shape.colour && col != Black {
                    func(col, &mut colour, &mut cnt);
                }
            }
            if c == 0 {
                func(NoColour, &mut colour, &mut cnt);
            } else {
                let col = self.cells[(r,c-1)].colour;

                if col != shape.colour && col != Black {
                    func(col, &mut colour, &mut cnt);
                }
            }
            if r == self.cells.rows - 1 {
                func(NoColour, &mut colour, &mut cnt);
            } else {
                let col = self.cells[(r+1,c)].colour;

                if col != shape.colour && col != Black {
                    func(col, &mut colour, &mut cnt);
                }
            }
            if c == self.cells.columns - 1 {
                func(NoColour, &mut colour, &mut cnt);
            } else {
                let col = self.cells[(r,c+1)].colour;

                if col != shape.colour && col != Black {
                    func(col, &mut colour, &mut cnt);
                }
            }

            if cnt == 2 { colour } else { NoColour }
        } else {
            NoColour
        }
    }

    // unfinished 0e671a1a
    pub fn trim_excess_mut(&mut self) {
        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                let colour = self.cells[(r,c)].colour;

                if colour == Black {
                    continue;
                }

                let rs = self.cells.rows;
                let cs = self.cells.columns;
                let mut state = 0;

                if r == 0 {
                    for r in 0 .. rs {
                        if self.cells[(r,c)].colour != Black && self.cells[(r,c-1)].colour == Black && self.cells[(r,c+1)].colour == Black {
                            self.cells[(r,c)].colour = if state == 0 {
                                Black
                            } else {
                                Colour::from_usize(state * 10) + self.cells[(r,c)].colour
                            };
                        } else {
                            state += 1;
                        }
                    } 
                } else if r == rs - 1 {
                    for r in (0 .. rs).rev() {
                        if self.cells[(r,c)].colour != Black && self.cells[(r,c-1)].colour == Black && self.cells[(r,c+1)].colour == Black {
                            self.cells[(r,c)].colour = if state == 0 {
                                Black
                            } else {
                                Colour::from_usize(state * 10) + self.cells[(r,c)].colour
                            };
                        } else {
                            state += 1;
                        }
                    } 
                } else if c == 0 {
                    for c in 0 .. cs {
                        if self.cells[(r,c)].colour != Black && self.cells[(r-1,c)].colour == Black && self.cells[(r+1,c)].colour == Black {
                            self.cells[(r,c)].colour = if state == 0 {
                                Black
                            } else {
                                Colour::from_usize(state * 10) + self.cells[(r,c)].colour
                            };
                        } else {
                            state += 1;
                        }
                    }
                } else if c == cs - 1 {
                    for c in (0 .. cs).rev() {
                        if self.cells[(r,c)].colour != Black && self.cells[(r-1,c)].colour == Black && self.cells[(r+1,c)].colour == Black {
                            self.cells[(r,c)].colour = if state == 0 {
                                Black
                            } else {
                                Colour::from_usize(state * 10) + self.cells[(r,c)].colour
                            };
                        } else {
                            state += 1;
                        }
                    }
                }
            }
        }
    }

    pub fn draw_line_row(&mut self, s1: &Shape, s2: &Shape, colour: Colour, overwrite: bool, fill: bool) {
        let r1 = s1.orow;
        let c1 = s1.ocol;
        let r2 = s2.orow;
        let c2 = s2.ocol;

        self.draw_line_row_coords(r1, c1, r2, c2, colour, overwrite, fill, s1.cells.columns);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_line_row_coords(&mut self, r1: usize, c1: usize, r2: usize, c2: usize, colour: Colour, overwrite: bool, fill: bool, thick: usize) {
        if r1 == r2 && c1 != c2 {
            for y in c1.min(c2)+1 .. c1.max(c2) {
                for t in 0 .. thick {
                    if self.cells.rows <= r1+t || self.cells.columns <= y {
                        break;
                    }
                    if overwrite || self.cells[(r1+t,y)].colour == Black {
                        if overwrite && fill && self.cells[(r1+t,y)].colour != Black {
                            self.flood_fill_bg_mut(r1+t, y, self.cells[(r1+t,y)].colour, NoColour, colour);
                        } else {
                            if overwrite && fill && self.cells[(r1+t-1,y)].colour != Black {
                                self.flood_fill_bg_mut(r1+t-1, y, self.cells[(r1+t-1,y)].colour, NoColour, colour);
                            }
                            self.cells[(r1+t,y)].colour = colour;
                            if overwrite && fill && self.cells[(r1+t+1,y)].colour != Black {
                                self.flood_fill_bg_mut(r1+t+1, y, self.cells[(r1+t+1,y)].colour, NoColour, colour);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn draw_line_col(&mut self, s1: &Shape, s2: &Shape, colour: Colour, overwrite: bool, fill: bool) {
        let r1 = s1.orow;
        let c1 = s1.ocol;
        let r2 = s2.orow;
        let c2 = s2.ocol;

        self.draw_line_col_coords(r1, c1, r2, c2, colour, overwrite, fill, s1.cells.rows);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn draw_line_col_coords(&mut self, r1: usize, c1: usize, r2: usize, c2: usize, colour: Colour, overwrite: bool, fill: bool, thick: usize) {
        if c1 == c2 && r1 != r2 {
            for x in r1.min(r2)+1 .. r1.max(r2) {
                for t in 0 .. thick {
                    if self.cells.rows <= x || self.cells.columns <= c1+t {
                        break;
                    }

                    if overwrite || self.cells[(x,c1+t)].colour == Black {
                        if overwrite && fill && self.cells[(x,c1+t)].colour != Black {
                            self.flood_fill_bg_mut(x, c1+t, self.cells[(x,c1+t)].colour, NoColour, colour);
                        } else {
                            if overwrite && fill && self.cells[(x,c1+t-1)].colour != Black {
                                self.flood_fill_bg_mut(x, c1+t-1, self.cells[(x,c1+t-1)].colour, NoColour, colour);
                            }
                            self.cells[(x,c1+t)].colour = colour;
                            if overwrite && fill && self.cells[(x,c1+t+1)].colour != Black {
                                self.flood_fill_bg_mut(x, c1+t+1, self.cells[(x,c1+t+1)].colour, NoColour, colour);
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn extend_border(&self) -> Self {
        let mut grid = Self::new(self.cells.rows + 2, self.cells.columns + 2, Black);

        for ((r, c), cell) in self.cells.items() {
            grid.cells[(r + 1,c + 1)].row = r + 1;
            grid.cells[(r + 1,c + 1)].col = c + 1;
            grid.cells[(r + 1,c + 1)].colour = cell.colour;
        }

        grid.colour = self.colour;

        let rows = grid.cells.rows;
        let cols = grid.cells.columns;

        for r in 0 .. rows {
            grid.cells[(r,0)].row = r;
            grid.cells[(r,0)].col = 0;
            grid.cells[(r,0)].colour = grid.cells[(r,1)].colour;

            grid.cells[(r,cols-1)].row = r;
            grid.cells[(r,cols-1)].col = cols-2;
            grid.cells[(r,cols-1)].colour = grid.cells[(r,cols-2)].colour;
        }

        for c in 0 .. cols {
            grid.cells[(0,c)].row = 0;
            grid.cells[(0,c)].col = c;
            grid.cells[(0,c)].colour = grid.cells[(1,c)].colour;

            grid.cells[(rows-1,c)].row = rows-2;
            grid.cells[(rows-1,c)].col = c;
            grid.cells[(rows-1,c)].colour = grid.cells[(rows-2,c)].colour;
        }

        grid
    }

    fn extend(&self, lr: bool) -> Self {
        self.extend_by(lr, 2)
    }

    fn extend_by(&self, lr: bool, amount: usize) -> Self {
        let rows = self.cells.rows;
        let cols = self.cells.columns;
        let mut grid = if lr {
            Self::new(rows, cols * amount, Black)
        } else {
            Self::new(rows * amount, cols, Black)
        };

        for ((r, c), cell) in self.cells.items() {
            grid.cells[(r,c)].row = r;
            grid.cells[(r,c)].col = c;
            grid.cells[(r,c)].colour = cell.colour;
        }

        grid.colour = self.colour;

        grid
    }

    /*
    fn extend_inc(&self, lr: bool, amount: usize) -> Self {
        let rows = self.cells.rows;
        let cols = self.cells.columns;
        let mut grid = if lr {
            Self::new(rows, cols + amount, Black)
        } else {
            Self::new(rows + amount, cols, Black)
        };

        for ((r, c), cell) in self.cells.items() {
            grid.cells[(r,c)].row = r;
            grid.cells[(r,c)].col = c;
            grid.cells[(r,c)].colour = cell.colour;
        }

        grid.colour = self.colour;

        grid
    }
    */

    pub fn extend_right(&self) -> Self {
        self.extend(true)
    }

    pub fn extend_left(&self) -> Self {
        self.mirrored_cols().extend(true).mirrored_cols()
    }

    pub fn extend_down(&self) -> Self {
        self.extend(false)
    }

    pub fn extend_up(&self) -> Self {
        self.mirrored_rows().extend(false).mirrored_rows()
    }

    pub fn extend_right_by(&self, amount: usize) -> Self {
        self.extend_by(true, amount)
    }

    pub fn extend_left_by(&self, amount: usize) -> Self {
        self.mirrored_cols().extend_by(true, amount).mirrored_cols()
    }

    pub fn extend_down_by(&self, amount: usize) -> Self {
        self.extend_by(false, amount)
    }

    pub fn extend_up_by(&self, amount: usize) -> Self {
        self.mirrored_rows().extend_by(false, amount).mirrored_rows()
    }

    pub fn dup_func(&self, lr: bool, func: &dyn Fn(&Self) -> Self) -> Self {
        let rows = self.cells.rows;
        let cols = self.cells.columns;

        // not efficient!
        let mut grid = if lr {
            let temp = self.mirrored_cols().extend_right();
            let temp = &func(&temp);

            temp.mirrored_cols()
        } else {
            let temp = self.mirrored_rows().extend_down();
            let temp = &func(&temp);

            temp.mirrored_rows()
        };

        for r in 0 .. rows {
            for c in 0 .. cols {
                grid.cells[(r,c)].row = r;
                grid.cells[(r,c)].col = c + cols;
                grid.cells[(r,c)].colour = self.cells[(r,c)].colour;
            }
        }

        grid.colour = self.colour;

        grid
    }

    fn dup(&self, lr: bool) -> Self {
        self.dup_func(lr, &|g| g.clone())
    }

    pub fn dup_right(&self) -> Self {
        self.dup(true)
    }

    pub fn dup_left(&self) -> Self {
        self.mirrored_cols().dup(true).mirrored_cols()
    }

    pub fn dup_down(&self) -> Self {
        self.dup(false)
    }

    pub fn dup_up(&self) -> Self {
        self.mirrored_rows().dup(false).mirrored_rows()
    }

    pub fn dup_right_func(&self, func: &dyn Fn(&Self) -> Self) -> Self {
        self.dup_func(true, &func)
    }

    pub fn dup_left_func(&self, func: &dyn Fn(&Self) -> Self) -> Self {
        self.mirrored_cols().dup_func(true, &func).mirrored_cols()
    }

    pub fn dup_down_func(&self, func: &dyn Fn(&Self) -> Self) -> Self {
        self.dup_func(false, &func)
    }

    pub fn dup_up_func(&self, func: &dyn Fn(&Self) -> Self) -> Self {
        self.mirrored_rows().dup_func(false, &func).mirrored_rows()
    }

    pub fn mirror_dir_func(&self, lr: bool, func: &dyn Fn(&Self) -> Self) -> Self {
        let rows = self.cells.rows;
        let cols = self.cells.columns;

        let mut grid = if lr {
            let temp = self.extend_right();
            let temp = &func(&temp);

            temp.mirrored_cols()
        } else {
            let temp = self.extend_down();
            let temp = &func(&temp);

            temp.mirrored_rows()
        };

        for r in 0 .. rows {
            for c in 0 .. cols {
                if r < grid.cells.rows && c < grid.cells.columns {
                    grid.cells[(r,c)].row = r;
                    grid.cells[(r,c)].col = c;
                    grid.cells[(r,c)].colour = self.cells[(r,c)].colour;
                }
            }
        }

        grid.colour = self.colour;

        grid
    }

    pub fn mirror_dir(&self, lr: bool) -> Self {
        self.mirror_dir_func(lr, &|g| g.clone())
    }

    pub fn mirror_right(&self) -> Self {
        self.mirror_dir(true)
    }

    pub fn mirror_left(&self) -> Self {
        self.mirrored_cols().mirror_dir(true).mirrored_cols()
    }

    pub fn mirror_down(&self) -> Self {
        self.mirror_dir(false)
    }

    pub fn mirror_up(&self) -> Self {
        self.mirrored_rows().mirror_dir(false).mirrored_rows()
    }

    pub fn mirror_right_func(&self, func: &dyn Fn(&Self) -> Self) -> Self {
        self.mirror_dir_func(true, &func)
    }

    pub fn mirror_left_func(&self, func: &dyn Fn(&Self) -> Self) -> Self {
        self.mirrored_cols().mirror_dir_func(true, &func).mirrored_cols()
    }

    pub fn mirror_down_func(&self, func: &dyn Fn(&Self) -> Self) -> Self {
        self.mirror_dir_func(false, &func)
    }

    pub fn mirror_up_func(&self, func: &dyn Fn(&Self) -> Self) -> Self {
        self.mirrored_rows().mirror_dir_func(false, &func).mirrored_rows()
    }

    pub fn pixels(&self) -> usize {
        self.cells.values()
            .filter(|c| c.colour != Black).
            count()
    }

    pub fn transform(&self, trans: Transformation) -> Self {
        match trans {
            NoTrans          => self.clone(),
            MirrorRow          => self.mirrored_rows(),
            MirrorCol          => self.mirrored_cols(),
            Trans            => self.transposed(),
            Rotate90         => self.rot_rect_90(),
            Rotate180        => self.rot_rect_180(),
            Rotate270        => self.rot_rect_270(),
            Rotate90MirrorRow  => self.rot_rect_90().mirrored_rows(),
            Rotate180MirrorRow => self.rot_rect_180().mirrored_rows(),
            Rotate270MirrorRow => self.rot_rect_270().mirrored_rows(),
            Rotate90MirrorCol  => self.rot_rect_90().mirrored_cols(),
            Rotate180MirrorCol => self.rot_rect_180().mirrored_cols(),
            Rotate270MirrorCol => self.rot_rect_270().mirrored_cols(),
            MirrorRowRotate90  => self.mirrored_rows().rot_rect_90(),
            MirrorRowRotate180 => self.mirrored_rows().rot_rect_180(),
            MirrorRowRotate270 => self.mirrored_rows().rot_rect_270(),
            MirrorColRotate90  => self.mirrored_cols().rot_rect_90(),
            MirrorColRotate180 => self.mirrored_cols().rot_rect_180(),
            MirrorColRotate270 => self.mirrored_cols().rot_rect_270(),
        }
    }

    pub fn inverse_transform(&self, trans: Transformation) -> Self {
        let trans = Transformation::inverse(&trans);

        self.transform(trans)
    }

    pub fn inverse_colour(&self) -> Self {
        let colour = self.colour();
        let mut inv = self.clone();

        for cell in inv.cells.values_mut() {
            if cell.colour == Black {
                cell.colour = colour;
            } else {
                cell.colour = Black;
            }
        }

        inv
    }

    pub fn colour(&self) -> Colour {
        let mut colour = Black;

        for c in self.cells.values() {
            if colour == Black && c.colour != Black {
                colour = c.colour;
            } else if c.colour != Black && c.colour != colour {
                return self.colour;
            }
        }

        colour
    }

    pub fn size(&self) -> usize {
        self.cells.columns * self.cells.rows
    }

    pub fn same_size(&self, other: &Self) -> bool {
        self.size() == other.size()
    }

    pub fn same_shape(&self, other: &Self) -> bool {
        self.cells.rows == other.cells.rows && self.cells.columns == other.cells.columns
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.cells.rows, self.cells.columns)
    }

    pub fn add_border(&self, n: usize) -> Self {
        self.add_border_colour(n, Black)
    }

    pub fn add_border_colour(&self, n: usize, colour: Colour) -> Self {
        let mut g = Self::new(self.cells.rows + n * 2, self.cells.columns + n * 2, colour);

        g.copy_to_position_mut(self, n, n);

        g
    }

    pub fn remove_border(&self, n: usize) -> Self {
        if self.cells.rows < n * 2 || self.cells.columns < n *  2 {
            return self.clone();
        }

        self.subgrid(n, self.cells.rows - n * 2, n, self.cells.columns - n * 2)
    }

    pub fn max_dim(&self) -> usize {
        self.cells.rows.max(self.cells.columns)
    }

    pub fn derive_missing_rule(&self, other: &Self) -> Self {
        if self.cells.rows == 1 || self.cells.columns == 0 ||other.cells.rows % self.cells.rows != 0 || other.cells.columns % self.cells.columns != 0 {
            return Self::trivial();
        }

        let sg = other.subgrid(0, self.cells.rows, 0, self.cells.columns);

        match self.diff(&sg) {
            Some(diff) => diff,
            None => Self::trivial(),
        }
    }

    pub fn apply_missing_rule(&self, other: &Self) -> Self {
        let mut grid = self.clone();

        for ((r, c), cell) in self.cells.items() {
            if cell.colour != Black {
                grid.copy_centre(r, c, other);
            }
        }

        grid
    }

    pub fn copy_centre(&mut self, cr: usize, cc: usize, other: &Self) {
        if other == &Self::trivial() {
            return;
        }
        let centre_r = (other.cells.rows / 2) as isize;
        let centre_c = (other.cells.columns / 2) as isize;

        let or = centre_r - cr as isize;
        let oc = centre_c - cc as isize;
        let rs = if or < 0 { 0 } else { or as usize };
        let re = if or < 0 {
            // not happy with this
            let incdec = if or < -1 { 1 } else { -1 };
            (other.cells.rows as isize + or + incdec) as usize
        } else {
            other.cells.rows
        }; 
        let cs = if oc < 0 { 0 } else { oc as usize };
        let ce = if oc < 0 {
            // not happy with this
            let incdec = if oc < -1 { 1 } else { -1 };
            (other.cells.columns as isize + oc + incdec) as usize
        } else {
            other.cells.columns
        }; 

        for r in rs .. re {
            for c in cs .. ce {
                let colour = other.cells[(r, c)].colour;
                let sr = (r as isize - or) as usize;
                let sc = (c as isize - oc) as usize;

                // untidy but necessary
                if sr >= self.cells.rows || sc >= self.cells.columns {
                    continue;
                }

                if !colour.is_same() && colour != Black && self.cells[(sr, sc)].colour == Black {
                    self.cells[(sr, sc)].colour = colour.to_base();
                }
            }
        }
    }

    pub fn fill_border(&mut self) {
        let hr = self.cells.rows / 2;
        let hc = self.cells.columns / 2;

        for layer in 0 .. hr.min(hc) {
            let mut cc = NoColour;

            for i in layer .. self.cells.rows - layer {
                let c = self.cells[(i,layer)].colour;

                if c != Black && cc == NoColour {
                    cc = c;
                } else if c != Black && c != cc {
                    return;
                }
            }
            for i in layer .. self.cells.columns - layer {
                let c = self.cells[(layer,i)].colour;

                if c != Black && cc == NoColour {
                    cc = c;
                } else if c != Black && c != cc {
                    return;
                }
            }
//println!("Layers: {layer} {:?}", cc);

            for i in layer .. self.cells.rows - layer {
                let c = self.cells[(i,layer)].colour;

                if c == Black {
                    self.cells[(i,layer)].colour = cc;
                }
            }
            for i in layer .. self.cells.columns - layer {
                let c = self.cells[(layer, i)].colour;

                if c == Black {
                    self.cells[(layer,i)].colour = cc;
                }
            }
        }
    }

    /*
    pub fn find_xy_seq(&self, xseq: &[Colour], yseq: &[Colour]) -> (bool, (usize, usize)) {

        if xseq.is_empty() || Colour::single_colour_vec(&xseq) {
            (false, self.find_y_seq(0, 0, yseq, xseq.len()))
        } else if !yseq.is_empty() {
            (true, self.find_x_seq(0, 0, xseq, yseq.len()))
        } else {
            (false, (0, 0))
        }
/*
println!("1 #### {}/{} {}/{}", xsx, xsy, ysx, ysy);

        while (xsx != ysx || xsy != ysy) && xsx != usize::MAX && ysx != usize::MAX {
            if xsx < ysx || xsy < ysy {
                (xsx, xsy) = self.find_x_seq(xsx, xsy + 1, xseq, yseq.len());
            } else {
                (ysx, ysy) = self.find_y_seq(ysx + 1, ysy, yseq, xseq.len());
            }
println!("2 #### {}/{} {}/{}", xsx, xsy, ysx, ysy);
        }

        if xsx != usize::MAX {
           (xsx, xsy)
        } else {
           (ysx, ysy)
        }
*/
        //(xsx, xsy)
        //(ysx, ysy)
        //self.find_x_seq(xsx, xsy + 1, xseq, yseq.len())
        //self.find_y_seq(ysx + 1, ysy, yseq, xseq.len())
    }
    */

    pub fn has_colour(&self, tlr: usize, tlc: usize, rlen: usize, clen: usize, colour: Colour) -> bool {
        for r in tlr .. rlen {
            for c in tlc .. clen {
                if self.cells[(r,c)].colour == colour {
                    return true;
                }
            }
        }

        false
    }

    pub fn find_row_seq(&self, sr: usize, sc: usize, seq: &[Colour], width: usize) -> (usize, usize) {
        let mut cnt = 0;
        let mut xp = 0;
        let mut rs = usize::MAX;
        let mut cs = usize::MAX;

        'outer:
        for r in 0 .. self.cells.columns - width {
            for c in 0 .. self.cells.rows - seq.len() {
                if c == sr+1 && r <= sc { continue 'outer}; 
                let cell = self.cells[(c,r)].clone();
                if xp != r {
                    xp = r;
                    cnt = 0;
                    rs = usize::MAX;
                    cs = usize::MAX;
                }
                if seq[cnt] == cell.colour {
                    if !self.has_colour(c, r, seq.len(), width, Black) {
                        if cnt == 0 {
                            rs = c;
                            cs = r;
                        }
//println!("{} {} == {} {xs}/{ys} {x} {y}", cnt, Colour::to_usize(seq[cnt]), Colour::to_usize(c.colour));
//println!("{sy} ==== {cnt} {:?}", c.colour);
                        if cnt == seq.len() - 1 {
                            break 'outer;
                        }
                        cnt += 1;
                    }
                } else if cnt > 0 {
                    cnt = 0;
                    rs = usize::MAX;
                    cs = usize::MAX;
                }
            }
        }

        (rs, cs)
    }

    pub fn find_col_seq(&self, sr: usize, sc: usize, seq: &[Colour], length: usize) -> (usize, usize) {
        let mut cnt = 0;
        let mut rp = 0;
        let mut rs = usize::MAX;
        let mut cs = usize::MAX;

        'outer:
        for r in 0 .. self.cells.rows - length {
            for c in 0 .. self.cells.columns - seq.len() {
                if c == sc+1 && r <= sr { continue 'outer}; 
                let cell = self.cells[(r,c)].clone();
                if rp != r {
                    rp = r;
                    cnt = 0;
                    rs = usize::MAX;
                    cs = usize::MAX;
                }
                if seq[cnt] == cell.colour {
                    if !self.has_colour(r, c, length, seq.len(), Black) {
                        if cnt == 0 {
                            rs = r;
                            cs = c;
                        }
                        cnt += 1;
                        if cnt == seq.len() {
                            break 'outer;
                        }
                    }
                } else if cnt > 0 {
                    cnt = 0;
                    rs = usize::MAX;
                    cs = usize::MAX;
                }
            }
        }

        (rs, cs)
    }

    pub fn colour_every_nxn_for_m(colour: Colour, side: usize, n: usize, m: usize) -> Self {
        if m == 0 || n == 0 {
            return Self::trivial();
        }
        let mut grid = Self::new(side, side, Black);
        let mut count = 0;

        'outer:
        for r in 0 .. side {
            for c in 0 .. side {
                if (r + grid.cells.rows * c) % n == 0 {
                    grid.cells[(r, c)].colour = colour;
                    count += 1;
                }
                if count == m {
                    break 'outer;
                }
            }
        }

        grid
    }

    pub fn colour_dimensions(&self, colour: Colour) -> (usize, usize) {
        let mut min_r: usize = usize::MAX;
        let mut max_r: usize = 0;
        let mut min_c: usize = usize::MAX;
        let mut max_c: usize = 0;

        for ((x, y), c) in self.cells.items() {
            if c.colour == colour {
                min_r = x.min(min_r);
                max_r = x.max(max_r);
                min_c = y.min(min_c);
                max_c = y.max(max_c);
            }
        }

        (max_r - min_r + 1, max_c - min_c + 1)
    }

    pub fn bigger(&self, other: &Self) -> bool {
        self.size() > other.size()
    }

    pub fn smaller(&self, other: &Self) -> bool {
        self.size() < other.size()
    }

    pub fn cell_count(&self) -> usize {
        self.cell_count_colour(Black)
    }

    pub fn cell_count_colour(&self, colour: Colour) -> usize {
        self.cells.values().filter(|c| c.colour != colour).count()
    }

    pub fn flood_fill(&self, x: usize, y: usize, ignore_colour: Colour, new_colour: Colour) -> Self {
        self.flood_fill_bg(x, y, ignore_colour, Black, new_colour)
    }

    pub fn flood_fill_bg(&self, r: usize, c: usize, ignore_colour: Colour, bg: Colour, new_colour: Colour) -> Self {
        let mut grid = self.clone();

        grid.flood_fill_bg_mut(r, c, ignore_colour, bg, new_colour);

        grid
    }

    pub fn flood_fill_mut(&mut self, r: usize, c: usize, ignore_colour: Colour, new_colour: Colour) {
        self.flood_fill_bg_mut(r, c, ignore_colour, Black, new_colour)
    }

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
    }

    pub fn flood_fill_bg_mut(&mut self, r: usize, c: usize, ignore_colour: Colour, bg: Colour, new_colour: Colour) {
        let reachable = self.cells.bfs_reachable((r, c), false, |i| self.cells[i].colour == bg || self.cells[i].colour == ignore_colour);

        reachable.iter().for_each(|&i| self.cells[i].colour = new_colour);
    }

    pub fn flood_fill_from_seeds(&self, ignore_colour: Colour, new_colour: Colour) -> Self {
        let mut grid = self.clone();

        let coloured: Vec<(usize, usize)> = grid.cells.items()
            .filter(|(_, cell)| cell.colour == ignore_colour)
            .map(|(i, _)| i)
            .collect();

        coloured.iter()
            .for_each(|(r, c)| grid.flood_fill_mut(*r, *c, ignore_colour, new_colour));

        grid
    }

    pub fn subgrid(&self, tlr: usize, sr: usize, tlc: usize, sc: usize) -> Self {
        if sr == 0 || sc == 0 {
            return Grid::trivial();
        }

        let mut m = Matrix::new(sr, sc, Cell::new(0, 0, 0));

        for r in 0 ..  sr {
            for c in 0 .. sc {
                m[(r,c)].row = self.cells[(r + tlr,c + tlc)].row;
                m[(r,c)].col = self.cells[(r + tlr,c + tlc)].col;
                m[(r,c)].colour = self.cells[(r + tlr,c + tlc)].colour;
            }
        }

        Self::new_from_matrix(&m)
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

    pub fn cell_colour_posn_map(&self) -> BTreeMap<Colour, Vec<(usize, usize)>>  {
        let mut h: BTreeMap<Colour, Vec<(usize, usize)>> = BTreeMap::new();

        for c in self.cells.values() {
            if c.colour != Black {
                h.entry(c.colour).or_default().push((c.row, c.col));
            }
        }

        h
    }

    pub fn majority_colour(&self) -> Colour {
        let cccm = self.cell_colour_cnt_map();

        let mx = cccm.iter().map(|(c,n)| (n,c)).max();

        if let Some(mx) = mx {
            *mx.1
        } else {
            NoColour
        }
    }
    
    pub fn minority_colour(&self) -> Colour {
        let cccm = self.cell_colour_cnt_map();

        let mn = cccm.iter().map(|(c,n)| (n,c)).min();

        if let Some(mn) = mn {
            *mn.1
        } else {
            NoColour
        }
    }

    pub fn get_diff_colour(&self, other: &Self) -> Colour {
        let mut in_colour = Black;

        if let Some(diff) = self.diff(other) {
            let colour = diff.cell_colour_cnt_map_diff();

            if colour.len() != 1 {
                return in_colour;
            }

            if let Some((colour, _)) = colour.first_key_value() {
                in_colour = colour.to_base()
            }
        }

        in_colour
    }

    pub fn cell_colour_cnt_map_diff(&self) -> BTreeMap<Colour, usize>  {
        let mut h: BTreeMap<Colour, usize> = BTreeMap::new();

        for c in self.cells.values() {
            if c.colour >= DiffBlack && c.colour <= DiffBrown {
                *h.entry(c.colour).or_insert(0) += 1;
            }
        }

        h
    }

    pub fn even_rows(&self) -> bool {
        self.cells.rows % 2 == 0
    }

    pub fn even_columns(&self) -> bool {
        self.cells.columns % 2 == 0
    }

    pub fn border_top(&self) -> bool {
        self.cells.items()
            .filter(|((r, _), cell)| *r == 0 && cell.colour != Black)
            .count() == self.cells.columns - 1
    }

    pub fn border_bottom(&self) -> bool {
        self.cells.items()
            .filter(|((r, _), cell)| *r == self.cells.columns - 1 && cell.colour != Black)
            .count() == self.cells.columns - 1
    }

    pub fn border_left(&self) -> bool {
        self.cells.items()
            .filter(|((_, c), cell)| *c == 0 && cell.colour != Black)
            .count() == self.cells.rows - 1
    }

    pub fn border_right(&self) -> bool {
        self.cells.items()
            .filter(|((_, c), cell)| *c == self.cells.rows - 1 && cell.colour != Black)
            .count() == self.cells.rows - 1
    }

    pub fn has_border(&self) -> bool {
        self.border_top() && self.border_bottom() && self.border_left() && self.border_right()
    }

    pub fn mirrored_rows(&self) -> Self {
        let mut m = self.cells.flipped_ud();

        for (r, c) in self.cells.keys() {
            m[(r, c)].row = r;
            m[(r, c)].col = c;
        }
        
        Self::new_from_matrix(&m)
    }

    pub fn mirrored_cols(&self) -> Self {
        let mut m = self.cells.flipped_lr();

        for (x, y) in self.cells.keys() {
            m[(x, y)].row = x;
            m[(x, y)].col = y;
        }
        
        Self::new_from_matrix(&m)
    }

    fn compare_colour(this: &Matrix<Cell>, other: &Matrix<Cell>) -> bool {
        if this.columns == 1 || this.rows == 1 || this.columns != other.columns || this.rows != other.rows {
            return false;
        }

        for x in 0 .. this.rows {
            for y in 0 .. this.columns {
                if this[(x, y)].colour != other[(x, y)].colour {
                    return false
                }
            }
        }

        true
    }

    pub fn populate_skew_edge_lr(&self, shape: &Shape, colour: Colour) -> Shape {
        let rs = self.row_skew();
        let mut new_shape = shape.clone();

        for r in 0 .. rs as usize {
            for c in 0 .. self.cells.columns {
                if self.cells[(r,c)].colour == colour {
                    if shape.cells.rows >= r + 1 {
                        new_shape.cells[(r - shape.orow,c - shape.ocol)].colour = self.cells[(shape.cells.rows - r - 1,c)].colour;
                    } else {
                        new_shape.cells[(r - shape.orow,c - shape.ocol)].colour = self.cells[(r,c)].colour;
                    }
                } else if self.cells[(c,r)].colour == colour {
                    if shape.cells.columns >= c + 1 {
                        new_shape.cells[(c - shape.orow,r - shape.ocol)].colour = self.cells[(r,shape.cells.columns - c - 1)].colour;
                    } else {
                        new_shape.cells[(c - shape.orow,r - shape.ocol)].colour = self.cells[(r,c)].colour;
                    }
                }
            }
        }

        new_shape
    }

    pub fn populate_skew_edge_tb(&self, shape: &Shape, colour: Colour) -> Shape {
        let grid = self.rotated_270(1);
        let mut shape = shape.rot_rect_270();

        let tr = shape.orow;

        shape.orow = shape.ocol;
        shape.ocol = tr;

        let shape = grid.populate_skew_edge_lr(&shape, colour);

        let mut shape = shape.rot_rect_90();

        let tr = shape.orow;

        shape.orow = shape.ocol;
        shape.ocol = tr;

        shape
    }

    pub fn is_mirror_rows(&self) -> bool {
        if self.cells.len() <= 4 {
            return false
        }
        let copy = self.cells.flipped_ud();

        Self::compare_colour(&self.cells, &copy)
    }

    pub fn is_mirror_cols(&self) -> bool {
        if self.cells.len() <= 4 {
            return false
        }
        let copy = self.cells.flipped_lr();

        Self::compare_colour(&self.cells, &copy)
    }

    pub fn is_symmetric(&self) -> bool {
        self.is_mirror_rows() && self.is_mirror_cols()
    }

    pub fn is_mirror_offset_rows(&self, skew: i32) -> bool {
        let copy = if skew < 0 {
            self.cells.slice(0 .. (self.cells.rows as i32 + skew) as usize, 0 .. self.cells.columns)
        } else {
            self.cells.slice((skew as usize) .. self.cells.rows, 0 .. self.cells.columns)
        };

        if copy.is_err() { return false; };

        let this = copy.unwrap();
        let flipped_copy = this.flipped_ud();

        Self::compare_colour(&this, &flipped_copy)
    }

    pub fn is_mirror_offset_cols(&self, skew: i32) -> bool {
        let copy = if skew < 0 {
            self.cells.slice(0 .. self.cells.rows, 0 .. (self.cells.columns - skew.unsigned_abs() as usize))
        } else {
            self.cells.slice(0 .. self.cells.rows, (skew as usize) .. self.cells.columns)
        };

        if copy.is_err() { return false; };

        let this = copy.unwrap();
        let flipped_copy = this.flipped_lr();

        Self::compare_colour(&this, &flipped_copy)
    }

    pub fn is_panelled_rows(&self) -> bool {
        let len: usize = self.cells.rows;
        let half: usize = len / 2;

        if len < 4 || half == 0 { return false }

        let offset: usize = if len % 2 == 0 { 0 } else { 1 };

        for c in 0 .. self.cells.columns {
            for r in 0 .. half {
                if self.cells[(r, c)].colour != self.cells[(half + offset + r, c)].colour {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_panelled_cols(&self) -> bool {
        let len: usize = self.cells.columns;
        let half: usize = len / 2;

        if len < 4 || half == 0 { return false }

        let offset: usize = if len % 2 == 0 { 0 } else { 1 };

        for r in 0 .. self.cells.rows {
            for c in 0 .. half {
                if self.cells[(r, c)].colour != self.cells[(r, half + offset + c)].colour {
                    return false;
                }
            }
        }

        true
    }

    pub fn transpose(&mut self) {
        if self.cells.rows != self.cells.columns {
            return;
        }

        self.cells.transpose();
    }

    pub fn transposed(&self) -> Self {
        if self.cells.rows != self.cells.columns {
            return self.clone();
        }

        let mut m: Matrix<Cell> = self.cells.clone();

        m.transpose();

        for (r, c) in self.cells.keys() {
            m[(r, c)].row = r;
            m[(r, c)].col = c;
        }
        
        Self::new_from_matrix(&m)
    }

    pub fn inv_transposed(&self) -> Self {
        Self::new_from_matrix(&self.cells.rotated_cw(2).transposed())
    }

    /*
    pub fn rotate_90(&mut self, times: usize) {
        if self.cells.rows == self.cells.columns {
            return;
        }

        self.cells.rotate_cw(times);

        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                self.cells[(r, c)].row = r;
                self.cells[(r, c)].col = c;
            }
        }
    }

    pub fn rotate_270(&mut self, times: usize) {
        if self.cells.rows != self.cells.columns {
            return;
        }

        self.cells.rotate_ccw(times);

        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                self.cells[(r, c)].row = r;
                self.cells[(r, c)].col = c;
            }
        }
    }
    */

    pub fn rotated_90(&self, times: usize) -> Self {
        if self.cells.rows != self.cells.columns || times == 0 {
            return self.clone();
        }

        let mut m: Matrix<Cell> = self.cells.clone();

        m.rotate_cw(times);

        for (r, c) in self.cells.keys() {
            m[(r, c)].row = r;
            m[(r, c)].col = c;
        }
        
        Self::new_from_matrix(&m)
    }

    pub fn rotated_270(&self, times: usize) -> Self {
        if self.cells.rows != self.cells.columns {
            return self.clone();
        }

        let mut m: Matrix<Cell> = self.cells.clone();

        m.rotate_ccw(times);

        for (r, c) in self.cells.keys() {
            m[(r, c)].row = r;
            m[(r, c)].col = c;
        }
        
        Self::new_from_matrix(&m)
    }

    pub fn rotate_90(&self, times: usize) -> Self {
        if times == 1 {
            self.rot_rect_90()
        } else if times == 2 {
            self.rot_rect_90().rot_rect_90()
        } else if times == 3 {
            self.rot_rect_90().rot_rect_90().rot_rect_90()
        } else {
            self.clone()
        }
    }

    pub fn rot_00(&self) -> Self {      // Identity rotation
        self.clone()
    }

    pub fn rot_90(&self) -> Self {
        self.rotated_90(1)
    }

    pub fn rot_180(&self) -> Self {
        self.rotated_90(2)
    }

    pub fn rot_270(&self) -> Self {
        self.rotated_270(1)
    }

    pub fn rot_rect_90(&self) -> Self {
        if self.cells.rows == self.cells.columns {
            self.rotated_90(1)
        } else {
            let mut rot = Self::new(self.cells.columns, self.cells.rows, Black);
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
            self.rotated_90(2)
        } else {
            self.rot_rect_90().rot_rect_90()
        }
    }

    pub fn rot_rect_270(&self) -> Self {
        if self.cells.rows == self.cells.columns {
            self.rotated_270(1)
        } else {
            self.rot_rect_90().rot_rect_90().rot_rect_90()
        }
    }

    pub fn colours(&self) -> BTreeMap<Colour, usize> {
        let mut h: BTreeMap<Colour, usize> = BTreeMap::new();

        for cell in self.cells.values() {
            if cell.colour != Black {
                *h.entry(cell.colour).or_insert(0) += 1;
            }
        }

        h
    }

    pub fn min_colour(&self) -> Colour {
        let h = self.colours();

        match h.iter().min_by(|col, c| col.1.cmp(c.1)) {
            None => NoColour,
            Some((colour,_)) => *colour
        }
    }

    pub fn max_colour(&self) -> Colour {
        let h = self.colours();

        match h.iter().max_by(|col, c| col.1.cmp(c.1)) {
            None => NoColour,
            Some((colour,_)) => *colour
        }
    }

    pub fn no_colours(&self) -> usize {
        self.colours().len()
    }

    pub fn single_colour(&self) -> Colour {
        let h = self.colours();

        if h.is_empty() {
            NoColour
        } else if h.len() == 1 {
            //h.keys().max().unwrap().clone()
            *h.keys().max().unwrap()
        } else {
            Mixed
        }
    }

    fn draw_row(&mut self, from_c: usize, to_c: usize, x: usize, colour: Colour) -> usize {
        for c in from_c .. to_c {
            self.cells[(x, c)].colour = colour;
        }

        if to_c > 0 { to_c - 1 } else { 0 }
    }

    fn draw_col(&mut self, from_r: usize, to_r: usize, y: usize, colour: Colour) -> usize {
        for r in from_r .. to_r {
            self.cells[(r, y)].colour = colour;
        }

        if to_r > 0 { to_r - 1 } else { 0 }
    }

    pub fn do_circle(&self, colour: Colour, spiral: bool) -> Self {
        let inc = if spiral { 2 } else { 0 };
        let mut copy = self.clone();
        let mut cinc = 0;
        let mut sr = 0;
        let mut sc = 1;
        let mut rows = self.cells.rows;
        let mut cols = self.cells.columns;

        // First round
        let mut cc = copy.draw_row(sr, cols, 0, colour);
        let mut cr = copy.draw_col(sc, rows, cc, colour);
        cc = copy.draw_row(sr, cols - 1, cr, colour);
        if spiral { sc += 1};
        copy.draw_col(sc, rows - 1, 0, colour);

        if spiral {
            while sr + 1 < cc { 
                sr += 1;
                sc += 1;
                rows -= inc;
                cols -= inc;
                cinc += inc;

                cc = copy.draw_row(sr, cols, cinc, colour);
                cr = copy.draw_col(sc, rows, cc, colour);
                sr += 1;
                cc = copy.draw_row(sr, cols - 1, cr, colour);
                sc += 1;
                copy.draw_col(sc, rows - 1, cinc, colour);
            }
        }

        Self::new_from_matrix(&copy.cells)
    }

    pub fn circle(&self, colour: Colour) -> Self {
        self.do_circle(colour, false)
    }

    pub fn spiral(&self, colour: Colour) -> Self {
        self.do_circle(colour, true)
    }

    pub fn show_matrix(m: &Matrix<Cell>, diff: bool, io: bool) {
        let mut px = 0;
        for ((r, c), cell) in m.items() {
            if r != px {
                println!();
                px = r;
            } else if c != 0 {
                print!(" ");
            }

            let colour = cell.colour.to_usize();

            if colour == 100 {
                print!("{}", if !diff && io { "##" } else { "#" });
            } else if colour == 101 {
                print!("{}", if !diff && io { "**" } else { "*" });
            } else if diff && !io {
                if colour >= 10 { print!("#", ) } else { print!("{colour}") };
            } else if !diff && io {
                print!("{colour:0>2}");
            } else if !diff && !io {
                if colour >= 20 { print!("#") } else { print!("{}", colour % 10) };
            } else {
                print!("{}", colour % 10);
            }
        }
    }

    fn show_any(&self, diff: bool, io: bool) {
        println!("--------Grid--------");
        Self::show_matrix(&self.cells, diff, io);
        println!();
    }

    pub fn show_summary(&self) {
        println!("0/0: {}/{} {:?}", self.cells.rows, self.cells.columns, self.colour);
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

    pub fn has_bg_grid(&self) -> Colour {
        self.has_bg_grid_impl(true, true)
    }

    pub fn has_bg_grid_coloured(&self) -> Colour {
        self.has_bg_grid_impl(false, true)
    }

    pub fn has_bg_grid_not_sq(&self) -> Colour {
        self.has_bg_grid_impl(true, false)
    }

    pub fn has_bg_grid_coloured_not_sq(&self) -> Colour {
        self.has_bg_grid_impl(false, false)
    }

    fn has_bg_grid_impl(&self, can_be_black: bool, square: bool) -> Colour {
        let mut colour = NoColour;

//println!("---- {colour:?} {square}");
        if self.cells.len() <= 9 || (square && self.cells.rows != self.cells.columns) {
            return colour;
        }

        let mut nox = false;    // may be only a y axis

        'outerr:
        for x in 0 .. self.cells.rows {
            if can_be_black || self.cells[(x, 0)].colour != Black {
                colour = self.cells[(x, 0)].colour;
            } else {
                continue;
            }

            if colour == self.cells[(x, 0)].colour {
                for c in 0 .. self.cells.columns {
                    if colour != self.cells[(x, c)].colour {
                        continue 'outerr;
                    }
                    nox = true;
                }
                break;  // we have found a candidate
            }
        }

        'outerc:
        for y in 0 .. self.cells.columns {
            if nox {
                if can_be_black || self.cells[(0, y)].colour != Black {
                    colour = self.cells[(0, y)].colour;
                } else {
                    continue;
                }
            }
            if colour == self.cells[(0, y)].colour {
                for r in 0 .. self.cells.rows {
                    if colour != self.cells[(r, y)].colour {
                        continue 'outerc;
                    }
                }

                if !can_be_black && colour == Black {
                    break;
                }

                return colour;  // we have found one
            }
        }

        NoColour
    }

    // TODO
    pub fn ray(&self, _direction: (i8, i8)) -> Shape {
        Shape::new_empty()
    }

    // TODO
    pub fn r#move(&self, _direction: (i8, i8)) -> Shape {
        Shape::new_empty()
    }

    pub fn template_shapes(&self, template: &Self) -> Shapes {
        let mut shapes = Shapes::new_sized(self.height(), self.width());

        if self.height() % template.height() != 0 || self.width() % template.width() != 0 {
            return shapes;
        }

        let rr = self.height() / template.height();
        let cr = self.width() / template.width();

        for (tr, r) in (0 .. self.height()).step_by(rr).enumerate() {
            for (tc, c) in (0 .. self.width()).step_by(cr).enumerate() {
                if template.cells[(tr,tc)].colour != Black {
                    let sg = self.subgrid(r, rr, c, cr);

                    shapes.shapes.push(sg.as_shape());
                }
            }
        }

        shapes
    }

    pub fn fill_template(&self, filler: &Shape) -> Self {
        let mut shapes = Shapes::new_sized(self.height() * filler.height(), self.width() * filler.width());

        let rr = filler.height();
        let cr = filler.width();

        for r in 0 .. self.height() {
            for c in 0 .. self.width() {
                if self.cells[(r,c)].colour != Black {
                    let s = filler.to_position(r * rr, c * cr);

                    shapes.shapes.push(s);
                }
            }
        }

        shapes.to_grid()
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

        Self::new_from_matrix(&cells)
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

        Self::new_from_matrix(&cells)
    }

    pub fn resize(&self, factor: usize) -> Self {
        let mut cells = Matrix::new(self.cells.rows * factor, self.cells.columns * factor, Cell::new(0, 0, 0));

        for ((r, c), cell) in self.cells.items() {
            let rf = r + factor;
            let cf = c + factor;

            cells[(rf, cf)].row = r;
            cells[(rf, cf)].col = c;
            cells[(rf, cf)].colour = cell.colour;
        }

        Self::new_from_matrix(&cells)
    }

    pub fn resize_inc(&self, factor: usize) -> Self {
        let mut cells = Matrix::new(self.cells.rows + factor * 2, self.cells.columns + factor * 2, Cell::new(0, 0, 0));

        for ((r, c), cell) in self.cells.items() {
            let rf = r + factor;
            let cf = c + factor;

            cells[(rf, cf)].row = r;
            cells[(rf, cf)].col = c;
            cells[(rf, cf)].colour = cell.colour;
        }

        Self::new_from_matrix(&cells)
    }

    pub fn find_patch(&self, patch: &Shape) -> (usize, usize) {
        if patch.size() < 4 {
            return (0, 0);
        }
        let colour00 = patch.cells[(0,0)].colour;
        let colour01 = patch.cells[(0,1)].colour;
        let colour10 = patch.cells[(1,0)].colour;
        let colour11 = patch.cells[(1,1)].colour;

        for ((rw, cl), cell) in self.cells.items() {
            if rw > patch.cells.rows || cl > patch.cells.columns {
                // find candidate
                if cell.colour == colour00 && rw + patch.cells.rows < self.cells.rows && cl + patch.cells.columns < self.cells.columns && self.cells[(rw+1,cl)].colour == colour10 && self.cells[(rw,cl+1)].colour == colour01 && self.cells[(rw+1,cl+1)].colour == colour11 {
                    let s = self.subgrid(rw, patch.cells.rows, cl, patch.cells.columns);

                    if s.equals(&patch.to_grid()) == Same {
                        return (rw, cl);
                    }
                }
            }
        }

        (0, 0)
    }

    pub fn find_axis_colour(&self, shape: &Shape) -> Colour {
        self.find_axis_colour_bg(shape, Black)
    }

    pub fn find_axis_colour_bg(&self, shape: &Shape, bg: Colour) -> Colour {
        if self.cells.rows <= shape.cells.rows || self.cells.columns <= shape.cells.columns || shape.ocol >= self.cells.columns || shape.orow >= self.cells.rows || shape.orow >= self.cells.rows || shape.ocol >= self.cells.columns {
            return NoColour;
        }

        for r in 0 .. shape.cells.rows {
            for c in shape.ocol .. shape.ocol + shape.cells.columns {
                if self.cells[(r,c)].colour != bg {
                    return self.cells[(r,c)].colour;
                }
            }
        }
        for c in 0 .. shape.cells.columns {
            for r in shape.orow .. shape.orow + shape.cells.rows {
                if self.cells[(r,c)].colour != bg {
                    return self.cells[(r,c)].colour;
                }
            }
        }

        NoColour
    }

    // Size 4 and 9
    pub fn colour_squares(&mut self, colour: Colour) {
        let grid = self.clone();

        for (r, c) in grid.cells.keys() {
            if surround_cnt(&grid.cells, r, c, Black) == (8, false) {
                self.cells[(r-1,c-1)].colour = colour;
                self.cells[(r-1,c)].colour = colour;
                self.cells[(r-1,c+1)].colour = colour;
                self.cells[(r,c-1)].colour = colour;
                self.cells[(r,c)].colour = colour;
                self.cells[(r,c+1)].colour = colour;
                self.cells[(r+1,c-1)].colour = colour;
                self.cells[(r+1,c)].colour = colour;
                self.cells[(r+1,c+1)].colour = colour;
            }
        }

        let grid = self.clone();

        for ((r, c), cell) in grid.cells.items() {
            let col = cell.colour;

            if col == Black && r < grid.cells.rows - 1 && c < grid.cells.columns - 1 && grid.cells[(r+1,c)].colour == Black && grid.cells[(r,c+1)].colour == Black && grid.cells[(r+1,c+1)].colour == Black {
                self.cells[(r,c)].colour = colour;
                self.cells[(r,c+1)].colour = colour;
                self.cells[(r+1,c)].colour = colour;
                self.cells[(r+1,c+1)].colour = colour;
            }
        }
    }

    fn find_colour_patches_core(&self, colour: Colour) -> Shapes {
        fn mshape(cells: &mut Matrix<Cell>, colour: Colour) -> Option<(usize, usize, Matrix<Cell>)> {
            // Find starting position
            let rc = cells.items().filter(|(_, cell)| cell.colour == colour).map(|(pos, _)| pos).min();
            //if let Some((0, 0)) = xy {    // This is allowed???
            //    return None;
            //}

            if let Some((r, c)) = rc {
                let reachable = cells.bfs_reachable((r, c), false, |i| cells[i].colour == colour);

                let tlr = *reachable.iter().map(|(r, _)| r).min().unwrap();
                let tlc = *reachable.iter().map(|(_, c)| c).min().unwrap();
                let brr = *reachable.iter().map(|(r, _)| r).max().unwrap();
                let brc = *reachable.iter().map(|(_, c)| c).max().unwrap();

                let mut m = Matrix::new(brr - tlr + 1, brc - tlc + 1, Cell::new(0, 0, 0));

                // Set all cells to correct position
                for r in tlr ..= brr {
                    for c in tlc ..= brc {
                        let cell = &mut m[(r - tlr, c - tlc)];

                        cell.row = r;
                        cell.col = c;
                        cell.colour = colour;
                    }
                }

                // Set cells to correct colour 
                reachable.iter().for_each(|(r, c)| {
                    cells[(*r, *c)].colour = NoColour;
                });

                Some((tlr, tlc, m))
            } else {
                None
            }
        }

        let mut shapes = Shapes::new_sized(self.cells.rows, self.cells.columns);
        let mut cells = self.cells.clone();

        while let Some((or, oc, m)) = mshape(&mut cells, colour) {
            let s = Shape::new(or, oc, &m);

            shapes.add(&s);
            /*  11e1fe23 trans ?
            if self.cells.rows == s.cells.rows && self.cells.columns == s.cells.columns {
                shapes.add(&s);
            } else {
                let sg = self.subgrid(s.orow, s.cells.rows, s.ocol, s.cells.columns);
//sg.show();
                let mut ss = sg.to_shapes_coloured();
                if ss.shapes.len() > 1 {
                    for s2 in ss.shapes.iter_mut() {
//println!("{or}/{oc}");
//s2.show_summary();
//s2.show();
                        s2.to_position_mut(or + s.orow, oc + s.ocol);

                        shapes.add(&s2);
                    }
                } else {
                    shapes.add(&s);
                }
            }
            */
        }

        shapes
    }

    /*
    // Assumes two offset rectangles
    pub fn find_touching(&self, colour: Colour) -> (bool, usize, usize) {
        let is_colour = self.cells[(0,0)].colour == colour;
        let mut rr = 0;
        let mut cc = 0;

        for r in 1 .. self.cells.rows {
            if !is_colour && self.cells[(r,0)].colour == colour ||
                is_colour && self.cells[(r,0)].colour != colour {
                rr = r - 1;
                break;
            }
        }
        for c in 1 .. self.cells.columns {
            if !is_colour && self.cells[(0,c)].colour == colour ||
                is_colour && self.cells[(0,c)].colour != colour {
                cc = c - 1;
                break;
            }
        }

        (is_colour, rr, cc)
    }
    */

    pub fn find_black_patches(&self) -> Shapes {
        self.find_black_patches_limit(12)
    }

    pub fn find_black_patches_limit(&self, limit: usize) -> Shapes {
        if self.cells.rows < limit && self.cells.columns < limit {
            return Shapes::new_sized(0, 0);
        }
        self.find_colour_patches_core(Black)
    }

    pub fn find_colour_patches(&self, colour: Colour) -> Shapes {
        // patches tend to be smaller
        self.find_colour_patches_limit(colour, 2)
    }

    pub fn find_colour_patches_limit(&self, colour: Colour, limit: usize) -> Shapes {
        if self.cells.rows < limit && self.cells.columns < limit {
            return Shapes::new_sized(0, 0);
        }
        self.find_colour_patches_core(colour)
    }

    fn find_gaps(&self, bg: Colour) -> (Vec<usize>, Vec<usize>) {
        let mut rs: Vec<usize> = Vec::new();
        let mut cs: Vec<usize> = Vec::new();

        rs.push(0);
        cs.push(0);

        let mut lastr: isize = -1;
        let mut lastc: isize = -1;

        for ((r, c), cell) in self.cells.items() {
            if cell.colour == bg {
                if c == 0 {
                    if (lastr + 1) as usize != r {
                        rs.push(r);
                    }
                    lastr = r as isize;
                }
                if r == 0 {
                    if (lastc + 1) as usize != c {
                        cs.push(c);
                    }
                    lastc = c as isize;
                }
            }
        }

        (rs, cs)
    }

    pub fn toddle_colour(&self, bg: Colour, fg: Colour) -> Self {
        let s = self.recolour(bg, ToBlack + bg).recolour(fg, bg);

        s.recolour(ToBlack + bg, fg)
    }

    pub fn to_shapes_base_bg(&self, bg: Colour) -> Shapes {
        let mut shapes: Vec<Shape> = Vec::new();
        let (mut rs, mut cs) = self.find_gaps(bg);

        if rs.len() >= 2 && rs[0] == rs[1] || cs.len() > 4 && cs[0] == cs[1] {
            return Shapes::new();   // Trivial shapes
        }

        if self.cells[(self.cells.rows - 1, 0)].colour != bg {
            rs.push(self.cells.rows);
        }
        if self.cells[(0, self.cells.columns - 1)].colour != bg {
            cs.push(self.cells.columns);
        }

        for i in 0 .. rs.len() - 1 {
            let mut sr = rs[i];
            if i > 0 {
                sr += 1;
                // Find start of x range
                for x in sr .. rs[i + 1] {
                    if self.cells[(x, 0)].colour != bg {
                        break;
                    }
                }
            }

            for j in 0 .. cs.len() - 1 {
                let mut sc = cs[j];
                if j > 0 {
                    sc += 1;
                    // Find start of y range
                    for y in sc .. cs[j + 1] {
                        if self.cells[(0, y)].colour != bg {
                            break;
                        }
                    }
                }
                // Find shape
                let xsize = rs[i + 1] - sr;
                let ysize = cs[j + 1] - sc;
                let mut m = Matrix::from_fn(xsize, ysize, |(_, _)| Cell::new_empty());

                for r in sr .. rs[i + 1] {
                    for c in sc .. cs[j + 1] {
                        m[(r - sr, c - sc)].row = r;
                        m[(r - sr, c - sc)].col = c;
                        m[(r - sr, c - sc)].colour = self.cells[(r, c)].colour;
                    }
                }

                shapes.push(Shape::new_cells(&m));
            }
        }
        
        Shapes::new_shapes(&shapes)
    }

    pub fn mid_div_colour(&self) -> Colour {
        let mut colour = NoColour;

        if self.cells.rows % 2 == 1 {
            colour = self.cells[(self.cells.rows / 2,0)].colour;

            for c in 1 .. self.cells.columns {
                if self.cells[(self.cells.rows / 2,c)].colour != colour {
                    return NoColour;
                }
            }
        } else if self.cells.columns % 2 == 1 {
            colour = self.cells[(0,self.cells.columns / 2)].colour;

            for r in 1 .. self.cells.rows {
                if self.cells[(r,self.cells.columns / 2)].colour != colour {
                    return NoColour;
                }
            }
        }

        colour
    }

    pub fn has_shapes_base_bg(&self) -> bool {
        match self.has_bg_grid() {
            NoColour => false,
            colour => {
              let shapes = self.to_shapes_base_bg(colour);
//println!("{shapes:?}"); 

              shapes.len() >= 4
            }
        }
    }

    pub fn copy_part_matrix(rs: usize, cs: usize, rows: usize, cols: usize, cells: &Matrix<Cell>) -> Matrix<Cell> {
        let mut m = Matrix::new(rows, cols, Cell::new(0, 0, 0));

        for r in 0 .. rows {
            for c in 0 .. cols {
                m[(r, c)].row = cells[(rs + r, cs + c)].row;
                m[(r, c)].col = cells[(rs + r, cs + c)].col;
                m[(r, c)].colour = cells[(rs + r, cs + c)].colour;
            }
        }

        m
    }

    // Split cell composed of 2 cells joined at a corner apart
    fn de_diag(or: usize, oc: usize, s: &Shape, shapes: &mut Shapes) {
        if s.cells.rows >= 4 && s.cells.rows % 2 == 0 && s.cells.rows == s.cells.columns {
            let hrow = s.cells.rows / 2;
            let hcol = s.cells.columns / 2;

            if s.cells[(hrow - 1, hcol - 1)].colour == Black && s.cells[(hrow, hcol)].colour == Black && s.cells[(hrow - 1, hcol)].colour != Black && s.cells[(hrow, hcol - 1)].colour != Black {
                let m = Self::copy_part_matrix(hrow, 0, hrow, hcol, &s.cells);
                shapes.add(&Shape::new(or + hrow, oc, &m));
                let m = Self::copy_part_matrix(0, hcol, hrow, hcol, &s.cells);
                shapes.add(&Shape::new(or, oc + hcol, &m));
            } else if s.cells[(hrow - 1, hcol - 1)].colour != Black && s.cells[(hrow, hcol)].colour != Black && s.cells[(hrow - 1, hcol)].colour == Black && s.cells[(hrow, hcol - 1)].colour == Black {
                let m = Self::copy_part_matrix(0, 0, hrow, hcol, &s.cells);
                shapes.add(&Shape::new(or, oc, &m));
                let m = Self::copy_part_matrix(hrow, hcol, hrow, hcol, &s.cells);
                shapes.add(&Shape::new(or + hrow, oc + hcol, &m));
            } else {    // Not diagonally opposed shapes
                shapes.add(s);
            }
        } else {
            shapes.add(s);
        }
    }

    // Find shapes in a grid
    fn to_shapes_base(&self, same_colour: bool, diag: bool, cons: bool, bg: Colour) -> Shapes {
        // TODO Fix training 2204b7a8 - left border
        fn mshape(same_colour: bool, bgc: Colour, bg: Colour, cells: &mut Matrix<Cell>, diag: bool) -> Option<(usize, usize, Matrix<Cell>)> {
            // Find starting position
            let rc = cells.items().filter(|(_, c)| c.colour != bgc && c.colour != bg).map(|(xy, _)| xy).min();

            if let Some((r, c)) = rc {
                let start_colour = cells[(r, c)].colour;
                let reachable = cells.bfs_reachable((r, c), diag, |i| cells[i].colour != bgc && cells[i].colour != bg && (!same_colour || cells[i].colour == start_colour));
                //let mut other: Vec<(usize, usize)> = Vec::new();

                let tlr = *reachable.iter().map(|(r, _)| r).min().unwrap();
                let tlc = *reachable.iter().map(|(_, c)| c).min().unwrap();
                let brr = *reachable.iter().map(|(r, _)| r).max().unwrap();
                let brc = *reachable.iter().map(|(_, c)| c).max().unwrap();

                let mut m = Matrix::new(brr - tlr + 1, brc - tlc + 1, Cell::new(0, 0, 0));

                // S cells to correct position
                for r in tlr ..= brr {
                    for c in tlc ..= brc {
                        let cell = &mut m[(r - tlr, c - tlc)];

                        cell.row = r;
                        cell.col = c;

                        // Find other same colour objects in bounding box
                        // and add to reachable???
                        /*
                        if !reachable.contains(&(x, y)) && (!same_colour || start_colour == cells[(x, y)].colour) {
                            reachable.insert((x, y));
                        }
                        */
                    }
                }
                // Set cells to correct colour 
                reachable.iter().for_each(|(r, c)| {
                    m[(r - tlr, c - tlc)].colour = cells[(*r, *c)].colour;
                    cells[(*r, *c)].colour = bgc;
                });

                Some((tlr, tlc, m))
            } else {
                None
            }
        }

        let mut shapes = Shapes::new_sized(self.cells.rows, self.cells.columns);
        let mut cells = self.cells.clone();
        let bg = if bg == NoColour {
            match self.has_bg_grid_not_sq() {
                NoColour => bg,
                colour => colour
            }
        } else {
            bg
        };

        if bg == NoColour || bg == Black {
            let bgc = Black;

            while let Some((or, oc, m)) = mshape(same_colour, bgc, bg, &mut cells, diag) {
                let s = Shape::new(or, oc, &m);

                /*
                // Quit if Background not Black
                let mut bg = bg;
                let mut bgc = bgc;
                if self.size() == s.size() {
println!("BG not Black {:?}", s.colour);
                    return shapes;
                }
                if self.size() == s.size() && s.colour != Mixed {
//println!("{} == {} {:?} {:?}", self.size(), s.size(), self.colour, s.colour);
                    bgc = s.colour;
                    bg = NoColour;
//println!("--- {:?}", self.cells);
                    cells = self.cells.clone();

                    if let Some((ox, oy, m)) = mshape(same_colour, bgc, bg, &mut cells, diag) {
                        let s = Shape::new(ox, oy, &m);

                        Self::de_diag(ox, oy, &s, &mut shapes);
                    }
                    continue;
                }
                */

                Self::de_diag(or, oc, &s, &mut shapes);
            }
        } else {
            shapes = self.to_shapes_base_bg(bg);

            // Hum, not grid, try ordinary shape conversion
            if shapes.is_empty() {
                // Not Right???
                //while let Some((ox, oy, m)) = mshape(same_colour, Black, Black, &mut cells, diag) {
                while let Some((or, oc, m)) = mshape(same_colour, Black, bg, &mut cells, diag) {
                    let s = Shape::new(or, oc, &m);

                    Self::de_diag(or, oc, &s, &mut shapes);
                }
            }
        }
            //// unfinished 0e671a1a
//shapes.show();

        //shapes.categorise_shapes();

        //shapes.patch_shapes()
        if cons {
            shapes = shapes.consolidate_shapes();
        }

        shapes.shapes.sort();

        shapes
    }

    pub fn to_shapes_coloured(&self) -> Shapes {
        self.to_shapes_base(false, true, false, Black)
    }

    pub fn to_shapes(&self) -> Shapes {
        self.to_shapes_base(true, true, false, Black)
    }

    pub fn to_shapes_cons(&self) -> Shapes {
        self.to_shapes_base(true, true, true, Black)
    }

    pub fn to_shapes_coloured_bg(&self, bg: Colour) -> Shapes {
        self.to_shapes_base(false, true, false, bg)
    }

    pub fn to_shapes_bg(&self, bg: Colour) -> Shapes {
        self.to_shapes_base(true, true, false, bg)
    }

    pub fn to_shapes_bg_cons(&self, bg: Colour) -> Shapes {
        self.to_shapes_base(true, true, true, bg)
    }

    pub fn to_shapes_coloured_cbg(&self) -> Shapes {
        self.to_shapes_base(false, true, false, NoColour)
    }

    pub fn to_shapes_cbg(&self) -> Shapes {
        self.to_shapes_base(true, true, false, NoColour)
    }

    pub fn to_shapes_coloured_sq(&self) -> Shapes {
        self.to_shapes_base(false, false, false, Black)
    }

    pub fn to_shapes_sq(&self) -> Shapes {
        self.to_shapes_base(true, false, false, Black)
    }

    pub fn to_shapes_coloured_bg_sq(&self, bg: Colour) -> Shapes {
        self.to_shapes_base(false, false, false, bg)
    }

    pub fn to_shapes_bg_sq(&self, bg: Colour) -> Shapes {
        self.to_shapes_base(true, false, false, bg)
    }

    pub fn to_shapes_coloured_cbg_sq(&self) -> Shapes {
        self.to_shapes_base(false, false, false, NoColour)
    }

    pub fn to_shapes_cbg_sq(&self) -> Shapes {
        self.to_shapes_base(true, false, false, NoColour)
    }

    pub fn to_shapes_from_grid_gap(&self, gap: usize) -> Shapes {
        self.to_shapes_from_grid_gap_border(gap, 0)
    }

    pub fn to_shapes_from_grid_border(&self, border: usize) -> Shapes {
        self.to_shapes_from_grid_gap_border(1, border)
    }

    pub fn to_shapes_from_grid(&self) -> Shapes {
        self.to_shapes_from_grid_gap_border(1, 0)
    }

    pub fn to_shapes_from_grid_gap_border(&self, gap: usize, border: usize)  -> Shapes {
        let rs = self.height() - border * 2;
        let cs = self.width() - border * 2;
        let r = (rs as f32).sqrt().abs() as usize;
        let c = (cs as f32).sqrt().abs() as usize;

        let mut shapes = Shapes::new_sized(self.height(), self.width());

        for ri in (border .. rs).step_by(r + gap) { 
            for ci in (border .. cs).step_by(c + gap) { 
                if ri + r > rs || ci + c > cs {
                    return Shapes::trivial();
                }
                let sg = self.subgrid(ri, r, ci, c);

                shapes.shapes.push(sg.as_shape_position(ri, ci));
            }
        }

        shapes
    }

    pub fn as_shape(&self) -> Shape {
        if self.size() == 0 {
            return Shape::trivial();
        }

        Shape::new_cells(&self.cells)
    }

    pub fn as_shape_position(&self, r: usize, c: usize) -> Shape {
        if self.size() == 0 {
            return Shape::trivial();
        }

        Shape::new(r, c, &self.cells)
    }

    pub fn as_shapes(&self) -> Shapes {
        if self.size() == 0 {
            return Shapes::new();
        }

        Shapes::new_from_shape(&Shape::new_cells(&self.cells))
    }

    pub fn as_shapes_position(&self, r: usize, c: usize) -> Shapes {
        if self.size() == 0 {
            return Shapes::new();
        }

        Shapes::new_from_shape(&Shape::new(r, c, &self.cells))
    }

    pub fn as_pixel_shapes(&self) -> Shapes {
        if self.size() == 0 {
            return Shapes::new();
        }

        let mut shapes = Shapes::new_sized(self.cells.rows, self.cells.columns);

        for ((r, c), cell) in self.cells.items() {
            if cell.colour != Black {
                let s = Shape::new_sized_coloured_position(r, c, 1, 1, cell.colour);

                shapes.shapes.push(s);
            }
        }

        shapes
    }

    pub fn repeat_rows(&self, start: usize, colour: Colour) -> Vec<usize> {
        let mut res: Vec<usize> = vec![start];
        let row: Vec<Colour> = (0 .. self.cells.rows)
            .map(|r| self.cells[(r,start)].colour)
            .collect();

        for c in start + 1 .. self.cells.columns {
            for r in 0 .. self.cells.rows {
                let lcol = self.cells[(r,c)].colour;

                if row[r] != lcol && lcol != colour {
                    break;
                }
                if r == self.cells.rows - 1 {
                    res.push(c);
                }
            }
        }

        res
    }

    // ea959feb trans - does not work
    pub fn cover_rows(&self, start: usize, colour: Colour) -> Self {
        let reps = self.repeat_rows(start, colour);
        let mut grid = self.clone();

        for rc in 1 .. reps.len() - 1 {
            for c in 0 .. self.cells.columns {
                let r = reps[rc];
//println!("{r}");
                let r1 = reps[rc + 1];
                let curr = &mut grid.cells[(r,c)];
                let curr1 = &self.cells[(r1,c)];

                if curr.colour == colour && curr1.colour != colour {
                    curr.colour = curr1.colour;
                }
            }
        }

        grid
    }

    pub fn repeat_cols(&self, start: usize, colour: Colour) -> Vec<usize> {
        let mut res: Vec<usize> = vec![start];
        let col: Vec<Colour> = (0 .. self.cells.columns)
            .map(|c| self.cells[(start,c)].colour)
            .collect();

        for r in start + 1 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                let lcol = self.cells[(r,c)].colour;

                if col[c] != lcol && lcol != colour {
                    break;
                }
                if c == self.cells.columns - 1 {
                    res.push(r);
                }
            }
        }

        res
    }

    pub fn is_square(&self) -> bool {
        self.cells.rows > 1 && self.cells.rows == self.cells.columns
    }

    pub fn square(&self) -> usize {
        if self.is_square() {
            self.cells.rows * self.cells.columns
        } else {
            0
        }
    }

    pub fn height(&self) -> usize {
        self.cells.rows
    }

    pub fn width(&self) -> usize {
        self.cells.columns
    }

    pub fn row_colour(&self, r: usize) -> Colour {
        let colour = self.cells[(r,0)].colour;

        Self::row_colour_matrix(&self.cells, r, colour)
    }

    pub fn row_colour_matrix(m: &Matrix<Cell>, r: usize, colour: Colour) -> Colour {
        for y in 1 .. m.columns {
            if m[(r,y)].colour != colour {
                return Mixed;
            }
        }

        colour
    }

    pub fn find_pixel(&self, colour: Colour) -> (usize, usize) {
        for ((r, c), cell) in self.cells.items() {
            if cell.colour == colour {
                return (r, c);
            }
        }

        (0, 0)
    }

    pub fn invert_colour(&self) -> Self {
        self.invert_colour_new(self.colour)
    }

    pub fn invert_colour_new(&self, colour: Colour) -> Self {
        if self.colour == Mixed {
            return Self::trivial();
        }

        let mut grid = self.clone();

        for cell in grid.cells.values_mut() {
            cell.colour = if cell.colour == Black {
                colour
            } else {
                Black
            };
        }

        grid
    }

    pub fn blank(&self) -> Self {
        let mut shape = self.clone();

        for cell in shape.cells.values_mut() {
            cell.colour = Black;
        }

        shape
    }

    pub fn col_colour(&self, c: usize) -> Colour {
        let colour = self.cells[(0,c)].colour;

        Self::col_colour_matrix(&self.cells, c, colour)
    }

    pub fn col_colour_matrix(m: &Matrix<Cell>, c: usize, colour: Colour) -> Colour {
        for x in 1 .. m.rows {
            if m[(x,c)].colour != colour {
                return Mixed;
            }
        }

        colour
    }

    pub fn div9(&self) -> bool {
        self.cells.rows == 3 && self.cells.columns == 3
    }

    pub fn is_3x3(&self) -> bool {
        self.cells.rows % 3 == 0 && self.cells.columns % 3 == 0
    }

    pub fn is_full(&self) -> bool {
        for (r, c) in self.cells.keys() {
            if self.cells[(r, c)].colour == Black {
                return false;
            }
        }

        true
    }

    pub fn has_marker(&self, shape: &Shape) -> bool {
        self.has_marker_colour(shape, Black)
    }

    pub fn has_marker_colour(&self, s: &Shape, colour: Colour) -> bool {
        for (r, c) in s.cells.keys() {
            if r == 0 && s.orow >= 3 &&
                self.cells[(s.orow - 1, s.ocol + c)].colour != colour &&
                self.cells[(s.orow - 2, s.ocol + c)].colour != colour &&
                self.cells[(s.orow - 3, s.ocol + c)].colour != colour ||
                c == 0 && s.ocol >= 3 &&
                self.cells[(s.orow + r, s.ocol - 1)].colour != colour &&
                self.cells[(s.orow + r, s.ocol - 2)].colour != colour &&
                self.cells[(s.orow + r, s.ocol - 3)].colour != colour ||
                r == s.cells.rows - 1 &&
                s.orow + s.cells.rows < self.cells.rows - 3 &&
                self.cells[(s.orow + 1, s.ocol + c)].colour != colour &&
                self.cells[(s.orow + 2, s.ocol + c)].colour != colour &&
                self.cells[(s.orow + 3, s.ocol + c)].colour != colour ||
                c == s.cells.columns - 1 &&
                s.ocol + s.cells.columns < self.cells.columns - 3 &&
                self.cells[(s.orow + r, s.ocol + c + 1)].colour != colour &&
                self.cells[(s.orow + r, s.ocol + c + 2)].colour != colour &&
                self.cells[(s.orow + r, s.ocol + c + 3)].colour != colour {
                    return true;
                }
        }

        false
    }

    pub fn find_rectangles(&self) -> Shapes {
        let grid = self.recolour(Black, NoColour);

        let shapes = grid.to_shapes();
        let mut new_shapes = shapes.clone_base();

        for s in shapes.shapes.iter() {
            if s.is_full() && grid.has_marker_colour(s, NoColour) {
                let s = s.recolour(NoColour, Black);
                new_shapes.shapes.push(s.clone());
            }
        }

        new_shapes
    }

    pub fn has_gravity(&self, orientation: usize) -> bool {
        self.has_gravity_colour(orientation, Black)
    }

    pub fn has_gravity_colour(&self, orientation: usize, colour: Colour) -> bool {
        // Must be same shape
        if self.cells.rows != self.cells.columns {
            return false;
        }

        let mut grid = self.clone();
        let mut has_colour = false;

        grid = grid.rotate_90(orientation);

        for r in 0 .. grid.cells.rows {
            let mut prev = grid.cells[(r, 0)].colour;

            if !has_colour {
                has_colour = prev == colour;
            }

            for c in 1 .. grid.cells.columns {
                let next = grid.cells[(r, c)].colour;

                if prev != next && next == colour {
                    return false;
                }

                prev = next;
            }
        }

        has_colour
    }

    pub fn has_gravity_down(&self) -> bool {
        self.has_gravity(3)
    }

    pub fn has_gravity_up(&self) -> bool {
        self.has_gravity(1)
    }

    pub fn has_gravity_left(&self) -> bool {
        self.has_gravity(0)
    }

    pub fn has_gravity_right(&self) -> bool {
        self.has_gravity(0)
    }

    pub fn equals(&self, other: &Self) -> Colour {
        if self.size() == 0 || other.size() == 0 || self.cells.columns != other.cells.columns || self.cells.rows != other.cells.rows {
            return DiffShape;
        }

        for (c1, c2) in self.cells.values().zip(other.cells.values()) {
            if c1.colour.to_base() != c2.colour.to_base() {
                return DiffBlack + c2.colour;
            }
        }

        Same
    }

    pub fn bleach(&self) -> Self {
        let mut shape = self.clone();

        if self.size() == 0 {
            return shape;
        }

        shape.colour = NoColour;

        for cell in shape.cells.values_mut() {
            if cell.colour != Black {
                cell.colour = NoColour;
            }
        }

        shape
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

        for ((sr, sc), (or, oc)) in self.cells.keys().zip(other.cells.keys()) {
            if sr != or || sc != oc {
                return false;
            }
        }

        true
    }

    /*
    pub fn diff_only_same(&self) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if Colour::to_usize(c.colour) / 30 < 10 {
                c.colour = Colour::from_usize(Colour::to_usize(c.colour) / 30);
            } else {
                c.colour = Black;
            }
        }

        g
    }
    */

    pub fn diff_only_diff(&self) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if Colour::to_usize(c.colour) / 40 < 10 {
                c.colour = Colour::from_usize(Colour::to_usize(c.colour) / 40);
            } else {
                c.colour = Black;
            }
        }

        g
    }

    // Experiiments on difference allowed
    /*
    pub fn diff_only(&self, colour: Colour, n: usize) -> Self {
        match n {
           0 => self.diff_only_and(colour),
           1 => self.diff_only_or(colour),
           2 => self.diff_only_xor(colour),
           _ => Self::trivial()
        }
    }
    */

    pub fn diff_only_and(&self, colour: Colour) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if c.colour.is_same() {
                c.colour = colour;
            } else {
                c.colour = Black;
            }
        }

        g
    }

    pub fn diff_only_or(&self, colour: Colour) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if Colour::to_usize(c.colour) > 0 {
                c.colour = colour;
            } else {
                c.colour = Black;
            }
        }

        g
    }

    pub fn diff_only_xor(&self, colour: Colour) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if c.colour == Black || Colour::to_usize(c.colour) > 30 {
                c.colour = Black;
            } else {
                c.colour = colour;
            }
        }

        g
    }

    pub fn diff_only_not(&self, colour: Colour) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if c.colour == Black {
                c.colour = colour;
            } else {
                c.colour = Black;
            }
        }

        g
    }

    pub fn diff_only_same(&self) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if c.colour.is_same() {
                c.colour = c.colour.to_base();
            } else {
                c.colour = Black;
            }
        }

        g
    }

    pub fn diff_black_same(&self, colour: Colour) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if Colour::to_usize(c.colour) == 30 {
                c.colour = colour;
            } else {
                c.colour = Black;
            }
        }

        g
    }

    pub fn diff_other_same(&self, colour: Colour) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if Colour::to_usize(c.colour) > 30 && Colour::to_usize(c.colour) < 40 {
                c.colour = colour;
            } else {
                c.colour = Black;
            }
        }

        g
    }

    // GEN
    pub fn find_unique_colours(&self) -> Vec<Colour> {
        let mut unique_colours = Vec::new();
        for row in 0..self.cells.rows {
            for col in 0..self.cells.columns {
                let colour = self.cells[(row, col)].colour;
                if !unique_colours.contains(&colour) {
                    unique_colours.push(colour);
                }
            }
        }
        unique_colours
    }

    pub fn find_colour_row_order(&self) -> BTreeMap<usize, Colour> {
        let mut colours: BTreeMap<usize, Colour> = BTreeMap::new();
        let mut col_set: Vec<Colour> = Vec::new();

        for row in 0 .. self.cells.rows {
            let colour = self.cells[(row, 0)].colour;

            if colour != Black && !col_set.contains(&colour){
                colours.insert(row, colour);
                col_set.push(colour);
            }

            let colour = self.cells[(row, self.cells.columns - 1)].colour;

            if colour != Black && !col_set.contains(&colour){
                colours.insert(row, colour);
                col_set.push(colour);
            }
        }

        colours
    }

    pub fn diff_only_transparent(&self) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if c.colour != Black {
                c.colour = Colour::to_base(c.colour);
            }
        }

        g
    }

    pub fn diff(&self, other: &Self) -> Option<Self> {
        self.diff_impl(other, true)
    }

    pub fn diff_orig(&self, other: &Self) -> Option<Self> {
        self.diff_impl(other, false)
    }

    pub fn diff_impl(&self, other: &Self, diff: bool) -> Option<Self> {
        // Must be same shape
        if self.cells.columns != other.cells.columns || self.cells.rows != other.cells.rows {
            return None;
        }

        let mut newg = other.clone();

        for (((r, c), c1), ((_, _), c2)) in self.cells.items().zip(other.cells.items()) {
            if c1.colour != c2.colour {
                newg.cells[(r, c)].colour = if c1.colour == Black { 
                    c2.colour + ToBlack
                } else if c2.colour == Black { 
                    c1.colour + FromBlack
                } else if diff {
                    c2.colour + DiffBlack
                } else {
                    c1.colour + OrigBlack
                };
            } else if c1.colour != Black && c1.colour == c2.colour { 
                newg.cells[(r, c)].colour = c1.colour + SameBlack;
            }
        }

        Some(newg)
    }

    pub fn distance(&self, other: &Self) -> f64 {
        let diff = self.diff_impl(other, true);

        // different shapes
        if diff.is_none() {
            return -1.0;
        }

        let diff = diff.unwrap();
        let mut cnt = 0;

        for cell in diff.cells.values() {
            if cell.colour == Black || cell.colour.is_same() {
                cnt += 1;
            }
        }

        cnt as f64 / diff.size() as f64
    }

/*
    pub fn diff_map(&self, to: &Self) -> HashMap<Colour, Colour> {
        let mut map: HashMap<Colour, Colour> = HashMap::new();
        if self.cells.rows != to.cells.rows || self.cells.columns != to.cells.columns {
            return map;
        }
        let inp = self.diff_orig(to);
        let inp = if let None = inp {
            return map;
        } else {
            inp.unwrap()
        };
        let out = self.diff(to);
        let out = if let None = out {
            return map;
        } else {
            out.unwrap()
        };


        //for (((x, y), ic), ((_, _), oc)) in inp.cells.items().zip(out.cells.items()) {
        for ((x, y), c) in inp.cells.items() {
            match Colour::to_usize(c.colour) {
                units @ 0 ..= 9 => {
                    let units = Colour::from_usize(units);
                    map.insert(units, units)
                },
                tens @ 10 ..= 19 => {
                    //let from = Colour::from_usize(tens);
                    let from = self.cells[(x,y)].colour;
                    let to = Colour::from_usize(tens % 10);
                    map.insert(from, to)
                },
                twenties @ 20 ..= 29 => {
                    let from = self.cells[(x,y)].colour;
                    map.insert(from, Black)
                },
                thirties @ 30 ..= 39 => {
                    let from = self.cells[(x,y)].colour;
                    let to = Colour::from_usize(thirties % 10);
                    map.insert(from, to)
                },
                forties @ 40 ..= 49 => {    // Should not happen
                    let from = self.cells[(x,y)].colour;
                    let to = Colour::from_usize(forties % 10);
                    map.insert(from, to)
                },
                fifties @ 50 ..= 59 => {
                    let from = self.cells[(x,y)].colour;
                    let to = Colour::to_usize(out.cells[(x,y)].colour);
                    let to = Colour::from_usize(to % 10);
                    map.insert(from, to)
                },
                other => {
                    let other = Colour::from_usize(other);
                    map.insert(other, other)
                },
            };
        }

        map
    }

    pub fn diff_update(&self, map: HashMap<Colour, Colour>) -> Self {
        let mut ans = self.clone();

        for ((x, y), c) in self.cells.items() {
            if let Some(colour) = map.get(&c.colour) {
                ans.cells[(x,y)].colour = *colour;
            } else {
            }
        }

        ans
    }
*/

    pub fn centre_of(&self) -> (usize, usize) {
        (self.cells.rows / 2, self.cells.columns / 2)
    }

    pub fn centre_of_symmetry(&self) -> (usize, usize) {
        let (min_r, min_c, max_r, max_c) = self.corners();

        (min_r + (max_r - min_r) / 2, min_c + (max_c - min_c) / 2)
    }

    pub fn corners(&self) -> (usize, usize, usize, usize) {
        let mut min_r = usize::MAX;
        let mut min_c = usize::MAX;
        let mut max_r = 0;
        let mut max_c = 0;

        for cell in self.cells.values() {
            if cell.colour != Black {
                min_r = min_r.min(cell.row);
                min_c = min_c.min(cell.col);
                max_r = max_r.max(cell.row);
                max_c = max_c.max(cell.col);
            }
        }

        if min_r == usize::MAX || min_c == usize::MAX {
            return (0, 0, 0, 0);
        }

        (min_r, min_c, max_r, max_c)
    }

    pub fn row_skew(&self) -> isize {
        fn same(cells1: &Matrix<Cell>, cells2: &Matrix<Cell>, offset: usize) -> bool {
            for (c1, c2) in (offset .. cells1.rows).zip(0 .. cells2.rows) {
                if cells1[(c1,0)].colour != cells2[(cells2.rows - 1 - c2,0)].colour {
                    return false;
                }
            }

            true
        }

        if self.cells.rows != self.cells.columns || self.cells.rows < 12 {
            return 0;
        }
        let rows = self.cells.rows;
        let dim = 6;

        let top = self.subgrid(0, dim, 0, 1);
        let bot = self.subgrid(rows - dim, dim, 0, 1);

        for i in 0 .. 3 {
            if same(&top.cells, &bot.cells, i) {
                return i as isize;
            }
        }

        for i in 0 .. 3 {
            if same(&bot.cells, &top.cells, i) {
                return -(i as isize);
            }
        }

        0
    }

    pub fn col_skew(&self) -> isize {
        fn same(cells1: &Matrix<Cell>, cells2: &Matrix<Cell>, offset: usize) -> bool {
            for (c1, c2) in (offset .. cells1.columns).zip(0 .. cells2.columns) {
                if cells1[(0,c1)].colour != cells2[(0,cells2.columns - 1 - c2)].colour {
                    return false;
                }
            }

            true
        }

        if self.cells.rows != self.cells.columns || self.cells.rows < 12 {
            return 0;
        }
        let rows = self.cells.rows;
        let dim = 6;

        let left = self.subgrid(0, 1, 0, dim);
        let right = self.subgrid(0, 1, rows - dim, dim);

        for i in 0 .. 3 {
            if same(&left.cells, &right.cells, i) {
                return i as isize;
            }
        }

        for i in 0 .. 3 {
            if same(&right.cells, &left.cells, i) {
                return -(i as isize);
            }
        }

        0
    }

    pub fn count_diff(&self) -> (usize, usize) {
        let mut total = 0;
        let mut count = 0;

        for c in self.cells.values() {
            if c.colour == NoColour {
                count += 1;
            }

            total += 1;
        }

        (count, total)
    }

    pub fn tile_mut(&mut self, tile: &Self) {
        for r in (0 .. self.height()).step_by(tile.height()) {
            for c in (0 .. self.width()).step_by(tile.width()) {
                for ((tr, tc), cell) in tile.cells.items() {
                    if r + tr < self.height() && c + tc < self.width() {
                        self.cells[(r + tr, c + tc)].colour = cell.colour;
                    }
                }
            }
        }
    }
    
    pub fn roll_right(&self) -> Self {
        let mut grid = self.clone();
        let rows = self.cells.rows;
        let cols = self.cells.columns;
        let first = self.cells.slice(0 .. rows, 0 .. 1);

        if let Ok(first) = first {
            for r in 0 .. rows {
                for c in 1 .. cols {
                    grid.cells[(r,c-1)].colour = self.cells[(r,c)].colour;
                }
            }
            for r in 0 .. rows {
                grid.cells[(r,cols-1)].colour = first[(r,0)].colour;
            }
        }

        grid
    }

    pub fn roll_left(&self) -> Self {
        self.rotate_90(2).roll_right().rotate_90(2)
    }

    pub fn roll_up(&self) -> Self {
        self.rotate_90(3).roll_right().rotate_90(1)
    }

    pub fn roll_down(&self) -> Self {
        self.rotate_90(3).roll_right().rotate_90(3)
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

    pub fn to_vec(&self) -> Vec<Vec<usize>> {
        let mut grid: Vec<Vec<usize>> = vec![vec![0; self.cells.columns]; self.cells.rows];

        for r in 0 .. self.cells.rows {
            for c in 0 .. self.cells.columns {
                let colour: usize = self.cells[(r, c)].colour.to_usize();

                grid[r][c] = colour;
            }
        }

        grid
    }

    pub fn corner_colours(&self) -> (Colour, Colour, Colour, Colour) {
        (self.cells[(0,0)].colour, self.cells[(self.cells.rows - 1, 0)].colour, self.cells[(0, self.cells.columns - 1)].colour, self.cells[(self.cells.rows - 1, self.cells.columns - 1)].colour)
    }

    pub fn corner_idx(&self) -> (Self, Direction) {
        let g = self.subgrid(0, 2, 0, 2);
        if g.no_colours() == 4 {
            return (g, FromUpLeft);
        }
        let g = self.subgrid(self.cells.rows - 2, 2, self.cells.columns - 2, 2);
        if g.no_colours() == 4 {
            return (g, FromDownRight);
        }
        let g = self.subgrid(self.cells.rows - 2, 2, 0, 2);
        if g.no_colours() == 4 {
            return (g, FromDownLeft);
        }
        let g = self.subgrid(0, 2, self.cells.columns - 2, 2);
        if g.no_colours() == 4 {
            return (g, FromUpRight);
        }

        (self.clone(), Other)
    }

    pub fn corner_body(&self, dir: Direction) -> Self {
        let hr = self.cells.rows - 2;
        let hc = self.cells.columns - 2;

        match dir {
            FromUpLeft => self.subgrid(0, hr, 0, hc),
            FromDownRight => self.subgrid(self.cells.rows - hr, hr, self.cells.columns - hc, hc),
            FromDownLeft => self.subgrid(self.cells.rows - hr, hr, 0, hc),
            FromUpRight => self.subgrid(0, hr, self.cells.columns - hc, hc),
            _ => self.clone(),
        }
    }

    pub fn split_n_horizontal(&self, n: usize) -> Vec<Self> {
        if n == 0 {
            return Vec::new();
        }
        let hr = self.cells.rows;
        let hc = self.cells.columns / n;
        let mut gs: Vec<Grid> = Vec::new();

        if hr == 0 || hc == 0 {
            return Vec::new();
        }

        for c in 0 .. n {
            let s = self.subgrid(0, hr, c * hc, hc);

            gs.push(s);
        }
//gs.iter().for_each(|g| g.show());

        gs
    }

    pub fn split_n_vertical(&self, n: usize) -> Vec<Self> {
        if n == 0 {
            return Vec::new();
        }
        let hr = self.cells.rows / n;
        let hc = self.cells.columns;
        let mut gs: Vec<Grid> = Vec::new();

        if hr == 0 || hc == 0 {
            return Vec::new();
        }

        for r in 0 .. n {
            let s = self.subgrid(r * hr, hr, 0, hc);

            gs.push(s);
        }

        gs
    }

    pub fn split_4(&self) -> Vec<Self> {
        let hr = self.cells.rows / 2;
        let hc = self.cells.columns / 2;

        if hr == 0 || hc == 0 || self.cells.rows % 2 != 0 || self.cells.columns % 2 != 0 {
            return Vec::new();
        }

        let s1 = self.subgrid(0, hr, 0, hc);
        let s2 = self.subgrid(0, hr, self.cells.columns - hc, hc);
        let s3 = self.subgrid(self.cells.rows - hr, hr, 0, hc);
        let s4 = self.subgrid(self.cells.rows - hr, hr, self.cells.columns - hc, hc);

        vec![s1, s2, s3, s4]
    }

    pub fn split_4_inline(&self, delimiter: bool) -> Shapes {
        let rows = self.cells.rows;
        let cols = self.cells.columns;
        let mut dr = if delimiter { 1 } else { 0 };
        let mut dc = if delimiter { 1 } else { 0 };
        let mut shapes = Shapes::new_sized(rows, cols);

        if rows % 4 != 0 && cols % 4 != 0 || rows < dr * 3 || cols < dc * 3 {
            return shapes;
        }
//println!("1 {rows} {cols}");
        // too strict?
        if ((rows - dr * 3) % 4 != 0 || cols % 4 != 0) &&
            (rows % 4 != 0 || (cols - dc * 3) % 4 != 0) {
            //return shapes;
            dr = 0;
            dc = 0;
        }
//println!("2 {rows} {cols}");
        /*
        */

        let (s1, s2, s3, s4) = if rows > cols {
            let qr = rows / 4;
            if qr * 4 >= rows {
                dr = 0;
            }

            (self.subgrid(0, qr, 0, cols),
            self.subgrid(qr + dr, qr, 0, cols),
            self.subgrid((qr + dr) * 2, qr, 0, cols),
            self.subgrid((qr + dr) * 3, qr, 0, cols))
        } else {
            let qc = cols / 4;
            if qc * 4 >= cols {
                dc = 0;
            }

            (self.subgrid(0, rows, 0, qc),
            self.subgrid(0, rows, qc + dc, qc),
            self.subgrid(0, rows, (qc + dc) * 2, qc),
            self.subgrid(0, rows, (qc + dc) * 3, qc))
        };

        shapes.shapes = vec![s1.as_shape(), s2.as_shape(), s3.as_shape(), s4.as_shape()];

        shapes
    }

    pub fn full_row(&self, shape: &Shape) -> bool {
        shape.orow == 0 && shape.cells.rows == self.cells.rows
    }

    pub fn full_col(&self, shape: &Shape) -> bool {
        shape.ocol == 0 && shape.cells.columns == self.cells.columns
    }

    pub fn full_dim_split(&self, shapes: &Shapes) -> (Colour, Shapes) {
        let mut div_colour = NoColour;
        let mut horizontal = false;
        let mut shapes = shapes.clone();

        for s in shapes.shapes.iter_mut() {
            if self.full_row(s) {
                horizontal = true;
                //s.to_position_mut(s.orow, s.ocol * 2 + 1);
                div_colour = s.colour;
            } else if self.full_col(s) {
                horizontal = false;
                //s.to_position_mut(s.orow * 2 + 1, s.ocol);
                div_colour = s.colour;
            }
        }

        if horizontal {
            shapes.shapes.sort_by(|a,b| (a.ocol,a.orow).cmp(&(b.ocol,b.orow)));
        } else {
            shapes.shapes.sort_by(|a,b| (a.orow,a.ocol).cmp(&(b.orow,b.ocol)));
        }

        (div_colour, shapes)
    }

    pub fn split_2(&self) -> Shapes {
        if self.cells.rows % 2 != 0 && self.cells.columns % 2 != 0 {
            return Shapes::new();
        }

        let mut shapes = Shapes::new_sized(self.cells.rows, self.cells.columns);
        let hr = self.cells.rows / 2;
        let hc = self.cells.columns / 2;

        //let s1 = if self.cells.rows % 2 == 0 {
        let s1 = if self.cells.rows > self.cells.columns {
            self.subgrid(0, hr, 0, self.cells.columns)
        } else {
            self.subgrid(0, self.cells.rows, 0, hc)
        };
        let s2 = if self.cells.rows > self.cells.columns {
            self.subgrid(hr, self.cells.rows - hr, 0, self.cells.columns)
        } else {
            self.subgrid(0, self.cells.rows, hc, self.cells.columns - hc)
        };

        shapes.shapes = vec![s1.as_shape(), s2.as_shape()];

        shapes
    }

    pub fn full(&self) -> bool {
        if self.cells.rows == 0 {
            return false;
        }
        for cell in self.cells.values() {
            if cell.colour == Black {
                return false;
            }
        }

        true
    }

    pub fn get_patch(&self, r: usize, c: usize, rows: usize, cols: usize) -> Shape {
        match self.cells.slice(r .. r + rows, c .. c + cols) {
            Ok(m) => Shape::new(r, c, &m),
            Err(_e) => {
                //eprintln!("{e}");

                Shape::trivial()
            }
        }

    }

    pub fn fill_patch_mut(&mut self, other: &Shape, or: usize, oc: usize) {
        if self.size() <= other.size() {
            return;
        }

        for (r, c) in other.cells.keys() {
            if self.cells[(or + r, oc + c)].colour == Black {
                self.cells[(or + r, oc + c)].colour = other.cells[(r, c)].colour;
            }
        }
    }


    pub fn fill_patch_coord_mut(&mut self, or: usize, oc: usize, rs: usize, cs: usize, colour: Colour) {
        for r in or .. or + rs {
            for c in oc .. oc + cs {
                self.cells[(r, c)].colour = colour;
            }
        }
    }

    // TODO
    /*
    fn compress_1d(&self) -> Self {
        self.clone()
    }

    fn compress_2d(&self) -> Self {
        self.clone()
    }
    */

    /**
     * Code constraint solver predicates
     */
    pub fn used_in_row(&self, r: usize, colour: Colour) -> bool {
        for c in 0 .. self.cells.columns {
            if self.cells[(r,c)].colour == colour {
                return true;
            }
        }

        false
    }

    pub fn used_in_col(&self, c: usize, colour: Colour) -> bool {
        for r in 0 .. self.cells.rows {
            if self.cells[(r,c)].colour == colour {
                return true;
            }
        }

        false
    }

    // Only needed for Suduko type problems
    /*
    pub fn used_in_subgrid(&self, sr: usize, sc: usize, colour: Colour) -> bool {
        let rl = self.cells.rows;
        let cl = self.cells.columns;
        let rs = (rl as f32).sqrt().abs() as usize;
        let cs = (cl as f32).sqrt().abs() as usize;

        for r in 0 .. rs {
            for c in 0 .. cs {
                if self.cells[(r + sr,c + sc)].colour == colour {
                    return true;
                }
            }
        }

        false
    }

    fn is_valid_move(&self, r: usize, c: usize, colour: Colour) -> bool {
        //!self.used_in_row(r, colour) && !self.used_in_col(c, colour) && !self.used_in_subgrid(r - r % 3, c - c % 3, colour)
        !self.used_in_row(r, colour) && !self.used_in_col(c, colour)
    }
    */

    // Expensive so only call when sure
    pub fn solve(&mut self, pred: &dyn Fn(&Self, usize, usize, Colour) -> bool) -> bool {
        self.solve_depth(pred, 6)
    }

    pub fn solve_depth(&mut self, pred: &dyn Fn(&Self, usize, usize, Colour) -> bool, depth: usize) -> bool {
        fn find_empty_cell(grid: &Grid) -> Option<(usize, usize)> {
            for ((r, c), cell) in grid.cells.items() {
                if cell.colour == Black {
                    return Some((r, c));
                }
            }

            None
        }

        // Might be a bit too conmservative on recursive depth!
        if self.cells.rows != self.cells.columns || self.cells.rows > 9 || depth == 0 {
            return false;
        }

        if let Some((r, c)) = find_empty_cell(self) {
            for n in 1 ..= self.cells.rows {
                let colour = Colour::from_usize(n);

                if pred(self, r, c, colour) {
                    self.cells[(r,c)].colour = colour;

                    if self.solve_depth(pred, depth - 1) {
                        return true;
                    }

                    self.cells[(r,c)].colour = Black;
                }
            }
            return false;
        }
        true
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualifiedGrid {
    pub grid: Grid,
    pub bg: Colour,
    pub shapes: Shapes,
    pub coloured_shapes: Shapes,
    pub black: Shapes,
}

impl QualifiedGrid {
    pub fn new(data: &[Vec<usize>]) -> Self {
        let grid = Grid::duplicate(data);
        let bg = grid.has_bg_grid();
        let shapes = if bg == NoColour {
            grid.to_shapes()
        } else {
            grid.to_shapes_bg(bg)
        };
        let coloured_shapes = if bg == NoColour {
            grid.to_shapes_coloured()
        } else {
            grid.to_shapes_coloured_bg(bg)
        };
        let black = grid.find_black_patches();

        Self { grid: grid.clone(), bg, shapes, coloured_shapes, black }
    }

    pub fn new_cons(data: &[Vec<usize>]) -> Self {
        let grid = Grid::duplicate(data);
        let bg = grid.has_bg_grid();
        let shapes = if bg == NoColour {
            grid.to_shapes_cons()
        } else {
            grid.to_shapes_bg_cons(bg)
        };
        let coloured_shapes = if bg == NoColour {
            grid.to_shapes_coloured()
        } else {
            grid.to_shapes_coloured_bg(bg)
        };
        let black = grid.find_black_patches();

        Self { grid: grid.clone(), bg, shapes, coloured_shapes, black }
    }

    pub fn trivial() -> Self {
        Self { grid: Grid::trivial(), bg: NoColour, shapes: Shapes::trivial(), coloured_shapes: Shapes::trivial(), black: Shapes::trivial() }
    }

    pub fn single_shape_count(&self) -> usize {
        self.shapes.len()
    }

    pub fn colour_shape_count(&self) -> usize {
        self.coloured_shapes.len()
    }

    pub fn has_bg_shape(&self) -> bool {
        for s in self.shapes.shapes.iter() {
            if s.orow == 0 && s.ocol == 0 && s.cells.rows == self.grid.cells.rows && s.cells.columns == self.grid.cells.columns {
                return true;
            }
        }

        false
    }

    pub fn has_bg_coloured_shape(&self) -> bool {
        for s in self.coloured_shapes.shapes.iter() {
            if s.orow == 0 && s.ocol == 0 && s.cells.rows == self.grid.cells.rows && s.cells.columns == self.grid.cells.columns {
                return true;
            }
        }

        false
    }

    /*
    // TODO: improve!
    pub fn grid_likelyhood(&self) -> f32 {
        let shapes_any = self.to_shapes_cbg();
        if shapes_any.len() == 1 { return 1.0; }
        let shapes = self.to_shapes_single_colour_cbg();
        let count = shapes.shapes.len();
        if shapes_any.shapes.len() as f32 / count as f32 > 10.0 { return 1.0; }
        let singles = shapes.shapes.iter().filter(|s| s.size() == 1).count();

        if count == 0 {
            1.0
        } else {
            singles as f32 / count as f32
        }
    }
    */
}
