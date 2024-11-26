use std::collections::{BTreeMap, HashMap};
use pathfinding::prelude::Matrix;
use crate::cats::{Colour, Direction};
use crate::cell::*;
use crate::shape::*;

#[derive(Debug, Hash, Clone, PartialEq, Eq)]
pub struct Grid {
    pub colour: Colour,
    pub cells: Matrix<Cell>,
    //pub cats: BTreeSet<ShapeCategory>,
}

impl Grid {
    pub fn new(rows: usize, cols: usize, colour: Colour) -> Self {
        if cols > 1000 || rows > 1000 {
            return Self::trivial();
        }

        let cells: Matrix<Cell> = Matrix::from_fn(rows, cols, |(x, y)| Cell::new_colour(x, y, colour));

        Self { colour, cells }
    }

    pub fn trivial() -> Self {
        Grid::new(0, 0, Colour::Black)
    }

    pub fn dummy() -> Self {
        Grid::new(2, 2, Colour::Black)
    }

    pub fn is_trivial(&self) -> bool {
        self.cells.rows == 0 && self.cells.columns == 0 && self.colour == Colour::Black
    }

    pub fn is_dummy(&self) -> bool {
        self.cells.rows == 2 && self.cells.columns == 2 && self.colour == Colour::Black
    }

    pub fn new_from_matrix(cells: &Matrix<Cell>) -> Self {
        let mut colour = Colour::NoColour;

        for c in cells.values() {
            if c.colour != Colour::Black {
                if colour == Colour::NoColour {
                    colour = c.colour;
                } else if colour != c.colour {
                    colour = Colour::Mixed;
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

    pub fn duplicate(grid: &[Vec<usize>]) -> Self {
        let x: usize = grid.len();
        let y: usize = grid[0].len();
        let mut colour = Colour::NoColour;
        let cells: Matrix<Cell> = Matrix::from_fn(x, y, |(x, y)| Cell::new(x, y, grid[x][y]));

        for c in cells.values() {
            if c.colour != Colour::Black {
                if colour == Colour::NoColour {
                    colour = c.colour;
                } else if colour != c.colour {
                    colour = Colour::Mixed;
                    break;
                }
            }
        }

        Self { colour, cells }
    }

    pub fn is_empty(&self) -> bool {
        for c in self.cells.values() {
            if c.colour != Colour::Black {
                return false
            }
        }

        true
    }

    pub fn find_colour(&self, colour: Colour) -> Vec<Cell> {
        self.cells.values().filter(|c| c.colour == colour).map(|c| c.clone()).collect()
    }

    pub fn find_colours(&self) -> BTreeMap<Colour, usize> {
        let mut c: BTreeMap<Colour, usize> = BTreeMap::new();

        for cell in self.cells.values() {
            *c.entry(cell.colour).or_insert(0) += 1;
        }

        c
    }

    pub fn find_min_colour(&self) -> Colour {
        let cols = self.find_colours();
        let col = cols.iter()
            .filter(|(&col, _)| col != Colour::Black)
            .min_by(|col, c| col.1.cmp(c.1))
            .map(|(col, _)| col);

        if let Some(col) = col {
            col.clone()
        } else {
            Colour::NoColour
        }
    }

    pub fn find_max_colour(&self) -> Colour {
        let cols = self.find_colours();
        let col = cols.iter()
            .filter(|(&col, _)| col != Colour::Black)
            .max_by(|col, c| col.1.cmp(c.1))
            .map(|(col, _)| col);

        if let Some(col) = col {
            col.clone()
        } else {
            Colour::NoColour
        }
    }

    // TODO crap improve
    pub fn stretch_down(&self) -> Grid {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for y in 0 .. self.cells.columns {
            let mut colour = Colour::NoColour;

            for x in 1 .. self.cells.rows {
                if self.cells[(x - 1,y)].colour != Colour::Black {
                    if colour == Colour::NoColour {
                        colour = self.cells[(x - 1,y)].colour;
                        m[(x - 1,y)].colour = colour;
                    } else {
                        continue;
                    }
                }
                m[(x,y)].row = x;
                m[(x,y)].col = y;
                m[(x,y)].colour = if colour == Colour::NoColour {
                    self.cells[(x,y)].colour
                } else {
                    colour
                };
            }
        }

        Grid::new_from_matrix(&m)
    }

    // diagonal to Edge
    pub fn diagonal_mut(&mut self, x: usize, y: usize, dir: Direction, colour: Colour) {
        match dir {
            Direction::Up => {
                for xi in 0 ..= x {
                    self.cells[(xi, y)].colour = colour;
                }
            },
            Direction::Down => {
                for xi in x .. self.cells.rows - 1 {
                    self.cells[(xi, y)].colour = colour;
                }
            },
            Direction::Left => {
                for yi in 0 ..= y {
                    self.cells[(x, yi)].colour = colour;
                }
            },
            Direction::Right => {
                for yi in y .. self.cells.columns - 1 {
                    self.cells[(x, yi)].colour = colour;
                }
            },
            Direction::TopRight => {
                for (xi, yi) in (0 ..= x).rev().zip(y-1 .. self.cells.columns) {
                    self.cells[(xi, yi)].colour = colour;
                }
            },
            Direction::TopLeft => {
                for (xi, yi) in (0 ..= x).zip(0 ..= y) {
                    self.cells[(xi, yi)].colour = colour;
                }
            },
            Direction::BottomRight => {
                for (xi, yi) in (x .. self.cells.rows).zip(y .. self.cells.columns) {
                    self.cells[(xi, yi)].colour = colour;
                }
            },
            Direction::BottomLeft => {
                for (xi, yi) in (x-1 .. self.cells.rows).zip((0 ..= y).rev()) {
                    self.cells[(xi, yi)].colour = colour;
                }
            },
            Direction::Other => todo!(),
        }
    }

    pub fn is_diag_origin(&self) -> bool {
        if !self.is_square() || self.colour == Colour::Mixed {
            return false;
        }

        for ((r, c), cell) in self.cells.items() {
            if r != c {
                if cell.colour != Colour::Black {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_diag_not_origin(&self) -> bool {
        self.rot_90().is_diag_origin()
    }

    /*
    pub fn stretch_up(&self) -> Grid {
        self.mirrored_x().stretch_up().mirrored_x()
    }
    */

    pub fn gravity_down(&self) -> Grid {
        let mut values: Vec<Colour> = vec![Colour::Black;self.cells.columns];
        let mut counts: Vec<usize> = vec![0;self.cells.columns];

        for ((x, y), c) in self.cells.items() {
            if self.cells[(x,y)].colour != Colour::Black {
                if values[y] == Colour::Black {
                    values[y] = c.colour;
                }

                counts[y] += 1;
            }
        }

        let mut m = self.cells.clone();

        for (x, y) in self.cells.keys() {
            if self.cells[(x,y)].colour == Colour::Black {
               m[(x,y)].colour = values[y];
            }
            if self.cells.rows - x > counts[y] {
               m[(x,y)].colour = Colour::Black;
            }
        }

        Grid::new_from_matrix(&m)
    }

    pub fn gravity_up(&self) -> Grid {
        self.mirrored_x().gravity_down().mirrored_x()
    }

    pub fn move_down(&self) -> Grid {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for y in 0 .. self.cells.columns {
            for x in 1 .. self.cells.rows {
                m[(x,y)].row = x;
                m[(x,y)].col = y;
                m[(x,y)].colour = self.cells[(x - 1,y)].colour;
            }
        }

        Grid::new_from_matrix(&m)
    }

    /*
     * move_up
     * stretch down
     * stretch_up
     */

    pub fn move_up(&self) -> Grid {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for y in 0 .. self.cells.columns {
            for x in 1 .. self.cells.rows {
                m[(x,y)].row = x;
                m[(x,y)].col = y;
                m[(x - 1,y)].colour = self.cells[(x,y)].colour;
            }
        }

        Grid::new_from_matrix(&m)
    }

    pub fn move_right(&self) -> Grid {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for x in 0 .. self.cells.rows {
            for y in 1 .. self.cells.columns {
                m[(x,y)].row = x;
                m[(x,y)].col = y;
                m[(x,y)].colour = self.cells[(x,y - 1)].colour;
            }
        }

        Grid::new_from_matrix(&m)
    }

    pub fn move_left(&self) -> Grid {
        let mut m = Matrix::new(self.cells.rows, self.cells.columns, Cell::new(0, 0, 0));

        for x in 0 .. self.cells.rows {
            for y in 1 .. self.cells.columns {
                m[(x,y)].row = x;
                m[(x,y)].col = y;
                m[(x,y - 1)].colour = self.cells[(x,y)].colour;
            }
        }

        Grid::new_from_matrix(&m)
    }

    pub fn recolour(&self, from: Colour, to: Colour) -> Self {
        let mut grid = self.clone();

        grid.recolour_mut(from, to);

        grid
    }

    pub fn recolour_mut(&mut self, from: Colour, to: Colour) {
        for c in self.cells.values_mut() {
            if c.colour == from || from == Colour::NoColour {
                c.colour = to;
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

    pub fn copy_shape_to_grid(&mut self, shape: &Shape) {
        let ox = shape.cells.rows / 2;
        let oy = shape.cells.columns / 2;

        for (x, y) in shape.cells.keys() {
            // Clip
            if shape.orow < ox || shape.ocol < oy || shape.orow-ox+x >= self.cells.rows || shape.ocol-oy+y >= self.cells.columns {
                continue;
            }

            self.cells[(shape.orow-ox+x, shape.ocol-oy+y)].colour = shape.cells[(x,y)].colour;
        }
    }

    pub fn draw_lines(&mut self, shapes: &[&Shape], overwrite: bool, hv: bool) {
        for (j, s) in shapes.iter().enumerate() {
            for i in j .. shapes.len() {
                if hv {
                    self.draw_line_x(s, shapes[i], s.colour, false);
                } else {
                    self.draw_line_y(s, shapes[i], s.colour, false);
                }
            }
        }
        for (j, s) in shapes.iter().enumerate() {
            for i in j .. shapes.len() {
                if hv {
                    self.draw_line_y(s, shapes[i], s.colour, overwrite);
                } else {
                    self.draw_line_x(s, shapes[i], s.colour, overwrite);
                }
            }
        }
    }

    pub fn draw_line_x(&mut self, s1: &Shape, s2: &Shape, colour: Colour, overwrite: bool) {
        let x1 = s1.orow;
        let y1 = s1.ocol;
        let x2 = s2.orow;
        let y2 = s2.ocol;

        if x1 == x2 && y1 != y2 {
            let thick = s1.cells.columns;

            for y in y1.min(y2)+1 .. y1.max(y2) {
                for t in 0 .. thick {
                    if overwrite || self.cells[(x1+t,y)].colour == Colour::Black {
                        self.cells[(x1+t,y)].colour = colour;
                    }
                }
            }
        }
    }

    pub fn draw_line_y(&mut self, s1: &Shape, s2: &Shape, colour: Colour, overwrite: bool) {
        let x1 = s1.orow;
        let y1 = s1.ocol;
        let x2 = s2.orow;
        let y2 = s2.ocol;

        if y1 == y2 && x1 != x2 {
            let thick = s1.cells.rows;

            for x in x1.min(x2)+1 .. x1.max(x2) {
                for t in 0 .. thick {
                    if overwrite || self.cells[(x,y1+t)].colour == Colour::Black {
                        self.cells[(x,y1+t)].colour = colour;
                    }
                }
            }
        }
    }

    pub fn pixels(&self) -> usize {
        self.cells.values()
            .filter(|c| c.colour != Colour::Black).
            count()
    }

    pub fn size(&self) -> usize {
        self.cells.columns * self.cells.rows
    }

    pub fn same_size(&self, other: &Self) -> bool {
        self.size() == other.size()
    }

    pub fn dimensions(&self) -> (usize, usize) {
        (self.cells.columns, self.cells.rows)
    }

    pub fn fill_border(&mut self) {
        let hr = self.cells.rows / 2;
        let hc = self.cells.columns / 2;

        for layer in 0 .. hr.min(hc) {
            let mut cc = Colour::NoColour;

            for i in layer .. self.cells.rows - layer {
                let c = self.cells[(i,layer)].colour;

                if c != Colour::Black && cc == Colour::NoColour {
                    cc = c;
                } else if c != Colour::Black && c != cc {
                    return;
                }
            }
            for i in layer .. self.cells.columns - layer {
                let c = self.cells[(layer,i)].colour;

                if c != Colour::Black && cc == Colour::NoColour {
                    cc = c;
                } else if c != Colour::Black && c != cc {
                    return;
                }
            }
//println!("Layers: {layer} {:?}", cc);

            for i in layer .. self.cells.rows - layer {
                let c = self.cells[(i,layer)].colour;

                if c == Colour::Black {
                    self.cells[(i,layer)].colour = cc;
                }
            }
            for i in layer .. self.cells.columns - layer {
                let c = self.cells[(layer, i)].colour;

                if c == Colour::Black {
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

    pub fn has_colour(&self, tlx: usize, tly: usize, xlen: usize, ylen: usize, colour: Colour) -> bool {
        for x in tlx .. xlen {
            for y in tly .. ylen {
                if self.cells[(x,y)].colour == colour {
                    return true;
                }
            }
        }

        false
    }

    pub fn find_x_seq(&self, sx: usize, sy: usize, seq: &[Colour], width: usize) -> (usize, usize) {
        let mut cnt = 0;
        let mut yp = 0;
        let mut xs = usize::MAX;
        let mut ys = usize::MAX;

        'outer:
        for y in 0 .. self.cells.columns - width {
            for x in 0 .. self.cells.rows - seq.len() {
                if x == sx+1 && y <= sy { continue 'outer}; 
                let c = self.cells[(x,y)].clone();
                if yp != y {
                    yp = y;
                    cnt = 0;
                    xs = usize::MAX;
                    ys = usize::MAX;
                }
                if seq[cnt] == c.colour {
                    if !self.has_colour(x, y, seq.len(), width, Colour::Black) {
                        if cnt == 0 {
                            xs = x;
                            ys = y;
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
                    xs = usize::MAX;
                    ys = usize::MAX;
                }
            }
        }

        (xs, ys)
    }

    pub fn find_y_seq(&self, sx: usize, sy: usize, seq: &[Colour], length: usize) -> (usize, usize) {
        let mut cnt = 0;
        let mut xp = 0;
        let mut xs = usize::MAX;
        let mut ys = usize::MAX;

        'outer:
        for x in 0 .. self.cells.rows - length {
            for y in 0 .. self.cells.columns - seq.len() {
                if y == sy+1 && x <= sx { continue 'outer}; 
                let c = self.cells[(x,y)].clone();
                if xp != x {
                    xp = x;
                    cnt = 0;
                    xs = usize::MAX;
                    ys = usize::MAX;
                }
                if seq[cnt] == c.colour {
                    if !self.has_colour(x, y, length, seq.len(), Colour::Black) {
                        if cnt == 0 {
                            xs = x;
                            ys = y;
                        }
                        cnt += 1;
                        if cnt == seq.len() {
                            break 'outer;
                        }
                    }
                } else if cnt > 0 {
                    cnt = 0;
                    xs = usize::MAX;
                    ys = usize::MAX;
                }
            }
        }

        (xs, ys)
    }

    pub fn colour_every_nxn_for_m(colour: Colour, side: usize, n: usize, m: usize) -> Grid {
        if m == 0 || n == 0 {
            return Grid::trivial();
        }
        let mut grid = Grid::new(side, side, Colour::Black);
        let mut count = 0;

        'outer:
        for x in 0 .. side {
            for y in 0 .. side {
                if (x + grid.cells.rows * y) % n == 0 {
                    grid.cells[(x, y)].colour = colour;
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
        let mut min_x: usize = usize::MAX;
        let mut max_x: usize = 0;
        let mut min_y: usize = usize::MAX;
        let mut max_y: usize = 0;

        for ((x, y), c) in self.cells.items() {
            if c.colour == colour {
                min_x = x.min(min_x);
                max_x = x.max(max_x);
                min_y = y.min(min_y);
                max_y = y.max(max_y);
            }
        }

        (max_x - min_x + 1, max_y - min_y + 1)
    }

    pub fn bigger(&self, other: &Self) -> bool {
        self.size() > other.size()
    }

    pub fn smaller(&self, other: &Self) -> bool {
        self.size() < other.size()
    }

    pub fn cell_count(&self) -> usize {
        self.cells.values().filter(|c| c.colour != Colour::Black).count()
    }

    pub fn flood_fill(&self, x: usize, y: usize, ignore_colour: Colour, new_colour: Colour) -> Self {
        let mut grid = self.clone();

        grid.flood_fill_in_situ(x, y, ignore_colour, new_colour);

        grid
    }

    pub fn flood_fill_in_situ(&mut self, x: usize, y: usize, ignore_colour: Colour, new_colour: Colour) {
        let reachable = self.cells.bfs_reachable((x, y), false, |i| self.cells[i].colour == Colour::Black || self.cells[i].colour == ignore_colour);

        reachable.iter().for_each(|&i| self.cells[i].colour = new_colour);
    }

    pub fn flood_fill_from_seeds(&self, ignore_colour: Colour, new_colour: Colour) -> Self {
        let mut grid = self.clone();

        let coloured: Vec<(usize, usize)> = grid.cells.items()
            .filter(|(_, c)| c.colour == ignore_colour)
            .map(|(i, _)| i)
            .collect();

        coloured.iter()
            .for_each(|(x, y)| grid.flood_fill_in_situ(*x, *y, ignore_colour, new_colour));

        grid
    }

    pub fn subgrid(&self, tlx: usize, sx: usize, tly: usize, sy: usize) -> Self {
        let mut m = Matrix::new(sx, sy, Cell::new(0, 0, 0));

        for x in 0 ..  sx {
            for y in 0 .. sy {
                m[(x,y)].row = self.cells[(x + tlx,y + tly)].row;
                m[(x,y)].col = self.cells[(x + tlx,y + tly)].col;
                m[(x,y)].colour = self.cells[(x + tlx,y + tly)].colour;
            }
        }

        Self::new_from_matrix(&m)
    }

    pub fn subgrid2(&self, tlx: usize, sx: usize, tly: usize, sy: usize) -> Self {
//println!("{} {} {} {}", sx, sy, self.cells.rows, self.cells.columns);
        let mut m = Matrix::new(sx, sy, Cell::new(0, 0, 0));

        for x in 0 .. sx {
            for y in 0 .. sy {
                if sx < self.cells.rows {
                    m[(x,y)].row = self.cells[(x + tlx - 1,y)].row - sx - 1;
                    m[(x,y)].col = self.cells[(x + tlx - 1,y)].col;
                    m[(x,y)].colour = self.cells[(x + tlx - 1,y)].colour;
                } else {
                    m[(x,y)].row = self.cells[(x,y + tly - 1)].row;
                    m[(x,y)].col = self.cells[(x,y + tly - 1)].col - sy - 1;
                    m[(x,y)].colour = self.cells[(x,y + tly - 1)].colour;
                }
            }
        }
//Self::new_from_matrix(&m).show();

        Self::new_from_matrix(&m)
    }

    pub fn cell_colour_cnt_map(&self) -> BTreeMap<Colour, usize>  {
        let mut h: BTreeMap<Colour, usize> = BTreeMap::new();

        for c in self.cells.values() {
            if c.colour != Colour::Black {
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
            .filter(|((x, _), c)| *x == 0 && c.colour != Colour::Black)
            .count() == self.cells.columns - 1
    }

    pub fn border_bottom(&self) -> bool {
        self.cells.items()
            .filter(|((x, _), c)| *x == self.cells.columns - 1 && c.colour != Colour::Black)
            .count() == self.cells.columns - 1
    }

    pub fn border_left(&self) -> bool {
        self.cells.items()
            .filter(|((_, y), c)| *y == 0 && c.colour != Colour::Black)
            .count() == self.cells.rows - 1
    }

    pub fn border_right(&self) -> bool {
        self.cells.items()
            .filter(|((_, y), c)| *y == self.cells.rows - 1 && c.colour != Colour::Black)
            .count() == self.cells.rows - 1
    }

    pub fn mirrored_x(&self) -> Self {
        let mut m = self.cells.flipped_ud();

        for (x, y) in self.cells.keys() {
            m[(x, y)].row = x;
            m[(x, y)].col = y;
        }
        
        Self::new_from_matrix(&m)
    }

    pub fn mirrored_y(&self) -> Self {
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

    pub fn is_mirror_x(&self) -> bool {
        if self.cells.len() <= 4 {
            return false
        }
        let copy = self.cells.flipped_ud();

        Self::compare_colour(&self.cells, &copy)
    }

    pub fn is_mirror_y(&self) -> bool {
        if self.cells.len() <= 4 {
            return false
        }
        let copy = self.cells.flipped_lr();

        Self::compare_colour(&self.cells, &copy)
    }

    pub fn is_symmetric(&self) -> bool {
        self.is_mirror_x() && self.is_mirror_y()
    }

    pub fn is_mirror_offset_x(&self, skew: i32) -> bool {
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

    pub fn is_mirror_offset_y(&self, skew: i32) -> bool {
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

    pub fn is_panelled_x(&self) -> bool {
        let len: usize = self.cells.rows;
        let half: usize = len / 2;

        if len < 4 || half == 0 { return false }

        let offset: usize = if len % 2 == 0 { 0 } else { 1 };

        for y in 0 .. self.cells.columns {
            for x in 0 .. half {
                if self.cells[(x, y)].colour != self.cells[(half + offset + x, y)].colour {
                    return false;
                }
            }
        }

        true
    }

    pub fn is_panelled_y(&self) -> bool {
        let len: usize = self.cells.columns;
        let half: usize = len / 2;

        if len < 4 || half == 0 { return false }

        let offset: usize = if len % 2 == 0 { 0 } else { 1 };

        for x in 0 .. self.cells.rows {
            for y in 0 .. half {
                if self.cells[(x, y)].colour != self.cells[(x, half + offset + y)].colour {
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

        for (x, y) in self.cells.keys() {
            m[(x, y)].row = x;
            m[(x, y)].col = y;
        }
        
        Self::new_from_matrix(&m)
    }

    pub fn inv_transposed(&self) -> Self {
        Self::new_from_matrix(&self.cells.rotated_cw(2).transposed())
    }

    pub fn rotate_90(&mut self, times: usize) {
        if self.cells.rows != self.cells.columns {
            return;
        }

        self.cells.rotate_cw(times);
    }

    pub fn rotate_270(&mut self, times: usize) {
        if self.cells.rows != self.cells.columns {
            return;
        }

        self.cells.rotate_ccw(times);
    }

    pub fn rotated_90(&self, times: usize) -> Self {
        if self.cells.rows != self.cells.columns {
            return self.clone();
        }

        let mut m: Matrix<Cell> = self.cells.clone();

        m.rotate_cw(times);

        for (x, y) in self.cells.keys() {
            m[(x, y)].row = x;
            m[(x, y)].col = y;
        }
        
        Self::new_from_matrix(&m)
    }

    pub fn rotated_270(&self, times: usize) -> Self {
        if self.cells.rows != self.cells.columns {
            return self.clone();
        }

        let mut m: Matrix<Cell> = self.cells.clone();

        m.rotate_ccw(times);

        for (x, y) in self.cells.keys() {
            m[(x, y)].row = x;
            m[(x, y)].col = y;
        }
        
        Self::new_from_matrix(&m)
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

    pub fn colours(&self) -> HashMap<Colour, usize> {
        let mut h: HashMap<Colour, usize> = HashMap::new();

        for c in self.cells.values() {
            if c.colour != Colour::Black {
                *h.entry(c.colour).or_insert(0) += 1;
            }
        }

        h
    }

    pub fn min_colour(&self) -> Colour {
        let h = self.colours();

        match h.iter().min_by(|col, c| col.1.cmp(c.1)) {
            None => Colour::NoColour,
            Some((colour,_)) => *colour
        }
    }

    pub fn max_colour(&self) -> Colour {
        let h = self.colours();

        match h.iter().max_by(|col, c| col.1.cmp(c.1)) {
            None => Colour::NoColour,
            Some((colour,_)) => *colour
        }
    }

    pub fn no_colours(&self) -> usize {
        self.colours().len()
    }

    pub fn single_colour(&self) -> Colour {
        let h: HashMap<Colour, usize> = self.colours();

        if h.is_empty() {
            Colour::NoColour
        } else if h.len() == 1 {
            //h.keys().max().unwrap().clone()
            *h.keys().max().unwrap()
        } else {
            Colour::Mixed
        }
    }

    fn draw_x(&mut self, from_y: usize, to_y: usize, x: usize, colour: Colour) -> usize {
        for y in from_y .. to_y {
            self.cells[(x, y)].colour = colour;
        }

        if to_y > 0 { to_y - 1 } else { 0 }
    }

    fn draw_y(&mut self, from_x: usize, to_x: usize, y: usize, colour: Colour) -> usize {
        for x in from_x .. to_x {
            self.cells[(x, y)].colour = colour;
        }

        if to_x > 0 { to_x - 1 } else { 0 }
    }

    pub fn do_circle(&self, colour: Colour, spiral: bool) -> Self {
        let inc = if spiral { 2 } else { 0 };
        let mut copy = self.clone();
        let mut cinc = 0;
        let mut sx = 0;
        let mut sy = 1;
        let mut rows = self.cells.rows;
        let mut cols = self.cells.columns;

        // First round
        let mut cy = copy.draw_x(sx, cols, 0, colour);
        let mut cx = copy.draw_y(sy, rows, cy, colour);
        cy = copy.draw_x(sx, cols - 1, cx, colour);
        if spiral { sy += 1};
        copy.draw_y(sy, rows - 1, 0, colour);

        if spiral {
            while sx + 1 < cy { 
                sx += 1;
                sy += 1;
                rows -= inc;
                cols -= inc;
                cinc += inc;

                cy = copy.draw_x(sx, cols, cinc, colour);
                cx = copy.draw_y(sy, rows, cy, colour);
                sx += 1;
                cy = copy.draw_x(sx, cols - 1, cx, colour);
                sy += 1;
                copy.draw_y(sy, rows - 1, cinc, colour);
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
        for ((x, y), c) in m.items() {
            if x != px {
                println!();
                px = x;
            } else if y != 0 {
                print!(" ");
            }

            let c = c.colour.to_usize();

            if c == 100 {
                print!("{}", if !diff && io { "##" } else { "#" });
            } else if c == 101 {
                print!("{}", if !diff && io { "**" } else { "*" });
            } else if diff && !io {
                if c >= 10 { print!("#", ) } else { print!("{c}") };
            } else if !diff && io {
                print!("{c:0>2}");
            } else if !diff && !io {
                if c >= 20 { print!("#") } else { print!("{}", c % 10) };
            } else {
                print!("{}", c % 10);
            }
        }
    }

    fn show_any(&self, diff: bool, io: bool) {
        println!("--------Grid--------");
        Grid::show_matrix(&self.cells, diff, io);
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
        let mut colour = Colour::NoColour;

//println!("---- {colour:?} {square}");
        if self.cells.len() <= 9 || (square && self.cells.rows != self.cells.columns) {
            return colour;
        }

        let mut nox = false;    // may be only a y axis

        'outerx:
        for x in 0 .. self.cells.rows {
            if can_be_black || self.cells[(x, 0)].colour != Colour::Black {
                colour = self.cells[(x, 0)].colour;
            } else {
                continue;
            }

            if colour == self.cells[(x, 0)].colour {
                for c in 0 .. self.cells.columns {
                    if colour != self.cells[(x, c)].colour {
                        continue 'outerx;
                    }
                    nox = true;
                }
                break;  // we have found a candidate
            }
        }

        'outery:
        for y in 0 .. self.cells.columns {
            if nox {
                if can_be_black || self.cells[(0, y)].colour != Colour::Black {
                    colour = self.cells[(0, y)].colour;
                } else {
                    continue;
                }
            }
            if colour == self.cells[(0, y)].colour {
                for r in 0 .. self.cells.rows {
                    if colour != self.cells[(r, y)].colour {
                        continue 'outery;
                    }
                }

                if !can_be_black && colour == Colour::Black {
                    break;
                }

                return colour;  // we have found one
            }
        }

        Colour::NoColour
    }

    // TODO
    pub fn ray(&self, _direction: (i8, i8)) -> Shape {
        Shape::new_empty()
    }

    // TODO
    pub fn r#move(&self, _direction: (i8, i8)) -> Shape {
        Shape::new_empty()
    }

    pub fn find_colour_patches(&self, colour: Colour) -> Shapes {
        fn mshape(cells: &mut Matrix<Cell>, colour: Colour) -> Option<(usize, usize, Matrix<Cell>)> {
            // Find starting position
            let xy = cells.items().filter(|(_, c)| c.colour == colour).map(|(pos, _)| pos).min();
            if let Some((0, 0)) = xy { return None; }

            if let Some((x, y)) = xy {
                let reachable = cells.bfs_reachable((x, y), false, |i| cells[i].colour == colour);

                let tlx = *reachable.iter().map(|(x, _)| x).min().unwrap();
                let tly = *reachable.iter().map(|(_, y)| y).min().unwrap();
                let brx = *reachable.iter().map(|(x, _)| x).max().unwrap();
                let bry = *reachable.iter().map(|(_, y)| y).max().unwrap();

                let mut m = Matrix::new(brx - tlx + 1, bry - tly + 1, Cell::new(0, 0, 0));

                // Set all cells to correct position
                for x in tlx ..= brx {
                    for y in tly ..= bry {
                        let cell = &mut m[(x - tlx, y - tly)];

                        cell.row = x;
                        cell.col = y;
                    }
                }

                // Set cells to correct colour 
                reachable.iter().for_each(|(x, y)| {
                    cells[(*x, *y)].colour = Colour::NoColour;
                });

                Some((tlx, tly, m))
            } else {
                None
            }
        }

        let mut shapes = Shapes::new_sized(self.cells.rows, self.cells.columns);
        let mut cells = self.cells.clone();

        while let Some((ox, oy, m)) = mshape(&mut cells, colour) {
            let s = Shape::new(ox, oy, &m);

            shapes.add(&s);
        }

        shapes
    }

    pub fn find_black_patches(&self) -> Shapes {
        if self.cells.rows < 12 && self.cells.columns < 12 {
            //return Shapes::new_sized(self.cells.rows, self.cells.columns);
            return Shapes::new_sized(0, 0);
        }
        self.find_colour_patches(Colour::Black)
    }

    fn find_gaps(&self, bg: Colour) -> (Vec<usize>, Vec<usize>) {
        let mut xs: Vec<usize> = Vec::new();
        let mut ys: Vec<usize> = Vec::new();

        xs.push(0);
        ys.push(0);

        let mut lastx: isize = -1;
        let mut lasty: isize = -1;

        for ((x, y), c) in self.cells.items() {
            if c.colour == bg {
                if y == 0 {
                    if (lastx + 1) as usize != x {
                        xs.push(x);
                    }
                    lastx = x as isize;
                }
                if x == 0 {
                    if (lasty + 1) as usize != y {
                        ys.push(y);
                    }
                    lasty = y as isize;
                }
            }
        }

        (xs, ys)
    }

    pub fn toddle_colour(&self, bg: Colour, fg: Colour) -> Self {
        let s = self.recolour(bg, Colour::ToBlack + bg).recolour(fg, bg);

        s.recolour(Colour::ToBlack + bg, fg)
    }

    pub fn to_shapes_base_bg(&self, bg: Colour) -> Shapes {
        let mut shapes: Vec<Shape> = Vec::new();
        let (mut xs, mut ys) = self.find_gaps(bg);

        if xs.len() >= 2 && xs[0] == xs[1] || ys.len() > 4 && ys[0] == ys[1] {
            return Shapes::new();   // Trivial shapes
        }

        if self.cells[(self.cells.rows - 1, 0)].colour != bg {
            xs.push(self.cells.rows);
        }
        if self.cells[(0, self.cells.columns - 1)].colour != bg {
            ys.push(self.cells.columns);
        }

        for i in 0 .. xs.len() - 1 {
            let mut sx = xs[i];
            if i > 0 {
                sx += 1;
                // Find start of x range
                for x in sx .. xs[i + 1] {
                    if self.cells[(x, 0)].colour != bg {
                        break;
                    }
                }
            }

            for j in 0 .. ys.len() - 1 {
                let mut sy = ys[j];
                if j > 0 {
                    sy += 1;
                    // Find start of y range
                    for y in sy .. ys[j + 1] {
                        if self.cells[(0, y)].colour != bg {
                            break;
                        }
                    }
                }
                // Find shape
                let xsize = xs[i + 1] - sx;
                let ysize = ys[j + 1] - sy;
                let mut m = Matrix::from_fn(xsize, ysize, |(_, _)| Cell::new_empty());

                for x in sx .. xs[i + 1] {
                    for y in sy .. ys[j + 1] {
                        m[(x - sx, y - sy)].row = x;
                        m[(x - sx, y - sy)].col = y;
                        m[(x - sx, y - sy)].colour = self.cells[(x, y)].colour;
                    }
                }

                shapes.push(Shape::new_cells(&m));
            }
        }
        
        Shapes::new_shapes(&shapes)
    }

    pub fn has_shapes_base_bg(&self) -> bool {
        match self.has_bg_grid() {
            Colour::NoColour => false,
            colour => {
              let shapes = self.to_shapes_base_bg(colour);
//println!("{shapes:?}"); 

              shapes.len() >= 4
            }
        }
    }

    pub fn copy_part_matrix(xs: usize, ys: usize, rows: usize, cols: usize, cells: &Matrix<Cell>) -> Matrix<Cell> {
        let mut m = Matrix::new(rows, cols, Cell::new(0, 0, 0));

        for x in 0 .. rows {
            for y in 0 .. cols {
                m[(x, y)].row = cells[(xs + x, ys + y)].row;
                m[(x, y)].col = cells[(xs + x, ys + y)].col;
                m[(x, y)].colour = cells[(xs + x, ys + y)].colour;
            }
        }

        m
    }

    // Split cell composed of 2 cells joined at a corner apart
    fn de_diag(ox: usize, oy:usize, s: &Shape, shapes: &mut Shapes) {
        if s.cells.rows >= 4 && s.cells.rows % 2 == 0 && s.cells.rows == s.cells.columns {
            let hrow = s.cells.rows / 2;
            let hcol = s.cells.columns / 2;

            if s.cells[(hrow - 1, hcol - 1)].colour == Colour::Black && s.cells[(hrow, hcol)].colour == Colour::Black && s.cells[(hrow - 1, hcol)].colour != Colour::Black && s.cells[(hrow, hcol - 1)].colour != Colour::Black {
                let m = Self::copy_part_matrix(hrow, 0, hrow, hcol, &s.cells);
                shapes.add(&Shape::new(ox + hrow, oy, &m));
                let m = Self::copy_part_matrix(0, hcol, hrow, hcol, &s.cells);
                shapes.add(&Shape::new(ox, oy + hcol, &m));
            } else if s.cells[(hrow - 1, hcol - 1)].colour != Colour::Black && s.cells[(hrow, hcol)].colour != Colour::Black && s.cells[(hrow - 1, hcol)].colour == Colour::Black && s.cells[(hrow, hcol - 1)].colour == Colour::Black {
                let m = Self::copy_part_matrix(0, 0, hrow, hcol, &s.cells);
                shapes.add(&Shape::new(ox, oy, &m));
                let m = Self::copy_part_matrix(hrow, hcol, hrow, hcol, &s.cells);
                shapes.add(&Shape::new(ox + hrow, oy + hcol, &m));
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
            let xy = cells.items().filter(|(_, c)| c.colour != bgc && c.colour != bg).map(|(xy, _)| xy).min();

            if let Some((x, y)) = xy {
                let start_colour = cells[(x, y)].colour;
                let reachable = cells.bfs_reachable((x, y), diag, |i| cells[i].colour != bgc && cells[i].colour != bg && (!same_colour || cells[i].colour == start_colour));
                //let mut other: Vec<(usize, usize)> = Vec::new();

                let tlx = *reachable.iter().map(|(x, _)| x).min().unwrap();
                let tly = *reachable.iter().map(|(_, y)| y).min().unwrap();
                let brx = *reachable.iter().map(|(x, _)| x).max().unwrap();
                let bry = *reachable.iter().map(|(_, y)| y).max().unwrap();

                let mut m = Matrix::new(brx - tlx + 1, bry - tly + 1, Cell::new(0, 0, 0));

                // S cells to correct position
                for x in tlx ..= brx {
                    for y in tly ..= bry {
                        let cell = &mut m[(x - tlx, y - tly)];

                        cell.row = x;
                        cell.col = y;

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
                reachable.iter().for_each(|(x, y)| {
                    m[(x - tlx, y - tly)].colour = cells[(*x, *y)].colour;
                    cells[(*x, *y)].colour = bgc;
                });

                Some((tlx, tly, m))
            } else {
                None
            }
        }

        let mut shapes = Shapes::new_sized(self.cells.rows, self.cells.columns);
        let mut cells = self.cells.clone();
        let bg = if bg == Colour::NoColour {
            match self.has_bg_grid_not_sq() {
                Colour::NoColour => bg,
                colour => colour
            }
        } else {
            bg
        };

        if bg == Colour::NoColour || bg == Colour::Black {
            let bgc = Colour::Black;

            while let Some((ox, oy, m)) = mshape(same_colour, bgc, bg, &mut cells, diag) {
                let s = Shape::new(ox, oy, &m);

                /*
                // Quit if Background not Black
                let mut bg = bg;
                let mut bgc = bgc;
                if self.size() == s.size() {
println!("BG not Black {:?}", s.colour);
                    return shapes;
                }
                if self.size() == s.size() && s.colour != Colour::Mixed {
//println!("{} == {} {:?} {:?}", self.size(), s.size(), self.colour, s.colour);
                    bgc = s.colour;
                    bg = Colour::NoColour;
//println!("--- {:?}", self.cells);
                    cells = self.cells.clone();

                    if let Some((ox, oy, m)) = mshape(same_colour, bgc, bg, &mut cells, diag) {
                        let s = Shape::new(ox, oy, &m);

                        Self::de_diag(ox, oy, &s, &mut shapes);
                    }
                    continue;
                }
                */

                Self::de_diag(ox, oy, &s, &mut shapes);
            }
        } else {
            shapes = self.to_shapes_base_bg(bg);

            // Hum, not grid, try ordinary shape conversion
            if shapes.is_empty() {
                // Not Right???
                //while let Some((ox, oy, m)) = mshape(same_colour, Colour::Black, Colour::Black, &mut cells, diag) {
                while let Some((ox, oy, m)) = mshape(same_colour, Colour::Black, bg, &mut cells, diag) {
                    let s = Shape::new(ox, oy, &m);

                    Self::de_diag(ox, oy, &s, &mut shapes);
                }
            }
        }
//shapes.show();

        //shapes.categorise_shapes(same_colour);

        //shapes.patch_shapes()
        if cons {
            shapes = shapes.consolidate_shapes();
        }

        shapes.shapes.sort();

        shapes
    }

    pub fn to_shapes_coloured(&self) -> Shapes {
        self.to_shapes_base(false, true, false, Colour::Black)
    }

    pub fn to_shapes(&self) -> Shapes {
        self.to_shapes_base(true, true, false, Colour::Black)
    }

    pub fn to_shapes_cons(&self) -> Shapes {
        self.to_shapes_base(true, true, true, Colour::Black)
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
        self.to_shapes_base(false, true, false, Colour::NoColour)
    }

    pub fn to_shapes_cbg(&self) -> Shapes {
        self.to_shapes_base(true, true, false, Colour::NoColour)
    }

    pub fn to_shapes_coloured_sq(&self) -> Shapes {
        self.to_shapes_base(false, false, false, Colour::Black)
    }

    pub fn to_shapes_sq(&self) -> Shapes {
        self.to_shapes_base(true, false, false, Colour::Black)
    }

    pub fn to_shapes_coloured_bg_sq(&self, bg: Colour) -> Shapes {
        self.to_shapes_base(false, false, false, bg)
    }

    pub fn to_shapes_bg_sq(&self, bg: Colour) -> Shapes {
        self.to_shapes_base(true, false, false, bg)
    }

    pub fn to_shapes_coloured_cbg_sq(&self) -> Shapes {
        self.to_shapes_base(false, false, false, Colour::NoColour)
    }

    pub fn to_shapes_cbg_sq(&self) -> Shapes {
        self.to_shapes_base(true, false, false, Colour::NoColour)
    }

    pub fn as_shape(&self) -> Shape {
        if self.size() == 0 {
            return Shape::trivial();
        }

        Shape::new_cells(&self.cells)
    }

    pub fn as_shape_position(&self, x: usize, y: usize) -> Shape {
        if self.size() == 0 {
            return Shape::trivial();
        }

        Shape::new(x, y, &self.cells)
    }

    pub fn is_square(&self) -> bool {
        self.cells.rows > 1 && self.cells.rows == self.cells.columns
    }

    pub fn square(&self) -> usize {
        self.cells.rows * self.cells.columns
    }

    pub fn height(&self) -> usize {
        self.cells.rows
    }

    pub fn width(&self) -> usize {
        self.cells.columns
    }

    pub fn row_colour(&self, x: usize) -> Colour {
        let colour = self.cells[(x,0)].colour;

        Self::row_colour_matrix(&self.cells, x, colour)
    }

    pub fn row_colour_matrix(m: &Matrix<Cell>, x: usize, colour: Colour) -> Colour {
        for y in 1 .. m.columns {
            if m[(x,y)].colour != colour {
                return Colour::Mixed;
            }
        }

        colour
    }

    pub fn col_colour(&self, y: usize) -> Colour {
        let colour = self.cells[(0,y)].colour;

        Self::col_colour_matrix(&self.cells, y, colour)
    }

    pub fn col_colour_matrix(m: &Matrix<Cell>, y: usize, colour: Colour) -> Colour {
        for x in 1 .. m.rows {
            if m[(x,y)].colour != colour {
                return Colour::Mixed;
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
        for (x, y) in self.cells.keys() {
            if self.cells[(x, y)].colour == Colour::Black {
                return false;
            }
        }

        true
    }

    pub fn has_gravity(&self, orientation: usize) -> bool {
        // Must be same shape
        if self.cells.rows != self.cells.columns {
            return false;
        }

        let mut grid = self.clone();

        grid.rotate_90(orientation);

        let mut has_black = false;

        for x in 0 .. grid.cells.rows {
            let mut prev = grid.cells[(x, 0)].colour;

            if !has_black {
                has_black = prev == Colour::Black;
            }

            for y in 1 .. grid.cells.columns {
                let next = grid.cells[(x, y)].colour;

                if prev != next && next == Colour::Black {
                    return false;
                }

                prev = next;
            }
        }

        has_black
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
            return Colour::DiffShape;
        }

        for (c1, c2) in self.cells.values().zip(other.cells.values()) {
            if c1.colour.to_base() != c2.colour.to_base() {
                return Colour::DiffBlack + c2.colour;
            }
        }

        Colour::Same
    }

    pub fn bleach(&self) -> Grid {
        let mut shape = self.clone();

        if self.size() == 0 {
            return shape;
        }

        shape.colour = Colour::NoColour;

        for c in shape.cells.values_mut() {
            if c.colour != Colour::Black {
                c.colour = Colour::NoColour;
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

        for ((sx, sy), (ox, oy)) in self.cells.keys().zip(other.cells.keys()) {
            if sx != ox || sy != oy {
                return false;
            }
        }

        true
    }

    /*
    */
    pub fn diff_only_same(&self) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if Colour::to_usize(c.colour) / 30 < 10 {
                c.colour = Colour::from_usize(Colour::to_usize(c.colour) / 30);
            } else {
                c.colour = Colour::Black;
            }
        }

        g
    }

    pub fn diff_only_diff(&self) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if Colour::to_usize(c.colour) / 40 < 10 {
                c.colour = Colour::from_usize(Colour::to_usize(c.colour) / 40);
            } else {
                c.colour = Colour::Black;
            }
        }

        g
    }

    // Experiiments on difference allowed
    pub fn diff_only(&self, colour: Colour, n: usize) -> Self {
        match n {
           0 => self.diff_only_and(colour),
           1 => self.diff_only_or(colour),
           2 => self.diff_only_xor(colour),
           _ => Self::trivial()
        }
    }

    pub fn diff_only_and(&self, colour: Colour) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if c.colour.is_same() {
                c.colour = colour;
            } else {
                c.colour = Colour::Black;
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
                c.colour = Colour::Black;
            }
        }

        g
    }

    pub fn diff_only_xor(&self, colour: Colour) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if c.colour == Colour::Black || Colour::to_usize(c.colour) > 30 {
                c.colour = Colour::Black;
            } else {
                c.colour = colour;
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
                c.colour = Colour::Black;
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
                c.colour = Colour::Black;
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

            if colour != Colour::Black && !col_set.contains(&colour){
                colours.insert(row, colour);
                col_set.push(colour);
            }

            let colour = self.cells[(row, self.cells.columns - 1)].colour;

            if colour != Colour::Black && !col_set.contains(&colour){
                colours.insert(row, colour);
                col_set.push(colour);
            }
        }

        colours
    }

    pub fn diff_only_transparent(&self) -> Self {
        let mut g = self.clone();

        for c in g.cells.values_mut() {
            if c.colour != Colour::Black {
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

        for (((x, y), c1), ((_, _), c2)) in self.cells.items().zip(other.cells.items()) {
            if c1.colour != c2.colour {
                newg.cells[(x, y)].colour = if c1.colour == Colour::Black { 
                    c2.colour + Colour::ToBlack
                } else if c2.colour == Colour::Black { 
                    c1.colour + Colour::FromBlack
                } else if diff {
                    c2.colour + Colour::DiffBlack
                } else {
                    c1.colour + Colour::OrigBlack
                };
            } else if c1.colour != Colour::Black && c1.colour == c2.colour { 
                newg.cells[(x, y)].colour = c1.colour + Colour::SameBlack;
            }
        }

        Some(newg)
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
                    map.insert(from, Colour::Black)
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

    pub fn count_diff(&self) -> (usize, usize) {
        let mut total = 0;
        let mut count = 0;

        for c in self.cells.values() {
            if c.colour == Colour::NoColour {
                count += 1;
            }

            total += 1;
        }

        (count, total)
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

    pub fn to_vec(&self) -> Vec<Vec<usize>> {
        let mut grid: Vec<Vec<usize>> = vec![vec![0; self.cells.columns]; self.cells.rows];

        for x in 0 .. self.cells.rows {
            for y in 0 .. self.cells.columns {
                let colour: usize = self.cells[(x, y)].colour.to_usize();

                grid[x][y] = colour;
            }
        }

        grid
    }

    pub fn corner_idx(&self) -> (Self, Direction) {
        let g = self.subgrid(0, 2, 0, 2);
        if g.no_colours() == 4 {
            return (g, Direction::TopLeft);
        }
        let g = self.subgrid(self.cells.rows - 2, 2, self.cells.columns - 2, 2);
        if g.no_colours() == 4 {
            return (g, Direction::BottomRight);
        }
        let g = self.subgrid(self.cells.rows - 2, 2, 0, 2);
        if g.no_colours() == 4 {
            return (g, Direction::BottomLeft);
        }
        let g = self.subgrid(0, 2, self.cells.columns - 2, 2);
        if g.no_colours() == 4 {
            return (g, Direction::TopRight);
        }

        (self.clone(), Direction::Other)
    }

    pub fn corner_body(&self, dir: Direction) -> Self {
        let hx = self.cells.rows - 2;
        let hy = self.cells.columns - 2;

        match dir {
            Direction::TopLeft => self.subgrid(0, hx, 0, hy),
            Direction::BottomRight => self.subgrid(self.cells.rows - hx, hx, self.cells.columns - hy, hy),
            Direction::BottomLeft => self.subgrid(self.cells.rows - hx, hx, 0, hy),
            Direction::TopRight => self.subgrid(0, hx, self.cells.columns - hy, hy),
            _ => self.clone(),
        }
    }

    pub fn split_4(&self) -> Vec<Self> {
        let hx = self.cells.rows / 2;
        let hy = self.cells.columns / 2;

        if hx == 0 || hy == 0 || self.cells.rows % 2 != 0 || self.cells.columns % 2 != 0 {
            return Vec::new();
        }

        let s1 = self.subgrid(0, hx, 0, hy);
        let s2 = self.subgrid(0, hx, self.cells.columns - hy, hy);
        let s3 = self.subgrid(self.cells.rows - hx, hx, 0, hy);
        let s4 = self.subgrid(self.cells.rows - hx, hx, self.cells.columns - hy, hy);

        vec![s1, s2, s3, s4]
    }

    pub fn split_4_inline(&self, delimiter: bool) -> Shapes {
        let rows = self.cells.rows;
        let cols = self.cells.columns;
        let mut dx = if delimiter { 1 } else { 0 };
        let mut dy = if delimiter { 1 } else { 0 };
        let mut shapes = Shapes::new_sized(rows, cols);

        if rows % 4 != 0 && cols % 4 != 0 || rows < dx * 3 || cols < dy * 3 {
            return shapes;
        }
//println!("1 {rows} {cols}");
        // too strict?
        if ((rows - dx * 3) % 4 != 0 || cols % 4 != 0) &&
            (rows % 4 != 0 || (cols - dy * 3) % 4 != 0) {
            //return shapes;
            dx = 0;
            dy = 0;
        }
//println!("2 {rows} {cols}");
        /*
        */

        let (s1, s2, s3, s4) = if rows > cols {
            let qx = rows / 4;
            if qx * 4 >= rows {
                dx = 0;
            }

            (self.subgrid(0, qx, 0, cols),
            self.subgrid(qx + dx, qx, 0, cols),
            self.subgrid((qx + dx) * 2, qx, 0, cols),
            self.subgrid((qx + dx) * 3, qx, 0, cols))
        } else {
            let qy = cols / 4;
            if qy * 4 >= cols {
                dy = 0;
            }

            (self.subgrid(0, rows, 0, qy),
            self.subgrid(0, rows, qy + dy, qy),
            self.subgrid(0, rows, (qy + dy) * 2, qy),
            self.subgrid(0, rows, (qy + dy) * 3, qy))
        };

        shapes.shapes = vec![s1.as_shape(), s2.as_shape(), s3.as_shape(), s4.as_shape()];

        shapes
    }

    pub fn split_2(&self) -> Shapes {
        if self.cells.rows % 2 != 0 && self.cells.columns % 2 != 0 {
            return Shapes::new();
        }

        let mut shapes = Shapes::new_sized(self.cells.rows, self.cells.columns);
        let hx = self.cells.rows / 2;
        let hy = self.cells.columns / 2;

        let s1 = if self.cells.rows % 2 == 0 {
            self.subgrid(0, hx, 0, self.cells.columns)
        } else {
            self.subgrid(0, self.cells.rows, 0, hy)
        };
        let s2 = if self.cells.rows % 2 == 0 {
            self.subgrid(hx, self.cells.rows - hx, 0, self.cells.columns)
        } else {
            self.subgrid(0, self.cells.rows, hy, self.cells.columns - hy)
        };

        shapes.shapes = vec![s1.as_shape(), s2.as_shape()];

        shapes
    }

    pub fn full(&self) -> bool {
        if self.cells.rows == 0 {
            return false;
        }
        for c in self.cells.values() {
            if c.colour == Colour::Black {
                return false;
            }
        }

        true
    }

    pub fn get_patch(&self, x: usize, y: usize, rows: usize, cols: usize) -> Shape {
        match self.cells.slice(x .. x + rows, y .. y + cols) {
            Ok(m) => Shape::new(x, y, &m),
            Err(_e) => {
                //eprintln!("{e}");

                Shape::trivial()
            }
        }

    }

    pub fn fill_patch_mut(&mut self, other: &Shape, ox: usize, oy: usize) {
        if self.size() <= other.size() {
            return;
        }
//println!("{} {}", ox, oy);

        for (x, y) in other.cells.keys() {
//println!("{:?}", self.cells[(ox + x, oy + y)].colour);
            if self.cells[(ox + x, oy + y)].colour == Colour::Black {
                self.cells[(ox + x, oy + y)].colour = other.cells[(x, y)].colour;
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
        let shapes = if bg == Colour::NoColour {
            grid.to_shapes()
        } else {
            grid.to_shapes_bg(bg)
        };
        let coloured_shapes = if bg == Colour::NoColour {
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
        let shapes = if bg == Colour::NoColour {
            grid.to_shapes_cons()
        } else {
            grid.to_shapes_bg_cons(bg)
        };
        let coloured_shapes = if bg == Colour::NoColour {
            grid.to_shapes_coloured()
        } else {
            grid.to_shapes_coloured_bg(bg)
        };
        let black = grid.find_black_patches();

        Self { grid: grid.clone(), bg, shapes, coloured_shapes, black }
    }

    pub fn trivial() -> Self {
        Self { grid: Grid::trivial(), bg: Colour::NoColour, shapes: Shapes::new(), coloured_shapes: Shapes::new(), black: Shapes::new() }
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
        for s in self.shapes.coloured_shapes.iter() {
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

