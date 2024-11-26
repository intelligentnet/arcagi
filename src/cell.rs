use std::ops::Add;
use std::cmp::Ordering;
use num_traits::identities::Zero;
use crate::cats::{Colour, CellCategory};

#[derive(Debug, Clone, Eq, Hash)]
pub struct Cell {
    pub row: usize,
    pub col: usize,
    pub colour: Colour,
    pub cat: CellCategory,
}

impl Add for Cell {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        rhs
    }
}

impl Zero for Cell {
    fn zero() -> Self {
        Self {
            row: 0,
            col: 0,
            colour: Colour::Black,
            cat: CellCategory::BG,
        }
    }

    fn is_zero(&self) -> bool {
        self.row == 0 && self.col == 0
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.row, &self.col).cmp(&(other.row, &other.col))
    }
}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        (self.row, &self.col) == (other.row, &other.col)
    }
}

impl Cell {
    pub fn new(x: usize, y: usize, colour: usize) -> Self {
        Cell { row: x, col: y, colour: Colour::new(colour), cat: CellCategory::BG }
    }

    pub fn new_empty() -> Self {
        Cell { row: 0, col: 0, colour: Colour::NoColour, cat: CellCategory::BG }
    }

    pub fn new_colour(x: usize, y: usize, colour: Colour) -> Self {
        Cell { row: x, col: y, colour, cat: CellCategory::BG }
    }

    pub fn above(&self, other: &Self) -> bool {
        other.col < self.col
    }

    pub fn below(&self, other: &Self) -> bool {
        other.col > self.col
    }

    pub fn left(&self, other: &Self) -> bool {
        other.row < self.row
    }

    pub fn right(&self, other: &Self) -> bool {
        other.row > self.row
    }

    #[allow(clippy::nonminimal_bool)]
    pub fn next(&self, other: &Self) -> bool {
        let self_row: i16 = self.row as i16;
        let self_col: i16 = self.col as i16;
        let other_row: i16 = other.row as i16;
        let other_col: i16 = other.col as i16;

        other_col == self_col - 1 && other_row == self_row ||
        other_col == self_col + 1 && other_row == self_row ||
        other_row == self_row - 1 && other_col == self_col ||
        other_row == self_row + 1 && other_col == self_col
    }

    #[allow(clippy::nonminimal_bool)]
    pub fn adjacent(&self, other: &Self) -> bool {
        let self_row: i16 = self.row as i16;
        let self_col: i16 = self.col as i16;
        let other_row: i16 = other.row as i16;
        let other_col: i16 = other.col as i16;

        other_col == self_col - 1 && other_row == self_row - 1 ||
        other_col == self_col - 1 && other_row == self_row + 1 ||
        other_col == self_col + 1 && other_row == self_row - 1 ||
        other_col == self_col + 1 && other_row == self_row + 1
    }

    pub fn next_colour(&self, other: &Self) -> bool {
        self.colour == other.colour && self.next(other)
    }

    pub fn adjacent_colour(&self, other: &Self) -> bool {
        self.colour == other.colour && self.adjacent(other)
    }
}

