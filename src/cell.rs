use std::ops::Add;
use std::cmp::Ordering;
use num_traits::identities::Zero;
use crate::cats::{Colour, CellCategory};

#[derive(Debug, Clone, Eq, Hash)]
pub struct Cell {
    pub x: usize,
    pub y: usize,
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
            x: 0,
            y: 0,
            colour: Colour::Black,
            cat: CellCategory::BG,
        }
    }

    fn is_zero(&self) -> bool {
        self.x == 0 && self.y == 0
    }
}

impl Ord for Cell {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.x, &self.y).cmp(&(other.x, &other.y))
    }
}

impl PartialOrd for Cell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Cell {
    fn eq(&self, other: &Self) -> bool {
        (self.x, &self.y) == (other.x, &other.y)
    }
}

impl Cell {
    pub fn new(x: usize, y: usize, colour: usize) -> Self {
        Cell { x, y, colour: Colour::new(colour), cat: CellCategory::BG }
    }

    pub fn new_empty() -> Self {
        Cell { x: 0, y: 0, colour: Colour::NoColour, cat: CellCategory::BG }
    }

    pub fn new_colour(x: usize, y: usize, colour: Colour) -> Self {
        Cell { x, y, colour, cat: CellCategory::BG }
    }

    pub fn above(&self, other: &Self) -> bool {
        other.y < self.y
    }

    pub fn below(&self, other: &Self) -> bool {
        other.y > self.y
    }

    pub fn left(&self, other: &Self) -> bool {
        other.x < self.x
    }

    pub fn right(&self, other: &Self) -> bool {
        other.x > self.x
    }

    #[allow(clippy::nonminimal_bool)]
    pub fn next(&self, other: &Self) -> bool {
        let self_x: i16 = self.x as i16;
        let self_y: i16 = self.y as i16;
        let other_x: i16 = other.x as i16;
        let other_y: i16 = other.y as i16;

        other_y == self_y - 1 && other_x == self_x ||
        other_y == self_y + 1 && other_x == self_x ||
        other_x == self_x - 1 && other_y == self_y ||
        other_x == self_x + 1 && other_y == self_y
    }

    #[allow(clippy::nonminimal_bool)]
    pub fn adjacent(&self, other: &Self) -> bool {
        let self_x: i16 = self.x as i16;
        let self_y: i16 = self.y as i16;
        let other_x: i16 = other.x as i16;
        let other_y: i16 = other.y as i16;

        other_y == self_y - 1 && other_x == self_x - 1 ||
        other_y == self_y - 1 && other_x == self_x + 1 ||
        other_y == self_y + 1 && other_x == self_x - 1 ||
        other_y == self_y + 1 && other_x == self_x + 1
    }

    pub fn next_colour(&self, other: &Self) -> bool {
        self.colour == other.colour && self.next(other)
    }

    pub fn adjacent_colour(&self, other: &Self) -> bool {
        self.colour == other.colour && self.adjacent(other)
    }
}

