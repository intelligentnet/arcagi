//use std::collections::BTreeMap;
//use pathfinding::prelude::Matrix;
use itertools::Itertools;
use crate::cats::*;
//use crate::cell::*;
use crate::examples::*;
use crate::grid::*;
use crate::shape::*;
//use crate::rules::Colour::Mixed;
//use crate::experiments::*;

pub fn move_only(grid: &Grid, n: &mut usize) -> Grid {
    let func = [Grid::move_down, Grid::move_up, Grid::move_right, Grid::move_left];
    if *n == usize::MAX {
        *n = func.len();
    }
    if *n == 0 {
        Grid::trivial()
    } else {
        func[func.len() - *n](grid)
    }
}

// Experiments on gravity
pub fn gravity_only(grid: &Grid, n: &mut usize) -> Grid {
    let func = [Grid::stretch_down, Grid::gravity_down, Grid::gravity_up];
    if *n == usize::MAX {
        *n = func.len();
    }
    if *n == 0 {
        Grid::trivial()
    } else {
        func[func.len() - *n](grid)
    }
}

// Experiments on mirroring and duplicating
pub fn mirror_only(grid: &Grid, n: &mut usize) -> Grid {
    let func = [Grid::mirror_right, Grid::mirror_left, Grid::mirror_down, Grid::mirror_up, Grid::dup_right, Grid::dup_left, Grid::dup_down, Grid::dup_up];
    if *n == usize::MAX {
        *n = func.len();
    }
    if *n == 0 {
        Grid::trivial()
    } else {
        func[func.len() - *n](grid)
    }
}

/*
pub fn gravity_only(grid: &Grid, n: &mut usize) -> Grid {
    let func = [&Grid::stretch_down, &Grid::gravity_down, &Grid::gravity_up];
    apply(grid, n, &func)
}

pub fn apply(grid: &Grid, n: &mut usize, func: &[&dyn Fn(&Grid) -> Grid]) -> Grid {
    if *n == usize::MAX {
        *n = func.len();
    }
    if *n == 0 {
        Grid::trivial()
    } else {
        func[func.len() - *n](grid)
    }
}
*/

pub fn diff_only(grid: &Grid, colour: Colour, n: &mut usize) -> Grid {
    let func = [Grid::diff_only_and, Grid::diff_only_or, Grid::diff_only_xor, Grid::diff_black_same, Grid::diff_other_same];
    if *n == usize::MAX {
        *n = func.len();
    }
    if *n == 0 {
        Grid::trivial()
    } else {
        func[func.len() - *n](grid, colour)
    }
}

pub fn transform_only(grid: &Grid, n: &mut usize) -> Grid {
    let func = [Grid::rot_00,  Grid::rot_90, Grid::rot_180, Grid::rot_270, Grid::transposed, Grid::mirrored_rows, Grid::mirrored_cols];

    if *n == usize::MAX {
        *n = func.len();
    }
    if *n == 0 {
        Grid::trivial()
    } else {
        func[func.len() - *n](grid)
    }
}

/*
pub fn try_colours(grid: &Grid, n: &mut usize, func: &(dyn Fn(&Grid, Colour) -> Grid)) -> Grid {
    //let colours = Colour::colours();
    const COLOURS: &[Colour]  = Colour::base_colours();

    if *n == usize::MAX {
        *n = COLOURS.len();
    }
    if *n == 0 {
        Grid::trivial()
    } else {
        func(grid, COLOURS[*n])
    }
}
*/

pub fn simple_diffs(exs: &Examples) -> Option<Vec<Grid>> {
    let mut diffs: Vec<Grid> = Vec::new();

    for e in exs.examples.iter() {
        if let Some(diff) = e.input.grid.diff(&e.output.grid) {
            diffs.push(diff);
        } else {
            return None;
        }
    }

    Some(diffs)
}

pub fn difference(exs: &Examples) {
//let mut done = false;
    for e in exs.examples.iter() {
        if let Some(diff) = e.input.grid.diff(&e.output.grid) {
            diff.show_full();
//done = true;
        }
        /*
        let (count, total) = diff.count_diff();
        println!("{file} counts: {count} of {total} {}", 
                 if count < 9 { "easy" } else { "hard" });
        */
    }

    exs.tests.iter().for_each(|tests| tests.input.grid.show());
//if done { println!("++++"); }
}

pub fn permutations(v: &[usize]) -> Vec<Vec<&usize>> {
    let mut perms: Vec<Vec<&usize>> = Vec::new();

    for perm in v.iter().permutations(v.len()).unique() {
        perms.push(perm);
    }

    perms
}

pub fn difference_shapes(exs: &Examples, coloured: bool) {
    for shapes in &exs.examples {
        if let Some(diff) = shapes.input.grid.diff(&shapes.output.grid) {
            diff.show_full();
        }
        let mut used: Vec<Shape> = Vec::new();
        let inp = if coloured { &shapes.input.coloured_shapes } else { &shapes.input.shapes };
        let out = if coloured { &shapes.output.coloured_shapes } else { &shapes.output.shapes };

        for si in inp.shapes.iter() {
            for so in out.shapes.iter() {
                if si.orow == so.orow && si.ocol == so.ocol {
                    if let Some(diff) = si.diff(so) {
                        println!("Same size");
                        diff.show_full();
                    } else {
                        println!("Different sizes");
                        si.show();
                        so.show();
                    }
                } else if !used.contains(so) {
                    used.push(so.clone());
                    println!("Other shapes");
                    so.show();
                }
            }
        }
    }

    println!("Test shapes");
    for test in exs.tests.iter() {
        for ex in test.input.shapes.shapes.iter() {
            ex.show();
        }
    }
}

/*
pub fn repeat_pattern(ex: &Example, colour: Colour) -> Grid {
    let shape = &ex.input.grid.as_shape();
    let posns = shape.colour_position(colour);
    let rows = shape.cells.rows;
    let cols = shape.cells.columns;
println!("{rows}/{cols} {colour:?}");
shape.show();
    let mut shapes = Shapes::new_sized(rows.pow(2), cols.pow(2));

    for (r, c) in posns.iter() {
        let or = r * 3;
        let oc = c * 3;

        shapes.shapes.push(shape.translate_absolute(or, oc));
    }
ex.output.grid.show();
shapes.to_grid().show();
    shapes.to_grid()
}
*/
pub fn repeat_pattern(ex: &Example, colour: Colour) -> Grid {
    let shape = ex.input.grid.as_shape();
    let (rs, cs) = ex.input.grid.dimensions();

    if colour == Colour::Black {
        shape.chequer(rs, cs, &|r,c| ex.input.grid.cells[(r / rs,c / cs)].colour != colour, &|s| s.clone(), true).to_grid()
    } else {
        shape.chequer(rs, cs, &|r,c| ex.input.grid.cells[(r / rs,c / cs)].colour == colour, &|s| s.clone(), true).to_grid()
    }
}

/* Diff
 *
 * 1* and 3*    1* is add
 *
 * fill same between x and y    22168020.json
 * fill same between x and y    22eb0ac0.json
 * line to limits x and y       178fcbfb.json
 * line to limits x and y       23581191.json
 * diag to limits or other      508bd3b6.json
 *
 * 2* and 3*    2* is delete
 *
 * take middle x and y          d23f8c26.json
 */

/*
   Priors
   ------

   1. Object permanence, they can move, be occluded, rotate etc. but remain
      the same object.
   2. Agentness, object with 'purpose'. It has 'goals' and can influence other       object and/or agents.
   3. Simple physics, distance, movement, 
   4. Number, basic counting
   5. Rotation, symmetry etc.
*/
