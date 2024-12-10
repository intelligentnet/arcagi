//use std::collections::{BTreeSet, BTreeMap};
//use crate::cats::*;
//use crate::rules::*;
//use crate::data::{dir, Data};
use std::panic::RefUnwindSafe;
use crate::examples::*;
use crate::grid::*;
//use crate::shape::*;
use crate::cats::Colour;
use crate::cats::Transformation;
//use crate::cats::Transformation::NoTrans;

pub fn check_examples(ex: &Examples, func: &dyn Fn(&Example) -> bool) -> bool {
    for e in &ex.examples {
        if !func(e) {
            return false;
        }
    }

    true
}

pub fn check_input_grids(ex: &Examples, func: &dyn Fn(&Grid) -> bool) -> bool {
    for e in &ex.examples {
        if !func(&e.input.grid) {
            return false;
        }
    }

    true
}

pub fn check_output_grids(ex: &Examples, func: &dyn Fn(&Grid) -> bool) -> bool {
    for e in &ex.examples {
        if !func(&e.output.grid) {
            return false;
        }
    }

    true
}

/*
pub fn experiment_grid(ex: &Examples, len: usize, func: &dyn Fn(&Grid, &Grid, usize) -> Grid) -> Grid {
    let mut n = 0;

    for e in &ex.examples {
        let grid = &e.input.grid;
        let target = &e.output.grid;

        loop {
            if n >= len {
                return Grid::trivial();
            } else {
                if func(&grid, &target, n).equals(target) != Colour::Same {
                    n += 1;
                } else {
                    break;
                }
            }
        }
    }

    func(&ex.test.input.grid, &ex.examples[0].output.grid, n)
}
*/

pub fn experiment_grid(ex: &Examples, file: &str, experiment: usize, _trans: Transformation, func: &(dyn Fn(&Grid, &Grid, &mut usize) -> Grid + RefUnwindSafe)) -> Vec<Grid> {
    let mut n = usize::MAX;

    for (attempts, e) in ex.examples.iter().enumerate() {
        let grid = &e.input.grid;
        let target = &e.output.grid;

        if target.size() == 0 {
            return ex.tests.iter().map(|_| Grid::trivial()).collect::<Vec<_>>();
        }

        loop {
            if n == 0 {
                return ex.tests.iter().map(|_| Grid::trivial()).collect::<Vec<_>>();
            } else {
                let ans = func(grid, target, &mut n);
                /*
                let ans = if trans == NoTrans {
                    ans
                } else {
                    ans.inverse_transform(trans)
                };
                */
                if ans.equals(target) != Colour::Same {
                    if n == usize::MAX {
                        if attempts > 0 {
                            println!("{file} {experiment:<4}: {attempts} worked out of {}", ex.examples.len());
                        }
                        return ex.tests.iter().map(|_| Grid::trivial()).collect::<Vec<_>>();
                    }

                    n -= 1;
                } else {
                    break;
                }
            }
        }
    }

    // if experiments work then run tests
    ex.tests.iter().map(|test| {
        let ans = func(&test.input.grid, &ex.examples[0].output.grid, &mut n);

        /*
        if trans == NoTrans {
            ans
        } else {
            ans.inverse_transform(trans)
        }
        */
        ans
    })
    .collect()
}

pub fn experiment_example(ex: &Examples, file: &str, experiment: usize, _trans: Transformation, func: &(dyn Fn(&Example) -> Grid + RefUnwindSafe)) -> Vec<Grid> {
    for (attempts, e) in ex.examples.iter().enumerate() {
        //if attempts < 1 { continue; }
        let target = &e.output.grid;
        let ans = func(e);

        /*
        let ans = if trans == NoTrans {
            ans
        } else {
            ans.inverse_transform(trans)
        };
        */
//ans.show();

        if ans == Grid::trivial() || ans.equals(target) != Colour::Same {
            if attempts > 0 {
                println!("{file} {experiment:<4}: {attempts} worked out of {}", ex.examples.len());
            }
            return ex.tests.iter().map(|_| Grid::trivial()).collect::<Vec<_>>();
        }
    }
    
    // if experiments work then run tests
    ex.tests.iter().map(|test| {
        let ans = func(test);

        /*
        if trans == NoTrans {
            ans
        } else {
            ans.inverse_transform(trans)
        }
        */
        ans
    })
    .collect()
}
