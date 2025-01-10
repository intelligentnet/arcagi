use std::str::FromStr;
use peak_alloc::PeakAlloc;
#[global_allocator]
static PEAK: PeakAlloc = PeakAlloc;
use strum::IntoEnumIterator;
use std::panic;
use std::panic::RefUnwindSafe;
use std::collections::{BTreeSet, BTreeMap};
use pathfinding::prelude::Matrix;
use array_tool::vec::Uniq;
use crate::cats::Colour::*;
use crate::cats::GridCategory::*;
use crate::cats::Direction::*;
use crate::cats::Transformation::*;
use crate::cats::*;
use crate::examples::*;
use crate::experiments::*;
use crate::rules::*;
//use crate::oldrules::*;
use crate::grid::*;
use crate::shape::*;
use crate::cell::*;
use crate::data::*;

pub fn runner(data: &str, task_name: &str, experiment: &str, trans: &str, all: bool) {
    //eprintln!("{data} {task_name} {experiment} {trans} {all}");
    let start = std::time::Instant::now();
    let tdata = load_files(data);
    let is_test = data == "test";
    let mut cnt = 0;
    let mut output: BTreeMap<String, Vec<OutputData>> = BTreeMap::new();
    let mut cap_cats: BTreeMap<GridCategory, i32> = BTreeMap::new();
    let mut cap_todo: BTreeMap<GridCategory, i32> = BTreeMap::new();
    let mut rule_tasks: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    let mut done: BTreeSet<String> = BTreeSet::new();
    let mut tries: usize = 0;

    // Main processing outer loop
    for (file_name, data) in tdata.iter() {
        let task = file_name.to_string();

        // already done or just one???
        if done.contains(&task) || (!task_name.is_empty() && task_name != "all" && task != task_name) {
            continue;
        }

        let mut examples = Examples::new(data);

        // Do some shapes overay others?
        if examples.cat.contains(&OverlayInSame) || examples.cat.contains(&OverlayOutSame) {
            examples = Examples::new_cons(data);
        }
        println!("{task}: {:?}", examples.cat);

        let transform = if trans == "all" || trans == "NoTrans" {
            NoTrans
        } else if let Ok(tr) = Transformation::from_str(trans) {
            examples = examples.transformation(tr);

            tr
        } else {
            NoTrans
        };

        // Base pass with no transformations
        let rule = pass(all, &task, experiment, &examples, transform, is_test, &mut output, &mut cap_cats, &mut cap_todo, &mut done, &mut tries);

        if done.contains(&task) {
            if let Some(rule) = rule {
                if rule != usize::MAX {
                    rule_tasks.entry(rule).or_default().push(task.clone());
                }
            }
        }

        // Experimental - needs more work!
        if trans == "all" {
            // Now check for transformed grids to make closures more generic
            for transform in Transformation::iter() {
                if transform == NoTrans || transform >= MirrorRowRotate90 { continue; }
                //println!("{:?}", transform);

                let ex_copy = examples.transformation(transform);

                let _rule = pass(all, &task, experiment, &ex_copy, transform, is_test, &mut output, &mut cap_cats, &mut cap_todo, &mut done, &mut tries);
            }
        }

        cnt += 1;
    }

    println!("Totals: {cap_cats:?}");

    let cap_done: BTreeMap<GridCategory, i32> = cap_cats
        .iter()
        .map(|(cat, i)| {
            if let Some(j) = cap_todo.get(cat) {
                (*cat, *i - j)
            } else {
                (*cat, *i)
            }
        })
        .collect();

    if is_test {
        create_output(true, &output);
    }

    println!("Complete: {cap_done:?}");
    //println!("{done:?}");
    if rule_tasks.len() > 0 {
        format(&rule_tasks);
        println!("Generalisation Ratio: {:.4}", done.len() as f64 / rule_tasks.len() as f64);
    }
    println!("Done = {}, To Do = {}, tries = {tries}", done.len(), cnt - done.len());

    let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

    println!("Elapsed time: {timing} secs");
    println!("Used memory : {:.2} Mb", PEAK.current_usage_as_mb());
}

fn pass(all: bool, task: &str, experiment: &str, examples: &Examples, trans: Transformation, is_test: bool, output: &mut BTreeMap<String, Vec<OutputData>>, cap_cats: &mut BTreeMap<GridCategory, i32>, cap_todo: &mut BTreeMap<GridCategory, i32>, done: &mut BTreeSet<String>, tries: &mut usize) -> Option<usize> {
    let cat = &examples.cat;
    let targets: Vec<Grid> = examples.tests.iter().map(|test| test.output.grid.clone()).collect();

    // used by many tasks
    let colour_diffs = examples.io_colour_diff();
    let colour_common = examples.io_colour_common();

{
        //let (in_rs, in_cs) = examples.examples[0].input.grid.dimensions();
        //let (out_rs, out_cs) = examples.examples[0].output.grid.dimensions();
        //let rs = out_rs / in_rs;
        //let cs = out_cs / in_cs;
        
        // testing function
        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            ex.input.grid.scale_up(ex.input.grid.height())
        };

        //if let Some(rule) = run_experiment(file, 10000, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
}

    let gc = BlankIn;
    if all || cat.contains(&gc) { // 2 done
        *cap_cats.entry(gc).or_insert(0) += 1;

        let colour = examples.examples[0].output.grid.colour;

        let func = &|ex: &Example| {
            let odd = ex.cat.contains(&InOutSquareSameSizeOdd);
            let mut grid;

            if odd {
                // TODO: Might be improved by using permutation of output grid?
                grid = ex.input.grid.clone();

                for (r, c) in grid.cells.keys() {
                    if r % 2 == 0 || c % 2 == 0 || r == grid.cells.rows - 1 || c == grid.cells.columns - 1 {
                        grid.cells[(r,c)].colour = colour;
                    }
                }
            } else {
                let sq = ex.cat.contains(&InOutSquareSameSize);

                grid = ex.input.grid.do_circle(colour, sq);
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 0, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = BlackPatches;
    if all || cat.contains(&gc) { // 3?
        *cap_cats.entry(gc).or_insert(0) += 1;

        // TODO: 981571dc.json and af22c60d.json mirrored
        // 1e97544e.json e95e3d8e.json e95e3d8e.json 0dfd9992.json + test c3f564a4.json 29ec7d0e.json
        let func = &|ex: &Example| {
            if ex.input.black.is_empty() {
                return Grid::trivial();
            }

            //let mut grid = ex.input.grid.clone();
            let mut grid = ex.input.grid.clone();
            let mut black_patches = ex.input.black.clone();
            //grid.fill_border();

            for it in 0 .. 2 {   // may need more that one iteration
                'second:
                //for bp in ex.input.black.shapes.iter() {
                for bp in black_patches.shapes.iter() {
                    let r1 = if bp.orow > 0 { bp.orow - 1 } else { bp.orow };
                    let r2 = if bp.orow + bp.cells.rows < grid.cells.rows { bp.orow + bp.cells.rows + 1 } else { bp.orow + bp.cells.rows };
                    let c1 = if bp.ocol > 0 { bp.ocol - 1 } else { bp.ocol };
                    let c2 = if bp.ocol + bp.cells.columns < grid.cells.columns { bp.ocol + bp.cells.columns + 1 } else { bp.ocol + bp.cells.columns };
                    let m = grid.cells.slice(r1 .. r2, c1 .. c2);

                    if let Ok(m) = m {
                        let s = Shape::new(bp.orow, bp.ocol, &m);
//s.show();
                        let l = c2 - c1;
                        let fw = m.windows(l).next().unwrap();
                        let sc: Vec<_> = fw.iter().map(|c| c.colour).collect();
                        for w in grid.cells.windows(l) {
                            let fc: Vec<_> = w.iter().map(|c| c.colour).collect();
                            if sc == fc && (fw[0].row != w[0].row || fw[0].col != w[0].col) {
                                let patch = grid.get_patch(w[0].row, w[0].col, m.rows, m.columns);
                                if patch.is_full() && s.same_patch(&patch) {
                                    let sor = if s.orow > 0 { s.orow - 1 } else { s.orow };
                                    let soc = if s.ocol > 0 { s.ocol - 1 } else { s.ocol };

                                    grid.fill_patch_mut(&patch, sor, soc);

                                    continue 'second;
                                }
                            }
                        }
                    } else {
                        return Grid::trivial();
                    }
                }
                if it % 2 == 0 {
                    grid = ex.input.grid.rot_180();
                    black_patches = grid.find_black_patches();
                } else {
                    grid = grid.rot_180();
                }
                if grid.full() {
                    break
                }
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 1, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = &|ex: &Example| {
            if ex.input.black.is_empty() {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();

            grid.fill_border();
            let mut pr = 0;
            let mut pc = 0;
            let mut lc = 0;     // if all else fails break the loop!

            loop {
                let black = grid.find_black_patches();

                if black.is_empty() || lc >= 10 {
                    break;
                }

                for p in black.shapes.iter() {
                    let ra = if p.ocol == 0 { 
                        Vec::new()
                    } else if let Ok(xs) = grid.cells.slice(p.orow .. p.orow+p.cells.rows, p.ocol-1 .. p.ocol) {
                        let xa: Vec<_> = xs.values().map(|c| c.colour).collect();
                        if Colour::single_colour_vec(&xa) {
                            Vec::new()
                        } else {
                            xa
                        }

                    } else {
                        Vec::new()
                    };
                    let ca = if p.orow == 0 {
                        Vec::new()
                    } else if let Ok(ys) = grid.cells.slice(p.orow-1 .. p.orow, p.ocol .. p.ocol+p.cells.columns) {
                        let ya: Vec<_> = ys.values().map(|c| c.colour).collect();
                        if Colour::single_colour_vec(&ya) {
                            Vec::new()
                        } else {
                            ya
                        }
                    } else {
                        Vec::new()
                    };
                    if !ra.is_empty() && ra.len() >= ca.len() {
                        let (xo, yo) = grid.find_row_seq(p.orow, p.ocol, &ra, p.cells.columns);
                        if xo == usize::MAX && yo == usize::MAX || pr == xo && pc == yo {
                            return Grid::trivial();
                        }
                        for r in 0 .. p.cells.rows {
                            for c in 0 .. p.cells.columns {
                                if grid.cells[(p.orow+r,p.ocol+c)].colour == Black && xo+r < grid.cells.rows && yo+1+c < grid.cells.columns {
                                    grid.cells[(p.orow+r,p.ocol+c)] = grid.cells[(xo+r,yo+1+c)].clone();
                                }
                            }
                        }

                        pr = xo;
                        pc = yo;
                    }
                    else if !ca.is_empty() {
                        let (xo, yo) = grid.find_col_seq(p.orow, p.ocol, &ca, p.cells.rows);
                        if xo == usize::MAX && yo == usize::MAX || pr == xo && pc == yo {
                            return Grid::trivial();
                        }
                        for r in 0 .. p.cells.rows {
                            for c in 0 .. p.cells.columns {
                                if grid.cells[(p.orow+r,p.ocol+c)].colour == Black && xo+1+r < grid.cells.rows && yo+c < grid.cells.columns {
                                    grid.cells[(p.orow+r,p.ocol+c)] = grid.cells[(xo+1+r,yo+c)].clone();
                                }
                            }
                        }

                        pr = xo;
                        pc = yo;
                    } else {
                        return Grid::trivial();
                    }
                }
                lc += 1;
            }
//grid.show();
            
            grid
        };

        if let Some(rule) = run_experiment(task, 4, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if !examples.examples[0].input.black.is_empty() && !examples.examples[0].output.shapes.shapes.is_empty() {
            let mut ccm = examples.examples[0].output.grid.as_shape().cell_colour_cnt_map();
            // assume grid background and border shapes same colour for all tests
            let bg = examples.examples[0].output.grid.cells[(0,0)].colour;
            let border = examples.examples[0].output.shapes.shapes[0].colour;

            ccm.remove(&bg);
            ccm.remove(&border);

            let func = |ex: &Example| {
                if ccm.len() != 1 || !ex.input.black.shapes[0].is_square() {
                    return Grid::trivial();
                }

                // Assume square
                let side = ex.input.black.shapes[0].cells.rows;
                let Some((inner, _)) = ccm.first_key_value() else { todo!() };
                let mut grid = ex.input.grid.clone();

                for s in ex.input.black.shapes.iter() {
                    if s.orow == 0 || s.ocol == 0 {
                        return Grid::trivial();
                    }
                    if s.orow <= 2 || s.ocol <= 2 || s.orow >= grid.cells.rows - side - 2 || s.ocol >= grid.cells.columns - side - 2 {
                        grid.flood_fill_mut(s.orow, s.ocol, NoColour, border);
                    } else {
                        grid.flood_fill_mut(s.orow, s.ocol, NoColour, *inner);
                    }
                }
//grid.show();

                grid
            };

            if let Some(rule) = run_experiment(task, 3, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let colour = colour_diffs[0];
            let g = &ex.input.grid;
            let mut grid = g.clone();

            grid.recolour_mut(Black, colour);

            let shapes = grid.to_shapes_sq();

            for s in shapes.shapes.iter() {
                if s.is_pixel() && s.colour == colour {
                    grid.cells[(s.orow, s.ocol)].colour = Black;
                }
            }

            grid
        };

        let (rs, cs) = examples.examples[0].output.grid.dimensions();

        if let Some(rule) = run_experiment(task, 4, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !ex.cat.contains(&NxNIn(15)) || !ex.cat.contains(&NxNOut(3)) || !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let mut grid = Grid::new(rs, cs, ex.input.grid.colour);

            for s in ex.input.black.shapes.iter() {
                grid.cells[(s.orow / 5,s.ocol / 5)].colour = Black;
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 5, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();

            shapes.shapes = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                let (dir,_) = s.has_arm(5);

                if dir == Other {
                    let sn = s.recolour(Black, s.colour);

                    shapes.shapes.push(sn.clone());
                }
            }

            for s in ex.input.shapes.shapes.iter() {
                let (dir,len) = s.has_arm(5);

                if dir == Other {
                    continue;
                }

                let sn = match dir {
                    Up => {
                        let s = s.subshape(len, s.cells.rows - len, 0, s.cells.columns);

                        if s.orow >= len {
                            s.to_position(s.orow - len, s.ocol)
                        } else {
                            return Grid::trivial();
                        }
                    },
                    Down => {
                        let s = s.subshape(0, s.cells.rows - len, 0, s.cells.columns);
                        s.to_position(s.orow + len, s.ocol)
                    },
                    Left => {
                        let s = s.subshape(0, s.cells.rows, len, s.cells.columns - len);
                        if s.ocol >= len {
                            s.to_position(s.orow, s.ocol - len)
                        } else {
                            return Grid::trivial();
                        }
                    },
                    Right => {
                        let s = s.subshape(0, s.cells.rows, 0, s.cells.columns - len);
                        s.to_position(s.orow, s.ocol + len)
                    },
                    _ => todo!(),   // Should never happen!
                };

                shapes.shapes.push(sn.clone());
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 6, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // TODO BlackPatches okay but cannot find test!!!
        let func = |ex: &Example| {
            if !ex.cat.contains(&OverlayInDiff) || !ex.cat.contains(&OverlayOutDiff) || ex.input.black.shapes.len() != 1 {
                return Grid::trivial();
            }

            let g = &ex.input.grid;
            let black = &ex.input.black.shapes[0];
            let mut grid = ex.input.grid.clone();

            for (r, c) in g.cells.keys() {
                if g.cells[(r,c)].colour == Black {
                    if g.cells[(g.cells.rows - r - 1,c)].colour != Black {
                        grid.cells[(r,c)].colour = g.cells[(g.cells.rows - r - 1,c)].colour;
                    } else {
                        grid.cells[(r,c)].colour = g.cells[(r,g.cells.columns - c - 1)].colour;
                    }
                }
            }

//ex.output.grid.show();
            grid = grid.subgrid(black.orow, black.cells.rows, black.ocol, black.cells.columns);
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 6, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if colour_diffs.len() == 2 {
            for i in 0 .. 2 {
                let func = |ex: &Example| {
                    if ex.input.shapes.shapes.is_empty() {
                        return Grid::trivial();
                    }
                    let mut shapes = ex.input.shapes.clone();
        //println!("{colour_diffs:?}");

                    for s in ex.input.shapes.shapes.iter() {
                        let (dir, _) = s.has_arm(1);
                        let mut ns = if dir == Up || dir == Down {
                            let colour = colour_diffs[if i == 0 { 0 } else { 1 }];
                            s.mirrored_r().recolour(s.colour, colour)
                        } else if dir == Left || dir == Right {
                            let colour = colour_diffs[if i == 1 { 0 } else { 1 }];
                            s.mirrored_c().recolour(s.colour, colour)
                        } else {
                            s.clone()
                        };

                        match dir {
                            Left => ns.to_position_mut(s.orow, s.ocol + s.cells.columns + 1),
                            Right => if s.ocol > s.cells.columns {
                                ns.to_position_mut(s.orow, s.ocol - s.cells.columns - 1)
                            } else {
                                return Grid::trivial();
                            },
                            Down => if s.orow > s.cells.rows {
                                ns.to_position_mut(s.orow - s.cells.rows - 1, s.ocol)
                            } else {
                                return Grid::trivial();
                            },
                            Up => ns.to_position_mut(s.orow + s.cells.rows + 1, s.ocol),
                            _ => (),
                        }

                        shapes.shapes.push(ns);
                    }
        //shapes.trim_to_grid().show();

                    shapes.trim_to_grid()
                };

                if let Some(rule) = run_experiment(task, 7, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }
        }

        // Should work, bug in data
        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() || colour_common.len() != 3 {
                return Grid::trivial();
            }
            let mut colour_common = colour_common.clone();
            let mut orig_colour = NoColour;

            let (div_colour, mut shapes) = ex.input.grid.full_dim_split(&ex.input.shapes);

            colour_common.retain(|&c| c != div_colour);

            for s in shapes.shapes.iter_mut() {
                if orig_colour == NoColour {
                    colour_common.retain(|&c| c != s.colour);
                    orig_colour = s.colour;
                }

                if colour_common.is_empty() {
                    return Grid::trivial();
                }

                if s.colour == div_colour {
                    break
                }

                s.recolour_mut(colour_common[0], orig_colour);
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 8, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut cc = ex.input.shapes.colour_cnt();
            let mut shapes = ex.input.shapes.clone();

            shapes.shapes = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                if !s.is_pixel() {
                    cc.remove(&s.colour);
                }
            }

            for s in ex.input.shapes.shapes.iter() {
                if !s.is_pixel() {
                    let mut ns = s.clone();
                    let bg_cnt = s.bg_count();
                    let colour: Vec<_> = cc.iter().filter(|(&c,&s)|c != ns.colour && (s == if bg_cnt % 2 == 0 { bg_cnt / 2 } else { bg_cnt / 2 + 1})).collect();
                    
                    if colour.len() != 1 {
                        return Grid::trivial();
                    }

                    // Every other pixel set
                    let mut toddle = true;

                    for ((r, c), cell) in ns.cells.items_mut() {
                        if cell.colour == Black {
                            // reset per line
                            if r % 2 == 0 && c <= 1 {
                                toddle = false;
                            }
                            if toddle {
                                cell.colour = *colour[0].0;
                            }
                            toddle = !toddle;
                        }
                    }

                    shapes.shapes.push(ns);
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 9, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();

            shapes.shapes.sort_by(|a, b| {
                let (_, a_cc) = a.colour_cnt(true);
                let (_, b_cc) = b.colour_cnt(true);

                a_cc.cmp(&b_cc)
            });

            let mut pos = 0;

            for s in shapes.shapes.iter_mut() {
                if s.cells.rows > s.cells.columns {
                    s.to_position_mut(s.orow, pos);

                    pos += s.cells.columns + 1;
                } else {
                    s.to_position_mut(pos, s.ocol);

                    pos += s.cells.rows + 1;
                }
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 10, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = InToSquaredOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 11, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.grid.in_to_squared_out(), output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = BGGridInBlack;
    if all || cat.contains(&gc) && !cat.contains(&BGGridOutBlack) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 15, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_min().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 20, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_max_colour_count().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 30, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.has_mirror_r().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 40, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.has_mirror_c().to_grid(), output) { return Some(rule); };
        //if let Some(rule) = run_experiment_examples(&file, 1000, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return Some(rule); };

        //-if let Some(rule) = run_experiment_examples(&file, 1010, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_shape_fill(&exs), output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }

    let gc = SingleColourOut2xIn;
    if all || cat.contains(&gc) { // 3?
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment_tries(task, 800, experiment, trans, is_test, examples, &targets, done, tries, &|ex, _, n| mirror_only(ex, n), output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }

    let gc = SymmetricOut;
    if all || cat.contains(&gc) { // 3?
        *cap_cats.entry(gc).or_insert(0) += 1;

        let func = |ex: &Example| {
            let xc = ex.input.shapes.shapes.iter().filter(|s| s.orow == 0).count();

            if xc == 0 {
                return Grid::trivial();
            }

            Grid::new(ex.input.shapes.len() / xc, xc, ex.input.shapes.shapes[0].colour)
        };

        if let Some(rule) = run_experiment(task, 31, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = InOutSameShapes;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        let func = &|ex: &Example| {
            let mut border = Shape::trivial();
            let mut colours: BTreeSet<Colour> = BTreeSet::new();
            let border_colour = ex.input.shapes.biggest_shape().colour;

            for s in ex.input.shapes.shapes.iter() {
                if s.colour == border_colour {
                    border = s.clone();
                } else {
                    colours.insert(s.colour);
                }
            }

            if colours.len() != 2 || border == Shape::trivial() {
                return Grid::trivial();
            }
            let sg = ex.input.grid.subgrid(border.orow, border.cells.rows, border.ocol, border.cells.columns);
            let colours: Vec<&Colour> = colours.iter().collect();
//sg.toddle_colour(*colours[0], *colours[1]).show();

            let s = sg.toddle_colour(*colours[0], *colours[1]);

            let mut shapes = Shapes::new_sized(ex.input.grid.cells.rows, ex.input.grid.cells.columns);

            shapes.shapes.push(ex.input.grid.as_shape());
            shapes.shapes.push(s.as_shape());
//shapes.to_grid().show();

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 41, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let common_colour = examples.io_common_row_colour();

        let func = |ex: &Example| {
            if ex.input.shapes.is_empty() || !ex.cat.contains(&InSameCountOut) {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();
            let mut row = 0;

            for s in shapes.shapes.iter() {
                if s.colour == common_colour {
                    row = s.orow;

                    break;
                }
            }

            for s in shapes.shapes.iter_mut() {
                s.to_position_mut(row, s.ocol);
            }
//shapes.to_grid().show();

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 42, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();

            for s in ex.input.shapes.shapes.iter() {
                if s.is_pixel() {
                    if s.orow > 0 && grid.cells[(s.orow-1,s.ocol)].colour == Black {
                        grid.flood_fill_mut(s.orow-1, s.ocol, NoColour, s.colour);
                    }
                    if s.ocol > 0 && grid.cells[(s.orow,s.ocol-1)].colour == Black {
                        grid.flood_fill_mut(s.orow, s.ocol-1, NoColour, s.colour);
                    }
                    if s.orow < grid.cells.rows - 1 && grid.cells[(s.orow+1,s.ocol)].colour == Black {
                        grid.flood_fill_mut(s.orow+1, s.ocol, NoColour, s.colour);
                    }
                    if s.ocol < grid.cells.columns - 1 && grid.cells[(s.orow,s.ocol+1)].colour == Black {
                        grid.flood_fill_mut(s.orow, s.ocol+1, NoColour, s.colour);
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 43, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if colour_diffs.len() == 2 {
            let first = &examples.examples[0].input.grid.to_shapes_sq().shapes[0];
            let first_colour = examples.examples[0].output.shapes.shapes[0].colour;
            let second_colour = if colour_diffs[0] == first_colour {
                colour_diffs[1]
            } else {
                colour_diffs[0]
            };
            let (rs, cs) = examples.examples[0].output.grid.dimensions();

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.is_empty() || rs.max(cs) != rs.min(cs) * 2 {
                    return Grid::trivial();
                }

                let mut s1 = if rs > cs {
                    Shape::new_sized_coloured(cs, cs, Black)
                } else {
                    Shape::new_sized_coloured(rs, rs, Black)
                };
                let mut s2 = if rs > cs {
                    Shape::new_sized_coloured_position(0, cs, cs, cs, Black)
                } else {
                    Shape::new_sized_coloured_position(rs, 0, rs, rs, Black)
                };

                let mut size0 = 0;
                let mut size1 = 0;

                for s in ex.input.grid.to_shapes_sq().shapes.iter() {
                    if first.same_pixel_positions(s, false) {
                        size0 += 1;
                    } else {
                        size1 += 1;
                    }
                }
                
                s1.fill_corners_mut(size0, first_colour);
                s2.fill_corners_mut(size1, second_colour);

//Shapes::new_shapes(&[s1.clone(), s2.clone()]).to_grid().show();
                Shapes::new_shapes(&[s1.clone(), s2.clone()]).to_grid()
            };

            if let Some(rule) = run_experiment(task, 44, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }
        *cap_todo.entry(gc).or_insert(0) += 1;
    }

    let gc = BGGridOutBlack;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if all || cat.contains(&IdenticalNoColours) {
            let func = &|ex: &Example| {
                if ex.input.shapes.shapes.len() % 2 != 0 || ex.input.shapes.shapes.len() == ex.input.coloured_shapes.shapes.len() {
                    return Grid::trivial();
                }
                let mut shapes = ex.input.shapes.clone();
                shapes.shapes = Vec::new(); // Zoroize shapes
//ex.input.shapes.show();

                for shape in ex.input.coloured_shapes.shapes.iter() {
                    let inner_shapes = shape.to_shapes();
                    let mut small = inner_shapes.smallest();
                    let mut large = inner_shapes.largest();

                    if shape.cells.rows != shape.cells.columns || inner_shapes.len() != 2 || !large.can_contain(&small) {
                        return Grid::trivial();
                    }

                    let small_colour = small.colour;

                    small.recolour_mut(small_colour, large.colour);
                    large.recolour_mut(large.colour, small_colour);

                    // Assume rows == columns for small
                    let enclose = large.surround(small.cells.rows, small.colour, false, false);

                    shapes.shapes = [shapes.shapes, vec![large, small, enclose]].concat();
                }
//shapes.show();

                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 50, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let func = &|ex: &Example| {
            let rows = ex.input.grid.cells.rows;
            let cols = ex.input.grid.cells.columns;
            let h = ex.input.grid.cell_colour_cnt_map();
//print!("{h:?} -> ");
//let h2 = ex.input.grid.cell_colour_cnt_map();
//println!("{h2:?}");
            let mut grid = Grid::new(rows, cols, Black);

            for (col, size) in h.iter() {
                let c = Colour::to_usize(*col) - 1;

                if *size >= rows || c >= cols {
                    return Grid::trivial();
                }

                for r in 0 .. *size {
                    let r = rows - r - 1;

                    grid.cells[(r,c)].row = r;
                    grid.cells[(r,c)].col = c;
                    grid.cells[(r,c)].colour = *col;
                }
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 60, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |gi: &Example| {
//gi.input.shapes.show();
            let mut rc: Vec<(usize, usize, Colour)> = Vec::new();
            let mut colour = NoColour;
            let mut shapes = gi.input.shapes.clone();

            for s in gi.input.shapes.shapes.iter() {
                if colour == NoColour {
                    colour = s.colour;
                }
                if colour != s.colour {
                    rc.push((s.cells[(0,0)].row, s.cells[(0,0)].col, s.colour));

                    break;
                }
            }

            for ss in shapes.shapes.iter_mut() {
                for (rr, cc, colour) in rc.iter() {
                    if ss.cells[(0,0)].row == *rr || ss.cells[(0,0)].col == *cc {
                        ss.colour = *colour;

                        for i in ss.cells.keys() {
                            ss.cells[i].colour = *colour;
                        }
                    }
                }
            }
//shapes.to_grid().show();

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 70, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = FullyPopulatedOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 80, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_max().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 90, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.largest().to_grid(), output) { return Some(rule); };
//target.show();
//ans.show();

        if let Some(rule) = run_experiment(task, 100, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_min().to_grid(), output) { return Some(rule); };

        let common_colours = examples.find_output_colours();

        for colour in common_colours {
            if let Some(rule) = run_experiment(task, 110, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.cell_colour_cnts(colour).to_grid(), output) { return Some(rule); };
        }

        if let Some(rule) = run_experiment(task, 120, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_largest_count().to_grid(), output) { return Some(rule); };

        let func = |ex: &Example| {
            let mut cnt = 0;
            let mut shapes: Vec<Shape> = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                if s.size() >= 9 && s.colour != Mixed && s.dense() {
                    cnt += 1;

                    shapes.push(s.clone())
                }
            }

            let height = cnt / 3;

            if height == 0 {
                return Grid::trivial();
            }

            let mut grid = Grid::new(height, 3, Black);
            let mut r = 0;

            // Get right ordering by munging row coord then sorting
            for (i, s) in shapes.iter_mut().enumerate() {
                if i > 0 && (i % 3) == 0 {
                    r += 1;
                }

                s.orow = r;
            }

            shapes.sort_by(|a, b| (a.orow, a.ocol).cmp(&(b.orow, b.ocol)));
//println!("{} by {}", ex.input.grid.cells.rows, ex.input.grid.cells.columns);

            for (i, s) in shapes.iter().enumerate() {
                let r = i / 3;
                let c = i % 3;
//println!("{:?} {i} {}/{} -> {}/{}", s.colour, s.orow, s.ocol, x, y);

                if r >= grid.cells.rows || c >= grid.cells.columns {
                    return Grid::trivial();
                }

                grid.colour = s.colour;

                grid.cells[(r,c)].row = i / 3;
                grid.cells[(r,c)].col = i / height;
                grid.cells[(r,c)].colour = s.colour;
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 121, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !ex.input.grid.is_square() {
                return Grid::trivial();
            }

            let colours = ex.input.grid.find_colour_row_order();
            let len = colours.len();

            if len == 0 {
                return Grid::trivial();
            }
            let colours: BTreeMap<usize, Colour> = colours.iter()
                .map(|(k,&v)| (k % len, v))
                .collect();

            let mut grid = ex.input.grid.clone();

            for ((r, c), cell) in ex.input.grid.cells.items() {
                if cell.colour == Black {
                    let idx = (r + c) % len;

                    if let Some(colour) = colours.get(&idx) {
                        grid.cells[(r,c)].colour = *colour;
                    }
                }
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 122, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let ex = &examples.examples[0];
        let in_colour = ex.output.grid.get_diff_colour(&ex.input.grid);
        let out_colour = ex.input.grid.get_diff_colour(&ex.output.grid);

        let func = |ex: &Example| {
            ex.input.grid.recolour(in_colour, out_colour)
        };

        if let Some(rule) = run_experiment(task, 123, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let mut colours = ex.input.shapes.find_shape_colours();

            let mut width = 0;
            let mut first = true;

            for s in ex.input.shapes.shapes.iter() {
                if !first && s.ocol == 0 {
                    break;
                }

                first = false;
                width += 1;
            }

            if colours.len() % 2 == 1 && (colours.len() + 1) % width == 0 {
                let mut ps = Shape::trivial();
                
                for (n, s) in ex.input.shapes.shapes.iter().enumerate() {
                    if s.cells.rows > ps.cells.rows {
                        if ps.cells.rows > 0 {
                            if n + n + 1 > colours.len() {
                                return Grid::trivial();
                            }

                            colours.insert(n + n + 1, s.colour);

                            break;
                        }
                        ps = s.clone();
                    }
                }
            }

            if width == 0 {
                return Grid::trivial();
            }

            let mut grid = Grid::new(colours.len() / width, width, Black);

            for (i, cell) in grid.cells.values_mut().enumerate() {
                cell.colour = colours[i];
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 124, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let (rs, cs) = examples.examples[0].output.grid.dimensions();

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let (div_colour, shapes) = ex.input.grid.full_dim_split(&ex.input.shapes);

            let mut colours: BTreeMap<Colour, Vec::<usize>> = BTreeMap::new();

            for s in shapes.shapes.iter() {
                if s.colour != div_colour {
                    colours.entry(s.colour).and_modify(|size| size.push(s.size())).or_insert(vec![s.size()]);
                }
            }

            let mut colour = NoColour;

            for (k,v) in colours.iter() {
                if v.len() != 2 {
                    return Grid::trivial();
                }

                if v[1] > v[0] + 2 {
                    colour = *k;
                    break;
                } else if v[0] + 2 == v[1] {
                    colour = *k;
                }
            }

            if colour == NoColour {
                return Grid::trivial();
            }

            Grid::new(rs, cs, colour)
        };

        if let Some(rule) = run_experiment(task, 125, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = MirrorCIn;
    if all || cat.contains(&gc) || cat.contains(&MirrorCOut) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut row = ex.input.shapes.shapes[0].orow;
            let mut shapes = ex.input.shapes.clone();

            for s in shapes.shapes.iter_mut().rev() {
                s.to_position_mut(row, s.ocol);
                row += s.cells.rows;
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 125, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // Also done by 142
        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let shapes = ex.input.grid.to_shapes_cons();
            let mut new_shapes = shapes.clone();

            new_shapes.shapes = Vec::new();

            for s in shapes.shapes.iter() {
                let ns = s.mirrored_r();

                new_shapes.shapes.push(ns);
            }

            new_shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 126, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = MirrorROut;
    if all || cat.contains(&gc) || cat.contains(&MirrorCOut) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        let func = |gi: &Example| {
            if gi.input.coloured_shapes.len() != 1 {
                return Grid::trivial();
            }

            let s = gi.input.grid.as_shape();

            let rows = s.cells.rows;
            let cols = s.cells.columns;
            let mut shapes = Shapes::new_sized(rows * 2, cols * 2);

            shapes.shapes.push(s.mirrored_r().mirrored_c());
            shapes.shapes.push(s.mirrored_c().translate_absolute(rows, 0));
            shapes.shapes.push(s.mirrored_r().translate_absolute(0, cols));
            shapes.shapes.push(s.translate_absolute(rows, cols));

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 280, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let bg = ex.input.grid.max_colour();
            let shapes = ex.input.grid.to_shapes_sq();
            let shapes = shapes.colour_groups_to_shapes(bg);

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 281, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Div9In;
    if all || cat.contains(&gc) && !cat.contains(&Div9Out) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        //-if let Some(rule) = run_experiment_examples(&file, 1020, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_9_in_out(&exs), output) { return Some(rule); };
        //
        let out_colours = examples.find_all_output_colours();
        let (rs, cs) = examples.examples[0].output.grid.dimensions();

        if out_colours.len() == 2 && examples.examples[0].input.grid.is_3x3() {
            let func = |ex: &Example| {
               if ex.input.grid.cells[(0,1)].colour == Black && ex.input.grid.cells[(2,1)].colour == Black {
                   Grid::new(rs,cs,out_colours[0])
               } else {
                   Grid::new(rs,cs,out_colours[1])
               }
            };

            if let Some(rule) = run_experiment(task, 285, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Div9Out;
    if all || cat.contains(&Is3x3In) && cat.contains(&gc) && !cat.contains(&Div9In){ 
        *cap_cats.entry(gc).or_insert(0) += 1;

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Div9In;
    if all || cat.contains(&gc) && cat.contains(&Is3x3Out) && !cat.contains(&Div9Out) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        //-if let Some(rule) = run_experiment_examples(&file, 1030, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return Some(rule); };

        //-if let Some(rule) = run_experiment_examples(&file, 1040, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_double(&exs), output) { return Some(rule); };

        //-if let Some(rule) = run_experiment_examples(&file, 1050, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_expand_3x3(&exs), output) { return Some(rule); };

        if examples.examples[0].input.grid.size() * 9 == examples.examples[0].output.grid.size() {
            for n in 0 .. 2 {
                let colour = if n == 1 {
                    let shapes_out = examples.examples[0].output.coloured_shapes.shapes.len();
                    let h = examples.examples[0].input.grid.cell_colour_cnt_map();
                    let colours: Vec<Colour> = h.iter().filter(|(_, &v)| v == shapes_out).map(|(&k,_)| k).collect();

                    if colours.len() != 1 {
                        NoColour
                    } else {
                        colours[0]
                    }
                } else {
                    NoColour
                };

                let func = |ex: &Example| {
                    let shape = &ex.input.grid.as_shape();
                    let colour = if n == 0 {
                        let (colour, _) = shape.colour_cnt(false);

                        colour
                    } else {
                        colour
                    };

                    let posns = shape.colour_position(colour);
                    let mut shapes = Shapes::new_sized(shape.cells.rows * 3, shape.cells.columns * 3);

                    for (r, c) in posns.iter() {
                        let or = r * 3;
                        let oc = c * 3;

                        shapes.shapes.push(shape.translate_absolute(or, oc));
                    }
//shapes.to_grid().show();
                    shapes.to_grid()
                };

                if let Some(rule) = run_experiment(task, 130, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = IdenticalNoPixels;
    if all || cat.contains(&gc) { // 164
        *cap_cats.entry(gc).or_insert(0) += 1;

        /*
        let func = |ex: &Example| {
            let grid = &ex.input.grid;
            if grid.cells.rows != grid.cells.columns {
                return Grid::trivial();
            }

            let shapes = &ex.input.shapes.shapes;
            let trivial = Shape::trivial();
            let mut border1: &Shape = &trivial;
            let mut border2: &Shape = &trivial;
            let mut x = false;
//ex.input.shapes.show();

            for s in shapes {
                if s.size() == 1 {
                    continue;
                }
                if s.orow == 0 && s.cells.columns == grid.cells.columns {
                    x = true;
                    border1 = s;
                } else if s.orow == grid.cells.rows - 1 && s.cells.columns == grid.cells.columns {
                    x = true;
                    border2 = s;
                } else if s.ocol == 0 && s.cells.rows == grid.cells.rows {
                    border1 = s;
                } else if s.ocol == grid.cells.columns - 1 && s.cells.rows == grid.cells.rows {
                    border2 = s;
                }
            }

            let mut new_shapes = ex.input.shapes.clone();

            for s in new_shapes.shapes.iter_mut() {
                if s.size() == 1 {
                    if x && s.distance_x(border1) < s.distance_x(border2) || !x && s.distance_y(border1) < s.distance_y(border2){
                        s.force_recolour_mut(border1.colour);
                    } else {
                        s.force_recolour_mut(border2.colour);
                    }
                }
            }
println!("139 --- {file}");

            new_shapes.to_grid()
        };
        */

        // Not working???
        //if let Some(rule) = run_experiment(file, 139, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let grid = &ex.input.grid;
            if grid.cells.rows != grid.cells.columns {  // Must be square
                return Grid::trivial();
            }
            // Do manually as will miss shapes starting in 0,0!
            let shapes = grid.to_shapes();

            if shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let x_axis = shapes.shapes[0].height() > 1;

            let borders: Vec<_> = shapes.shapes.iter().filter(|s| s.size() > 1).collect();
            if borders.len() != 2 {
                return Grid::trivial();
            }

            let c1 = borders[0].colour;
            let c2 = borders[1].colour;

            let mut new_grid = grid.clone();

            for s in ex.input.shapes.shapes.iter() {
                if s.size() == 1 {
                    if x_axis {
                        new_grid.cells[(s.orow, s.ocol)].colour = if s.ocol < grid.cells.columns / 2 { c1 } else { c2 };
                    } else {
                        new_grid.cells[(s.orow, s.ocol)].colour = if s.orow < grid.cells.rows / 2 { c1 } else { c2 };
                    }
                }
            }
//new_grid.show();

            new_grid
        };

        if let Some(rule) = run_experiment(task, 141, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // Also done by 126
        let func = |ex: &Example| {
            let mut grid = ex.input.grid.clone();

            for (r, c) in ex.input.grid.cells.keys() {
                if ex.input.grid.cells[(r, 0)].colour != Black && grid.cells[(r, c)].colour != Black {
                    grid.cells[(r, c)].colour = ex.input.grid.cells[(r, 0)].colour;
                }
            }
            
            grid
        };

        if let Some(rule) = run_experiment(task, 142, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let out_colours = examples.find_output_colours();

        let func = |ex: &Example| {
            if out_colours.len() != 1 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();
            // get them in right order to alternate
            shapes.shapes.sort_by_key(|a| a.size());
            let mut toddle = false;

            for shape in shapes.shapes.iter_mut() {
                for cell in shape.cells.values_mut() {
                    if cell.colour != Black {
                        if toddle {
                            cell.colour = out_colours[0];
                        }
                        toddle = !toddle;
                    }
                }
            }
            // reverse order to lay on top on one another
            shapes.shapes.sort_by_key(|b| std::cmp::Reverse(b.size()));

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 145, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.grid.to_shapes_sq();

            for shape in shapes.shapes.iter_mut() {
                if shape.is_pixel() {
                    shape.force_recolour_mut(colour_diffs[0]);
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 146, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.grid.to_shapes_sq();

            shapes.shapes.sort_by(|a, b| (a.ocol, a.orow).cmp(&(b.ocol, b.orow)));

            let mut toddle = ex.input.grid.cells.columns % 2 != 0;

            for shape in shapes.shapes.iter_mut() {
                if toddle {
                    shape.force_recolour_mut(colour_diffs[0]);
                }
                toddle = !toddle;
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 147, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let ex = &examples.examples[0];
        let smallest = ex.input.shapes.smallest();
        let in_colour = smallest.colour;

        if ex.output.grid.height() >= ex.input.grid.height() && ex.output.grid.width() >= ex.input.grid.width() {
            let out_colour = ex.output.grid.cells[(smallest.orow,smallest.ocol)].colour;
            let mut diag_colour = colour_diffs.clone();

            diag_colour.retain(|&c| c != out_colour);

            // testing function
            let func = |ex: &Example| {
                if diag_colour.len() != 1 {
                    return Grid::trivial();
                }

                let mut grid = ex.input.grid.clone();
                let bg = grid.has_bg_grid_not_sq();

                for s in ex.input.shapes.shapes.iter() {
                    if s.is_pixel() {
                        grid.draw_bg_mut(Up, s.orow, s.ocol, in_colour, bg);
                        grid.draw_bg_mut(Down, s.orow, s.ocol, in_colour, bg);
                        grid.draw_bg_mut(Left, s.orow, s.ocol, in_colour, bg);
                        grid.draw_bg_mut(Right, s.orow, s.ocol, in_colour, bg);
                    }
                }
                
                let mut shapes = Shapes::new_sized(grid.cells.rows, grid.cells.columns);

                shapes.shapes.push(grid.as_shape());

                for s in ex.input.shapes.shapes.iter() {
                    if s.is_pixel() {
                        let ns = s.recolour(in_colour, out_colour);

                        shapes.shapes.push(ns);

                        let surround = s.surround(1, diag_colour[0], false, true);

                        shapes.shapes.push(surround);
                    }
                }

//shapes.to_grid().show();
                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 148, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();
            let mut index: Shape = Shape::trivial();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.size() < 10 && (s.cells.rows == 1 || s.cells.columns == 1) {
                    if s.cells.rows > 1 {
                        // normalise index
                        index = s.to_grid().rot_rect_270().as_shape();
                    } else {
                        index = s.clone();
                    }
                    break;
                }
            }

            if index == Shape::trivial() {
                return Grid::trivial();
            }

            let idx_len = index.cells.columns;

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.size() < 10 {
                    continue;
                }
                // Does it lean right or left!
                let right = s.cells[(s.cells.rows - 1,0)].colour == Black;
                let size = if right {   // Assume it's square!
                    let mut i: usize = 0;

                    for _ in 0 .. s.cells.rows {
                        if s.cells[(i,0)].colour == Black {
                            break;
                        }
                        i += 1;
                    }
                    
                    i
                } else {
                    let mut i: usize = 0;

                    for _ in (0 .. s.cells.rows).rev() {
                        if s.cells[(i,s.cells.columns - 1)].colour == Black {
                            break;
                        }
                        i += 1;
                    }

                    i
                };

                let mut idx = 0;
                let mut ns = s.clone();

                // Now do the real work
                for i in 0 .. s.cells.rows - size + 1 {
                    for pos in 0 .. size {
                        if i + pos >= s.cells.rows || i + pos >= s.cells.columns {
                            return Grid::trivial();
                        }
                        let colour = index.cells[(0,idx)].colour;
                        if right {
                            ns.cells[(i+pos,i+pos)].colour = colour;

                            for inc in 1 .. size {
                                if i+pos+inc < s.cells.rows {
                                    ns.cells[(i+pos+inc,i+pos)].colour = colour;
                                }
                                if i+pos+inc < s.cells.columns {
                                    ns.cells[(i+pos,i+pos+inc)].colour = colour;
                                }
                            }
                        } else {
                            ns.cells[(i+pos,s.cells.columns-1-pos-i)].colour = colour;
                            for inc in 1 .. size {
                                if i+pos+inc < s.cells.rows {
                                    ns.cells[(i+pos+inc,s.cells.columns-1-pos-i)].colour = colour;
                                }
                                if i+pos+inc < s.cells.columns {
                                    ns.cells[(i+pos,s.cells.columns-1-pos-i-inc)].colour = colour;
                                }
                            }
                        }
                    };

                    idx = (idx + 1) % idx_len;
                }

                shapes.shapes.push(ns);
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 149, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Is3x3In;
    if all || cat.contains(&gc) && cat.contains(&Is3x3Out){ 
        *cap_cats.entry(gc).or_insert(0) += 1;

//println!("#### {file}");
        //-if let Some(rule) = run_experiment_examples(&file, 1060, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return Some(rule); };

        let func = |gi: &Example| {
            let mut big: Shape = Shape::trivial();
            let mut n = 0;

            if gi.input.grid.cells.rows != 9 || gi.input.grid.cells.columns != 9 {
                return Grid::trivial();
            }

            for s in &gi.input.shapes.shapes {
                if s.size() > 1 {
                    if s.cells.rows > 3 || s.cells.columns > 3 {
                        return Grid::trivial();
                    }
                    if s.cells.rows == 3 && s.cells.columns == 3 {
                        if big != Shape::trivial() {
                            return Grid::trivial();
                        }
                        big = s.clone();
                    } else if s.cells.rows == 2 && s.cells.columns == 3 {
                        let mut m = Matrix::new(3, 3, Cell::new(0, 0, 0));
                        for c in 1 .. 3 {
                            m[(0,c)].row = 0;
                            m[(0,c)].col = c;
                        }
                        for r in 1 .. 3 {
                            for c in 0 .. 3 {
                                m[(r,c)].row = r;
                                m[(r,c)].col = c;
                                m[(r,c)].colour = s.cells[(r-1,c)].colour;
                            }
                        }
                        big = Shape::new(0, 0, &m);
                    } else if s.cells.rows == 3 && s.cells.columns == 2 {
                        let mut m = Matrix::new(3, 3, Cell::new(0, 0, 0));
                        for r in 1 .. 3 {
                            m[(r,0)].row = r;
                            m[(r,0)].col = 0;
                        }
                        for r in 0 .. 3 {
                            for c in 1 .. 3 {
                                m[(r,c)].row = r;
                                m[(r,c)].col = c;
                                m[(r,c)].colour = s.cells[(r,c-1)].colour;
                            }
                        }
                        big = Shape::new(0, 0, &m);
                    }
                } else {
                    n += 1;
                }
            }
//big.show();

            let rows = big.cells.rows;
            let cols = big.cells.columns;
//println!("{rows}/{cols}");
            let mut shapes = Shapes::new_sized(rows, cols * n);
            let big = big.to_origin();

            shapes.shapes.push(big.clone());
//shapes.show();

            for i in 1 .. n {
                shapes.shapes.push(big.translate_absolute(0, cols * i));
            }
//println!("{:?}", shapes);

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 140, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = InOutSquareSameSize;
    if all || cat.contains(&gc) { // 164
        *cap_cats.entry(gc).or_insert(0) += 1;

        if all || cat.contains(&InSameCountOut) || cat.contains(&InSameCountOutColoured) {
            //if let Some(rule) = run_experiment_examples(&file, 1070, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_shape_substitute(&exs), output) { return Some(rule); };

            let func = |ex: &Example| {
                let biggest = ex.input.coloured_shapes.biggest_shape();
                if biggest.size() != ex.input.grid.size() {
                    return Grid::trivial();
                }
                let (idx, dir) = ex.input.grid.corner_idx();
                if dir == Other {
                    return Grid::trivial();
                }
                let mut shapes = Shapes::new_from_shape(&biggest);
                let body = ex.input.grid.corner_body(dir.inverse());
                let four = body.split_4();

                if four.is_empty() {
                    return Grid::trivial();
                }

                four.iter().zip(idx.cells.values())
                    .for_each(|(s,c)| shapes.add(&s.recolour(s.colour, c.colour).as_shape()));

                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 141, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.len() > 20 {
                    return Grid::trivial();
                }

                let mut shapes = ex.input.shapes.clone();

                shapes.shapes = Vec::new();

                //for s in &ex.input.shapes.consolidate_shapes().shapes {
                for s in &ex.input.shapes.shapes {
                    shapes.shapes.push(s.mirrored_r());
                }
//shapes.to_grid().show();

                shapes.to_grid()
            };

            // evaluation only
            if let Some(rule) = run_experiment(task, 142, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        if all || cat.contains(&InLessCountOut) {
            //// see 045e512c.json
//println!("#### {}", examples.examples[0].input.shapes.len());
//examples.examples[0].input.shapes.show();
        }

        // 0ca9ddb6 4258a5f9 913fb3ed 95990924 b60334d2 test
        if let Some(rule) = run_experiment_tries(task, 0, experiment, trans, is_test, examples, &targets, done, tries, &|ex, _, n| transform_only(ex, n), output) { return Some(rule); };
        //-if let Some(rule) = run_experiment_examples(&file, 1100, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return Some(rule); };

        let func = |ex: &Example| {
            let mut grid = ex.input.grid.clone();

            for s in ex.input.shapes.shapes.iter() {
                if s.size() == 1 {
                    grid.cells[(ex.input.grid.cells.rows - 1,s.ocol)].colour = s.colour;
                    grid.cells[(s.orow,s.ocol)].colour = Black;
                }
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 220, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = &|ex: &Example| {
            fn do_diag(grid: &mut Grid, s: &Shape) {
                grid.draw_mut(FromUpLeft, s.orow, s.ocol,s.cells[(s.cells.rows-1,0)].colour);
                grid.draw_mut(FromUpRight, s.orow, s.ocol + s.cells.columns, s.cells[(0,0)].colour);
                grid.draw_mut(FromDownRight, s.orow + s.cells.rows-1, s.ocol + s.cells.columns-1, s.cells[(0,s.cells.columns-1)].colour);
                grid.draw_mut(FromDownLeft, s.orow + s.cells.rows, s.ocol, s.cells[(s.cells.rows-1,s.cells.columns-1)].colour);
            }

            let mut grid = ex.input.grid.clone();
//ex.input.shapes.show();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.cells.rows != 8 || s.cells.columns != 6 {
                    return Grid::trivial();
                }
//s.show();
                if s.size() > 16 {
                    // TODO: Parameterise it
                    let ss1 = s.subshape_remain(0, 4, 0, 6).shrink_coloured();
                    let ss2 = s.subshape_remain(4, 4, 0, 6).shrink_coloured();

                    do_diag(&mut grid, &ss1);
                    do_diag(&mut grid, &ss2);
                } else {
                    // Rotated 90 degrees
                    do_diag(&mut grid, s);
                }
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 163, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let colour = colour_diffs[0];
            let g = &ex.input.grid;
            let mut grid = g.clone();

            for ((r, c), cell) in g.cells.items() {
                if c > grid.cells.columns / 2 {
                    continue;
                }

                if cell.colour != Black && cell.colour == g.cells[(r, g.cells.columns - 1 - c)].colour {
                    grid.cells[(r, c)].colour = colour;
                    grid.cells[(r, g.cells.columns - 1 - c)].colour = colour;
                }
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 165, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let g = &ex.input.grid;
            let s = &ex.input.shapes.shapes[0];

            if s.cells.rows < 2 || s.cells.columns < 2 {
                return Grid::trivial();
            }

            let border = s.colour;
            let idx = g.subgrid(0, s.cells.rows - 1, 0, s.cells.columns - 1);
            let mut idx = idx.as_shape();

            if idx.cells.rows < 2 || idx.cells.columns < 2 {
                return Grid::trivial();
            }

            let (r, c) = g.find_patch(&idx);

            if r == 0 && c == 0 {
                return Grid::trivial();
            }

            idx.to_position_mut(r, c);

            let idx = idx.add_border(border);
            let mut shapes = g.as_shapes();

            shapes.shapes.push(idx);
//shapes.to_grid().show();

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 166, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = InOutSquare;
    if all || cat.contains(&gc) { // 27
        *cap_cats.entry(gc).or_insert(0) += 1;

        //if let Some(rule) = run_experiment_examples(&file, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_shape_fill(&exs), output) { return Some(rule); };

        //-if let Some(rule) = run_experiment_examples(&file, 1050, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_expand_3x3(&exs), output) { return Some(rule); };

        let func = |ex: &Example| {
            if !ex.input.grid.is_square() {
                return Grid::trivial();
            }

            let height = ex.input.grid.height();
            let width = ex.input.grid.width();

            let mut shapes = Shapes::new_sized(height * 2, width * 2);

            let shape = ex.input.grid.as_shape();
            shapes.shapes.push(shape);

            let shape = ex.input.grid.as_shape_position(0, width).rotated_270();
            shapes.shapes.push(shape);

            let shape = ex.input.grid.as_shape_position(height, 0).rotated_180();
            shapes.shapes.push(shape);

            let shape = ex.input.grid.as_shape_position(height, width).rotated_90();
            shapes.shapes.push(shape);

            shapes.to_grid()
        };
        if let Some(rule) = run_experiment(task, 160, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // TODO: Combine 160 and 162
        let func = |ex: &Example| {
            if !ex.input.grid.is_square() {
                return Grid::trivial();
            }

            let height = ex.input.grid.height();
            let width = ex.input.grid.width();

            let mut shapes = Shapes::new_sized(height * 2, width * 2);

            let shape = ex.input.grid.as_shape_position(height, width);
            shapes.shapes.push(shape);

            let shape = ex.input.grid.as_shape_position(0, width).mirrored_r();
            shapes.shapes.push(shape);

            let shape = ex.input.grid.as_shape_position(height, 0).mirrored_c();
            shapes.shapes.push(shape);

            let shape = ex.input.grid.as_shape().mirrored_r().mirrored_c();
            shapes.shapes.push(shape);

            shapes.to_grid()
        };
        if let Some(rule) = run_experiment(task, 162, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = &|ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() != 1 || !ex.input.coloured_shapes.shapes[0].is_square() || ex.input.coloured_shapes.shapes[0].size() != 36 {
                return Grid::trivial();
            }

            let shape = &ex.input.coloured_shapes.shapes[0];
            let ss = shape.subshape(0, 3, 0, 3);

            ss.to_grid()
        };

        if let Some(rule) = run_experiment(task, 163, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
//a.show();
    }
    let gc = InOutSameSize;
    if all || cat.contains(&gc) { // 97
        *cap_cats.entry(gc).or_insert(0) += 1;

        /* Causes problems probably not
        //if let Some(rule) = run_experiment_examples(&file, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_shape_fill(&exs), output) { return Some(rule); };

        //if let Some(rule) = run_experiment_examples(&file, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return Some(rule); };

        // 5
        if all || cat.contains(&InSameCountOut) {
            //if let Some(rule) = run_experiment_examples(&file, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_move_together(&exs), output) { return Some(rule); };
        }
        */

        if cat.contains(&IsPanelledROut) {
            let func = |ex: &Example| {
                let grid_shape = ex.input.grid.as_shape();
                let colour = grid_shape.majority_colour();
                let bg_shapes = ex.input.grid.to_shapes_base_bg(colour);
                let not_empty: Vec<Shape> = bg_shapes.shapes.iter().filter(|s| !s.is_empty()).cloned().collect();

                if not_empty.len() != 1 {
                    return Grid::trivial();
                }

                let mut shapes = ex.input.shapes.clone();

                shapes.shapes = Vec::new();

                shapes.shapes.push(grid_shape);

                for s in &bg_shapes.shapes {
                    if s.is_empty() {
                        shapes.shapes.push(not_empty[0].copy_into(s));
                    } else {
                        shapes.shapes.push(s.clone());
                    }
                }
//shapes.to_grid().show();

                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 500, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        //if let Some(rule) = run_experiment(file, 150, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.toddle_colours().to_grid(), output) { return Some(rule); };

        // Clunky and not very generic 42918530 in eval
        let func = |gi: &Example| {
            let mut h: BTreeMap<Colour, Shape> = BTreeMap::new();
            let mut size = 0;

            for s in gi.input.shapes.shapes.iter() {
//s.show();
                if size == 0 {
                    size = s.size();
                }
                if size != 25 {
                    return Grid::trivial();
                } else if s.size() < size {
                    if let Some(ls) = h.get(&s.colour) {
                        let mut ls = ls.clone();

                        if s.contained_by(&ls) {
//println!("heer {:?}", s.move_in(ls));
//s.move_in(ls).show();
                            if s.size() == 1 && size == 25 {
                                ls.cells[(2, 2)].colour = s.colour;
//ls.show();
                                h.insert(s.colour, ls.clone());
                                continue;
                            }
                        }
                    }
                    return Grid::trivial();
                }
                if let Some(shape) = h.get(&s.colour) {
                    if shape.pixels() <= s.pixels() {
                        h.insert(s.colour, s.clone());
                    }
                } else {
                    h.insert(s.colour, s.clone());
                }
            }

            let mut shapes = Shapes::new_sized(gi.input.shapes.nrows, gi.input.shapes.ncols);

            for s in gi.input.shapes.shapes.iter() {
                if s.size() < size { continue; }
//s.show_summary();
                if let Some(shape) = h.get(&s.colour) {
                    if shape.pixels() >= s.pixels() {
                        let mut new_shape = s.clone();
//shape.show_summary();
                        for ((r, c), cell) in shape.cells.items() {
                            if r >= new_shape.cells.rows || c >= new_shape.cells.columns {
                                return Grid::trivial();
                            }
                            new_shape.cells[(r,c)].colour = cell.colour;
                        }
                        shapes.add(&new_shape);
                    } else {
                        shapes.add(shape);
                    }
                }
            }
//println!("{:?}", h);
//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 161, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if !cat.contains(&FullyPopulatedIn) {
            let func = |gi: &Example| {
                let mut si = gi.clone();
                let mut dr = 0;
                let mut dc = 0;

                for s in &gi.input.shapes.shapes {
                    if dr == 0 {
                        dr = s.orow;
                    } else if s.orow % dr != 0 {
                        return Grid::trivial();
                    }
                    if dc == 0 {
                        dc = s.ocol;
                    } else if s.ocol % dc != 0 {
                        return Grid::trivial();
                    }
//println!("{} {}", s.ox, s.oy);
                }

                if dr != 0 && dc != 0 {
                    return Grid::trivial();
                }

                si.input.shapes.shapes.sort_by_key(|b| std::cmp::Reverse(b.pixels()));

                for (i, s) in si.input.shapes.shapes.iter_mut().enumerate() {
                    s.orow = i * dr;
                    s.ocol = i * dc;

                    for (r, c) in s.cells.keys() {
                        s.cells[(r, c)].row = s.orow + r;
                        s.cells[(r, c)].col = s.ocol + c;
                    }
                }
//si.input.shapes.to_grid().show();

                si.input.shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 170, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let func = |gi: &Example| {
            if gi.input.shapes.is_empty() || gi.input.shapes.len() > 21 {
                return Grid::trivial();
            }

            let mut colours: Vec<Colour> = Vec::new();
            let mut si = gi.input.shapes.clone();

            let mut full_size = false;

            for s in si.shapes.iter() {
                if s.size() == gi.input.grid.size() {
                    full_size = true;
                }
                if s.size() == 1 {
                    colours.push(s.colour);
                }
            }

            if colours.is_empty() || !full_size {
                return Grid::trivial();
            }

            let mut i = 0;

            for s in si.shapes.iter_mut() {
                if s.size() != 1 && s.size() != gi.input.grid.size() {
                    if i >= colours.len() {
                        return Grid::trivial();
                    }
                    s.mut_recolour(s.colour, colours[i]);
                    
                    /* Not Necessary
                    if !s.is_square() {
                        s.cells = s.make_square().cells;
                    }
                    */
                    i += 1;
                }
            }

            // Should not be necessary???
            si.shapes.sort_by_key(|b| std::cmp::Reverse(b.size()));

            si.to_grid()
        };

        if let Some(rule) = run_experiment(task, 180, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = &|ex: &Example| {
            let mut grid = ex.input.grid.clone();
            let cnts = ex.input.shapes.shape_colour_cnt_map();

            for a_colour in Colour::all_colours().iter() {
                if let Some(cs) = cnts.get(a_colour) {
                    if cs.len() == 2 || cs.len() == 4 {
                        let size = cs[0].size();
                        if cs.len() == 2 || cs.iter().filter(|s| s.size() == 1).count() == size {
                            // Now draw the lines
                            let v = Vec::from_iter(cs);

                            grid.draw_lines(&v, cs.len() == 2, true);
                        }
                    }
                }
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 51, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !ex.cat.contains(&NoShapesIn(5)) || !ex.cat.contains(&NoShapesOut(6)) || colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let shapes = &ex.input.shapes;
            let corners = shapes.corners();
            let centre = (corners.0 + (corners.2 - corners.0) / 2, corners.1 + (corners.3 - corners.1) / 2);

            let mut dist = f32::MAX;

            for s in shapes.shapes.iter() {
                let d = s.distance_from(centre.0, centre.1);

                if s.orow > corners.0 && s.orow < corners.2 && s.ocol > corners.1 && s.ocol < corners.3 && d < dist {
                    dist = d;
                }
            }

            let mut grid = ex.input.grid.clone();

            for s in shapes.shapes.iter() {
                if s.orow == corners.0 {
                    grid.draw_line_row_coords(corners.0, corners.1 - 1, corners.0, corners.3, s.colour, true, false, 1);
                } else if s.orow == corners.2 - 1 {
                    grid.draw_line_row_coords(corners.2 - 1, corners.1 - 1, corners.2 - 1, corners.3, s.colour, true, false, 1);
                } else if s.ocol == corners.1 {
                    grid.draw_line_col_coords(corners.0, corners.1, corners.2 - 1, corners.1, s.colour, true, false, 1);
                } else if s.ocol == corners.3 - 1 {
                    grid.draw_line_col_coords(corners.0, corners.3 - 1, corners.2 - 1, corners.3 - 1, s.colour, true, false, 1);
                } else {
                    grid.draw_line_row_coords(s.orow, corners.1, s.orow, corners.3 - 1, colour_diffs[0], false, false, 1);
                    grid.draw_line_col_coords(corners.0, s.ocol, corners.2 - 1, s.ocol, colour_diffs[0], false, false, 1);
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 182, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }
            let border = ex.input.shapes.shapes.iter()
                .map(|s| s.cells.rows.max(s.cells.columns))
                .max().unwrap() * 2;
            let mut grid = ex.input.grid.add_border(border);

            for s in ex.input.shapes.shapes.iter() {
                let mut tlr = s.orow + border + s.cells.rows;
                let mut tlc = s.ocol + border - s.cells.columns;

                //while tlr < grid.cells.rows && tlc >= s.cells.columns {
                while tlr < grid.cells.rows && tlc >= s.cells.columns {
                    grid.copy_shape_to_grid_position_mut(s, tlr, tlc);

                    tlr += s.cells.rows;
                    tlc -= s.cells.columns;
                }

                let mut brr = s.orow + s.cells.rows + border;
                let mut brc = s.ocol + s.cells.columns + border;

                while brr < grid.cells.rows && brc < grid.cells.columns {
                    grid.copy_shape_to_grid_position_mut(s, brr, brc);

                    brr += s.cells.rows;
                    brc += s.cells.columns;
                }
            }

            grid = grid.remove_border(border);
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 183, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();
            let mut pixels: Vec<Shape> = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                if s.is_pixel() && s.on_edge(&grid) {
                    pixels.push(s.clone());
                }
            }

            if pixels.len() % 2 != 0 {
                return Grid::trivial();
            }

            for s in pixels.iter() {
                if s.orow == 0 || s.orow == grid.cells.rows - 1 {
                    continue;
                }
                    
                let os = s.find_same_row(&pixels);

                if os != Shape::trivial() {
//println!("row {}/{} -> {}/{}", s.orow, s.ocol, os.orow, os.ocol);
                    grid.draw_line_row(s, &os, s.colour, true, true);
                }
            }
            for s in pixels.iter() {
                if s.ocol == 0 || s.ocol == grid.cells.columns - 1 {
                    continue;
                }
                    
                let os = s.find_same_col(&pixels);

                if os != Shape::trivial() {
                    grid.draw_line_col(s, &os, s.colour, true, true);
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 184, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Double;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        //-if let Some(rule) = run_experiment_examples(&file, 1110, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_double(&exs), output) { return Some(rule); };

        let func = |gi: &Example| {
            /*
            if gi.input.shapes.shapes.len() != 1 {
                return Grid::trivial();
            }
            */

            let s = &gi.input.grid.as_shape();
            let s = s.toddle_colour(Black, s.colour);
            let rows = s.cells.rows;
            let cols = s.cells.columns;
            let mut shapes = Shapes::new_sized(rows * 2, cols * 2);

            shapes.shapes.push(s.clone());
            shapes.shapes.push(s.translate_absolute(rows, 0));
            shapes.shapes.push(s.translate_absolute(0, cols));
            shapes.shapes.push(s.translate_absolute(rows, cols));

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 190, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        /*
        // TODO Important 10fcaaa3
        let common_colour = examples.find_output_colours();

        let func = |ex: &Example| {
            if common_colour.is_empty() {
                return Grid::trivial();
            }

            let s = &ex.input.grid.as_shape();
            let rows = s.cells.rows;
            let cols = s.cells.columns;
            let mut shapes = Shapes::new_sized(rows * 2, cols * 2);

            shapes.shapes.push(s.clone());
            shapes.shapes.push(s.translate_absolute(rows, 0));
            shapes.shapes.push(s.translate_absolute(0, cols));
            shapes.shapes.push(s.translate_absolute(rows, cols));

            let mut copy = shapes.clone();
            copy.shapes = Vec::new();

            for s in shapes.to_grid().to_shapes_sq().shapes.iter() {
                let enclose = s.new9(true, common_colour[0]);

                copy.shapes.push(enclose);
            }
            for s in shapes.to_grid().to_shapes_sq().shapes.iter() {
                copy.shapes.push(s.clone());
            }
//ex.output.grid.show();
//ex.output.grid.diff(&copy.trim_to_grid()).unwrap().show_full();
//copy.trim_to_grid().show();

            copy.trim_to_grid()
        };

        if let Some(rule) = run_experiment(file, 191, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        */

    let colours = examples.io_colour_diff();

    let func = |ex: &Example| {
        if colours.len() != 1 {
            return Grid::trivial();
        }

        let grid = ex.input.grid.dup_right().dup_down();
        let shapes = grid.as_pixel_shapes();

        let mut shapes = Shapes::new_given(grid.cells.rows, grid.cells.columns, &shapes.shapes);

        shapes = shapes.embellish(colours[0]);

        for s in grid.to_shapes_sq().shapes.iter() {
            shapes.shapes.push(s.clone());
        }
        
        shapes.to_grid_transparent()
    };

    if let Some(rule) = run_experiment(task, 192, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = InLessThanOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        let (in_rs, in_cs) = examples.examples[0].input.grid.dimensions();
        let (out_rs, out_cs) = examples.examples[0].output.grid.dimensions();
        let rs = out_rs / in_rs;
        let cs = out_cs / in_cs;

        let func = |ex: &Example| {
            // TODO: Derive the predicate and function before hand
            ex.input.grid.as_shape().checker(rs, cs, &|r,_| r == ex.input.grid.cells.rows, &|s| s.mirrored_c(), false).to_grid()
        };

        if let Some(rule) = run_experiment(task, 600, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            // TODO: Derive the predicate and function before hand
            let shape = ex.input.grid.as_shape();

            shape.checker(rs, cs, &|r,c| shape.cells[(r / rs,c / cs)].colour != Black, &|s| s.invert_colour(), true).to_grid()
        };

        if let Some(rule) = run_experiment(task, 601, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let shape = ex.input.grid.as_shape();

            let has_band = |r,c| {
                match ex.input.shapes.has_band() {
                    (Down, pos) => r == pos * rs,
                    (Right, pos) => c == pos * cs,
                    _ => false,
                }
            };

            shape.checker(rs, cs, &has_band, &|s| s.clone(), true).to_grid()
        };

        if let Some(rule) = run_experiment(task, 602, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let shapes = examples.examples[0].output.grid.template_shapes(&examples.examples[0].input.grid);

        if !shapes.shapes.is_empty()  {
            if let Some(rule) = run_experiment(task, 603, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.grid.fill_template(&shapes.shapes[0]), output) { return Some(rule); };
        }

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let shape = &ex.input.shapes.shapes[0].to_origin();
            let (rs, cs) = shape.dimensions();
            let mr = rs.max(cs);
            let dim = rs.max(cs) * 2 + rs.min(cs);
            let mut shapes = Shapes::new_sized(dim, dim);

            shapes.shapes.push(shape.to_position(mr, 0));
            shapes.shapes.push(shape.rotate_90_pos(1, 0, mr));
            shapes.shapes.push(shape.rotate_90_pos(2, mr, rs + cs));
            shapes.shapes.push(shape.rotate_90_pos(3, rs + cs, mr));

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 604, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let shape = ex.input.grid.as_shape();
            let (rs, cs) = shape.dimensions();
            let mut grid = Grid::new(rs * rs, cs * cs, Black);
            let max_colour = ex.input.grid.find_max_colour();

            for (or, r) in (0 .. rs * rs).step_by(rs).enumerate() {
                for (oc, c) in (0 .. cs * cs).step_by(cs).enumerate() {
                    if or >= shape.cells.rows || oc >= shape.cells.columns {
                        return Grid::trivial();
                    }
                    if shape.cells[(or,oc)].colour == max_colour {
                        grid.copy_shape_to_grid_position_mut(&shape, r, c);
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 605, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() || !cat.contains(&InLessThanOut) || colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let (lrs, lcs) = ex.input.grid.dimensions();
            let mut shape = ex.input.grid.as_shape();

            for ((r, c), cell) in shape.clone().cells.items() {
                if cell.colour != Black {
                    if r > 0 && c > 0 {
                        shape.cells[(r - 1, c - 1)].colour = colour_diffs[0];
                    } else if r == 0 && c > 0 && lrs <= shape.cells.rows && shape.cells[(lrs - 1, c - 1)].colour == Black {
                        shape.cells[(lrs - 1, c - 1)].colour = colour_diffs[0];
                    } else if r > 0 && c == 0 && lcs <= shape.cells.columns && shape.cells[(r - 1, lcs - 1)].colour == Black {
                        shape.cells[(r - 1, lcs - 1)].colour = colour_diffs[0];
                    }
                }
            }

            shape.checker(rs, cs, &|_,_| true, &|s| s.clone(), false).to_grid()
        };

        if let Some(rule) = run_experiment(task, 606, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let g = &ex.input.grid;

            if ex.input.shapes.shapes.is_empty() || !cat.contains(&InLessThanOut) || out_rs % g.height() != 0 || out_cs % g.width() != 0 {
                return Grid::trivial();
            }

            if g.cells.rows > g.cells.columns {
                g.as_shape().checker(1, rs, &|r,_| r % (g.cells.rows * 2) == 0, &|s| s.mirrored_r(), false).to_grid()
            } else {
                g.as_shape().checker(1, rs, &|_,c| c % (g.cells.columns * 2) == 0, &|s| s.mirrored_c(), false).to_grid()
            }
        };

        if let Some(rule) = run_experiment(task, 608, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let shape = ex.input.grid.as_shape().recolour(ex.input.grid.colour, Black).add_border(ex.input.grid.colour);
            let (in_rs, in_cs) = shape.dimensions();
            let rs = (out_rs + 4) / in_rs;
            let cs = (out_cs + 4) / in_cs;
            let grid = Grid::new(in_rs, in_cs, ex.input.grid.colour);

            grid.as_shape().checker(rs, cs, &|_,_| true, &|_| shape.clone(), false).to_grid().trim(out_rs, out_cs)
        };

        if let Some(rule) = run_experiment(task, 609, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 || colour_common.len() != 2 || !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let colour = colour_diffs[0];
            let grid = &ex.input.grid;

            if grid.free_border(Left) {
                grid.mirror_right_func(&|g| g.invert_colour_new(colour))
            } else if grid.free_border(Right) {
                grid.mirror_left_func(&|g| g.invert_colour_new(colour))
            } else if grid.free_border(Up) {
                grid.mirror_down_func(&|g| g.invert_colour_new(colour))
            } else {
                grid.mirror_up_func(&|g| g.invert_colour_new(colour))
            }
        };

        if let Some(rule) = run_experiment(task, 610, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !ex.input.grid.is_square() || !colour_diffs.is_empty() || colour_common.len() != 1 || !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let (rs, cs) = ex.input.grid.dimensions();
            let grid = ex.input.grid.scale_up(ex.input.grid.height());
            let mut grid = grid.resize_inc(ex.input.grid.height());

            for ((r, c), cell) in ex.input.grid.cells.items() {
                if cell.colour != Black {
                    if r == 0 && c == 0 {
                        grid.copy_to_position_mut(&ex.input.grid, 0, cs);
                        grid.copy_to_position_mut(&ex.input.grid, rs, 0);
                    } else if r == 0 && c == cs - 1 {
                        grid.copy_to_position_mut(&ex.input.grid, rs, grid.cells.columns - cs);
                    } else if r == rs - 1 && c == 0 {
                        grid.copy_to_position_mut(&ex.input.grid, r * rs * 2 - rs, 0);
                    } else {
                        if r == 0 {
                            grid.copy_to_position_mut(&ex.input.grid, 0, c * cs * 2);
                        }
                        if c == 0 {
                            grid.copy_to_position_mut(&ex.input.grid, r * rs * 2, 0);
                        }
                    }

                    if r == rs - 1 {
                        grid.copy_to_position_mut(&ex.input.grid, grid.cells.rows - rs, c * cs + c + (cs - c));
                    }
                    if c == cs - 1 {
                        grid.copy_to_position_mut(&ex.input.grid, r * rs + r + (rs - r), grid.cells.columns - cs);
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 611, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let grid = &ex.input.grid;
            let toddle = grid.toddle_colour(Black, grid.colour);

            toddle.as_shape().checker(rs, cs, &|r,c| toddle.cells[(r / rs,c / cs)].colour != Black, &|s| s.clone(), true).to_grid()
        };

        if let Some(rule) = run_experiment(task, 612, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let grid = &ex.input.grid;
            let cnt = grid.cell_colour_cnt_map().len();

            grid.as_shape().checker(cnt, cnt, &|_,_| true, &|s| s.clone(), true).to_grid()
        };

        if let Some(rule) = run_experiment(task, 613, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let colour = examples.examples[0].input.grid.max_colour();
        let colour2 = examples.examples[1].input.grid.max_colour();
        
        // testing function
        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || colour != colour2 {
                return Grid::trivial();
            }

            let grid = &ex.input.grid;
            let (rs, cs) = grid.dimensions();
            let grid = grid.as_shape().checker(rs, cs, &|r,c| grid.cells[(r/ rs,c / cs)].colour == colour, &|s| s.clone(), true).to_grid();

//grid.show();
            grid.clone()
        };

        if let Some(rule) = run_experiment(task, 614, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let (out_rs, out_cs) = examples.examples[0].output.grid.dimensions();
        let all_colour_diffs = examples.io_all_colour_diff();
        
        // testing function
        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let shapes = ex.input.grid.to_shapes_base_bg(all_colour_diffs[0]);

            let in_shape = shapes.shapes[0].scale_down(2);
            let (in_rs, in_cs) = in_shape.dimensions();

            if shapes.shapes.len() != 2 || out_rs % (in_rs * 2) != 0 || out_cs % (in_cs * 2) != 0 {
                return Grid::trivial();
            }

            let rs = out_rs / in_rs;
            let cs = out_cs / in_cs;

            let grid = in_shape.checker(rs, cs, &|r,c| shapes.shapes[1].cells[(r / (rs / 2), c / (cs / 2))].colour != Black, &|s| s.clone(), true).to_grid();

//grid.show();
            grid.clone()
        };

        if let Some(rule) = run_experiment(task, 615, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || !colour_diffs.is_empty() {
                return Grid::trivial();
            }
            let shape = ex.input.grid.as_shape();

            let grid = shape.checker(cs, rs, &|_,c| c % (in_cs * 2) != 0, &|s| s.mirrored_c(), false).to_grid();

//grid.show();
            grid.clone()
        };

        if let Some(rule) = run_experiment(task, 616, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let (in_rs, in_cs) = ex.input.grid.dimensions();
            let dim = in_rs.max(in_cs);
            let mut grid = Grid::new(dim, dim, Black);
            let mut c_pos = 0;
            let mut colour = ex.input.grid.colour;

            for c in 0 .. dim {
                if ex.input.grid.cells[(0,c)].colour != Black {
                    colour = ex.input.grid.cells[(0,c)].colour;
                    c_pos = c;
                    break;
                }
            }

            grid.cells[(0,c_pos)].colour = colour;

            let mut left = c_pos;
            let mut right = c_pos;
            let mut pos = 0;

            for r in 1 .. dim {
                if left > 0 {
                    left -= 1;
                    grid.cells[(r,left)].colour = colour;
                }
                if right < grid.cells.rows - 1 {
                    right += 1;
                    grid.cells[(r,right)].colour = colour;
                }

                if r > 0 && r % 2 == 0 && grid.cells[(r,left)].colour == colour {
                    grid.draw_bg_mut(DownRight, r, left, colour_diffs[0], Black);
                    if r + 1 >= left {
                        pos = r + 1 - left;
                    }
                }

                if left == 0 {
                    break;
                }
            }

            // Not entirely happy with this!
            for r in (pos + 3 .. dim).step_by(4) {
                grid.draw_bg_mut(DownRight, r, 0, colour_diffs[0], Black);
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 617, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let shape = ex.input.grid.as_shape();
            let action = &|s: &Shape, r: usize, c: usize| {
                let r_edge = r == 0 || r == s.cells.rows * rs - rs;
                let c_edge = c == 0 || c == s.cells.columns * cs - cs;

                if r_edge && c_edge {
                    s.rotated_180()
                } else if r_edge && !c_edge {
                    s.mirrored_r()
                } else if !r_edge && c_edge {
                    s.mirrored_c()
                } else {
                    s.clone()
                }
            };

            shape.combined_checker(rs, cs, &action).to_grid()
        };

        if let Some(rule) = run_experiment(task, 618, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        /* Does not work yet c92b942c
        let rule = examples.derive_missing_rule();

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || colour_diffs.len() != 2 {
                return Grid::trivial();
            }

            let grid = ex.input.grid.apply_missing_rule(&rule);
//rule.show_full();
//grid.show_full();
            let grid = grid.as_shape().checker(rs, cs, &|_,_| true, &|s| s.clone(), false).to_grid();
grid.show();

            grid
        };

        if let Some(rule) = run_experiment(file, 819, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        */

        if let Some(rule) = run_experiment(task, 820, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.grid.extend_border(), output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.scale_up(2);
            let shapes = grid.to_shapes();

            for s in shapes.shapes.iter() {
                let mr = s.cells.rows - 1;
                let mc = s.cells.columns - 1;

                grid.draw_mut(UpLeft, s.cells[(0,0)].row, s.cells[(0,0)].col, colour_diffs[0]);
                grid.draw_mut(DownRight, s.cells[(mr,mc)].row, s.cells[(mr,mc)].col, colour_diffs[0]);
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 821, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let rows = ex.input.grid.cells.rows;
            let cols = ex.input.grid.cells.columns;
            let mut grid = ex.input.grid.as_shape().checker(rs, cs, &|r,c| r == 0 && c == 0 || r == rows && c == cols , &|s| s.clone(), true).to_grid();
            let shapes = grid.to_shapes();

            for s in shapes.shapes.iter().skip(1).step_by(2) {
                grid.draw_mut(Right, s.orow - 1, 0, colour_diffs[0]);
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 822, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 823, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.grid.scale_up(ex.input.grid.height()), output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = SingleShapeOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if all || cat.contains(&SingleShapeIn) {
            //if let Some(rule) = run_experiment_examples(&file, 1120, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return Some(rule); };

            //if let Some(rule) = run_experiment_examples(&file, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_move_same_colour(&exs), output) { return Some(rule); };

            let func = |gi: &Grid, go: &Grid, n: &mut usize| {
                let grid = gi.recolour(gi.colour, go.colour);

                move_only(&grid, n)
            };

            if let Some(rule) = run_experiment_tries(task, 310, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let func = |gi: &Example| {
                if gi.input.coloured_shapes.shapes.is_empty() {
                    return Grid::trivial();
                }

                gi.input.coloured_shapes.shapes[0].to_origin().to_grid()
            };

            if let Some(rule) = run_experiment(task, 200, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        if all || !cat.contains(&SingleShapeIn) {
            let r = examples.examples[0].output.grid.cells.rows;
            let c = examples.examples[0].output.grid.cells.columns;

            if r == 1 || c == 1 {
                let colour = examples.examples[0].output.grid.colour;

                let func = |ex: &Example| {
                    let mut i = 0;
                    let mut grid = Grid::new(r, c, Black);

                    for s in ex.input.shapes.shapes.iter() {
                        if i >= c {
                            return Grid::trivial();
                        }
                        if s.colour == colour && s.size() > 1 {
                            if r == 1 {
                                grid.cells[(0, i)].colour = colour;
                            } else {
                                grid.cells[(i, 0)].colour = colour;
                            }
                            i += 1;
                        }
                    }

                    grid
                };

                if let Some(rule) = run_experiment(task, 400, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = OutLessThanIn;
    if all || cat.contains(&gc) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 201, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_unique().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 205, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.first().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 206, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.last().to_grid(), output) { return Some(rule); };

        let func = |ex: &Example| {
            for s in &ex.input.shapes.shapes {
                if !s.is_mirror_r() && !s.is_mirror_c() {
                    return s.to_grid();
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 207, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            for s in &ex.input.shapes.shapes {
                if s.is_mirror_r() || s.is_mirror_c() {
                    return s.to_grid();
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 208, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let bordered = ex.input.shapes.border_only();

            if bordered == Shape::trivial() {
                return Grid::trivial();
            }

            ex.input.grid.subgrid(bordered.orow + 1, bordered.cells.rows - 2, bordered.ocol + 1, bordered.cells.columns - 2)
        };

        if let Some(rule) = run_experiment(task, 209, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = SingleColouredShapeOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        //if let Some(rule) = run_experiment_examples(&file, 1130, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return Some(rule); };

        //if let Some(rule) = run_experiment_examples(&file, 1140, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return Some(rule); };

        let edge_colour = examples.largest_shape_colour();

        let func = |ex: &Example| {
            let h = ex.input.grid.cell_colour_cnt_map();

            if let Some(colour) = h.iter().min_by(|(_, cnt1), (_, cnt2)| cnt1.cmp(cnt2)).map(|(k, _)| k) {
                let h = ex.input.shapes.shape_colour_cnt_map();

                if let Some(vc) = h.get(colour) {
                    if vc.len() != 1 || vc[0].size() != 1 || vc[0].orow == 0 || vc[0].ocol == 0 || vc[0].orow >= ex.input.grid.cells.rows - 1 || vc[0].ocol >= ex.input.grid.cells.columns - 1 {
                        return Grid::trivial();
                    }
//println!("{:?}", vc[0]);

                    let enclose = vc[0].surround(1, edge_colour, true, true);
//enclose.show();
                    let mut ss = ex.input.shapes.clone();

                    ss.shapes = vec![enclose.clone(), vc[0].clone()];

                    return ss.to_grid();
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 204, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.coloured_shapes.is_empty() {
                return Grid::trivial();
            }

            let shape = if ex.input.coloured_shapes.len() == 1 {
                ex.input.coloured_shapes.shapes[0].clone()
            } else {
                ex.input.coloured_shapes.to_shape()
            };

            let mut shapes = ex.input.coloured_shapes.clone();

            shapes.shapes[0] = shape.make_symmetric();

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 205, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if cat.contains(&InLessCountOut) && cat.contains(&InOutSameSize) {
            let colour = examples.examples[0].output.grid.min_colour();

            let func = |ex: &Example| {
                let consolidated = ex.input.grid.as_shape();
                let mut shapes = Shapes::new_from_shape(&consolidated);
                let consolidated = ex.input.shapes.to_shape();
                let s = consolidated.fill_lines(colour);

                shapes.shapes.push(s);

                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 206, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        if colour_diffs.len() == 1  {
            let (r, c) = examples.examples[0].output.grid.find_pixel(colour_diffs[0]);

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.is_empty() || colour_diffs.len() != 1  || ex.input.grid.cells.columns < ex.input.shapes.shapes.len() {
                    return Grid::trivial();
                }

                let cs = ex.input.grid.cells.columns - ex.input.shapes.shapes.len();
                let mut shapes = Shapes::new_sized(cs + 1, cs);
                let mut row = r;
                let mut col = c;

                for s in ex.input.shapes.shapes.iter() {
                    if s.width() == 2 {
                        if s.cells[(0,0)].colour != Black {
                            let ns = s.to_position(row, col);

                            shapes.shapes.push(ns);

                            row += 1;
                            col += 1;
                        } else {
                            let ns = s.to_position(row, col - 1);

                            shapes.shapes.push(ns);

                            row += 1;
                            col -= 1;
                        }
                    } else {
                        let ns = s.to_position(row + 1, col);

                        shapes.shapes.push(ns);

                        row += 2;
                    }
                }

                let ns = Shape::new_sized_coloured_position(r, c, 1, 1, colour_diffs[0]);
                shapes.shapes.push(ns);
                
                shapes.to_grid_transparent()
            };

            if let Some(rule) = run_experiment(task, 207, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = NoShapesIn(1);
    if all || cat.contains(&gc) && cat.contains(&NoColouredShapesOut(1)) && !cat.contains(&NoShapesOut(1)) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if cat.contains(&NoShapesIn(1)) && cat.contains(&NoShapesOut(6)) {
            // Important: all colour scan does not come back in correct order
            let in_colours = examples.examples[0].input.shapes.colours();
            let out_colours = examples.examples[0].output.shapes.colours();
            let colours = Uniq::uniq(&out_colours, in_colours);

            let func = |ex: &Example| {
//ex.output.shapes.show();
                let mut grid = ex.input.grid.clone();
                let fb = ex.input.shapes.shapes[0].find_first_blacks();
//println!("{colours:?} {fb:?}");
                let mut j = 0;

                for (i, (r, c)) in fb.iter().enumerate() {
                    // TODO better
                    if i != 0 && i != 2 && i != 6 && i != 8 {
                        grid = grid.flood_fill(*r, *c, NoColour, colours[j]);
                        j += 1;
                    }
                }
//grid.show();

                grid
            };

            if let Some(rule) = run_experiment(task, 610, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        //let black = examples.examples[0].input.grid.to_shapes_base_bg(Red);
        let min_colour = examples.examples[0].output.grid.find_min_colour();
        let max_colour = examples.examples[0].output.grid.find_max_colour();

        let func = |ex: &Example| {
            let grid = ex.input.grid.recolour(Black, NoColour);
//grid.show();
            let mut shapes = grid.to_shapes();
            let mut min_size = usize::MAX;
            let mut max_size = 0;

            for s in shapes.shapes.iter_mut() {
                if s.size() != grid.size() {
                    s.recolour_mut(NoColour, Black);
                    if s.size() > max_size {
                        max_size = s.size();
                    }
                    if s.size() < min_size {
                        min_size = s.size();
                    }
                }
            }

            for s in shapes.shapes.iter_mut() {
                if s.size() == max_size {
                    s.recolour_mut(NoColour, max_colour);
                }
                if s.size() == min_size {
                    s.recolour_mut(NoColour, min_colour);
                }
            }
//shapes.show_summary();
//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 621, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if cat.contains(&NoShapesIn(1)) && cat.contains(&NoShapesOut(2)) {
            let extra_colour = examples.examples[0].output.grid.find_min_colour();

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.is_empty() || !ex.input.grid.is_square() {
                    return Grid::trivial();
                }
                let colour = ex.input.grid.colour;
                let mut shapes = ex.input.shapes.clone();
                let shape = &ex.input.shapes.shapes[0];
                let shape = shape.to_square();
                let mut shape = shape.diff(&shape.rotated_90()).unwrap();

                shape.recolour_mut(ToBlack + colour, extra_colour);
                shape.uncolour_mut();

                shapes.shapes.push(shape);
//shapes.to_grid().show();
                
                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 622, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        if cat.contains(&NoShapesIn(1)) && cat.contains(&NoColouredShapesOut(1)) {
            let extra_colour = examples.examples[0].output.grid.find_min_colour();

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.is_empty() {
                    return Grid::trivial();
                }
                let mut shapes = ex.input.shapes.clone();

                shapes.shapes[0].recolour_mut(Black, extra_colour);
//shapes.to_grid().show();
                
                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 623, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.is_empty() && ex.input.shapes.shapes[0].cells.columns % 2 != 1 {
                    return Grid::trivial();
                }

                let mut shape = ex.input.shapes.shapes[0].clone();
                let mid_r = shape.cells.rows / 2;
                let cols: Vec<usize> = (0 .. shape.cells.columns).filter(|&c| shape.cells[(mid_r, c)].colour == Black).collect();

                if cols.is_empty() {
                    return Grid::trivial();
                }

                let mut down = true;

                for (i, &c) in cols.iter().enumerate() {
                    if i % 3 == 0 {
                        shape.flood_fill_mut(mid_r, c, NoColour, extra_colour);
                        down = !down;
                    }
                }

                if cols.len() % 3 == 0 {
                    let r = if down { mid_r + 1 } else { mid_r - 1 };
                    let c = shape.cells.columns - 1;

                    shape.flood_fill_mut(r, c, NoColour, extra_colour);
                }

                shape.to_grid()
            };

            if let Some(rule) = run_experiment(task, 624, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let colours = examples.examples[0].output.grid.colours();
            let in_colour = examples.examples[0].input.grid.colour;

/*
            if colours.len() == 4  {
                // 626c0bcc eval
                let func = |ex: &Example| {
println!("{colours:?} {in_colour:?} {:?}", ex.output.shapes.len());
println!("{:?}", ex.output.shapes.shapes[1]);
println!("{:?}", ex.input.shapes.shapes[0]);
println!("{:?}", ex.output.coloured_shapes.shapes[0]);
                    Grid::trivial()
                };

                if let Some(rule) = run_experiment(file, 635, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }
*/

            let fill_colour = colours.iter().filter(|(&k,_)| k != in_colour).map(|(k,v)| (v,k)).min().map(|(_,k)| k).unwrap();

            if colours.len() == 3  {
                let bg_colour = colours.iter().filter(|(&k,_)| k != in_colour).map(|(k,v)| (v,k)).max().map(|(_,k)| k).unwrap();

                let func = |ex: &Example| {
                    if ex.input.shapes.shapes.is_empty() && ex.input.shapes.shapes[0].cells.columns % 2 != 1 {
                        return Grid::trivial();
                    }

                    let mut shape = ex.input.grid.as_shape();
                    let rows = shape.cells.rows;
                    let cols = shape.cells.columns;
//shape.show();

                    for r in 0 .. rows {
                        for c in 0 .. cols {
                            if (r == 0 || c == 0 || r == rows - 1 || c == cols - 1) && shape.cells[(r, c)].colour == Black {
                                shape.flood_fill_mut(r, c, NoColour, *bg_colour);
                            }
                        }
                    }

                    shape.recolour_mut(Black, *fill_colour);

//shape.show();
                    shape.to_grid()
                };

                if let Some(rule) = run_experiment(task, 625, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }

            if colours.len() == 2 {
                let func = |ex: &Example| {
                    if ex.input.shapes.shapes.is_empty() && ex.input.shapes.shapes[0].cells.columns % 2 != 1 || !ex.input.shapes.shapes[0].hollow() {
                        return Grid::trivial();
                    }

                    let mut shapes = ex.input.shapes.clone();
//shapes.to_grid().show();

                    let mut rects = ex.input.grid.find_rectangles();
//rects.show();

                    for s in rects.shapes.iter_mut() {
                        s.recolour_mut(Black, *fill_colour);
                    }
//rects.show();

                    shapes.merge_mut(&rects);

//ex.output.grid.diff(&shapes.to_grid()).unwrap().show_full();
//ex.output.grid.show();
//shapes.to_grid().show();

                    shapes.to_grid()
                };

                if let Some(rule) = run_experiment(task, 626, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.is_empty() {
                    return Grid::trivial();
                }

                let mut grid = ex.input.grid.clone();

                let mut nr = 1;
                for r in 1 .. grid.cells.rows {
                    if grid.cells[(r,0)].colour != grid.cells[(0,0)].colour {
                        nr += 1;
                    } else {
                        break;
                    }
                }
                let mut nc = 1;
                for c in 1 .. grid.cells.columns {
                    if grid.cells[(0,c)].colour != grid.cells[(0,0)].colour {
                        nc += 1;
                    } else {
                        break;
                    }
                }
                let mut tile = grid.subgrid(0, nr, 0, nc);

                tile = tile.roll_right();
                grid.tile_mut(&tile);
//grid.show();

                grid
            };

            if let Some(rule) = run_experiment(task, 627, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            // a65b410d test and fails 5207a7b5 eval
            for inc in 1 ..= 1 {    // TODO Fix inc for > 1
                let func = |ex: &Example| {
                    if ex.input.shapes.shapes.len() != 1 || ex.input.shapes.shapes[0].size() >= ex.input.grid.size() || ex.input.shapes.shapes[0].is_pixel() || ex.input.shapes.shapes[0].ocol > 0 || !ex.input.shapes.is_line() {
                        return Grid::trivial();
                    }

                    let top_colour = colours.iter().filter(|(&k,_)| k != in_colour).map(|(k,v)| (v,k)).max().map(|(_,k)| k).unwrap();
                    let bottom_colour = colours.iter().filter(|(&k,_)| k != in_colour).map(|(k,v)| (v,k)).min().map(|(_,k)| k).unwrap();

                    let mut shapes = ex.input.shapes.clone();
                    let shape = &ex.input.shapes.shapes[0];
                    let rows = shape.cells.rows;
                    let cols = shape.cells.columns;
                    let bound = if inc == 1 { cols + shape.orow - 1
                    } else {
                        cols + shape.orow + inc
                    };

                    for (i, c) in (0 ..= bound).rev().step_by(inc).enumerate() {
                        if i != shape.orow {
                            let colour = if i < shape.orow {
                                top_colour
                            } else {
                                bottom_colour
                            };
                            let s = Shape::new_sized_coloured_position(i, 0, rows, c + 1, *colour);
//println!("{i} {} {}", c + 1, c);

                            shapes.shapes.push(s);
                        }
                    }
//ex.output.grid.show();
//println!("####");
//shapes.to_grid().show();
//shapes.to_grid().inverse_transform(trans).show();

                    shapes.to_grid()
                };

                if let Some(rule) = run_experiment(task, 628, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.len() != 1 {
                    return Grid::trivial();
                }

                let shape = &ex.input.shapes.shapes[0];
                let (dir, r, c) = shape.has_border_break();
                let mut grid = ex.input.grid.clone();

                grid.draw_mut(dir, r, c, *fill_colour);
                grid.flood_fill_mut(r, c, NoColour, *fill_colour);

//grid.show();
                grid
            };

            if let Some(rule) = run_experiment(task, 630, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.len() != 1 {
                    return Grid::trivial();
                }

                let shape = &ex.input.shapes.shapes[0];
                let mut grid = ex.input.grid.clone();
                let dist = grid.cells.columns - 1;

                // TODO: This should be 4 functions on 4 rectangles
                let mut rfrom = shape.orow;
                let mut cfrom = shape.ocol;
                let mut odd = true;

                while rfrom > 0 {
                    let dir = if odd { UpRight } else { UpLeft };

                    grid.draw_mut(dir, rfrom, cfrom, in_colour);

                    rfrom = rfrom.saturating_sub(dist);
                    cfrom = if cfrom == 0 { dist } else { 0 };
                    odd = !odd;
                }

                grid = grid.recolour(Black, *fill_colour);

//grid.show();
                grid
            };

            if let Some(rule) = run_experiment(task, 631, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }
//eprintln!("{file}");

        /*
        eprintln!("{file} yes 4");
        examples.examples[0].input.shapes.show();
        examples.examples[0].output.shapes.show();
        examples.examples[0].output.coloured_shapes.show();
        //examples.examples[0].input.shapes.shapes[0].diff(&examples.examples[0].output.coloured_shapes.shapes[0]).unwrap().show_full();
        examples.examples[1].input.shapes.show();
        examples.examples[1].output.shapes.show();
        examples.examples[1].output.coloured_shapes.show();
        //eprintln!("{:?}", examples.examples[0].input.shapes.shapes);
        */

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    /*
if cat.contains(&NoShapesOut(5)) && cat.contains(&ShapeMaxCntOut(4)) {
eprintln!("{file} yes 3");
}
let no = 1;
if cat.contains(&NoShapesIn(no)) && cat.contains(&NoShapesOut(no + 2)) {
eprintln!("yes");
}
if (cat.contains(&NoShapesIn(no)) || cat.contains(&NoShapesIn(no + no * 4))) && cat.contains(&NoShapesOut(no + no * 4)) {
eprintln!("yes 2");
}
if cat.contains(&InOutSameShapesColoured) && cat.contains(&InOutSameSize) && cat.contains(&ShapeMinCntIn(1)) && cat.contains(&ShapeMinCntOut(1)) {
let consolidated = examples.examples[0].input.shapes.consolidate_shapes();
consolidated.show();
eprintln!("yes 3");
}
    */
    let gc = NoShapesIn(2);
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() != 2 || colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let mut shapes = ex.input.shapes.clone();

            if shapes.shapes[0].orow + shapes.shapes[0].cells.rows + 1 == shapes.shapes[1].orow {
                if shapes.shapes[1].orow == 0 {
                    return Grid::trivial();
                }
                let s = Shape::new_sized_coloured_position(shapes.shapes[1].orow - 1, 0, 1, ex.input.grid.cells.columns, colour_diffs[0]);

                shapes.shapes.push(s);
            } else {
                if shapes.shapes[1].ocol == 0 {
                    return Grid::trivial();
                }
                let s = Shape::new_sized_coloured_position(0, shapes.shapes[1].ocol - 1, ex.input.grid.cells.rows, 1, colour_diffs[0]);

                shapes.shapes.push(s);
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 632, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = NoColouredShapesIn(2);
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        let func = |ex: &Example| {
           let idx = ex.input.coloured_shapes.smallest();
           let shape = ex.input.coloured_shapes.largest();

           if idx.width() == 0 {
               return Grid::trivial();
           }

           let mut big = idx.rotated_90().scale_up(shape.width() / idx.width());

           big.to_position_mut(shape.orow, shape.ocol);

           let mut shapes = ex.input.coloured_shapes.clone();

           
           shapes.shapes.push(big);
//shapes.to_grid().show();

           shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 633, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }

    let gc = NoColouredShapesIn(9);
    if all || cat.contains(&gc) && cat.contains(&NoColouredShapesOut(9)){ 
        *cap_cats.entry(gc).or_insert(0) += 1;

        let colours: Vec<_> = examples.examples[0].input.grid.find_colours()
            .keys()
            .copied()
            .collect();

        let order0 = [(0, 0), (0, 4), (0, 8), (4, 0), (4, 4), (4, 8), (8, 0), (8, 4), (8, 8)];
        let order1 = [(0, 8), (0, 4), (0, 0), (4, 8), (4, 4), (4, 0), (8, 8), (8, 4), (8, 0)];
        let order2 = [(8, 0), (8, 4), (8, 8), (4, 0), (4, 4), (4, 8), (0, 0), (0, 4), (0, 8)];
        let order3 = [(8, 8), (8, 4), (8, 0), (4, 8), (4, 4), (4, 0), (0, 8), (0, 4), (0, 0)];

        for colour in colours.into_iter() {
            for order in [order0, order1, order2, order3] {
                // testing function
                let func = |ex: &Example| {
                    if ex.input.shapes.shapes.is_empty() {
                        return Grid::trivial();
                    }
                    let mut shapes = ex.input.grid.to_shapes_base_bg(Black);

                    shapes.shapes.sort_by_key(|b| std::cmp::Reverse(b.cell_count_colour(colour)));

                    let mut grid = Grid::new(ex.input.grid.cells.rows, ex.input.grid.cells.columns, Black);

                    for ((r, c), s) in order.iter().zip(shapes.shapes.iter()) {
                        grid.copy_shape_to_grid_position_mut(s, *r, *c);
                    }

                    grid
                };

                if let Some(rule) = run_experiment(task, 634, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = OutLessThanIn;
    if all || cat.contains(&gc) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        if all || cat.contains(&SinglePixelOut) {
            let (rs, cs) = examples.examples[0].output.grid.dimensions();

            let func = |gi: &Example| {
                if gi.input.shapes.shapes.is_empty() {
                    return Grid::trivial();
                }

                let h = gi.input.shapes.shape_colour_cnt_map();
                let pair: Option<Vec<Shape>> = h.clone().into_values().filter(|p| p.len() == 2 && p[0].size() == p[1].size()).last();
                if let Some(pair) = pair {
                    // TODO fix coloured shape when mixed
                    for s in gi.input.coloured_shapes.shapes.iter() {
                        if s.is_contained(&pair[0]) && s.is_contained(&pair[1]) {
                            // find joining colour
                            let colour = h.keys().filter(|&&c| c != pair[0].colour && c != pair[1].colour).collect::<Vec<_>>();

                            return Grid::new(rs, cs, *colour[0]);
                        }
                    }
                }

                Grid::new(rs, cs, Black)
            };

            if let Some(rule) = run_experiment(task, 210, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let func = |ex: &Example| {
            let scm = ex.input.shapes.shape_colour_cnt_map();

            for v in scm.values() {
                if v.len() == 1 {
                    return v[0].to_grid();
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 222, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        //if let Some(rule) = run_experiment(file, 230, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_max().to_grid(), output) { return Some(rule); };
        if let Some(rule) = run_experiment(task, 230, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.shapes.find_max().to_grid(), output) { return Some(rule); };

        let colour = examples.examples[0].output.grid.colour;

        // Use when divider 
        let func = |gi: &Grid, _: &Grid, n: &mut usize| {
            let centre = gi.centre_of();
            let shapes = gi.to_shapes_base_bg(gi.cells[centre].colour);
            if shapes.is_empty() || shapes.shapes.len() != 2 {
                return Grid::trivial();
            }
            let diff = shapes.shapes[0].diff(&shapes.shapes[1]);

            if let Some(diff) = diff {
                diff_only(&diff.to_grid(), colour, n)
            } else {
                Grid::trivial()
            }
        };

        if let Some(rule) = run_experiment_tries(task, 320, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let shapes = ex.input.grid.split_2();
            if shapes.len() != 2 || !ex.cat.contains(&OutRInWidth(2)) || colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let g1 = shapes.shapes[0].to_grid();
            let g2 = shapes.shapes[1].to_grid();
            let grid = g1.diff_only_not(colour_diffs[0]).diff(&g2.diff_only_not(colour_diffs[0]));

            match grid {
                Some(grid) => grid.diff_only_same(),
                None => Grid::trivial(),
            }
        };

        if let Some(rule) = run_experiment(task, 321, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // Use when divider - ops may be asymetric TODO: unduplicate
        let func = |gi: &Grid, _: &Grid, n: &mut usize| {
            let centre = gi.centre_of();
            let shapes = gi.to_shapes_base_bg(gi.cells[centre].colour);
            if shapes.is_empty() || shapes.shapes.len() != 2 {
                return Grid::trivial();
            }
            let diff = shapes.shapes[1].diff(&shapes.shapes[0]);

            if let Some(diff) = diff {
                diff_only(&diff.to_grid(), colour, n)
            } else {
                Grid::trivial()
            }
        };

        if let Some(rule) = run_experiment_tries(task, 330, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // No divider
        let func = |gi: &Grid, _: &Grid, n: &mut usize| {
            let shapes = gi.split_2();
//shapes.show();
            if shapes.is_empty() || shapes.shapes.len() != 2 { return Grid::trivial(); }
            let diff = shapes.shapes[0].diff(&shapes.shapes[1]);

            if let Some(diff) = diff {
                diff_only(&diff.to_grid(), colour, n)
            } else {
                Grid::trivial()
            }
        };

        if let Some(rule) = run_experiment_tries(task, 340, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let colours = examples.examples[0].input.grid.cell_colour_cnt_map();
        let side = examples.examples[0].output.grid.cells.rows;
        let colour = examples.examples[0].output.grid.colour;

        for (&in_colour, _) in colours.iter() {
            let func = |ex: &Example| {
                let mut m = 0;
                let mut size = 0;

                for s in &ex.input.shapes.shapes {
                    if s.colour == in_colour && s.size() > 1 {
                        if size > 1 && size != s.size() {
                            return Grid::trivial();
                        } 
                        size = s.size();

                        m += 1;
                    }
                }

                Grid::colour_every_nxn_for_m(colour, side, size, m)
            };

            if let Some(rule) = run_experiment(task, 240, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let (rs, cs) = examples.examples[0].output.grid.dimensions();

        let func = |ex: &Example| {
            let mut colour = NoColour;

            for s in &ex.input.shapes.shapes {
                if !s.is_full() {
                    colour = s.colour;

                    break;
                }
            }

            Grid::new(rs, cs, colour)
        };

        if let Some(rule) = run_experiment(task, 241, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // 3d31c5b3 281123b4
        for order in permutations(&[0,1,2,3]) {
            let func = |gi: &Example| {
                let shapes = gi.input.grid.split_4_inline(true);
                if shapes.is_empty() || shapes.shapes.len() != 4 {
                    return Grid::trivial();
                }

                let mut diff = Some(shapes.shapes[*order[0]].clone());

                for i in &order {
                    if let Some(new_diff) = diff {
                        let clear = new_diff.diff_only_transparent();

                        diff = clear.diff(&shapes.shapes[**i]);
                    } else {
                        return Grid::trivial();
                    }
                }
                if let Some(new_diff) = diff {
                    new_diff.diff_only_transparent().to_grid()
                } else {
                    Grid::trivial()
                }
            };

            if let Some(rule) = run_experiment(task, 250, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = Div9Out;
    if all || cat.contains(&gc) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        //if let Some(rule) = run_experiment_examples(&file, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_div9(&exs), output) { return Some(rule); };

        let colour_map = examples.find_colour_io_map();

        let func = |ex: &Example| {
                let mut shapes = ex.input.shapes.clone();
                
                for s in shapes.shapes.iter_mut() {
                    if let Some(colour) = colour_map.get(&s.colour) {
                        s.recolour_mut(s.colour, *colour);
                    }
                }

                shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 251, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = GravityDown;
    if all || cat.contains(&gc) || cat.contains(&GravityUp) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment_tries(task, 350, experiment, trans, is_test, examples, &targets, done, tries, &|ex, _, n| gravity_only(ex, n), output) { return Some(rule); };

        let func = |ex: &Example| {
                let mut shapes = ex.input.shapes.clone();
                let sccm = shapes.shape_colour_cnt_map();
                let mut fill_colour = NoColour;

                shapes.shapes = Vec::new();
                
                for (col, ss) in sccm.iter() {
                    if ss.len() == 1 {
                        shapes.shapes.push(ss[0].clone())
                    } else {
                        fill_colour = *col;
                    } 
                }

                if shapes.shapes.is_empty() {
                    return Grid::trivial();
                }

                let grid = shapes.to_grid();

                grid.flood_fill(grid.cells.rows - 1, 0, NoColour, fill_colour)
        };

        if let Some(rule) = run_experiment(task, 251, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    /*
    let gc = SingleColouredShapeIn;
    if all || cat.contains(&gc) && cat.contains(&SingleColouredShapeOut) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        // d631b094.json
        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    */
    let gc = SingleColourIn;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        // ff28f65a.json
        if !cat.contains(&SingleColourOut) {
//println!("ffff");
//example.test.input.grid.show();
/*
            let func = |gi: &Grid, _: &Grid, n: &mut usize| {
                let shapes = gi.to_shapes_base_bg(gi.cells[(1,3)].colour);
                if shapes.len() < 2 { return Grid::trivial(); }
                let diff = shapes.shapes[0].diff(&shapes.shapes[1]).unwrap();
                let colour = examples.examples[0].output.grid.colour;

                diff.to_grid().diff_only(colour, *n)
            };

            if let Some(rule) = run_experiment_tries(file, 500, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
*/

            if !examples.examples.is_empty() && !examples.examples[0].input.shapes.shapes.is_empty() {
                // Global colour data for two rules
                let colour = examples.examples[0].input.shapes.shapes[0].colour;
                let mut new_colour = colour;

                for s in examples.examples[0].output.shapes.shapes.iter() {
                    if colour != s.colour {
                        new_colour = s.colour;

                        break;
                    }
                }

                let func = |gi: &Example| {
                    if gi.input.shapes.is_empty() {
                        return Grid::trivial();
                    }

                    let mut si = gi.input.shapes.clone();

                    si.shapes.sort_by(|a, b| (a.ocol, a.orow).cmp(&(b.ocol, b.orow)));
                    let mut other = si.shapes.len() % 2 != 0;

                    for s in si.shapes.iter_mut() {
                        if other {
                            s.mut_force_recolour(new_colour);
                        }

                        other = !other;
                    }

                    si.to_grid()
                };

                if let Some(rule) = run_experiment(task, 270, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

                let func = |gi: &Example| {
                    if gi.input.shapes.is_empty() {
                        return Grid::trivial();
                    }

                    let mut si = gi.input.shapes.clone();

                    for s in si.shapes.iter_mut() {
                        if s.pixels() >= 4 {
                            s.mut_recolour(colour, new_colour);
                        }
                    }

                    si.to_grid()
                };

                if let Some(rule) = run_experiment(task, 281, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }

            let ccm = examples.find_hollow_cnt_colour_map();

            let func = |ex: &Example| {
                if ex.input.shapes.is_empty() {
                    return Grid::trivial();
                }

                let mut shapes = ex.input.shapes.clone();

                for s in shapes.shapes.iter_mut() {
                    let (_, n) = s.hollow_colour_count();
                    
                    if let Some(colour) = ccm.get(&n) {
                        s.recolour_mut(s.colour, *colour);
                    }
                }

//shapes.to_grid().show();
                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 282, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            /* TODO 759f3fd3 translate_absolute_clip
            let sq_colour = if !examples.examples.is_empty() && !examples.examples[0].output.shapes.shapes.is_empty() {
                examples.examples[0].output.shapes.shapes[0].colour
            } else {
                NoColour
            };

            let func = |ex: &Example| {
                if ex.input.shapes.len() != 4 || ex.input.shapes.shapes[0].colour != NoColour {
                    return Grid::trivial();
                }
                let shape = ex.input.grid.as_shape();
                let first = &ex.input.shapes.shapes[0];
                let x = first.cells.rows;
                let y = first.cells.columns;
                let mut shapes = Shapes::new_sized(shape.cells.rows, shape.cells.columns);
                
                for i in (5 .. shape.cells.rows * 2).step_by(2) {
                    if i > shape.cells.rows && i > shape.cells.columns {
                        break;
                    }

                    let s = Shape::new_square(x, y, i, sq_colour);
//s.show_summary();
                    let s = s.translate_absolute_clip(x as isize - i as isize + 3, y as isize - i as isize + 3);
//println!("{:?}", s);
//s.show_summary();
                    shapes.shapes.push(s);
                }

                shapes.shapes.reverse();

                //shapes.shapes.push(shape.clone());
//shapes.show();
shapes.trim_to_grid().show();

                shapes.trim_to_grid()
            };

            if let Some(rule) = run_experiment(file, 280, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            */
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = SingleColourOut;
    if all || cat.contains(&gc) && !cat.contains(&SingleColourIn) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 290, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_pixels_min().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 300, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_pixels_max().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 301, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_max().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 302, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_min().to_grid(), output) { return Some(rule); };

        let out_colours = examples.find_output_colours();

        if out_colours.len() == 1 {
            let in_colours = examples.find_input_colours().uniq(out_colours);
            if in_colours.len() == 2 {
                let func = |ex: &Example| {
                  let c1s = ex.input.grid.find_colour(in_colours[0]);
                  let c2s = ex.input.grid.find_colour(in_colours[1]);
                  let len = c1s.len();
                  if len == c2s.len() {
                      let c_dim = c1s[0].col == c2s[0].col;
                      let mut grid: Grid;

                      if c_dim {
                          let width = c1s[0].row.min(c2s[0].row) - c1s[0].row.min(c2s[0].row) - 1; // assume all same

                          grid = Grid::new(width, len, Black);

                          for (c, (a, b)) in c1s.iter().zip(c2s.iter()).enumerate() {
                              for r in a.row + 1 .. b.row {
                                  grid.cells[(r - a.row - 1, c)].row = ex.input.grid.cells[(r,a.col)].row;
                                  grid.cells[(r - a.row - 1, c)].col = ex.input.grid.cells[(r,a.col)].col;
                                  grid.cells[(r - a.row - 1, c)].colour = ex.input.grid.cells[(r,a.col)].colour;
                              }
                          }
                      } else {
                          let height = c1s[0].col.max(c2s[0].col) - c1s[0].col.min(c2s[0].col) - 1; // assume all same

                          grid = Grid::new(len, height, Black);

                              for (r, (a, b)) in c1s.iter().zip(c2s.iter()).enumerate() {
                              for c in a.col + 1 .. b.col {
                                  grid.cells[(r, c - a.col - 1)].row = ex.input.grid.cells[(a.row,c)].row;
                                  grid.cells[(r, c - a.col - 1)].col = ex.input.grid.cells[(a.row,c)].col;
                                  grid.cells[(r, c - a.col - 1)].colour = ex.input.grid.cells[(a.row,c)].colour;
                              }
                          }
                      };
                      return grid;
                  }

                  Grid::trivial()
                };

                if let Some(rule) = run_experiment(task, 303, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }
        }

        let (rs, cs) = examples.examples[0].output.grid.dimensions();

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let largest = ex.input.shapes.largest();

            Grid::new(rs, cs, largest.colour)
        };

        if let Some(rule) = run_experiment(task, 304, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 

    // Special section for rules that might give false positives
    // These often solve the examples but not the final test(s)
    *cap_cats.entry(CatchAll).or_insert(0) += 1;

    let func = |ex: &Example| {
        let largest = ex.input.coloured_shapes.largest();

        if largest.cells.rows < 3 || largest.cells.columns < 3 {
            return Grid::trivial();
        }

        let shape = largest.subshape(0, 3, 0, 3);
        let mut grid = ex.input.grid.clone();

        for s in ex.input.coloured_shapes.shapes.iter() {
            if s.size() == 1 {
                if s.orow == 0 || s.ocol == 0 {
                    return Grid::trivial();
                }
                let new_shape = shape.translate_absolute(s.orow-1,s.ocol-1);

                grid.copy_shape_to_grid_mut(&new_shape);
            }
        }

        grid
    };

    if let Some(rule) = run_experiment(task, 1000, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    let out_shapes = examples.all_shapes_out();

    let func = |ex: &Example| {
        let mut shapes = ex.input.shapes.clone();

        for s1 in out_shapes.shapes.iter() {
            for s2 in shapes.shapes.iter_mut() {
                if s1.equal_shape(s2) {
                    s2.recolour_mut(s2.colour, s1.colour);
                    break;
                }
            }
        }

//ex.input.grid.show();
//shapes.to_grid().show();
        shapes.to_grid()
    };

    if let Some(rule) = run_experiment(task, 1001, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    let func = |ex: &Example| {
        let mut shapes = ex.input.shapes.clone();

        for s1 in ex.input.shapes.shapes.iter() {
            for s2 in shapes.shapes.iter_mut() {
                if s1.equal_shape(s2) && s1.colour > s2.colour {
                    s2.recolour_mut(s2.colour, s1.colour);

                    break;
                }
            }
        }

//ex.output.grid.show();
//shapes.to_grid().show();
        shapes.to_grid()
    };

    if let Some(rule) = run_experiment(task, 1002, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    // Cross example knowledge needed for closure
    let h = examples.bleached_io_map();

    let func = |ex: &Example| {
        if let Some(grid) = h.get(&ex.input.grid.bleach().to_json()) {
            grid.clone()
        } else {
            Grid::trivial()
        }
    };

    if let Some(rule) = run_experiment(task, 1003, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    if (all || cat.contains(&InOutSameShapesColoured)) && !examples.examples[0].output.shapes.shapes.is_empty() {
        let mut pairs: BTreeMap<Shape, Shape> = BTreeMap::new();

        for exs in examples.examples.iter() {
            let m = exs.output.shapes.contained_pairs();
            pairs.extend(m);
        }

        let mut ssm: BTreeMap<usize, Colour> = BTreeMap::new();

        for (ks, vs) in pairs.iter() {
            ssm.insert(ks.size(), vs.colour);
        }

        let func = &|ex: &Example| {
            let mut shapes = ex.input.shapes.clone();

            for s in shapes.shapes.iter_mut() {
                if let Some(colour) = ssm.get(&s.size()) {
                    let (r, c) = s.centre_of();

                    s.flood_fill_mut(r - s.orow, c - s.ocol, NoColour, *colour);
                }
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1004, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
    }

    let func = |ex: &Example| {
        let shape = ex.input.grid.as_shape().scale_up(2);

        Shapes::new_shapes(&[shape]).to_grid()
    };

    if let Some(rule) = run_experiment(task, 1005, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    let func = &|ex: &Example| {
        let len = ex.input.shapes.shapes.len();

//ex.input.grid.show();
        if len != 4 && len != 9 && len != 25 || !ex.input.shapes.shapes.is_empty() && (!ex.input.shapes.shapes[0].is_square() || ex.input.shapes.shapes[0].size() == 1) {
            return Grid::trivial();
        }

        let mut n = usize::MAX;
        let mut posn = 0;

        for (i, s) in ex.input.shapes.shapes.iter().enumerate() {
            let h = s.cell_colour_cnt_map();
            if n > h.len() {
                n = h.len();
                posn = i;
            }
        }
        let ans = ex.input.shapes.shapes[posn].clone();
        let mut shapes = ex.input.shapes.clone();

        shapes.shapes = Vec::new();

        for (i, s) in ex.input.shapes.shapes.iter().enumerate() {
            if ans.size() != s.size() || ans.colour == NoColour  || i / ans.cells.rows >= ans.cells.rows  {
                return Grid::trivial();
            }

            let colour = ans.cells[(i / ans.cells.rows, i % ans.cells.columns)].colour;
            let ns = s.force_recolour(colour);
            
            shapes.shapes.push(ns);
        }

        let bg = ex.input.grid.cells[(ans.cells.rows,ans.cells.columns)].colour;
//shapes.to_grid_colour(bg).show();

        shapes.to_grid_colour(bg)
    };

    if let Some(rule) = run_experiment(task, 1006, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    let in_colour = examples.examples[0].output.grid.get_diff_colour(&examples.examples[0].input.grid);
    let colour = examples.examples[0].io_colour_diff();

    let func = |ex: &Example| {
        let mut shapes = ex.input.grid.to_shapes_sq();

        for s in shapes.clone().shapes.iter_mut() {
            if !s.is_pixel() {
                shapes.shapes.push(s.recolour(in_colour, colour));
            }
        }

        shapes.to_grid()
    };

    if let Some(rule) = run_experiment(task, 1007, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    let in_grid = &examples.examples[0].input.grid;
    let out_grid = &examples.examples[0].output.grid;
    
    let func = |ex: &Example| {
        if ex.input.shapes.shapes.is_empty() || NxNIn(0).gte_value(&NxNOut(0), &cat) {
            return Grid::trivial();
        }

        let colour_map: BTreeMap<Colour,Colour> = in_grid.cells.values()
            .zip(ex.input.grid.cells.values())
            .map(|(k,v)| (k.colour,v.colour))
            .collect();

        let mut grid = out_grid.clone();

        for ((r, c), cell) in out_grid.cells.items() {
            match colour_map.get(&cell.colour) {
                Some(colour) => grid.cells[(r,c)].colour = *colour,
                None => return Grid::trivial()
            }
        }

        grid
    };

    if let Some(rule) = run_experiment(task, 1008, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    *cap_todo.entry(CatchAll).or_insert(0) += 1;

    None
}

#[allow(clippy::too_many_arguments)]
fn run_experiment(task: &str, experiment: usize, experiment_todo: &str, trans: Transformation, is_test: bool, examples: &Examples, targets: &[Grid], done: &mut BTreeSet<String>, tries: &mut usize, func: &(dyn Fn(&Example) -> Grid + std::panic::RefUnwindSafe), output: &mut BTreeMap<String, Vec<OutputData>>) -> Option<usize> {
    if !experiment_todo.is_empty() {
        if let Ok(ex) = experiment_todo.parse::<usize>() {
            if ex != experiment {
                return None;
            }
        } 
    }
    if done.contains(task) {   // already done???
        return Some(usize::MAX);
    }

    let ans = panic::catch_unwind(|| experiment_example(examples, task, experiment, trans, func));

    let ans = match ans {
        Ok(ans) => ans,
        Err(e) => {
            eprintln!("{task} / {experiment} Exception: {e:?}");
            vec![Grid::trivial()]
        },
    };

    *tries += 1;

    save(task, experiment, trans, is_test, &ans, targets, done, output)
}

#[allow(clippy::too_many_arguments)]
fn run_experiment_tries(task: &str, experiment: usize, experiment_todo: &str, trans: Transformation, is_test: bool, examples: &Examples, targets: &[Grid], done: &mut BTreeSet<String>, tries: &mut usize, func: &(dyn Fn(&Grid, &Grid, &mut usize) -> Grid + RefUnwindSafe), output: &mut BTreeMap<String, Vec<OutputData>>) -> Option<usize> {
    if !experiment_todo.is_empty() {
        if let Ok(ex) = experiment_todo.parse::<usize>() {
            if ex != experiment {
                return None;
            }
        } 
    }
    if done.contains(task) {   // already done???
        return Some(usize::MAX);
    }

    let ans = panic::catch_unwind(|| experiment_grid(examples, task, experiment, trans, func));

    let ans = match ans {
        Ok(ans) => ans,
        Err(e) => {
            eprintln!("{task} / {experiment} Exception: {e:?}");
            vec![Grid::trivial()]
        },
    };

    *tries += 1;

    save(task, experiment, trans, is_test, &ans, targets, done, output)
}

#[allow(clippy::too_many_arguments)]
fn save(task: &str, experiment: usize, trans: Transformation, is_test: bool, ans: &[Grid], targets: &[Grid], done: &mut BTreeSet<String>, results: &mut BTreeMap<String, Vec<OutputData>>) -> Option<usize> {
    let target_size: usize = targets.iter().map(|target| target.size()).sum();
    let ans_size: usize = ans.iter().map(|ans| ans.size()).sum();
    let same = if target_size > 0 && ans_size > 0 {
        targets.iter()
            .zip(ans.iter())
            .map(|(target, ans)| {
                if ans.equals(target) == Same {
                    1
                } else {
                    0
                }})
            .sum::<usize>() > 0
    } else {
        false
    };

    if !is_test {
        if same {
            add_real_output(task, ans, results);

            done.insert(task.to_string());
            println!("Success: {experiment:>05} {trans:?} / {task}");

            Some(experiment)
        } else if ans_size > 0 {
            add_dummy_output(task, ans.len(), results);

            let dist = ans[0].distance(&targets[0]);

            println!("Final Test Failed : {experiment:>05} {trans:?} / {task} by {dist:.4}");

            None
        } else {
            add_dummy_output(task, ans.len(), results);

            None
        }

    } else if ans_size > 0 {
        add_real_output(task, ans, results);

        done.insert(task.to_string());
        println!("Test Success: {experiment:>05} {trans:?} / {task}");

        Some(experiment)
    } else {
        add_dummy_output(task, ans.len(), results);

        None
    }
}

fn format(rule_tasks: &BTreeMap<usize, Vec<String>>) {
    println!();
    println!("Rule solving tasks:");
    println!("-------------------");
    for (k, v) in rule_tasks.iter() {
        println!("{k:<5}: {v:?}");
    }
    println!();
}
