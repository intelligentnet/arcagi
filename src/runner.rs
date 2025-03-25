use std::str::FromStr;
use std::hash::{DefaultHasher, Hash, Hasher};
use peak_alloc::PeakAlloc;
#[global_allocator]
static PEAK: PeakAlloc = PeakAlloc;
use strum::IntoEnumIterator;
use std::panic;
use std::panic::RefUnwindSafe;
use std::collections::{BTreeSet, BTreeMap};
use pathfinding::prelude::Matrix;
use array_tool::vec::{Uniq, Join};
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
    let mut cat_hash: BTreeMap<u64, usize> = BTreeMap::new();

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

        let mut hasher = DefaultHasher::new();
        examples.cat.hash(&mut hasher);
        let hash = hasher.finish();
        //println!("Hash is {:x}", hasher.finish());
        *cat_hash.entry(hash).or_insert(0) += 1;

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
//println!("{cat_hash:?}");

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
    // 50f325b5 : find shape in Grid
    let colour_diffs = examples.io_colour_diff();
    let all_colour_diffs = examples.io_all_colour_diff();
    let colour_common = examples.io_colour_common();

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

        if let Some(rule) = run_experiment(task, 187, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }

    let gc = BlackPatches;
    if all || cat.contains(&gc) { // 3?
        *cap_cats.entry(gc).or_insert(0) += 1;

        // TODO: 981571dc.json and af22c60d.json mirrored
        // 1e97544e.json e95e3d8e.json e95e3d8e.json 0dfd9992.json + test c3f564a4.json 29ec7d0e.json
        let func = &|ex: &Example| {
            if ex.input.black.is_empty() || !ex.input.grid.is_square() || ex.input.grid.size() < 16 * 16 {
                return Grid::trivial();
            }

            // Split up any adjacent black patches
            fn pop_bp(grid: &Grid, black_patches: &Shapes, out: &mut Shapes) {
                for bp in black_patches.shapes.iter() {
                    if grid.cells[(bp.orow,bp.ocol+bp.cells.columns-1)].colour != Black { 
                        let sg = grid.subgrid(bp.orow, bp.cells.rows, bp.ocol, bp.cells.columns);
                        let ss = sg.to_shapes_coloured();

                        for s in ss.shapes.iter() {
                            let b = if s.orow == 0 {
                                if bp.orow + s.cells.rows >= grid.cells.rows || bp.ocol + s.cells.columns + 1 >= grid.cells.columns {
                                    return;
                                }
                                let g = grid.subgrid(bp.orow, s.cells.rows, bp.ocol, s.cells.columns + 1);
                                g.as_shape().to_position(bp.orow, bp.ocol)
                            } else {
                                if bp.orow + s.orow + s.cells.rows >= grid.cells.rows || bp.ocol + s.cells.columns + s.cells.columns + 1 >= grid.cells.columns {
                                    return;
                                }
                                let g = grid.subgrid(bp.orow + s.orow, s.cells.rows, bp.ocol + s.cells.columns, s.cells.columns + 1);

                                g.as_shape().to_position(bp.orow + s.orow, bp.ocol + s.cells.columns)
                            };

                            out.shapes.push(b);
                        }
                    } else {
                        out.shapes.push(bp.clone());
                    }
                }
            }

            let mut grid = ex.input.grid.clone();
            let bp_in = ex.input.black.clone();
            let mut black_patches: Shapes = ex.input.black.clone_base();

            pop_bp(&grid, &bp_in, &mut black_patches);

            for it in 0 .. 4 {   // may need more that one iteration
                'second:
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
                                //if patch.is_full() && s.same_patch(&patch) {
                                if s.same_patch(&patch) {
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
                    let bp_in = black_patches.clone();
                    black_patches = grid.find_black_patches();
                    pop_bp(&grid, &bp_in, &mut black_patches);
                } else {
                    grid = grid.rot_180();
                }
                if grid.full() {
                    break
                }
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 292, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = &|ex: &Example| {
            if ex.input.black.is_empty() || !ex.input.grid.is_square() || ex.input.grid.size() < 20 * 20 {
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

        if let Some(rule) = run_experiment(task, 382, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 418, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 444, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 461, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone_base();

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

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 522, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 551, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

                if let Some(rule) = run_experiment(task, 597, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 633, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let mut cc = ex.input.shapes.colour_cnt();
            let mut shapes = ex.input.shapes.clone_base();

            for s in ex.input.shapes.shapes.iter() {
                if !s.is_pixel() {
                    cc.remove(&s.colour);
                }
            }

            for s in ex.input.shapes.shapes.iter() {
                if !s.is_pixel() {
                    let mut ns = s.clone();
                    let bg_cnt = s.bg_count();
                    let colour: Vec<_> = cc.iter().filter(|&(&c,&s)|c != ns.colour && (s == if bg_cnt % 2 == 0 { bg_cnt / 2 } else { bg_cnt / 2 + 1})).collect();
                    
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

        if let Some(rule) = run_experiment(task, 683, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 716, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            ex.input.grid.rip(colour_diffs[0])
        };

        if let Some(rule) = run_experiment(task, 726, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let ss = ex.input.coloured_shapes.split_size(12);

            if !colour_diffs.is_empty() && ss.shapes.len() < 2 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.coloured_shapes.clone_base();
            let mut last_size = usize::MAX;

            for s in ss.shapes.iter() {
                if s.size() > 12 {
                    let s = s.shrink_border_n(3);

                    if s.size() == 0 || last_size != usize::MAX && last_size != s.size() {
                        return Grid::trivial();
                    }

                    if last_size == usize::MAX {
                        last_size = s.size();
                    }

                    shapes.shapes.push(s);
                }
            }
            shapes.shapes.sort();
//shapes.show();

            let shape = shapes.majority_cell();

            if shape.size() == 0 {
                return Grid::trivial();
            }

            let (rrep, crep) = shapes.find_repeats();
            let (rgap, cgap) = shapes.find_gaps();

            shape.fit_chequer(rrep, crep, shapes.shapes[0].orow, shapes.shapes[0].ocol, rgap, cgap, ex.input.grid.cells.rows, ex.input.grid.cells.columns, &|s, _, _| s.clone()).to_grid()
        };

        if let Some(rule) = run_experiment(task, 768, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let pixels = ex.input.shapes.get_pixels();
            let mut all_shapes = ex.input.shapes.clone();

            if !colour_diffs.is_empty() || pixels.shapes.len() * 2 != all_shapes.shapes.len() {
                return Grid::trivial();
            }
            let mut i = 0;

            for ss in ex.input.shapes.shapes.iter() {
                if !ss.is_pixel() {
                    let mut shapes = ss.get_joined();

                    for s in shapes.shapes.iter_mut() {
                        s.fill_centre_mut(pixels.shapes[i].colour);

                        all_shapes.shapes.push(s.clone());
                    }

                    i += 1;
                }
            }
//all_shapes.to_grid_transparent().show();

            all_shapes.to_grid_transparent()
        };

        if let Some(rule) = run_experiment(task, 797, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 || ex.input.shapes.len() < 2 {
                return Grid::trivial();
            }

            let (rg, cg) = ex.input.shapes.find_gaps();
            let ss = &ex.input.shapes.shapes;
            let mut grid = ex.input.grid.clone();

            for (r, c) in (ss[ss.len() - 1].orow + rg + 1 .. grid.cells.rows).step_by(rg + 1).zip((ss[ss.len() - 1].ocol + cg + 1 .. grid.cells.columns).step_by(cg + 1)) {
                grid.cells[(r,c)].colour = colour_diffs[0];
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 816, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.len() != 5 {
                return Grid::trivial();
            }
            let in_shapes = ex.input.shapes.clone();
            let rows = in_shapes.shapes[1].cells.rows;
            let cols = in_shapes.shapes[1].cells.columns;
            let (cr, cc) = in_shapes.shapes[0].mid_pixel();

            let mut shapes = Shapes::new_sized(rows * 2, cols * 2);

            for s in in_shapes.shapes.iter() {
                if s.size() != ex.input.shapes.size() {
                    let ns = if s.orow < cr && s.ocol < cc {
                        s.to_position(0, 0)
                    } else if s.orow < cr && s.ocol >= cc {
                        s.to_position(0, cols)
                    } else if s.orow >= cr && s.ocol < cc {
                        s.to_position(rows, 0)
                    } else {
                        s.to_position(rows, cols)
                    };

                    shapes.shapes.push(ns);
                }
            }
//shapes.to_grid().show();

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 849, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();
            let (min_r, min_c, max_r, max_c) = grid.corners();
            let (r,c) = grid.centre_of_symmetry();

            if r == 0 || c == 0 || r >= grid.cells.rows - 1 || c >= grid.cells.columns - 1 {
                return Grid::trivial();
            }

            grid.cells[(r,c)].colour = colour_diffs[0];
            grid.cells[(r-1,c-1)].colour = grid.cells[(min_r,min_c)].colour;
            grid.cells[(r-1,c+1)].colour = grid.cells[(min_r,max_c)].colour;
            grid.cells[(r+1,c-1)].colour = grid.cells[(max_r,min_c)].colour;
            grid.cells[(r+1,c+1)].colour = grid.cells[(max_r,max_c)].colour;

            grid
        };

        if let Some(rule) = run_experiment(task, 873, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let out_csm = examples.colour_shape_map(true);
        let cam = examples.colour_attachment_map(true);

        let func = |ex: &Example| {
            if all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let (div_colour, mut shapes) = ex.input.grid.full_dim_split(&ex.input.shapes);

            shapes.shapes.sort_by(|a, b| (a.ocol,a.orow).cmp(&(b.ocol,b.orow)));

            let mut start = Shape::trivial();
            let mut div = Shape::trivial();

            for s in shapes.shapes.iter() {
                if s.is_pixel() {
                    start = s.clone();
                } else if s.cells.rows == ex.input.grid.cells.rows {
                    div = s.clone();
                }
            }

            let mut new_shapes = Shapes::new_sized(ex.input.grid.cells.rows, div.ocol);
            let mut r = 1;

            if start.ocol <= div.ocol {
                return Grid::trivial();
            }

            let mut c = start.ocol - div.ocol - 1;

            start.to_position_mut(0, c);
            new_shapes.shapes.push(start.clone());

            for s in shapes.shapes.iter() {
                if !s.is_pixel() && s.colour != div_colour {
                    if let Some(os) = out_csm.get(&s.colour) {
                        if let Some(left) = cam.get(&s.colour) {
                            let ns;

                            if *left {
                                ns = os.to_position(r, c);
                                c += os.cells.columns - 1;
                            } else {
                                if c < os.cells.columns - 1 {
                                    return Grid::trivial();
                                }

                                c -= os.cells.columns - 1;
                                ns = os.to_position(r, c);
                            };

                            new_shapes.shapes.push(ns);
                            r += os.cells.rows;
                        }
                    }
                }
            }

//new_shapes.to_grid().show();
            new_shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 938, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !all_colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let (div_colour, mut shapes) = ex.input.grid.full_dim_split(&ex.input.shapes);
            let div = shapes.find_by_colour(div_colour);
            let mut grid = ex.input.grid.clone();

            // get the sort order right!
            for s in shapes.shapes.iter() {
                if s.colour != div_colour {
                    if div.orow == ex.input.grid.cells.rows - 1 && div.cells.rows == 1 {
                        shapes.shapes.sort_by(|a, b| a.orow.cmp(&b.orow));
                    } else if div.ocol == ex.input.grid.cells.columns - 1 && div.cells.columns == 1 {
                        shapes.shapes.sort_by(|a, b| a.ocol.cmp(&b.ocol));
                    } else if div.orow == 0 && div.cells.rows == 1 {
                        shapes.shapes.sort_by(|a, b| b.orow.cmp(&a.orow));
                    } else {
                        shapes.shapes.sort_by(|a, b| b.ocol.cmp(&a.ocol));
                    }
                    break;
                }
            }

            for s in shapes.shapes.iter() {
                if s.colour != div_colour {
                    if div.orow == ex.input.grid.cells.rows - 1 && div.cells.rows == 1 {
                        for r in s.orow .. ex.input.grid.cells.rows - 1 {
                            for c in 0 .. s.cells.columns {
                                grid.cells[(r,c+s.ocol)].colour = s.colour;
                            }
                        }
                    } else if div.ocol == ex.input.grid.cells.columns - 1 && div.cells.columns == 1 {
                        for r in 0 .. s.cells.rows {
                            for c in s.ocol .. ex.input.grid.cells.columns - 1 {
                                grid.cells[(r+s.orow,c)].colour = s.colour;
                            }
                        }
                    } else if div.orow == 0 && div.cells.rows == 1 {
                        for r in (1 ..= s.orow).rev() {
                            for c in s.ocol .. s.ocol + s.cells.columns {
                                grid.cells[(r,c)].colour = s.colour;
                            }
                        }
                    } else {
                        for r in s.orow .. s.orow + s.cells.rows {
                            for c in (1 ..= s.ocol).rev() {
                                grid.cells[(r,c)].colour = s.colour;
                            }
                        }
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 999, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let mut shapes = ex.input.shapes.clone_base();

            for s in ex.input.shapes.shapes.iter() {
                for s2 in ex.input.shapes.shapes.iter() {
                    if s != s2 && s.container(&s2) {
                        let (s_r, _) = s.centre_of_exact();
                        let (s2_r, _) = s2.centre_of_exact();
                        shapes.shapes.push(s.clone());
                        shapes.shapes.push(s2.clone());

                        let sm = if s_r == s2_r {
                            let sm = s2.mirrored_c();

                            if sm.ocol + sm.cells.columns * 2 + 1 < s.ocol + s.cells.columns {
                                let extra = if s.cells.columns % 2 != 0 { 1 } else { 0 };
                                sm.to_position(sm.orow, s.ocol + s.cells.columns / 2 + extra)
                            } else {
                                sm.to_position(sm.orow, s.ocol + sm.cells.columns)
                            }
                        } else {
                            let sm = s2.mirrored_r();

                            if sm.orow + sm.cells.rows * 2 + 1 < s.orow + s.cells.rows {
                                let extra = if s.cells.rows % 2 != 0 { 1 } else { 0 };
                                sm.to_position(s.orow + s.cells.rows / 2 + extra, sm.ocol)
                            } else {
                                sm.to_position(s.orow + sm.cells.rows, sm.ocol)
                            }
                        };

                        shapes.shapes.push(sm);
                    }
                }
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1040, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }
            let sc = ex.input.shapes.size_cnt();
            if sc.len() != 1 {
                return Grid::trivial();
            }
            let Some((_, n)) = sc.first_key_value() else { todo!() };
            // TODO: strict assumption, make more generic
            if *n != 4 {
                return Grid::trivial();
            }
            let (rs, cs) = ex.input.shapes.shapes[0].dimensions();
            if rs != cs {
                return Grid::trivial();
            }

            let mut shapes = Shapes::new_sized(rs * 2 + 1, cs * 2 + 1);

            let corners = ex.input.shapes.all_corners();

            let mut row = 0;
            let mut col = 0;

            for (i, (r, c)) in corners.iter().enumerate() {
                let s = ex.input.shapes.nearest_shape(*r, *c);

                shapes.shapes.push(s.to_position(row, col));

                match i {
                    0 => col = cs + 1,
                    1 => row = rs + 1,
                    2 => col = 0,
                    3 => row = 0,
                    _ => todo!(),
                }
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1084, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let sc = ex.input.shapes.size_cnt();
            if sc.len() != 2 || ex.input.shapes.shapes.len() < 3 || all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let (rs, cs) = ex.input.shapes.shapes.iter()
                .filter(|s| s.colour != all_colour_diffs[0])
                .max().unwrap().dimensions();
            if rs != cs {
                return Grid::trivial();
            }

            let mut grid = Grid::new(2, 2, Black);

            let corners = ex.input.shapes.all_corners();

            for s in ex.input.shapes.shapes.iter() {
                if s.colour != all_colour_diffs[0] {
                    match s.nearest_point_idx(&corners) {
                        0 => grid.cells[(0,0)].colour = s.colour,
                        1 => grid.cells[(0,1)].colour = s.colour,
                        2 => grid.cells[(1,1)].colour = s.colour,
                        3 => grid.cells[(1,0)].colour = s.colour,
                        _ => todo!(),
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1121, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let ccm = ex.input.shapes.shape_colour_cnt_map();

            let mut grid = Grid::new(ex.input.grid.cells.rows, ex.input.grid.cells.columns, Black);

            let mut box_tlr = 0;
            let mut box_tlc = 0;

            for (col, sv) in ccm.iter() {
                if sv.len() == 1 {
                    box_tlr = sv[0].orow;
                    box_tlc = sv[0].ocol;
                    let box_brr = box_tlr + sv[0].cells.rows;
                    let box_brc = box_tlc + sv[0].cells.columns;

                    grid.fill_patch_coord_mut(box_tlr, box_tlc, box_brr - box_tlr, box_brc - box_tlc, *col);
                }
            }

            for (col, sv) in ccm.iter() {
                if sv.len() > 1 {
                    let (mut tlr, mut tlc, brr, brc) = Shapes::new_shapes(&sv).corners();
                    
                    let rlen = brr - tlr - 1;
                    let clen = brc - tlc - 1;

                    tlr = tlr.max(box_tlr);
                    tlc = tlc.max(box_tlc);

                    if grid.cells.rows < tlr + rlen || grid.cells.columns < tlc + clen {
                        return Grid::trivial();
                    }

                    grid.fill_patch_coord_mut(tlr, tlc, rlen, clen, *col);
                }
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 1168, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 5 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();
            let idx = &ex.input.shapes.shapes[0];
            let mut colour = NoColour;

            for s in ex.input.shapes.shapes.iter() {
                if s.size() == 1 && s.orow < idx.cells.rows && s.ocol < idx.cells.columns {
                    colour = s.colour;
                }
            }

            for s in shapes.shapes.iter_mut() {
                if s.colour == colour && (s.orow >= idx.cells.rows || s.ocol >= idx.cells.columns) {
                    s.force_recolour_mut(Black);
                }
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1194, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let colours = Uniq::uniq(&colour_common, all_colour_diffs.clone());
            if ex.input.shapes.shapes.len() < 2 || colour_diffs.len() != 1 || colours.len() != 1 {
                return Grid::trivial();
            }

            let (tlr, tlc, brr, brc) = ex.input.shapes.corners();
            let mut grid = ex.input.grid.clone();

            grid.draw_mut(Down, 0, tlc, colours[0]);
            grid.draw_mut(Down, 0, brc - 1, colours[0]);
            grid.draw_mut(Right, tlr, 0, colours[0]);
            grid.draw_mut(Right, brr - 1, 0, colours[0]);

            grid.flood_fill_mut(tlr + 1, tlc + 1, NoColour, colour_diffs[0]);

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1216, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if colour_diffs.len() == 1 {
            let mut gap = 0;

            for (i, s) in examples.examples[0].output.shapes.shapes.iter().enumerate() {
                if i == 0 && s.colour == colour_diffs[0] {
                    continue;
                } else if s.colour == colour_diffs[0] {
                    gap = i;
                    break;
                }
            }

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.len() < 2 || gap == 0 {
                    return Grid::trivial();
                }

                let mut shapes = ex.input.shapes.clone();

                for (i, s) in shapes.shapes.iter_mut().enumerate() {
                    if i % gap == 0 {
                        s.force_recolour_mut(colour_diffs[0]);
                    }
                }

//shapes.to_grid().show();
                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 1247, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 4 {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();
            let mut shapes: BTreeMap<Colour, Vec<&Shape>> = BTreeMap::new();

            for s in ex.input.shapes.shapes.iter() {
                if s.size() != ex.input.grid.size() {
                    shapes.entry(s.colour).and_modify(|sv| sv.push(s)).or_insert(vec![s]);
                }
            }

            for (_colour, shapes) in shapes.iter() {
                let origins = Shapes::origins(shapes);

                if origins.is_empty() {
                    return Grid::trivial();
                }

                for (i, (r, c)) in origins.iter().enumerate() {
                    if !Shapes::contains_origin(shapes, *r, *c) {
                        let posn = if i == 0 { 3 } else { i - 1 };

                        if posn >= shapes.len() {
                            return Grid::trivial();
                        }

                        let s = if posn % 2 == 0 {
                            shapes[posn].mirrored_c()
                        } else {
                            shapes[posn].mirrored_r()
                        };

                        let s = s.to_position(*r, *c);

                        grid.copy_shape_to_grid_mut(&s);
                    }
                }
            }
//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1295, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 10 || colour_common.len() < 5 {
                return Grid::trivial();
            }

            let mut cc: BTreeMap<Colour, usize> = BTreeMap::new();
            let mut corners: Vec<Shape> = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                if s.size() == 4 {
                    *cc.entry(s.colour).or_insert(0) += 1;
                    corners.push(s.clone());
                }
            }

            let (_, max_col) = if let Some(col) = cc.iter().map(|(&c, &n)| (n, c)).max() {
                col
            } else {
                (0, NoColour)
            };
            let (_, min_col) = if let Some(col) = cc.iter().map(|(&c, &n)| (n, c)).min() {
                col
            } else {
                (0, NoColour)
            };
            let (tlr, tlc, brr, brc) = Shapes::vec_corners(&corners);
            let mut grid = ex.input.grid.clone();

            if brr - tlr < 2 || brc - tlc < 2 || brr == tlr || brc == tlc || tlr == 0 || tlc == 0 || grid.cells.columns == brc || grid.cells.rows == brr {
                return Grid::trivial();
            }

            let centre = Shape::new_sized_coloured_position(tlr, tlc, brr - tlr, brc - tlc, max_col);
            let scentre = Shape::new_sized_coloured_position(tlr + 1, tlc + 1, brr - tlr - 2, brc - tlc - 2, min_col);
            let left = Shape::new_sized_coloured_position(tlr, 0, brr - tlr, tlc, min_col);
            let right = Shape::new_sized_coloured_position(tlr, brc, brr - tlr, grid.cells.columns - brc, min_col);
            let up = Shape::new_sized_coloured_position(0, tlc, tlr, brc - tlc, min_col);
            let down = Shape::new_sized_coloured_position(brr, tlc, grid.cells.rows - brr, brc - tlc, min_col);
            grid.copy_shape_to_grid_mut(&centre);
            grid.copy_shape_to_grid_mut(&scentre);
            grid.copy_shape_to_grid_mut(&left);
            grid.copy_shape_to_grid_mut(&right);
            grid.copy_shape_to_grid_mut(&up);
            grid.copy_shape_to_grid_mut(&down);

            for s in ex.input.shapes.shapes.iter() {
                if s.size() == 1 && grid.cells[(s.orow,s.ocol)].colour == min_col {
                    //grid.copy_shape_to_grid_mut(&s);
                    if s.orow <= tlr && (s.orow == 0 || grid.cells[(s.orow-1,s.ocol)].colour == min_col) {
                        grid.draw_bg_mut(Up, s.orow, s.ocol, s.colour, min_col);
                    } else if s.orow >= brr && (s.orow == grid.cells.rows - 1 || grid.cells[(s.orow+1,s.ocol)].colour == min_col) {
                        grid.draw_bg_mut(Down, s.orow, s.ocol, s.colour, min_col);
                    } else if s.ocol <= tlc && (s.ocol == 0 || grid.cells[(s.orow,s.ocol-1)].colour == min_col) {
                        grid.draw_bg_mut(Left, s.orow, s.ocol, s.colour, min_col);
                    } else if s.ocol >= brc {
                        grid.draw_bg_mut(Right, s.orow, s.ocol, s.colour, min_col);
                    }
                }
            }
            grid
        };

        if let Some(rule) = run_experiment(task, 1359, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 5 || ex.input.coloured_shapes.shapes.len() != ex.input.shapes.shapes.len() {
                return Grid::trivial();
            }

            let ccm = ex.input.shapes.shape_colour_cnt_map();
            let r = ccm.len();
            let c = if let Some(c) = ccm.iter().map(|(_,vs)| vs.len()).max() {
                c
            } else {
                0
            };

            if c == 0 {
                return Grid::trivial();
            }

            let mut grid = Grid::new(r, c, Black);
            let mut i = 0;

            let mut cs: Vec<(usize, Colour)> = ccm.iter().map(|(colour,vs)| (vs.len(), *colour)).collect();

            cs.sort_by(|a, b| b.cmp(a));

            for (len, colour) in cs.iter() {
                for j in 0 .. *len {
                    grid.cells[(i, c - len + j)].colour = *colour;
                }

                i += 1;
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1397, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() < 2 {
                return Grid::trivial();
            }
            let mut posn = 0;
            let mut horizontal = true;
            let mut shapes = ex.input.coloured_shapes.clone();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.orow == 0 && s.cells.rows == ex.input.grid.cells.rows {
                    horizontal = false;
                    posn = s.ocol;

                } else if s.ocol == 0 && s.cells.columns == ex.input.grid.cells.columns {
                    horizontal = true;
                    posn = s.orow;
                } 
            }

            for s in ex.input.coloured_shapes.shapes.iter() {
                if (s.orow != 0 || s.cells.rows != ex.input.grid.cells.rows) && (s.ocol != 0 || s.cells.columns != ex.input.grid.cells.columns) {
                    let ccm = s.cell_colour_cnt_map();

                    if ccm.len() != 2 {
                        return Grid::trivial();
                    }

                    let colours: Vec<Colour> = ccm.iter().map(|(c, _)| *c).collect();
                    shapes.shapes.push(s.toddle_colour(colours[0], colours[1]));

                    let s = if horizontal {
                        if posn > s.orow {
                            s.mirrored_r().to_position(posn + 2, s.ocol)
                        } else {
                            if posn < s.cells.rows + 1 {
                                return Grid::trivial();
                            }

                            s.mirrored_r().to_position(posn - 1 - s.cells.rows, s.ocol)
                        }
                    } else {
                        if posn > s.ocol {
                            s.mirrored_c().to_position(s.orow, posn + 2)
                        } else {
                            if posn < s.cells.columns + 1 {
                                return Grid::trivial();
                            }

                            s.mirrored_c().to_position(s.orow, posn - 1 - s.cells.columns)
                        }
                    };

                    shapes.shapes.push(s);
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1459, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() < 2 || all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let shapes = ex.input.grid.recolour(Black, NoColour).to_shapes();
            let mut shape = Shape::trivial();
            let mut bg = Shape::trivial();
            let mut pixel1 = Shape::trivial();
            let mut pixel2 = Shape::trivial();

            for s in shapes.shapes.iter() {
                if s.size() == ex.input.grid.size() {
                    continue;
                }
                if s.size() > 1 && s.colour != NoColour {
                    shape = s.clone();
                } else if s.is_pixel() {
                    if pixel1 == Shape::trivial() {
                        pixel1 = s.clone();
                    } else {
                        pixel2 = s.clone();
                    };
                } else if s.size() > 1 && s.colour == NoColour && s.colour_cnt(false).1 + 1 == s.size() {
                    bg = s.clone();
                }
            }
            if pixel1.colour != pixel2.colour {
                return Grid::trivial();
            }

            let mut grid = Grid::new(bg.cells.rows, bg.cells.columns, Black);

            let (pixel, shape_pixel) = if pixel1.contained_by(&bg) {
                (&pixel1, pixel2)
            } else {
                (&pixel2, pixel1)
            };

            if pixel.orow < bg.orow || pixel.ocol < bg.ocol {
                return Grid::trivial();
            }

            let r = pixel.orow - bg.orow;
            let c = pixel.ocol - bg.ocol;

            if shape_pixel.orow < shape.orow || shape_pixel.ocol < shape.ocol {
                return Grid::trivial();
            }

            let sr = shape_pixel.orow - shape.orow;
            let sc = shape_pixel.ocol - shape.ocol;

            if r < sr || c < sc {
                return Grid::trivial();
            }

            grid.copy_shape_to_grid_position_mut(&shape, r - sr, c - sc);
            grid.copy_shape_to_grid_position_mut(&pixel, r, c);

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1524, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let (out_r, out_c) = examples.examples[0].output.grid.dimensions();

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 20 {
                return Grid::trivial();
            }

            let mut cnt = 0;
            let mut ms = &Shape::trivial();

            for s in ex.input.shapes.shapes.iter() {
                if s.size() > 4 {
                    let (_, n) = if let Some(n) = s.cell_colour_cnt_map().pop_first() {
                        n
                    } else {
                        (NoColour, 0)
                    };
                    if n > cnt {
                        cnt = n;
                        ms = &s;
                    }
                }
            }
            let grid = Grid::new(out_r, out_c, ms.colour);

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1555, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let mut colour = NoColour;

        for s in examples.examples[0].input.black.shapes.iter() {
            if s.is_square() {
                colour = examples.examples[0].output.grid.find_axis_colour(&s);
            }
        }

        let func = |ex: &Example| {
            if ex.input.black.shapes.len() != 1 || colour == NoColour {
                return Grid::trivial();
            }
            let mut grid = ex.input.grid.clone();
            let patch = &ex.input.black.shapes[0];

            for r in 0 .. grid.cells.rows {
                for c in patch.ocol .. patch.ocol + patch.cells.columns {
                    if grid.cells[(r,c)].colour != colour {
                        grid.cells[(r,c)].colour = Black;
                    }
                }
            }

            for c in 0 .. grid.cells.columns {
                for r in patch.orow .. patch.orow + patch.cells.rows {
                    if grid.cells[(r,c)].colour != colour {
                        grid.cells[(r,c)].colour = Black;
                    }
                }
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 1591, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();

            for s in ex.input.shapes.shapes.iter() {
                grid.draw_mc_mut(Up, s.orow, s.ocol, s.colour);
                grid.draw_mc_mut(Down, s.orow, s.ocol, s.colour);
                grid.draw_mc_mut(Left, s.orow, s.ocol, s.colour);
                grid.draw_mc_mut(Right, s.orow, s.ocol, s.colour);
            }

            for s in ex.input.shapes.shapes.iter() {
                grid.recolour_mut(s.colour + ToBlack, colour_diffs[0]);
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1592, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();

            for s in ex.input.shapes.shapes.iter() {
                if s.is_pixel() {
                    let colour = ex.input.grid.shape_in_line(&s);

                    if colour != NoColour {
                        grid.flood_fill_bg_mut(s.orow, s.ocol, NoColour, all_colour_diffs[0], s.colour);
                    } else {
                        grid.flood_fill_bg_mut(s.orow, s.ocol, NoColour, Black, s.colour);
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1593, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 6 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone_base();

            for s in ex.input.shapes.shapes.iter() {
                if s.hollow() {
                    let mut cc: BTreeMap<Colour, usize> = BTreeMap::new();
                    let mut pix = Shape::trivial();

                    for s2 in ex.input.shapes.shapes.iter() {
                        if s2.is_pixel() && s2.contained_by(&s) {
                            *cc.entry(s2.colour).or_insert(0) += 1;
                            pix = s2.clone();
                        }
                    }
                    if let Some((_,c)) = cc.iter().map(|(k,v)| (v, k)).max() {
                        let ss = s.flood_fill(pix.orow - s.orow, pix.ocol - s.ocol, NoColour, *c);

                        shapes.shapes.push(ss);
                    }
                } else if !s.is_pixel() {
                        shapes.shapes.push(s.clone());
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1593, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let in_shapes = ex.input.coloured_shapes.clone();
            let mut shapes = in_shapes.clone();
            let mut cnt = usize::MAX;
            let mut del_shape = Shape::trivial();

            for s in in_shapes.shapes.iter() {
                let n = s.pixels_in_shape();

                if n == 0 {
                    return Grid::trivial();
                }

                if cnt > n {
                    cnt = n;
                    del_shape = s.clone();
                }
            }
 
            shapes.shapes.retain(|s| *s != del_shape);

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1594, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let centres = ex.input.shapes.centre_of();
            let mut grid = ex.input.grid.clone();
            // Only 1 direction or will skip too far
            let mut pairs: Vec<(usize, usize)> = Vec::new();

            for (r1, c1) in centres.keys() {
                for (r2, c2) in centres.keys() {
                    if r1 != r2 && c1 != c2 {
                        let l1 = (*r1 as isize - *r2 as isize).abs() as usize;
                        let l2 = (*c1 as isize - *c2 as isize).abs() as usize;

                        if l1 == l2 && !pairs.contains(&(l1, l2)) {
                            let dir = Grid::calc_direction(*r1, *c1, *r2, *c2);
                            if dir != Other && !pairs.contains(&(l1, l2)) {
                                let (r, c) = grid.skip_to(dir, *r1, *c1);
                                grid.draw_term_mut(dir, r, c, colour_diffs[0]);
                            }
                            pairs.push((l1, l2));
                        }
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1595, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();

            for s in ex.input.shapes.shapes.iter() {
                let (dir, r, c) = s.find_a_border_break();

                grid.draw_term_mut(dir, r, c, colour_diffs[0]);
                grid.flood_fill_mut(s.orow + s.cells.rows / 2, s.ocol + s.cells.columns / 2, NoColour, colour_diffs[0]);
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1596, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut grid = ex.input.shapes.biggest_shape().to_grid();
            let mut template = Shape::trivial();
            let mut idx = Shape::trivial();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.is_square() {
                    template = s.to_origin();
                } else if s.dimensions() != grid.dimensions() {
                    idx = s.clone();
                }
            }

            for s in idx.to_grid().to_shapes_sq().shapes.iter() {
                if s.colour != grid.colour {
                    let factor = grid.cells.rows as f64 / idx.cells.rows as f64 ;

                    template.recolour_mut(all_colour_diffs[0], grid.colour);
                    template.to_position_mut((s.orow as f64 * factor) as usize, (s.ocol as f64 * factor) as usize);
                    grid.copy_shape_to_grid_mut(&template);
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 1597, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if all_colour_diffs.is_empty() || !ex.input.grid.is_square() {
                return Grid::trivial();
            }

            let mut template = Shape::trivial();

            for s in ex.input.shapes.shapes.iter() {
                if s.colour != Black && s.ocol != 0 {
                    template = s.clone();
                }
            }

            let bg = ex.input.grid.has_bg_grid_not_sq();
            let mut colour = NoColour;
            let mut shapes = ex.input.shapes.clone_base();

            shapes.shapes.push(Shape::new_sized_coloured(ex.input.grid.cells.rows, ex.input.grid.cells.columns, bg));

            for s in ex.input.shapes.shapes.iter() {
                if colour != s.colour && s.colour != Black && s.colour != template.colour {
                    colour = s.colour;
                }

                template.to_position_mut(s.orow, s.ocol);
                shapes.shapes.push(template.recolour(template.colour, colour));

            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1598, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !all_colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let mut templates: Vec<Shape> = Vec::new();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if !s.is_pixel() {
                    templates.push(s.clone());
                }
            }

            if templates.is_empty() || templates[0].cells.rows != 1 || templates[0].cells.columns == 1 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone_base();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if templates.contains(s) {
                    shapes.shapes.push(s.clone());
                } else {
                    let mut c = 0;
                    let mut template = Shape::trivial();

                    templates.iter().for_each(|t| {
                        let cp = t.colour_position(s.colour);

                        if cp.len() > 0 {
                            let (_, cc) = cp[0];

                            c = cc;
                            template = t.clone();
                        }
                    });

                    let new_template = if s.ocol > c {
                        template.to_position(s.orow, s.ocol - c)
                    } else {
                        let sc = c - s.ocol;
                        let sub_temp = template.subshape(0, template.cells.rows, sc, template.cells.columns - sc);
                        sub_temp.to_position(s.orow, 0)
                    };

                    shapes.shapes.push(new_template);
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1599, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 || ex.input.shapes.shapes.len() < 4 {
                return Grid::trivial();
            }

            let mut rv: BTreeSet<usize> = BTreeSet::new();
            let mut cv: BTreeSet<usize> = BTreeSet::new();

            for s in ex.input.coloured_shapes.shapes.iter() {
                rv.insert(s.orow);
                cv.insert(s.ocol);
            }

            let mut shapes = ex.input.shapes.clone();
            let mut template = ex.input.shapes.shapes[0].clone();
            let cts = ex.input.shapes.coords_to_shape();

            for r in rv.iter() {
                for c in cv.iter() {
                    if !cts.contains_key(&(*r, *c)) {
                        template.recolour_mut(template.colour, colour_diffs[0]);
                        template.to_position_mut(*r, *c);

                        shapes.shapes.push(template.clone());
                    }
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1600, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            // Might have been neater using coloured shapes!
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let mut pixels: Vec<Shape> = Vec::new();
            let mut horizontal: Vec<Shape> = Vec::new();
            let mut vertical: Vec<Shape> = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                if s.is_pixel() {
                    pixels.push(s.clone());
                } else if s.is_horizontal_line() {
                    horizontal.push(s.clone());
                } else if s.is_vertical_line() {
                    vertical.push(s.clone());
                }
            }

            let mut pshapes: Vec<(Shape,Shape)> = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                if !s.is_pixel() && !s.is_line() {
                    for p in pixels.iter() {
                        if p.contained_by(&s) {
                            pshapes.push((p.clone(), s.clone()));
                        }
                    }
                }
            }

            let mut shapes = ex.input.shapes.clone_base();
            let mut used: Vec<&Shape> = Vec::new();

            for (p, s) in pshapes.iter() {
                for h in horizontal.iter() {
                    if h.adjacent_r_or_c(&s) && !used.contains(&h) {
                        let diff = (p.orow as isize - h.orow as isize).abs() as usize;
                        let ms;
                        let mp;

                        if h.above(&p) {
                            ms = s.mirrored_r().to_position(s.orow + s.cells.rows + 1, s.ocol);
                            mp = p.to_position(h.orow + diff, p.ocol);
                        } else {
                            if s.orow <= s.cells.rows || h.orow < diff {
                                return Grid::trivial();
                            }

                            ms = s.mirrored_r().to_position(s.orow - s.cells.rows - 1, s.ocol);
                            mp = p.to_position(h.orow - diff, p.ocol);
                        }
                        shapes.shapes.push(s.clone());
                        shapes.shapes.push(p.clone());
                        shapes.shapes.push(ms);
                        shapes.shapes.push(mp);
                        shapes.shapes.push(h.clone());

                        used.push(h);
                    }
                }
                for v in vertical.iter() {
                    if v.adjacent_r_or_c(&s) && !used.contains(&v){
                        let diff = (p.ocol as isize - v.ocol as isize).abs() as usize;
                        let ms;
                        let mp;

                        if v.left(&p) {
                            ms = s.mirrored_c().to_position(s.orow, s.ocol + s.cells.columns + 1);
                            mp = p.to_position(p.orow, v.ocol + diff);
                        } else {
                            if s.ocol <= s.cells.columns || v.ocol < diff {
                                return Grid::trivial();
                            }

                            ms = s.mirrored_c().to_position(s.orow, s.ocol - s.cells.columns - 1);
                            mp = p.to_position(p.orow, v.ocol - diff);
                        }

                        shapes.shapes.push(s.clone());
                        shapes.shapes.push(p.clone());
                        shapes.shapes.push(ms);
                        shapes.shapes.push(mp);
                        shapes.shapes.push(v.clone());

                        used.push(v);
                    }
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1601, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = InToSquaredOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 1599, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.grid.in_to_squared_out(), output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = BGGridInBlack;
    if all || cat.contains(&gc) && !cat.contains(&BGGridOutBlack) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 1607, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_min().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 1609, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_max_colour_count().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 1611, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.has_mirror_r().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 1613, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.has_mirror_c().to_grid(), output) { return Some(rule); };
        //if let Some(rule) = run_experiment_examples(&file, 1000, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return Some(rule); };

        //-if let Some(rule) = run_experiment_examples(&file, 1010, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_shape_fill(&exs), output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() != 1 || ex.input.coloured_shapes.shapes[0].dimensions() != (3, 3) {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();
            let ss = &ex.input.coloured_shapes.shapes[0];

            for c in ss.cells.values() {
                match (c.row - ss.orow, c.col - ss.ocol) {
                    (0, 0) => grid.draw_mut(UpLeft, c.row, c.col, c.colour),
                    (0, 1) => grid.draw_mut(Up, c.row, c.col, c.colour),
                    (0, 2) => grid.draw_mut(UpRight, c.row, c.col, c.colour),
                    (1, 0) => grid.draw_mut(Left, c.row, c.col, c.colour),
                    (1, 1) => (),
                    (1, 2) => grid.draw_mut(Right, c.row, c.col, c.colour),
                    (2, 0) => grid.draw_mut(DownLeft, c.row, c.col, c.colour),
                    (2, 1) => grid.draw_mut(Down, c.row, c.col, c.colour),
                    (2, 2) => grid.draw_mut(DownRight, c.row, c.col, c.colour),
                    _ => todo!()
                }
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 1645, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 3 || !ex.input.shapes.shapes[0].is_pixel() || !ex.input.shapes.shapes[1].is_pixel() {
                return Grid::trivial();
            }

            let cpm = ex.input.grid.cell_colour_posn_map();
            let mut grid = ex.input.grid.clone();

            for (c, vp) in cpm.iter() {
                if vp.len() == 2 {
                    let (_, c1) = vp[0];
                    let (r2, _) = vp[1];

                    grid.cells[(r2,c1)].colour = *c;
                }
            }

            grid.connect_dots();

            grid
        };

        if let Some(rule) = run_experiment(task, 1669, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let colour = colour_diffs[0];
            let grid = &ex.input.grid;
            let mut shapes = ex.input.shapes.clone_base();

            // TODO Fix hard coding
            for s in ex.input.shapes.shapes.iter() {
                if s.orow == 0 {
                    return Grid::trivial();
                }
                if s.cells.columns > 1 && s.cells[(0,1)].colour == Black {
                    shapes.shapes.push(s.flood_fill(0, 1, NoColour, colour));
                    shapes.shapes.push(Shape::new_sized_coloured_position(s.orow - 1, s.ocol + 1, 1, grid.cells.columns - s.ocol, colour));
                } else if s.cells.columns > 2 && s.cells[(0,2)].colour == Black {
                    shapes.shapes.push(s.flood_fill(0, 2, NoColour, colour));
                    shapes.shapes.push(Shape::new_sized_coloured_position(s.orow- 1, 0, 1, s.ocol + 3, colour));
                } else {
                    return Grid::trivial();
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1667, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }

    let gc = SingleColourOut2xIn;
    if all || cat.contains(&gc) { // 3?
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment_tries(task, 1678, experiment, trans, is_test, examples, &targets, done, tries, &|ex, _, n| mirror_only(ex, n), output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 1697, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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
//
            if *colours[0] == NoColour || *colours[1] == NoColour {
                return Grid::trivial();
            }

            let s = sg.toddle_colour(*colours[0], *colours[1]);

            let mut shapes = Shapes::new_sized(ex.input.grid.cells.rows, ex.input.grid.cells.columns);

            shapes.shapes.push(ex.input.grid.as_shape());
            shapes.shapes.push(s.as_shape());
//shapes.to_grid().show();

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1739, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 1767, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 1797, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

                let (mut s1,mut s2) = if rs < cs {
                    (Shape::new_sized_coloured(rs, rs, Black),
                    Shape::new_sized_coloured_position(0, rs, rs, rs, Black))
                } else {
                    (Shape::new_sized_coloured(cs, cs, Black),
                    Shape::new_sized_coloured_position(cs, 0, cs, cs, Black))
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

            if let Some(rule) = run_experiment(task, 1840, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }
            let mut shapes = ex.input.shapes.clone_base();

            let mut shape1 = Shape::trivial();
            let mut shape2 = Shape::trivial();

            for s in ex.input.shapes.shapes.iter() {
                if shape1 == Shape::trivial() {
                    shape1 = s.clone();
                } else if s.equals(&shape1) != Same {
                    shape2 = s.clone();
                }
            }
            for s in ex.input.shapes.shapes.iter() {
                let ms = if s.equals(&shape1) == Same {
                    shape2.to_position(s.orow, s.ocol)
                } else {
                    shape1.to_position(s.orow, s.ocol)
                };

                shapes.shapes.push(ms);
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1872, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 10 || ex.input.shapes.shapes.len() == ex.input.coloured_shapes.shapes.len(){
                return Grid::trivial();
            }
            let mut shapes = ex.input.shapes.clone_base();

            let mut idx = Shape::trivial();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.colour == Mixed {
                    idx = s.clone();
                } else {
                    shapes.shapes.push(s.clone());
                }
            }

            for (cell, s) in idx.cells.values().zip(shapes.shapes.iter_mut()) {
                s.recolour_mut(s.colour, cell.colour);
            }

            shapes.shapes.push(idx.clone());

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1899, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !all_colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();
            let smallest = ex.input.shapes.smallest();
            let largest = ex.input.shapes.largest();
            let edge = if smallest.cells.rows != smallest.cells.columns {
                smallest.cells.rows.max(smallest.cells.columns) / 2
            } else {
                smallest.cells.rows
            };

            if largest.orow < edge || largest.ocol < edge {
                return Grid::trivial();
            }

            let enclosing = Shape::new_sized_coloured_position(largest.orow - edge, largest.ocol - edge, largest.cells.rows + edge * 2, largest.cells.columns + edge * 2, smallest.colour);

            shapes.shapes.insert(0, enclosing);

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1927, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !all_colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let mut shapes = ex.input.coloured_shapes.clone_base();

            for s in ex.input.coloured_shapes.shapes.iter() {
                let cc = s.cell_colours();
                if cc.len() != 2 {
                    return Grid::trivial();
                }

                shapes.shapes.push(s.toddle_colour(cc[0], cc[1]));
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1948, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();
//shapes.show();

            for s in shapes.shapes.iter_mut() {
                for (i, r) in (0 .. s.cells.rows).rev().enumerate() {
                    for c in 0 .. s.cells.columns {
                        let cell = &mut s.cells[(r,c)];

                        if cell.col >= i {
                            cell.col -= i;
                        } else {
                            cell.colour = Black;
                        }
                    }
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 1988, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let mut shapes = ex.input.grid.to_shapes();
            let mut even = true;
            let mut bg = NoColour;

            for s in shapes.shapes.iter_mut() {
                if s.size() != ex.input.grid.size() {
                    if even {
                        if s.ocol == 0 {
                            return Grid::trivial();
                        }
                        s.to_position_mut(s.orow, s.ocol - 1);
                    } else {
                        if s.ocol >= s.cells.columns {
                            return Grid::trivial();
                        }
                        s.to_position_mut(s.orow, s.ocol + 1);
                    }

                    even = !even;
                } else {
                    bg = s.colour;
                }
            }

            let mut grid = shapes.to_grid();

            grid.recolour_mut(Black, bg);

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2026, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.is_empty() {
                return Grid::trivial();
            }
            let bg = ex.input.grid.majority_colour();
            let mut grid = ex.input.grid.clone();
            let mut cc: Vec<usize> = Vec::new();

            for cell in grid.cells.values() {
                if cell.colour != bg {
                    cc.push(cell.col);
                }
            }

            cc.sort();

            let cc = cc.unique();

            for cell in grid.cells.values_mut() {
                if cell.colour != bg {
                    if let Some(ncolour) = cc.iter().position(|n| *n == cell.col) {
                        cell.colour = Colour::from_usize(ncolour + 1);
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2027, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();
            let mut cc: Vec<Colour> = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                if s.ocol == 0 || s.ocol == grid.cells.columns {
                    return Grid::trivial();
                }
                // Down
                let mut r = s.orow + s.cells.rows;
                let mut cm = s.ocol - 1;
                let mut cp = s.ocol + 1;
                let mut l = s.cells.rows;
                let mut outleft = false;

                while r < grid.cells.rows { 
                    if grid.cells.rows - r <= l {
                        l = grid.cells.rows - r;
                    }

                    for rr in r .. r  + l {
                        if !cc.contains(&s.colour) {
                            if !outleft {
                                if grid.cells[(rr, cm)].colour != Black {
                                    grid.cells[(rr, cm)].colour = colour_diffs[0];
                                } else {
                                    grid.cells[(rr, cm)].colour = s.colour;
                                }
                            }
                        } else {
                            if cp < grid.cells.columns {
                                if grid.cells[(rr, cp)].colour != Black {
                                    grid.cells[(rr, cp)].colour = colour_diffs[0];
                                } else {
                                    grid.cells[(rr, cp)].colour = s.colour;
                                }
                            }
                        }
                    }
                    
                    if cm > 0 {
                        cm -= 1;
                    } else if cm == 0 {
                        outleft = true;
                    }
                    cp += 1;
                    r += l;
                }

                // Up
                let mut r = s.orow;
                let mut cm = s.ocol - 1;
                let mut cp = s.ocol + 1;
                let mut l = s.cells.rows;
                let mut outleft = false;

                while r > 0 { 
                    if r <= l {
                        l = r;
                    }

                    for rr in r - l .. r {
                        if !cc.contains(&s.colour) {
                            if !outleft {
                                if grid.cells[(rr, cm)].colour != Black {
                                    grid.cells[(rr, cm)].colour = colour_diffs[0];
                                } else {
                                    grid.cells[(rr, cm)].colour = s.colour;
                                }
                            }
                        } else {
                            if cp < grid.cells.columns {
                                if grid.cells[(rr, cp)].colour != Black {
                                    grid.cells[(rr, cp)].colour = colour_diffs[0];
                                } else {
                                    grid.cells[(rr, cp)].colour = s.colour;
                                }
                            }
                        }
                    }
                    
                    if cm > 0 {
                        cm -= 1;
                    } else if cm == 0 {
                        outleft = true;
                    }
                    cp += 1;
                    r -= if r >= l { l } else { r };
                }

                cc.push(s.colour);
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2028, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // TODO : factor out gravity into grid function
        let func = |ex: &Example| {
            if !colour_diffs.is_empty() || ex.input.shapes.shapes.len() < 4 {
                return Grid::trivial();
            }

            let grid = &ex.input.grid;
            let mut shapes = ex.input.grid.to_shapes_sq();
            let mut new_shapes = shapes.clone_base();

            shapes.shapes.sort_by(|a, b| (a.ocol,a.orow).cmp(&(b.ocol,b.orow)));

            let mut posns: BTreeMap<usize,usize> = (0 .. grid.cells.rows)
                .map(|r| (r, grid.cells.columns))
                .collect();

            for s in shapes.shapes.iter().rev() {
                let c = if let Some(c) = posns.get(&s.orow) { *c } else { 0 };

                // find min over shape
                let mut cs = c;

                for r in s.orow .. s.orow + s.cells.rows {
                    if let Some(c) = posns.get(&r) {
                        cs = cs.min(*c);
                    };
                }

                if cs < s.cells.columns {
                    return Grid::trivial();
                }

                new_shapes.shapes.push(s.to_position(s.orow, cs - s.cells.columns));

                // now update position
                for r in s.orow .. s.orow + s.cells.rows {
                    let mut blanks = 0;
                    let grid = new_shapes.to_grid();

                    // Don't count blanks
                    for c2 in cs - s.cells.columns .. c {
                        if grid.cells[(r,c2)].colour != Black {
                            break;
                        }
                        blanks += 1;
                    }
                    *posns.entry(r).or_insert(cs) = cs + blanks - s.cells.columns;
                }
            }

            new_shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 2029, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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
                let mut shapes = ex.input.shapes.clone_base();

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

            if let Some(rule) = run_experiment(task, 1988, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let func = |ex: &Example| {
                let mut read_shapes = ex.input.shapes.clone();
                let mut shapes = ex.input.shapes.clone_base();
                let mut row = 0;
                let mut col = 0;

                read_shapes.shapes.sort_by(|a, b| a.ocol.cmp(&b.ocol));

                for s in read_shapes.shapes.iter() {
                    shapes.shapes.push(s.to_position(row, col));

                    row += s.cells.rows - 1;
                    col += s.cells.columns - 1;
                }

//shapes.to_grid().show();
                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 2009, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 2040, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 2075, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !ex.input.grid.is_square() || colour_diffs.len() != 1 || all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let (rs, cs) = ex.input.grid.dimensions();
            let r = (rs as f32).sqrt().abs() as usize;
            let c = (cs as f32).sqrt().abs() as usize;
            let grid_shapes = ex.input.grid.to_shapes_from_grid();
            let mut base = Shape::trivial();
            let mut shapes = grid_shapes.clone_base();

            for s in grid_shapes.shapes.iter() {
                if s.colour != all_colour_diffs[0] && s.colour != Black {
                    if base == Shape::trivial() {
                        base = s.clone().to_origin();
                    } else if base.equals(s) != Same {
                        return Grid::trivial();
                    }
                }
            }

            if base.size() < 4 {
                return Grid::trivial();
            }

            let shape = base.clone();

            for s in grid_shapes.shapes.iter() {
                if s.colour != all_colour_diffs[0] && s.colour != Black {
                    let pr = s.orow / r;
                    let pc = s.ocol / c;
                    let mut ns = s.clone();

                    ns.cells[(pr,pc)].colour = colour_diffs[0];
                    base.cells[(pr,pc)].colour = Black;
                    shapes.shapes.push(ns);
                }
            }

            for cell in base.cells.values() {
                if cell.colour != Black {
                    let pr = cell.row * r + cell.row;
                    let pc = cell.col * c + cell.col;
                    let mut ns = shape.clone();

                    ns.cells[(cell.row,cell.col)].colour = colour_diffs[0];
                    shapes.shapes.push(ns.to_position(pr, pc));
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 2224, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = FullyPopulatedOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 2083, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_max().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 2085, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.largest().to_grid(), output) { return Some(rule); };
//target.show();
//ans.show();

        if let Some(rule) = run_experiment(task, 2089, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_min().to_grid(), output) { return Some(rule); };

        let common_colours = examples.find_output_colours();

        for colour in common_colours {
            if let Some(rule) = run_experiment(task, 2094, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.cell_colour_cnts(colour).to_grid(), output) { return Some(rule); };
        }

        if let Some(rule) = run_experiment(task, 2097, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_largest_count().to_grid(), output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 2152, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 2185, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let ex = &examples.examples[0];
        let in_colour = ex.output.grid.get_diff_colour(&ex.input.grid);
        let out_colour = ex.input.grid.get_diff_colour(&ex.output.grid);

        let func = |ex: &Example| {
            ex.input.grid.recolour(in_colour, out_colour)
        };

        if let Some(rule) = run_experiment(task, 2195, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 2244, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        // 67b4a34d - merge
        if let Some(rule) = run_experiment(task, 2286, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        /* Subsumed by 1193!
        let func = |ex: &Example| {
            let shape = ex.input.shapes.largest_solid();
            let other = if shape.orow > shape.ocol {
                ex.input.grid.subgrid(shape.orow, shape.cells.rows, ex.input.grid.cells.columns - (shape.ocol + shape.cells.columns), shape.cells.columns).mirrored_cols()
            } else {
                ex.input.grid.subgrid(ex.input.grid.cells.rows - (shape.orow + shape.cells.rows), shape.cells.rows, shape.ocol, shape.cells.columns).mirrored_rows()
            };
//other.show();

            other
        };

        if let Some(rule) = run_experiment(task, 2301, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // 0934a4d8 - merge
        let func = |ex: &Example| {
            let shape = ex.input.shapes.largest_solid();
            let cs = ex.input.grid.col_skew();
            let rs = ex.input.grid.row_skew();

            let other = if shape.ocol < cs as usize {
                ex.input.grid.subgrid(0, shape.cells.columns, shape.orow, shape.cells.rows).rot_rect_270().mirrored_rows()
            } else if shape.orow < rs as usize {    // Needs checking
                ex.input.grid.subgrid(shape.orow, shape.cells.rows, 0, shape.cells.columns).rot_rect_90().mirrored_cols()
            } else if shape.orow < shape.ocol {
                ex.input.grid.subgrid(shape.orow, shape.cells.rows, ex.input.grid.cells.columns - (shape.ocol + shape.cells.columns) + cs as usize, shape.cells.columns).mirrored_cols()
            } else {
                ex.input.grid.subgrid(ex.input.grid.cells.rows - (shape.orow + shape.cells.rows) + rs as usize, shape.cells.rows, shape.ocol, shape.cells.columns).mirrored_rows()
            };
//other.show();

            other
        };

        if let Some(rule) = run_experiment(task, 2323, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        */

        let func = |ex: &Example| {
            if all_colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();
            for i in 0 .. 2 {
                let shapes = grid.find_colour_patches(all_colour_diffs[0]);

                for shape in shapes.shapes.iter() {
                    let g = if i == 0 {
                        let cpos = ex.input.grid.cells.columns - (shape.ocol + shape.cells.columns);
                        ex.input.grid.subgrid(shape.orow, shape.cells.rows, cpos, shape.cells.columns).mirrored_cols()

                    } else {
                        let rpos = ex.input.grid.cells.rows - (shape.orow + shape.cells.rows);
                        ex.input.grid.subgrid(rpos, shape.cells.rows, shape.ocol, shape.cells.columns).mirrored_rows()
                    };

                    grid.copy_to_position_mut(&g, shape.orow, shape.ocol);
                }
            }
            
            grid
        };

        if let Some(rule) = run_experiment(task, 2352, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if all_colour_diffs.is_empty() || !ex.input.grid.is_square() {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();
            let shapes = grid.find_colour_patches(all_colour_diffs[0]);
            let gr = grid.cells.rows;
            let gc = grid.cells.columns;

            for shape in shapes.shapes.iter() {
                for cell in shape.cells.values() {
                    if grid.cells[(cell.row,cell.col)].colour == shape.colour {
                        if grid.cells[(gr - cell.row - 1,cell.col)].colour != shape.colour {
                            grid.cells[(cell.row,cell.col)].colour = grid.cells[(gr - cell.row - 1,cell.col)].colour;
                        } else if grid.cells[(cell.row,gc - cell.col - 1)].colour != shape.colour {
                            grid.cells[(cell.row,cell.col)].colour = grid.cells[(cell.row,gc - cell.col - 1)].colour;
                        } else if grid.cells[(gr - cell.row - 1,gc - cell.col - 1)].colour != shape.colour {
                            grid.cells[(cell.row,cell.col)].colour = grid.cells[(gr - cell.row - 1,gc - cell.col - 1)].colour;
                        } else if grid.cells[(cell.col,cell.row)].colour != shape.colour {
                            grid.cells[(cell.row,cell.col)].colour = grid.cells[(cell.col,cell.row)].colour;
//                        } else {
//println!("--- {} {} {:?}", cell.row, cell.col, cell.colour);
                        }
                    }
                }
            }
//grid.show();
            
            grid
        };

        if let Some(rule) = run_experiment(task, 2386, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !ex.input.grid.is_square() {
                return Grid::trivial();
            }

            let cnt = ex.input.grid.cell_colour_cnt_map();
            let grid = ex.input.grid.scale_up(cnt.len());
            
            grid
        };

        if let Some(rule) = run_experiment(task, 2399, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if all_colour_diffs.is_empty() || !ex.input.grid.is_square() {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();
            let gr = grid.cells.rows;
            let gc = grid.cells.columns;
            let rs = ex.input.grid.row_skew();
            let cs = ex.input.grid.col_skew();
            let rr = gr + rs as usize - 1;
            let cr = gc + cs as usize - 1;

            if rs != cs {
                return Grid::trivial();
            }

            let shapes = grid.find_colour_patches(all_colour_diffs[0]);

            for shape in shapes.shapes.iter() {
                let new_shape = if cs > 0 && (shape.ocol as isize) < cs {
                    grid.populate_skew_edge_lr(&shape, all_colour_diffs[0])
                } else {
                    shape.clone()
                };
                let new_shape = if rs > 0 && (shape.orow as isize) < rs {
                    grid.populate_skew_edge_tb(&new_shape, all_colour_diffs[0])
                } else {
                    new_shape
                };

                for cell in shape.cells.values() {
                    let r = cell.row;
                    let c = cell.col;

                    if grid.cells[(cell.row,cell.col)].colour == shape.colour {
                        if r > gr / 2 && grid.cells[(rr - r,c)].colour != shape.colour {
                            grid.cells[(r,c)].colour = grid.cells[(rr - r,c)].colour;
                        } else if c > gr / 2 && grid.cells[(r,cr - c)].colour != shape.colour {
                            grid.cells[(r,c)].colour = grid.cells[(r,cr - c)].colour;
                        } else if rr >= r && cr >= c && rr - r < gr && cr - c < gc && grid.cells[(rr - r,cr - c)].colour != shape.colour {
                            grid.cells[(r,c)].colour = grid.cells[(rr - r,cr - c)].colour;
                        } else if grid.cells[(c,r)].colour != shape.colour {
                            if r + rs as usize >= grid.cells.rows || c + cs as usize >= grid.cells.columns {
                                return Grid::trivial();
                            }

                            grid.cells[(r,c)].colour = grid.cells[(r + rs as usize,c + cs as usize)].colour;
                        }
                    }
                }
                let shape = grid.subgrid(shape.orow, shape.cells.rows, shape.ocol, shape.cells.columns).as_shape();
                let grid = new_shape.copy_not_colour(&shape, all_colour_diffs[0]).to_grid();

//grid.show();
                return grid;
            }
            
            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 2471, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let dir = ex.output.shapes.border_gravity();

        let func = |ex: &Example| {
            if !all_colour_diffs.is_empty() || colour_common.len() < 3 {
                return Grid::trivial();
            }

            let conts = ex.input.shapes.group_containers();
            let mut shapes = ex.input.shapes.clone_base();

            for (k, v) in conts.iter() {
                let nk = k.recolour(Black, v[0].colour);
                let nk = if let Some(dir) = dir.get(&k.colour) {
                    match dir {
                        Up => nk.gravity_up_colour(k.colour),
                        Down => nk.gravity_down_colour(k.colour),
                        Left => nk.gravity_left_colour(k.colour),
                        Right => nk.gravity_right_colour(k.colour),
                        _ => todo!(),
                    }
                } else {
                    return Grid::trivial();
                };

                shapes.shapes.push(nk);
            }

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 2503, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let mut out_colour: Vec<_> = examples.examples[0].output.grid.cell_colour_cnt_map()
            .into_iter()
            .map(|(k,v)| (v,k))
            .collect();
        out_colour.sort();
        out_colour.reverse();
        let out_colour: Vec<_> = out_colour.into_iter().map(|(_,v)| v).collect();
        // testing function
        let func = |ex: &Example| {
            if colour_diffs.len() != 3 || out_colour.len() != 5 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone_base();

            for s in ex.input.shapes.shapes.iter() {
                if s.size() == ex.input.grid.size() {
                    shapes.shapes.push(s.clone());
                    continue;
                }
                if s.colour == out_colour[0] {
                    continue;
                }
                let mut s = s.add_hugging_border(colour_diffs[0]);

                s.flood_fill_border_mut(NoColour, out_colour[0]);
                s.recolour_mut(Black, out_colour[4]);

                if s.contains_colour(out_colour[4]) {
                    s.recolour_mut(out_colour[1], out_colour[3]);
                }

                shapes.shapes.push(s);
            }

//grid.show();
            shapes.to_grid_colour_transparent(out_colour[0])
        };

        if let Some(rule) = run_experiment(task, 2544, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let mut h: BTreeMap<(Colour, u32), usize> = BTreeMap::new();

            for s in ex.input.shapes.shapes.iter() {
                *h.entry((s.colour, Shape::sid(&s.cells, false))).or_insert(0) += 1;
            }

            let mut colour = NoColour;

            for s in ex.input.shapes.shapes.iter() {
                if let Some(cnt) = h.get(&(s.colour, Shape::sid(&s.cells, false))) {
                    if *cnt > 1 && s.size() > 1 {
                        colour = s.colour;
                        break;
                    }
                }
            }

            let colours = Uniq::uniq(&colour_common, all_colour_diffs.clone());
            let colours = Uniq::uniq(&colours, vec![colour]);

            if colours.len() != 1 {
                return Grid::trivial();
            }

            for s in ex.input.shapes.shapes.iter() {
                if let Some(cnt) = h.get(&(s.colour, Shape::sid(&s.cells, false))) {
                    if s.colour == colour && *cnt == 1 {
                        return s.recolour(Black, colours[0]).add_border(colours[0]).to_grid();
                    }
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 2586, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let mut bg = ex.input.grid.has_bg_grid_not_sq();

            if bg == NoColour { // is this necessary
                bg = Black;
            }

            let mut colour = NoColour;
            let mut min_r = usize::MAX;
            let mut min_c = usize::MAX;
            let mut max_r = 0;
            let mut max_c = 0;

            for s in ex.input.shapes.shapes.iter() {
                if s.is_full() {
                    colour = s.colour;
                }
            }

            for s in ex.input.shapes.shapes.iter() {
                if s.colour == colour {
                    min_r = min_r.min(s.orow);
                    min_c = min_c.min(s.ocol);
                    max_r = max_r.max(s.orow + s.cells.rows - 1);
                    max_c = max_c.max(s.ocol + s.cells.columns - 1);
                }
            }

            for s in ex.input.shapes.shapes.iter() {
                if s.colour != colour && s.orow >= min_r && s.orow <= max_r && s.ocol >= min_c && s.ocol <= max_c {
//s.recolour(Black, bg).to_grid().show();
                    return s.recolour(Black, bg).to_grid();
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 2626, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let bg = ex.output.grid.has_bg_grid_not_sq();

            if all_colour_diffs.len() != 1 || ex.input.grid.cells.rows < 4 || ex.input.grid.cells.columns < 4 {
                return Grid::trivial();
            }

            let colour1 = ex.input.grid.cells[(1,1)].colour;
            let colour2 = ex.input.grid.cells[(2,2)].colour;
            let colour3 = ex.input.grid.cells[(3,3)].colour;

            if colour1 == bg {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();

            for ((r, c), cell) in ex.input.grid.cells.items() {
                if cell.colour == all_colour_diffs[0] {
                    grid.cells[(r, c)].colour = bg;
                }
                let one = colour1 != bg && colour2 == bg && colour3 != bg;
                let two = colour1 != bg && colour2 == bg && colour3 == bg;
                let three = colour1 != bg && colour2 != bg && colour3 == bg;

                if one && r % 2 == 1 && c % 2 == 1 ||
                   two && r % 3 == 1 && c % 3 == 1 ||
                   three && (r % 3 == 1 || r % 3 == 2) && (c % 3 == 1 || c % 3 == 2) {
                    grid.cells[(r, c)].colour = colour1;
                }
            }
            
//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2664, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 3 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();

            for (i, s) in ex.input.shapes.shapes.iter().enumerate() {
                if s.size() == ex.input.grid.size() {
                    shapes.shapes.remove(i);
                }
            }

            shapes.shapes.sort_by(|a, b| b.size().cmp(&a.size()));

            let mut new_shapes = Shapes::new_sized(shapes.shapes[0].cells.rows, shapes.shapes[0].cells.columns);

            for s in shapes.shapes.iter() {
                let mut ns = s.to_origin();

                ns.force_recolour_mut(s.colour);

                new_shapes.shapes.push(ns);
            }

//new_shapes.to_grid().show();
            new_shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 2695, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() || colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let bg = ex.input.grid.majority_colour();

            let mut grid = ex.input.grid.clone();

            for s in ex.input.shapes.shapes.iter() {
                if s.dimensions() != ex.input.grid.dimensions() {
                    let (dir, r, c) = s.has_border_break();

                    if s.orow == ex.input.grid.cells.rows - 1 || s.ocol == ex.input.grid.cells.columns - 1 {
                        return Grid::trivial();
                    }

                    grid.draw_bg_mut(dir, r, c, colour_diffs[0], bg);
                    grid.flood_fill_bg_mut(s.orow + 1, s.ocol + 1, NoColour, bg, colour_diffs[0]);
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2723, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let mut grid = ex.input.grid.clone();

            for r in 0 .. grid.cells.rows {
                for c in 0 .. grid.cells.columns {
                    if grid.cells[(r,c)].colour == grid.colour && c > 0 && grid.cells[(r,c-1)].colour == Black {
                        grid.cells[(r,c-1)].colour = grid.colour;
                        break;
                    }
                }
                for c in (0 .. grid.cells.columns).rev() {
                    if grid.cells[(r,c)].colour == grid.colour {
                        grid.cells[(r,c)].colour = Black;
                        break;
                    }
                }
            }

            grid.recolour_mut(Black, colour_diffs[0]);

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2752, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !all_colour_diffs.is_empty()  || ex.input.shapes.len() < 3 || ex.input.shapes.len() > 20 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();

            shapes.shapes.sort_by(|a, b| a.size().cmp(&b.size()));

            let mut size = 0;

            for s in shapes.shapes.iter() {
                if size == 0 {
                    size += s.cells.rows.max(s.cells.columns);
                } else {
                    size += 2;
                }
            }

            let mut grid = Grid::new(size, size, Black);

            shapes.shapes.sort_by(|a, b| b.size().cmp(&a.size()));

            for (i, s) in shapes.shapes.iter().enumerate() {
                for r in i .. size {
                    for c in i .. size {
                        grid.cells[(r, c)].colour = s.colour;
                    }
                }
                size -= 1;
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2790, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !all_colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let bg_colour = ex.input.shapes.largest().colour;
            let mut largest = ex.input.coloured_shapes.largest().to_grid();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.dimensions() != largest.dimensions() {
                    let colour = s.minority_colour();
                    let (r, c) = s.to_origin().find_colour_pixel_coords(colour);
                    let (lr, lc) = largest.find_colour_pixel_coords(colour);

                    if lr > 0 || lc > 0 {
                        if lr < r || lc < c {
                            return Grid::trivial();
                        }

                        largest.copy_shape_to_grid_position_mut(s, lr - r, lc - c);
                    }
                }
            }

            largest.recolour_mut(Black, bg_colour);

//largest.show();
            largest
        };

        if let Some(rule) = run_experiment(task, 2818, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let pred = |solver: &Grid, r, c, colour| !solver.used_in_row(r, colour) && !solver.used_in_col(c, colour);

            let mut grid = ex.input.grid.clone();

            grid.solve(&pred);

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2819, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 2842, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // Also done by 142
        let func = |ex: &Example| {
            if ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            let shapes = ex.input.grid.to_shapes_cons();
            let mut new_shapes = shapes.clone_base();

            for s in shapes.shapes.iter() {
                let ns = s.mirrored_r();

                new_shapes.shapes.push(ns);
            }

            new_shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 2862, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 2890, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let bg = ex.input.grid.max_colour();
            let shapes = ex.input.grid.to_shapes_sq();
            let shapes = shapes.colour_groups_to_shapes(bg);

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 2900, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 2922, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Div9Out;
    if all || cat.contains(&Is3x3In) && cat.contains(&gc) && !cat.contains(&Div9In){ 
        *cap_cats.entry(gc).or_insert(0) += 1;

        let func = |ex: &Example| {
            if colour_common.len() != 9 || !ex.input.grid.cells.is_square() {
                return Grid::trivial();
            }
            let div = ex.input.grid.cells.rows.isqrt();
            let mut cells: BTreeMap<(usize,usize),Colour> = BTreeMap::new();

            for cell in ex.input.grid.cells.values() {
                if cell.colour != Black {
                    cells.insert((cell.row / div, cell.col), cell.colour);
                }
            }
            let mut grid = Grid::new(3, 3, Black);

            for (cell, colour) in grid.cells.values_mut().zip(cells.values()) {
                cell.colour = *colour;
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 2923, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Div9In;
    if all || cat.contains(&gc) && cat.contains(&Is3x3Out) && !cat.contains(&Div9Out) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        //-if let Some(rule) = run_experiment_examples(&file, 1030, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return Some(rule); };

        //-if let Some(rule) = run_experiment_examples(&file, 1040, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_double(&exs), output) { return Some(rule); };

        //-if let Some(rule) = run_experiment_examples(&file, 1050, experiment, trans, is_test, examples, &targets, done, tries, &|exs| cat_expand_3x3(&exs), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 2943, experiment, trans, is_test, examples, &targets, done, tries, &|ex| repeat_pattern(&ex, Black), output) { return Some(rule); };

        let func = |ex: &Example| {
            let colour = if ex.input.grid.colour == Mixed {
                ex.input.grid.find_max_colour()
            } else {
                ex.input.grid.colour
            };

            repeat_pattern(ex, colour)
        };

        if let Some(rule) = run_experiment(task, 2955, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let colour = if ex.input.grid.colour == Mixed {
                ex.input.grid.find_min_colour()
            } else {
                ex.input.grid.colour
            };

            repeat_pattern(ex, colour)
        };

        if let Some(rule) = run_experiment(task, 2967, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if let Some(rule) = run_experiment_colours(task, 2969, experiment, is_test, examples, &targets, done, &|ex, colour| repeat_pattern(ex, colour), output) { return Some(rule); };

        /*
        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }
            let colour = if ex.input.grid.colour == Mixed {
                ex.input.grid.find_max_colour()
            } else {
                ex.input.grid.colour
            };
            let shape = ex.input.grid.as_shape();
            let (rs, cs) = ex.input.grid.dimensions();

            let shapes = shape.chequer(rs, cs, &|r,c| ex.input.grid.cells[(r / rs,c / cs)].colour == colour, &|s| s.clone(), true);

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 2989, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }
            let colour = if ex.input.grid.colour == Mixed {
                ex.input.grid.find_min_colour()
            } else {
                ex.input.grid.colour
            };
            let shape = ex.input.grid.as_shape();
            let (rs, cs) = ex.input.grid.dimensions();

            let shapes = shape.chequer(rs, cs, &|r,c| ex.input.grid.cells[(r / rs,c / cs)].colour == colour, &|s| s.clone(), true);

            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 3008, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if examples.examples[0].input.grid.size().pow(2) == examples.examples[0].output.grid.size() {
            let func = |ex: &Example| {
                let (colour, _) = ex.input.grid.as_shape().colour_cnt(false);

                repeat_pattern(ex, colour)
            };

            if let Some(rule) = run_experiment(task, 3017, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let shapes_out = examples.examples[0].output.coloured_shapes.shapes.len();
            let h = examples.examples[0].input.grid.cell_colour_cnt_map();
            let colours: Vec<Colour> = h.iter().filter(|(_, &v)| v == shapes_out).map(|(&k,_)| k).collect();

            let func = |ex: &Example| {
                if colours.len() != 1 {
                    return Grid::trivial();
                }

                repeat_pattern(ex, colours[0])
            };

            if let Some(rule) = run_experiment(task, 3031, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }
        */
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

        if let Some(rule) = run_experiment(task, 3129, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 3144, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 3175, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 3194, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 3218, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 3268, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 3371, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 3 || ex.input.shapes.shapes.len() % 2 == 0 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone();
            let mut rs: Vec<&Shape> = Vec::new();
            let mut ss: Vec<&Shape> = Vec::new();

            for s in ex.input.shapes.shapes.iter() {
                if s.size() == 1 {
                    rs.push(&s);
                }
            }

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.size() > 1 && !s.has_border() {
                    ss.push(&s);
                }
            }

            //rs.sort();
            //ss.sort();

            for (s, m) in rs.iter().zip(ss.iter()) {
                shapes.shapes.push(m.recolour(m.colour, s.colour));
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 3405, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let out_colours = examples.examples[0].output.grid.cell_colour_cnt_map();
        let func = |ex: &Example| {
            if all_colour_diffs.len() != 2 || out_colours.len() != 3 {
                return Grid::trivial();
            }
            let mut colour_order: Vec<(usize,Colour)> = out_colours.iter().map(|(k, v)| (*v, *k)).collect();
            colour_order.sort();
            let colours: Vec<Colour> = colour_order.iter().map(|(_, c)| *c).collect();
            let mut shapes = ex.input.shapes.clone();

            for s in shapes.shapes.iter_mut() {
                for i in 1 ..= s.cells.rows.max(s.cells.columns) / 2 {
                    let colour = colours[if i % 2 == 0 { 0 } else { 1 }];
                    s.nest_mut(i, colour);
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 3406, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Is3x3In;
    if all || cat.contains(&gc) && cat.contains(&Is3x3Out) { 
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

        if let Some(rule) = run_experiment(task, 3486, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.shapes.shapes.len() < 5 {
                return Grid::trivial();
            }

            //let mut one = Shape::trivial();
            let mut two = Shape::trivial();
            let mut three = Shape::trivial();
            let mut other = Shape::trivial();
            let mut shapes = Shapes::trivial();
            let mut rows = 0;

            for (i, s) in ex.input.shapes.shapes.iter().enumerate() {
                if s.ocol == 0 && s.cells.columns == ex.input.grid.cells.columns {
                    if s.orow < 1 || ex.input.grid.cells.rows < s.orow - 1 {
                        return Grid::trivial();
                    }
                    rows = ex.input.grid.cells.rows - s.orow - 1;
                    shapes = Shapes::new_sized(rows, ex.input.grid.cells.columns);
                    continue;
                }

                match i {
                    //0 => one = s.clone(),
                    0 => (),
                    1 => two = s.clone(),
                    2 => three = s.clone(),
                    _ => {
                        other = s.clone();

                        break
                    },
                }
            }

            if other == Shape::trivial() || shapes == Shapes::trivial() || two == Shape::trivial() {
                return Grid::trivial();
            }

            shapes.shapes.push(other.to_position(other.orow - (ex.input.grid.cells.rows - rows), other.ocol));

            if two.cells[(0,0)].colour == Black {
                shapes.shapes.push(three.to_position(other.orow - (ex.input.grid.cells.rows - rows) - other.cells.rows, other.ocol));
            } else {
                shapes.shapes.push(three.to_position(other.orow - (ex.input.grid.cells.rows - rows) + other.cells.rows, other.ocol));
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 3487, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 3520, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.len() > 20 {
                    return Grid::trivial();
                }

                let mut shapes = ex.input.shapes.clone_base();

                //for s in &ex.input.shapes.consolidate_shapes().shapes {
                for s in &ex.input.shapes.shapes {
                    shapes.shapes.push(s.mirrored_r());
                }
//shapes.to_grid().show();

                shapes.to_grid()
            };

            // evaluation only
            if let Some(rule) = run_experiment(task, 3539, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        if all || cat.contains(&InLessCountOut) {
            // see 045e512c.json TODO MORE
            let s1 = examples.all(false);
            let s2 = examples.all_coloured(true);

            let func = |ex: &Example| {
                let sc: BTreeMap<Shape, Shape> = s1.iter().zip(s2.iter()).map(|(s1, s2)| (s1.to_origin(), s2.to_origin())).collect();

                let mut shapes = ex.input.shapes.clone_base();

                for s in ex.input.shapes.shapes.iter() {
                    match sc.get(&s.to_origin()) {
                        Some(ns) => {
                            shapes.shapes.push(ns.to_position(s.orow, s.ocol));
                        },
                        None => (),
                    }
                }

                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 3568, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            /*
            // see 626c0bcc.json    map_coloured_shapes_to_shape
            let s1: Vec<Shape> = s2.iter().map(|s| s.recolour(s.colour, s1[0].colour)).collect();
s1.iter().for_each(|s| s.show());

            let func = |ex: &Example| {
                let sc: BTreeMap<Shape, Shape> = s1.iter().zip(s2.iter()).map(|(s1, s2)| (s1.to_origin(), s2.to_origin())).collect();

                let mut shapes = ex.input.shapes.clone_base();

                for s in ex.input.shapes.shapes.iter() {
                    match sc.get(&s.to_origin()) {
                        Some(ns) => {
                            shapes.shapes.push(ns.to_position(s.orow, s.ocol));
                        },
                        None => (),
                    }
                }

                shapes.to_grid()
            };

            if let Some(rule) = run_experiment(task, 3569, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            */
        }

        // 0ca9ddb6 4258a5f9 913fb3ed 95990924 b60334d2 test
        if let Some(rule) = run_experiment_tries(task, 3572, experiment, trans, is_test, examples, &targets, done, tries, &|ex, _, n| transform_only(ex, n), output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 3588, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 3623, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 3647, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 3686, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let min_colour = examples.examples[0].output.grid.minority_colour();

        let func = |ex: &Example| {
            if colour_diffs.len() != 2 {
                return Grid::trivial();
            }

            let mut other_colour = colour_diffs.clone();

            other_colour.retain(|&x| x != min_colour);

            let bg = ex.input.grid.colour;
            let (min_r, min_c, max_r, max_c) = ex.input.shapes.corners();
            let mut grid = ex.input.grid.clone();

            // Inside box is simple
            for r in min_r .. max_r {
                for c in min_c .. max_c {
                    if grid.cells[(r,c)].colour == Black {
                        grid.cells[(r,c)].colour = other_colour[0];
                    }
                }
            }

            // Outside box is harder
            for r in min_r .. max_r {
                for c in min_c .. max_c {
                    if grid.cells[(r,c)].colour == other_colour[0] {
                        for rr in r .. grid.cells.rows {
                            if grid.cells[(rr,c)].colour == bg {
                                break;
                            } else if grid.cells[(rr,c)].colour != other_colour[0] {
                                grid.cells[(rr,c)].colour = min_colour;
                            }
                        }
                        for cc in c .. grid.cells.columns {
                            if grid.cells[(r,cc)].colour == bg {
                                break;
                            } else if grid.cells[(r,cc)].colour != other_colour[0] {
                                grid.cells[(r,cc)].colour = min_colour;
                            }
                        }
                        for rr in (0 ..= r).rev() {
                            if grid.cells[(rr,c)].colour == bg {
                                break;
                            } else if grid.cells[(rr,c)].colour != other_colour[0] {
                                grid.cells[(rr,c)].colour = min_colour;
                            }
                        }
                        for cc in (0 ..= c).rev() {
                            if grid.cells[(r,cc)].colour == bg {
                                break;
                            } else if grid.cells[(r,cc)].colour != other_colour[0] {
                                grid.cells[(r,cc)].colour = min_colour;
                            }
                        }
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 3752, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() != 1 {
                return Grid::trivial();
            }

            let ccm = ex.input.coloured_shapes.shapes[0].cell_colour_cnt_map();
            let mut grid = ex.input.grid.clone();
            let div_colour = if let Some((_, colour)) = ccm.iter().map(|(c,n)| (n, c)).max() {
                *colour
            } else {
                NoColour
            };
            if let Some((_, colour)) = ccm.iter().filter(|&(&c, _)| c != div_colour).map(|(c,n)| (n, c)).max() {
                grid.cells[(ex.input.grid.cells.rows-1, ex.input.grid.cells.columns / 2)].colour = *colour;
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 3774, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let shapes = ex.input.grid.to_shapes_coloured_sq();
            let ss = &shapes.shapes;

            if !all_colour_diffs.is_empty() || ss.len() != 9 {
                return Grid::trivial();
            }

            let mut ns = shapes.clone_base();
            let ps = &mut ns.shapes;

            for s in shapes.shapes.iter() {
                match s.pixel_position(ex.input.grid.minority_colour()) {
                    UpLeft => ps.push(s.to_position(ss[0].orow, ss[0].ocol)),
                    Up => ps.push(s.to_position(ss[1].orow, ss[1].ocol)),
                    UpRight => ps.push(s.to_position(ss[2].orow, ss[2].ocol)),
                    Left => ps.push(s.to_position(ss[3].orow, ss[3].ocol)),
                    Middle => ps.push(s.to_position(ss[4].orow, ss[4].ocol)),
                    Right => ps.push(s.to_position(ss[5].orow, ss[5].ocol)),
                    DownLeft => ps.push(s.to_position(ss[6].orow, ss[6].ocol)),
                    Down => ps.push(s.to_position(ss[7].orow, ss[7].ocol)),
                    DownRight => ps.push(s.to_position(ss[8].orow, ss[8].ocol)),
                    _ => ()
                }
            }

//ns.to_grid().show();
            ns.to_grid()
        };

        if let Some(rule) = run_experiment(task, 3775, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let colour = ex.input.grid.minority_colour();
            //let mut shapes = ex.input.shapes.clone();
            let mut shapes = ex.input.grid.to_shapes();
            let mut idxes: Vec<Shape> = Vec::new();

            for s in shapes.shapes.iter() {
                if s.colour == colour {
                    idxes.push(s.clone());
                }
            }

            shapes.shapes.insert(0, ex.input.grid.as_shape());

            for s in shapes.shapes.iter_mut() {
                for idx in idxes.iter() {
                    if s.colour != colour && s.equal_shape(&idx) {
                        s.recolour_mut(s.colour, idx.colour);
                    }
                }
            }

//shapes.to_grid_transparent().show();
            shapes.to_grid_transparent()
        };

        if let Some(rule) = run_experiment(task, 3776, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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
        if let Some(rule) = run_experiment(task, 3810, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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
        if let Some(rule) = run_experiment(task, 3837, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = &|ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() != 1 || !ex.input.coloured_shapes.shapes[0].is_square() || ex.input.coloured_shapes.shapes[0].size() != 36 {
                return Grid::trivial();
            }

            let shape = &ex.input.coloured_shapes.shapes[0];
            let ss = shape.subshape(0, 3, 0, 3);

            ss.to_grid()
        };

        if let Some(rule) = run_experiment(task, 3850, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

                let mut shapes = ex.input.shapes.clone_base();

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

            if let Some(rule) = run_experiment(task, 3897, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 3966, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 4008, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 4058, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4082, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4124, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4164, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4212, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() || ex.input.shapes.shapes.len() < 3 {
                return Grid::trivial();
            }

            let rep = ex.input.shapes.shapes[0].cells.rows;
            let mut grid = ex.input.grid.clone();
            let (_, _, max_r, _) = grid.corners();

            for i in (max_r .. grid.cells.rows - rep).step_by(rep) {
                for r in 0 .. rep {
                    for c in 1 .. grid.cells.columns {
                        grid.cells[(i + r + 1,c)].colour = grid.cells[(r,c)].colour;
                    }
                }
            }

//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 4236, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let mut shapes = ex.input.grid.to_shapes_sq();

            for s in shapes.shapes.iter_mut() {
                if s.size() < 3 {
                    s.force_recolour_mut(colour_diffs[0]);
                }
            }

//shapes.to_grid_transparent().show();
            shapes.to_grid_transparent()
        };

        if let Some(rule) = run_experiment(task, 4254, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let mut grid = ex.input.grid.clone();

            for c in 0 .. grid.cells.columns {
                if grid.cells[(grid.cells.rows-1,c)].colour != Black {
                    let mut colour = grid.cells[(grid.cells.rows-1,c)].colour;

                    for r in (0 .. grid.cells.rows).rev() {
                        let col = grid.cells[(r,c)].colour;

                        if col != Black && col != colour {
                            colour = col;
                        } else if col == Black {
                            grid.cells[(r,c)].colour = colour;
                        }
                    }
                }
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 4278, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if examples.examples[0].input.grid.cells.columns != examples.examples[0].input.grid.cells.rows {
            let n = examples.examples[0].input.grid.cells.columns / examples.examples[0].input.grid.cells.rows;
            let s_to_s = examples.split_n_map_horizontal(n);

            let func = |ex: &Example| {
                let gc = &ex.input.grid.cells;

                if gc.columns % gc.rows != 0 || s_to_s.is_empty() {
                    return Grid::trivial();
                }
                let mut grid = ex.input.grid.clone();
                let reps = grid.split_n_horizontal(n);

                for (i, s) in reps.iter().enumerate() {
                    let os = s.to_origin();

                    if let Some(ns) = s_to_s.get(&os) {
                        grid.copy_to_position_mut(&ns, 0, i * s.cells.columns);
                    }
                }

                grid
            };

            if let Some(rule) = run_experiment(task, 4304, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let n = examples.examples[0].input.grid.cells.rows / examples.examples[0].input.grid.cells.columns;
            let s_to_s = examples.split_n_map_vertical(n);
            let func = |ex: &Example| {
                let gc = &ex.input.grid.cells;

                if gc.rows % gc.columns != 0 || s_to_s.is_empty() {
                    return Grid::trivial();
                }
                let mut grid = ex.input.grid.clone();
                let reps = grid.split_n_vertical(n);

                for (i, s) in reps.iter().enumerate() {
                    let os = s.to_origin();

                    if let Some(ns) = s_to_s.get(&os) {
                        grid.copy_to_position_mut(&ns, i * s.cells.rows, 0);
                    }
                }

                grid
            };

            if let Some(rule) = run_experiment(task, 4328, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

        let func = |ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() != 3 {
                return Grid::trivial();
            }

            let cnt = ex.input.coloured_shapes.shapes[0].size();
            let r = ex.input.coloured_shapes.shapes[1].orow;

            if cnt > r {
                return Grid::trivial();
            }

            let cc = ex.input.coloured_shapes.shapes[2].cell_colour_cnt_map();
            let mut grid = ex.input.grid.clone();

            for cell in ex.input.coloured_shapes.shapes[2].cells.values() {
                if let Some(n) = cc.get(&cell.colour) {
                    if *n == cnt {
                        for i in 1 ..= cnt {
                            grid.cells[(r - i, cell.col)].colour = cell.colour;
                        }
                    }
                }
            }

            grid
        };

        if let Some(rule) = run_experiment(task, 4359, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() < 2 {
                return Grid::trivial();
            }
            let mut shapes = ex.input.coloured_shapes.clone();
            let mut gr = 0;
            let mut gc = 0;
            let mut colour = NoColour;

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.size() == 1 {
                    gr = s.orow;
                    gc = s.ocol;
                    colour = s.colour;
                }
            }

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.size() > 1 {
                    for ((r, c), cell) in s.cells.items() {
                        if gr < r || gc < c {
                             return Grid::trivial();
                        }
                        if cell.colour == colour {
                            shapes.shapes.push(s.to_position(gr - r, gc - c).recolour(colour, Black));
                        }
                    }
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 4395, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 || ex.input.grid.height() < 8 || ex.input.grid.width() < 8 {
                return Grid::trivial();
            }
            let mut grid = ex.input.grid.clone();

            grid.colour_squares(colour_diffs[0]);

            grid
        };

        if let Some(rule) = run_experiment(task, 4408, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // aa18de87 but not 3490cc26
        let func = |ex: &Example| {
            if colour_diffs.len() != 1 || ex.input.shapes.shapes.len() != ex.input.coloured_shapes.shapes.len() {
                return Grid::trivial();
            }
            let mut grid = ex.input.grid.clone();

            grid.connect_dots_colour_pairs(colour_diffs[0]);

            grid
        };

        if let Some(rule) = run_experiment(task, 4422, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let mut shapes = ex.input.shapes.clone();

            for s in ex.input.shapes.shapes.iter() {
                if !s.is_pixel() {
                    let pixels = ex.input.shapes.pixels_in_shapes(&s);

                    if pixels.is_empty() {
                        return Grid::trivial();
                    }

                    let colour = pixels[0].colour;
                    let n = pixels.len();

                    if n == 0 || s.orow < n || s.ocol < n {
                        return Grid::trivial();
                    }

                    let ns = Shape::new_sized_coloured_position(s.orow - n, s.ocol - n, s.cells.rows + n * 2, s.cells.columns + n * 2, colour);

                    shapes.shapes.insert(0, ns);
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 4423, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // Assumes this is true for first test?
        let min_colour = examples.examples[0].input.grid.minority_colour();
        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            let bg = ex.input.grid.majority_colour();
            let all_colours = ex.input.grid.cell_colours();
            let colours = Uniq::uniq(&all_colours, vec![min_colour, bg]);
            let diag_colour = if colours.is_empty() {
                NoColour
            } else {
                colours[0]
            };
            let mut grid = ex.input.grid.clone();

            for s1 in ex.input.shapes.shapes.iter() {
                for s2 in ex.input.shapes.shapes.iter() {
                    if s1 != s2 && s1.is_diagonal(s2) {
                        let dir = s1.which_direction(s2);

                        if s1.colour == s2.colour && s1.colour == min_colour {
                            let r = if s1.orow == 0 { 0 } else { 1 };
                            let c = if s1.ocol == 0 { 0 } else { 1 };

                            match dir {
                                DownRight => grid.draw_bg_mc_term_other_mut(dir, s1.orow + 1, s1.ocol + 1, s1.colour, bg, false, true, diag_colour),
                                DownLeft => grid.draw_bg_mc_term_other_mut(dir, s1.orow + 1, s1.ocol - c, s1.colour, bg, false, true, diag_colour),
                                UpRight => grid.draw_bg_mc_term_other_mut(dir, s1.orow - r, s1.ocol + 1, s1.colour, bg, false, true, diag_colour),
                                UpLeft => grid.draw_bg_mc_term_other_mut(dir, s1.orow - r, s1.ocol - c, s1.colour, bg, false, true, diag_colour),
                                _ => (),
                            }
                        }
                    }
                }
            }

            for s1 in ex.input.shapes.shapes.iter() {
                for s2 in ex.input.shapes.shapes.iter() {
                    if s1 != s2 && s1.is_diagonal(s2) {
                        let dir = s1.which_direction(s2);

                        if s1.colour != s2.colour && s1.colour != min_colour {
                            let r = if s1.orow == 0 { 0 } else { 1 };
                            let c = if s1.ocol == 0 { 0 } else { 1 };
                            let dir_rot = dir.rot();

                            match dir_rot {
                                DownRight => 
                                    grid.draw_bg_mut(dir_rot, s1.orow + 1, s1.ocol + 1, s1.colour, bg),
                                DownLeft => 
                                    grid.draw_bg_mut(dir_rot, s1.orow + 1, s1.ocol - c, s1.colour, bg),
                                UpRight => 
                                    grid.draw_bg_mut(dir_rot, s1.orow - r, s1.ocol + 1, s1.colour, bg),
                                UpLeft => 
                                    grid.draw_bg_mut(dir_rot, s1.orow - r, s1.ocol - c, s1.colour, bg),
                                _ => (),
                            }
                        }
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 4424, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let sam = if colour_common.len() != 2 || examples.examples[0].input.shapes.shapes.len() % 2 != 0 {
            BTreeMap::new()
        } else {
            // This is expensive
            examples.shape_adjacency_map()
        };

        let func = |ex: &Example| {
            if sam.is_empty() || colour_common.len() != 2 || ex.input.shapes.shapes.len() % 2 != 0 {
                return Grid::trivial();
            }

            let mut shapes = ex.input.shapes.clone_base();

            for (s, colour) in sam.iter() {
                let es = s.find_equal_shape(&ex.input.shapes);
                let touching = es.find_touching(&ex.input.shapes);

                if touching != Shape::trivial() {
                    shapes.shapes.push(touching.recolour(touching.colour, *colour));
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 4425, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4453, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            let mut copy = shapes.clone_base();

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

    if let Some(rule) = run_experiment(task, 4515, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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
            ex.input.grid.as_shape().chequer(rs, cs, &|r,_| r == ex.input.grid.cells.rows, &|s| s.mirrored_c(), false).to_grid()
        };

        if let Some(rule) = run_experiment(task, 4533, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            // TODO: Derive the predicate and function before hand
            let shape = ex.input.grid.as_shape();

            shape.chequer(rs, cs, &|r,c| shape.cells[(r / rs,c / cs)].colour != Black, &|s| s.invert_colour(), true).to_grid()
        };

        if let Some(rule) = run_experiment(task, 4542, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let shape = ex.input.grid.as_shape();

            let has_band = |r,c| {
                match ex.input.shapes.has_band() {
                    (Down, pos) => r == pos * rs,
                    (Right, pos) => c == pos * cs,
                    _ => false,
                }
            };

            shape.chequer(rs, cs, &has_band, &|s| s.clone(), true).to_grid()
        };

        if let Some(rule) = run_experiment(task, 4558, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let shapes = examples.examples[0].output.grid.template_shapes(&examples.examples[0].input.grid);

        if !shapes.shapes.is_empty()  {
            if let Some(rule) = run_experiment(task, 4563, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.grid.fill_template(&shapes.shapes[0]), output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 4586, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4613, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            shape.chequer(rs, cs, &|_,_| true, &|s| s.clone(), false).to_grid()
        };

        if let Some(rule) = run_experiment(task, 4638, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let g = &ex.input.grid;

            if ex.input.shapes.shapes.is_empty() || !cat.contains(&InLessThanOut) || out_rs % g.height() != 0 || out_cs % g.width() != 0 {
                return Grid::trivial();
            }

            let grid = if g.cells.rows > g.cells.columns {
                g.as_shape().chequer(1, cs, &|r,_| r % (g.cells.rows * 2) == 0, &|s| s.mirrored_c(), false).to_grid()
            } else {
                g.as_shape().chequer(1, cs, &|_,c| c % (g.cells.columns * 2) == 0, &|s| s.mirrored_c(), false).to_grid()
            };

            grid
        };

        if let Some(rule) = run_experiment(task, 4656, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let shape = ex.input.grid.as_shape().recolour(ex.input.grid.colour, Black).add_border(ex.input.grid.colour);
            let (in_rs, in_cs) = shape.dimensions();
            let rs = (out_rs + 4) / in_rs;
            let cs = (out_cs + 4) / in_cs;
            let grid = Grid::new(in_rs, in_cs, ex.input.grid.colour);

            grid.as_shape().chequer(rs, cs, &|_,_| true, &|_| shape.clone(), false).to_grid().trim(out_rs, out_cs)
        };

        if let Some(rule) = run_experiment(task, 4672, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4693, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4735, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let grid = &ex.input.grid;
            let toddle = grid.toddle_colour(Black, grid.colour);

            toddle.as_shape().chequer(rs, cs, &|r,c| toddle.cells[(r / rs,c / cs)].colour != Black, &|s| s.clone(), true).to_grid()
        };

        if let Some(rule) = run_experiment(task, 4748, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) {
                return Grid::trivial();
            }

            let grid = &ex.input.grid;
            let cnt = grid.cell_colour_cnt_map().len();

            grid.as_shape().chequer(cnt, cnt, &|_,_| true, &|s| s.clone(), true).to_grid()
        };

        if let Some(rule) = run_experiment(task, 4761, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let colour = examples.examples[0].input.grid.max_colour();
        let colour2 = examples.examples[1].input.grid.max_colour();
        
        // testing function
        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || colour != colour2 {
                return Grid::trivial();
            }

            let grid = &ex.input.grid;
            let (rs, cs) = grid.dimensions();
            let grid = grid.as_shape().chequer(rs, cs, &|r,c| grid.cells[(r/ rs,c / cs)].colour == colour, &|s| s.clone(), true).to_grid();

//grid.show();
            grid.clone()
        };

        if let Some(rule) = run_experiment(task, 4780, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            let grid = in_shape.chequer(rs, cs, &|r,c| shapes.shapes[1].cells[(r / (rs / 2), c / (cs / 2))].colour != Black, &|s| s.clone(), true).to_grid();

//grid.show();
            grid.clone()
        };

        if let Some(rule) = run_experiment(task, 4808, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || !colour_diffs.is_empty() {
                return Grid::trivial();
            }
            let shape = ex.input.grid.as_shape();

            let grid = shape.chequer(rs, cs, &|_,c| c % (in_rs * 2) != 0, &|s| s.mirrored_c(), false).to_grid();

            grid.clone()
        };

        if let Some(rule) = run_experiment(task, 4821, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4878, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            shape.combined_chequer(rs, cs, &action).to_grid()
        };

        if let Some(rule) = run_experiment(task, 4904, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(file, 819, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        */

        if let Some(rule) = run_experiment(task, 4926, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.grid.extend_border(), output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 4947, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&InLessThanOut) || colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let rows = ex.input.grid.cells.rows;
            let cols = ex.input.grid.cells.columns;
            let mut grid = ex.input.grid.as_shape().chequer(rs, cs, &|r,c| r == 0 && c == 0 || r == rows && c == cols , &|s| s.clone(), true).to_grid();
            let shapes = grid.to_shapes();

            for s in shapes.shapes.iter().skip(1).step_by(2) {
                grid.draw_mut(Right, s.orow - 1, 0, colour_diffs[0]);
            }
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 4967, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 4969, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.grid.scale_up(ex.input.grid.height()), output) { return Some(rule); };

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

            if let Some(rule) = run_experiment_tries(task, 4988, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            let func = |gi: &Example| {
                if gi.input.coloured_shapes.shapes.is_empty() {
                    return Grid::trivial();
                }

                gi.input.coloured_shapes.shapes[0].to_origin().to_grid()
            };

            if let Some(rule) = run_experiment(task, 4998, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

                if let Some(rule) = run_experiment(task, 5029, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            }
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = OutLessThanIn;
    // Default shape parser sometimes gets it wrong!
    if all || cat.contains(&gc) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 5040, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_unique().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 5042, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.first().to_grid(), output) { return Some(rule); };

        //if let Some(rule) = run_experiment(task, 5044, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.last().to_grid(), output) { return Some(rule); };

        let func = |ex: &Example| {
            for s in &ex.input.shapes.shapes {
                if !s.is_mirror_r() && !s.is_mirror_c() {
                    return s.to_grid();
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 5056, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            for s in &ex.input.shapes.shapes {
                if s.is_mirror_r() || s.is_mirror_c() {
                    return s.to_grid();
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 5068, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let bordered = ex.input.shapes.border_only();

            if bordered == Shape::trivial() {
                return Grid::trivial();
            }

            ex.input.grid.subgrid(bordered.orow + 1, bordered.cells.rows - 2, bordered.ocol + 1, bordered.cells.columns - 2)
        };

        if let Some(rule) = run_experiment(task, 5080, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !cat.contains(&OutLessThanIn) {
                return Grid::trivial();
            }

            let sc = ex.input.coloured_shapes.shape_counts();

            if let Some(max) = sc.values().max() {
                let sid: Vec<_> = sc.iter().filter(|&(_,&v)| v == *max).map(|(k,_)| k).collect();
                for s in ex.input.coloured_shapes.shapes.iter() {
                    let this_sid = Shape::sid(&s.cells, true);

                    if *sid[0] == this_sid {
                        return s.to_grid();
                    }
                }
            }

            Grid::trivial()
        };

        if let Some(rule) = run_experiment(task, 5103, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let colour = ex.input.grid.mid_div_colour();
            let ns = ex.input.grid.to_shapes_base_bg(colour);

            if ns.shapes.len() != 2 {
                return Grid::trivial();
            }

            let mut nns = Shapes::new_sized(ns.shapes[0].cells.rows, ns.shapes[0].cells.columns);

            for s in ns.shapes.clone().iter() {
                nns.shapes.push(s.to_origin());
            }

            if nns.to_grid_transparent().is_full() {  
                nns.to_grid_transparent()
            } else {
                ns.shapes[0].to_grid()
            }
        };

        if let Some(rule) = run_experiment(task, 5126, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() || ex.input.shapes.shapes.len() < 3 {
                return Grid::trivial();
            }
            let big = ex.input.shapes.largest();
            let mut horizontal = true;
            let mut r = usize::MAX;

            for s in ex.input.shapes.shapes.iter() {
                if s.is_pixel() {
                    if r == usize::MAX {
                        r = s.orow;
                    } else if r != s.orow {
                        horizontal = false;
                    }
                }
            }

            let cnt = ex.input.shapes.shapes.len() - 1;
            let mut shapes = if horizontal {
                Shapes::new_sized(big.cells.rows, big.cells.columns * cnt)
            } else {
                Shapes::new_sized(big.cells.rows * cnt, big.cells.columns)
            };
            let mut offset = 0;

            for s in ex.input.shapes.shapes.iter() {
                if *s != big {
                    let mut shape = big.clone();

                    shape.recolour_mut(big.colour, s.colour);

                    if horizontal {
                        shape.to_position_mut(0, offset);
                        offset += big.cells.rows;
                    } else {
                        shape.to_position_mut(offset, 0);
                        offset += big.cells.columns;
                    };

                    shapes.shapes.push(shape);
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 5176, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }

            // Default shape parser sometimes gets it wrong!
            let colour = ex.input.grid.to_shapes().full_extent().colour;

            Grid::new(1, 1, colour)
        };

        if let Some(rule) = run_experiment(task, 5189, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // testing function
        let func = |ex: &Example| {
            if !colour_diffs.is_empty() {
                return Grid::trivial();
            }
            let s = ex.input.shapes.largest();

            if s.cells.rows <= 3 || s.cells.columns <= 3 {
                return Grid::trivial();
            }

            let s = s.shrink_border();

            let grid = ex.input.grid.subgrid(s.orow + 1, s.cells.rows - 2, s.ocol + 1, s.cells.columns - 2);
//grid.show();

            grid
        };

        if let Some(rule) = run_experiment(task, 5210, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let (rs, cs) = examples.examples[0].output.grid.dimensions();

        let func = |ex: &Example| {
            if all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }
            let mut shape = Shape::trivial();
            let mut ocol = 0;
            let mut left = false;
            let mut width = 0;
            let mut winc = 0;
            let mut wider = false;

            let grid = ex.input.grid.recolour(all_colour_diffs[0], Black);

            for s in grid.to_shapes_coloured().shapes.iter() {
                if s.colour != all_colour_diffs[0] {
                    left = s.ocol > ocol;
                    ocol = s.ocol;
                    if width < s.cells.columns {
                        winc = s.cells.columns - width;
                        width = s.cells.columns;
                        wider = true;
                    } else {
                        wider = false;
                    }

                    shape = s.clone();
                }
            }

            let mut new_shape = shape.clone();
            let mut shapes = Shapes::new_sized(rs, cs);

            if left {
                ocol += 1;
            } else if wider {
                new_shape = Shape::new_sized_coloured_position(shape.ocol, shape.orow, shape.cells.rows, shape.cells.columns + winc, shape.colour);
            } else {
                let mut c1 = NoColour;
                let mut cc = 0;

                'outer:
                for r in 0 .. shape.cells.rows {
                    for c in 0 .. shape.cells.columns {
                        let cell = &shape.cells[(r,c)];
                        if c1 == NoColour && cell.colour != Black {
                            c1 = cell.colour;
                        } else if c1 != NoColour && cell.colour != Black {
                            if c == 0 {
                                return Grid::trivial();
                            }

                            cc = c - 1;
                            break 'outer;
                        }
                    }
                }

                new_shape = shape.clone();

                for r in 0 .. shape.cells.rows {
                    let cell = &mut new_shape.cells[(r,cc)];
                    if cell.colour != Black {
                        cell.colour = c1;
                    }
                }
            }

            let orow = if shape.cells.rows == 1 { 1 } else { 0 };

            new_shape.to_position_mut(orow, ocol);

            shapes.shapes.push(new_shape);

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 5291, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if all_colour_diffs.len() != 1 {
                return Grid::trivial();
            }

            let colour = ex.input.grid.majority_colour();
            let mut pix = ex.input.shapes.clone_base();
            let mut rd = 0;
            let mut cd = 0;
            let mut big_r = 0;
            let mut big_c = 0;
            let mut max_r = 0;
            let mut max_c = 0;

            for s in ex.input.shapes.shapes.iter() {
                if s.colour != colour {
                    if rd < s.orow || cd < s.ocol {
                        return Grid::trivial();
                    }

                    pix.shapes.push(s.clone());

                    rd += (rd as isize - s.orow as isize).abs() as usize;
                    cd += (cd as isize - s.ocol as isize).abs() as usize;
                } else {
                    big_r = big_r.max(s.cells.rows);
                    big_c = big_c.max(s.cells.columns);
                    max_r += s.cells.rows;
                    max_c += s.cells.columns;
                }
            }

            let horizontal = rd < cd;
            let mut o_shapes = ex.input.shapes.clone();
            let mut shapes = if horizontal {
                o_shapes.shapes.sort_by(|a, b| (a.ocol,a.orow).cmp(&(b.ocol,b.orow)));
                Shapes::new_sized(big_r, max_c)
            } else {
                o_shapes.shapes.sort_by(|a, b| (a.orow,a.ocol).cmp(&(b.orow,b.ocol)));
                Shapes::new_sized(max_r, big_c)
            };
            let mut r = 0;
            let mut c = 0;

            for s in o_shapes.shapes.iter() {
                if s.colour == colour {
                    let (cr, cc) = s.centre_of();
                    let near = pix.nearest_shape(cr, cc);
                    let ns = s.recolour(s.colour, near.colour).to_position(r, c);
                    if horizontal { c += big_c } else { r += big_r };

                    shapes.shapes.push(ns);
                }
            }

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 5291, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let (gridr, gridc) = examples.examples[0].output.grid.dimensions();
        let func = |ex: &Example| {
            if colour_common.len() != 2 {
                return Grid::trivial();
            }

            let mut grid = Grid::new(gridr, gridc, Black);
            let mut bg = NoColour;

            for s in ex.input.shapes.shapes.iter() {
                if s.dimensions() == ex.input.grid.dimensions() {
                    bg = s.colour;
                }
            }

            let scc = ex.input.shapes.shape_colour_cnt_map();
            let mut ccs: Vec<(Colour,usize)> = scc.iter().filter(|(k,_)| **k != bg).map(|(k,v)| (*k, v.len())).collect();

            ccs.sort();
            ccs.reverse();

            let mut r = 0;
            let mut c = 0;

            for (col, sz) in ccs.iter() {
                for _ in 0 .. *sz {
                    if r >= grid.cells.rows || c >= grid.cells.columns {
                        return Grid::trivial();
                    }
                    grid.cells[(r,c)].colour = *col;

                    if c + 1 == gridc {
                        c = 0;
                        r += 1;
                    } else {
                        c += 1;
                    }
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 5292, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 5330, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 5350, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 5366, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

            if let Some(rule) = run_experiment(task, 5414, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

            if let Some(rule) = run_experiment(task, 5448, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
        }

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

        if let Some(rule) = run_experiment(task, 5486, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 5510, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

            if let Some(rule) = run_experiment(task, 5528, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 5562, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            let fill_colour = colours.iter().filter(|&(&k,_)| k != in_colour).map(|(k,v)| (v,k)).min().map(|(_,k)| k).unwrap();

            if colours.len() == 3  {
                let bg_colour = colours.iter().filter(|&(&k,_)| k != in_colour).map(|(k,v)| (v,k)).max().map(|(_,k)| k).unwrap();

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

                if let Some(rule) = run_experiment(task, 5611, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

                if let Some(rule) = run_experiment(task, 5640, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

            if let Some(rule) = run_experiment(task, 5675, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

            // a65b410d test and fails 5207a7b5 eval
            for inc in 1 ..= 1 {    // TODO Fix inc for > 1
                let func = |ex: &Example| {
                    if ex.input.shapes.shapes.len() != 1 || ex.input.shapes.shapes[0].size() >= ex.input.grid.size() || ex.input.shapes.shapes[0].is_pixel() || ex.input.shapes.shapes[0].ocol > 0 || !ex.input.shapes.is_line() {
                        return Grid::trivial();
                    }

                    let top_colour = colours.iter().filter(|&(&k,_)| k != in_colour).map(|(k,v)| (v,k)).max().map(|(_,k)| k).unwrap();
                    let bottom_colour = colours.iter().filter(|&(&k,_)| k != in_colour).map(|(k,v)| (v,k)).min().map(|(_,k)| k).unwrap();

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

                if let Some(rule) = run_experiment(task, 5717, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

            if let Some(rule) = run_experiment(task, 5736, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 5768, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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
//consolidated.show();
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

        if let Some(rule) = run_experiment(task, 5833, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let ss = examples.some(false, &|ss| ss.smallest());
        let sl = examples.some(true, &|ss| ss.largest());
        let sc: Vec<(Shape, Colour)> = ss.iter().zip(sl.iter()).map(|(s1, s2)| (s1.to_origin(), s2.colour)).collect();

        let func = |ex: &Example| {
            let largest = ex.input.shapes.largest();
            let smallest = ex.input.shapes.smallest().to_origin();
            let mut shapes = ex.input.shapes.clone_base();

            let pair: Vec<_> = sc.iter().filter(|(s, _)| s.equals(&smallest) == Same).collect();

            if pair.len() == 0 {
                return Grid::trivial();
            }

            shapes.shapes.push(largest.recolour(largest.colour, pair[0].1));

// shapes.to_grid().show();
            shapes.to_grid()
        };

        if let Some(rule) = run_experiment(task, 5856, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 5885, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            if ex.input.coloured_shapes.len() != 2 || !ex.input.coloured_shapes.shapes[0].is_square() || ex.input.coloured_shapes.shapes[0].size() != 4 {
                return Grid::trivial();
            }
            let mut ns = ex.input.coloured_shapes.clone();
            let idx = &ex.input.coloured_shapes.shapes[0].cells;

            let sm: BTreeMap<Colour, Colour> = idx.keys()
                .map(|(r,c)|
                    if r == 0 && c == 0 {
                        (idx[(r,c)].colour, idx[(r,c+1)].colour)
                    } else if r == 1 && c == 0 {
                        (idx[(r,c)].colour, idx[(r,c+1)].colour)
                    } else if r == 0 && c == 1 {
                        (idx[(r,c)].colour, idx[(r,c-1)].colour)
                    } else {
                        (idx[(r,c)].colour, idx[(r,c-1)].colour)
                    }
                )
            .collect();

            ns.shapes[1].swap_colours(&sm);

            ns.to_grid()
        };

        if let Some(rule) = run_experiment(task, 5913, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }

    let gc = NoColouredShapesIn(9);
    if all || cat.contains(&gc) && cat.contains(&NoColouredShapesOut(9)){ 
        *cap_cats.entry(gc).or_insert(0) += 1;

        let colours: Vec<_> = examples.examples[0].input.grid.find_all_colours()
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

                if let Some(rule) = run_experiment(task, 5952, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

            if let Some(rule) = run_experiment(task, 5987, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 6002, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 6004, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.shapes.find_max().to_grid(), output) { return Some(rule); };

        let colour = examples.examples[0].output.grid.colour;

        // Use when divider 
        let func = |g: &Grid, _: &Grid, n: &mut usize| {
            let colour = g.mid_div_colour();
            let shapes = g.to_shapes_base_bg(colour);
            //let centre = g.centre_of();
            //let shapes = g.to_shapes_base_bg(ex.cells[centre].colour);
            if shapes.shapes.len() != 2 {
                return Grid::trivial();
            }
            let diff = shapes.shapes[0].diff(&shapes.shapes[1]);

            if let Some(diff) = diff {
                diff_only(&diff.to_grid(), colour, n)
            } else {
                Grid::trivial()
            }
        };

        if let Some(rule) = run_experiment_tries(task, 6026, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        // Use when divider 
        let func = |g: &Grid, _: &Grid, n: &mut usize| {
            let colour = g.mid_div_colour();
            let shapes = g.to_shapes_base_bg(colour);

            if shapes.shapes.len() != 2 || colour == NoColour || colour_diffs.is_empty() {
                return Grid::trivial();
            }
            let diff = shapes.shapes[0].diff(&shapes.shapes[1]);

            if let Some(diff) = diff {
                diff_only(&diff.to_grid(), colour, n).recolour(colour, colour_diffs[0]).inverse_colour()
            } else {
                Grid::trivial()
            }
        };

        if let Some(rule) = run_experiment_tries(task, 6045, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let func = |ex: &Example| {
            let shapes = ex.input.grid.split_2();

            if shapes.len() != 2 || colour_diffs.is_empty() {
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

        if let Some(rule) = run_experiment(task, 6064, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment_tries(task, 6082, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment_tries(task, 6098, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 6123, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 6142, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

            if let Some(rule) = run_experiment(task, 6170, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 6195, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = GravityDown;
    if all || cat.contains(&gc) || cat.contains(&GravityUp) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment_tries(task, 6203, experiment, trans, is_test, examples, &targets, done, tries, &|ex, _, n| gravity_only(ex, n), output) { return Some(rule); };

        let func = |ex: &Example| {
                let sccm = ex.input.shapes.shape_colour_cnt_map();
                let mut fill_colour = NoColour;
                let mut shapes = ex.input.shapes.clone_base();
                
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

        if let Some(rule) = run_experiment(task, 6227, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

                if let Some(rule) = run_experiment(task, 6295, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

                if let Some(rule) = run_experiment(task, 6313, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

            if let Some(rule) = run_experiment(task, 6337, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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
//shapes.trim_to_grid().show();

                shapes.trim_to_grid()
            };

            if let Some(rule) = run_experiment(file, 280, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
            */
        }

        let func = |ex: &Example| {
            if colour_diffs.len() != 1 || ex.input.shapes.shapes.is_empty() {
                return Grid::trivial();
            }

            // Assume all shapes are the same size
            let (rs, cs) = ex.majority_dimensions();
            let mut grid = ex.input.grid.clone();

            grid.recolour_mut(Black, colour_diffs[0]);
            grid.recolour_mut(all_colour_diffs[0], Black);

            grid.background_border_mut();

            grid.row_dividers_mut(rs);
            grid.col_dividers_mut(cs);

            grid
        };

        if let Some(rule) = run_experiment(task, 6402, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        let out_colours = examples.examples[0].output.grid.cell_colour_cnt_map();
        let func = |ex: &Example| {
            if all_colour_diffs.len() != 2 || out_colours.len() != 3 {
                return Grid::trivial();
            }
            let mut colour_order: Vec<(usize,Colour)> = out_colours.iter().map(|(k, v)| (*v, *k)).collect();
            colour_order.sort();
            let colours: Vec<Colour> = colour_order.iter().map(|(_, c)| *c).collect();
            let mut grid = ex.input.grid.clone();

            for s in ex.input.shapes.shapes.iter() {
                let mut r = s.cells.rows + 1;

                for c in (0 .. s.ocol).rev() {
                    grid.draw_mut(Up, r, c, colours[2]);

                    r += if grid.cells.rows - r <= 2 {
                        grid.cells.rows - r - 1
                    } else {
                        2
                    };
                }

                if s.cells.rows < 2 + 1 {
                    return Grid::trivial();
                }

                let mut r = s.cells.rows - 2 - 1;

                for c in s.ocol + 1 .. grid.cells.columns {
                    grid.draw_mut(Up, r, c, colours[0]);

                    if r < 2 {
                        break;
                    }

                    r -= 2;
                }
            }

//grid.show();
            grid
        };

        if let Some(rule) = run_experiment(task, 6403, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = SingleColourOut;
    if all || cat.contains(&gc) && !cat.contains(&SingleColourIn) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if let Some(rule) = run_experiment(task, 6410, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_pixels_min().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 6412, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_pixels_max().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 6414, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_max().to_grid(), output) { return Some(rule); };

        if let Some(rule) = run_experiment(task, 6416, experiment, trans, is_test, examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_min().to_grid(), output) { return Some(rule); };

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

                if let Some(rule) = run_experiment(task, 6462, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
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

        if let Some(rule) = run_experiment(task, 6478, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

    if let Some(rule) = run_experiment(task, 6511, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    let out_shapes = examples.all_shapes_out();
    let out_shapes = out_shapes.shape_permutations();

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

        shapes.to_grid()
    };

    if let Some(rule) = run_experiment(task, 6531, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    let spc = examples.shape_pixels_to_colour();

    let func = |ex: &Example| {
        let mut shapes = ex.input.shapes.clone();

        for s in shapes.shapes.iter_mut() {
            if let Some(colour) = spc.get(&s.pixels()) {
                s.recolour_mut(s.colour, *colour);
            } else {
                return Grid::trivial();
            }
        }

        shapes.to_grid()
    };

    if let Some(rule) = run_experiment(task, 6549, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    let func = |ex: &Example| {
        if ex.input.shapes.is_empty() {
            return Grid::trivial();
        }
        let mut shapes = ex.input.shapes.clone();

        for s in shapes.shapes.iter_mut() {
            if !Colour::in_range(s.pixels()) {
                return Grid::trivial();
            }

            s.recolour_mut(s.colour, Colour::from_usize(s.pixels()));
        }
//shapes.to_grid().show();

        shapes.to_grid()
    };

    if let Some(rule) = run_experiment(task, 6569, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        shapes.to_grid()
    };

    if let Some(rule) = run_experiment(task, 6587, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

    // Cross example knowledge needed for closure
    let h = examples.bleached_io_map();

    let func = |ex: &Example| {
        if let Some(grid) = h.get(&ex.input.grid.bleach().to_json()) {
            grid.clone()
        } else {
            Grid::trivial()
        }
    };

    if let Some(rule) = run_experiment(task, 6600, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

        if let Some(rule) = run_experiment(task, 6630, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };
    }

    let func = |ex: &Example| {
        let shape = ex.input.grid.as_shape().scale_up(2);

        Shapes::new_shapes(&[shape]).to_grid()
    };

    if let Some(rule) = run_experiment(task, 100000, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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
        let mut shapes = ex.input.shapes.clone_base();

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

    if let Some(rule) = run_experiment(task, 100010, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

    if let Some(rule) = run_experiment(task, 100020, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

    if let Some(rule) = run_experiment(task, 100030, experiment, trans, is_test, examples, &targets, done, tries, &func, output) { return Some(rule); };

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

fn run_experiment_colours(task: &str, experiment: usize, experiment_todo: &str, is_test: bool, examples: &Examples, targets: &[Grid], done: &mut BTreeSet<String>, func: &(dyn Fn(&Example, Colour) -> Grid + RefUnwindSafe), output: &mut BTreeMap<String, Vec<OutputData>>) -> Option<usize> {
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

    let ans = panic::catch_unwind(|| experiment_colours(examples, func));

    let ans = match ans {
        Ok(ans) => ans,
        Err(e) => {
            eprintln!("{task} / {experiment} Exception: {e:?}");
            vec![Grid::trivial()]
        },
    };

    save(task, experiment, NoTrans, is_test, &ans, targets, done, output)
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

            let dist: Vec<_> = ans.iter().zip(targets.iter())
                .map(|(a,t)| format!("{:.4}", a.distance(t)))
                .collect();

            println!("Final Test Failed : {experiment:>05} {trans:?} / {task} by {}", dist.join(", "));

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
        println!("{k:<5}: {}", v.join(", "));
    }
    println!();
}
