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
    let mut done: BTreeSet<String> = BTreeSet::new();
    let mut tries: usize = 0;

    // Main processing outer loop
    for (file_name, task) in tdata.iter() {
        let file = file_name.to_string();

        // already done or just one???
        if done.contains(&file) || (!task_name.is_empty() && task_name != "all" && file != task_name) {
            continue;
        }

        let mut examples = Examples::new(task);

        // Do some shapes overay others?
        if examples.cat.contains(&OverlayInSame) || examples.cat.contains(&OverlayOutSame) {
            examples = Examples::new_cons(task);
        }
        println!("{file}: {:?}", examples.cat);

        let transform = if trans == "all" || trans == "NoTrans" {
            NoTrans
        } else {
            if let Ok(tr) = Transformation::from_str(trans)  {
                examples = examples.transformation(tr);

                tr
            } else {
                NoTrans
            }
        };

        // Base pass with no transformations
        pass(all, &file, experiment, &examples, transform, is_test, &mut output, &mut cap_cats, &mut cap_todo, &mut done, &mut tries);

        // Experimental - needs more work!
        if trans == "all" {
            // Now check for transformed grids to make closures more generic
            for transform in Transformation::iter() {
                if transform == NoTrans || transform >= MirrorXRotate90 { continue; }
                //println!("{:?}", transform);

                let ex_copy = examples.transformation(transform);

                pass(all, &file, experiment, &ex_copy, transform, is_test, &mut output, &mut cap_cats, &mut cap_todo, &mut done, &mut tries);
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
    println!("{done:?}");
    println!("Done = {}, To Do = {}, tries = {tries}", done.len(), cnt - done.len());

    let timing = start.elapsed().as_secs() as f64 + start.elapsed().subsec_millis() as f64 / 1000.0;

    println!("Elapsed time: {timing} secs");
    println!("Used memory : {:.2} Mb", PEAK.current_usage_as_mb());
}

fn pass(all: bool, file: &str, experiment: &str, examples: &Examples, trans: Transformation, is_test: bool, output: &mut BTreeMap<String, Vec<OutputData>>, cap_cats: &mut BTreeMap<GridCategory, i32>, cap_todo: &mut BTreeMap<GridCategory, i32>, done: &mut BTreeSet<String>, tries: &mut usize) {
    let cat = &examples.cat;
    let targets: Vec<Grid> = examples.tests.iter().map(|test| test.output.grid.clone()).collect();


    let in_colours = examples.examples[0].input.shapes.colours();
    let out_colours = examples.examples[0].output.shapes.colours();
    let colours = Uniq::uniq(&out_colours, in_colours.clone());

    // testing function
    /*
    let func = |ex: &Example| {
//println!("{:?}", ex.output.shapes.shapes[0]);
//ex.input.shapes.show();
//ex.output.shapes.show();
//println!("{colours:?}");
        if in_colours.len() != 1 || colours.len() != 1 {
            return Grid::trivial();
        }

        let grid = ex.input.grid.dup_right().dup_down();
        let shapes = grid.as_pixel_shapes();

        let mut shapes = Shapes::new_given(grid.cells.rows, grid.cells.columns, &shapes.shapes);

        shapes = shapes.embellish(colours[0]);

        for s in grid.to_shapes_sq().shapes.iter() {
            shapes.shapes.push(s.clone());
        }
        
        //Grid::trivial()
        shapes.to_grid_transparent()
    };

    if run_experiment(&file, 10000, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
    */

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

                for (x, y) in grid.cells.keys() {
                    if x % 2 == 0 || y % 2 == 0 || x == grid.cells.rows - 1 || y == grid.cells.columns - 1 {
                        grid.cells[(x,y)].colour = colour;
                    }
                }
            } else {
                let sq = ex.cat.contains(&InOutSquareSameSize);

                grid = ex.input.grid.do_circle(colour, sq);
            }

            grid
        };

        if run_experiment(&file, 0, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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
                    let x1 = if bp.orow > 0 { bp.orow - 1 } else { bp.orow };
                    let x2 = if bp.orow + bp.cells.rows < grid.cells.rows { bp.orow + bp.cells.rows + 1 } else { bp.orow + bp.cells.rows };
                    let y1 = if bp.ocol > 0 { bp.ocol - 1 } else { bp.ocol };
                    let y2 = if bp.ocol + bp.cells.columns < grid.cells.columns { bp.ocol + bp.cells.columns + 1 } else { bp.ocol + bp.cells.columns };
                    let m = grid.cells.slice(x1 .. x2, y1 .. y2);

                    if let Ok(m) = m {
                        let s = Shape::new(bp.orow, bp.ocol, &m);
//s.show();
                        let l = y2 - y1;
                        let fw = m.windows(l).next().unwrap();
                        let sc: Vec<_> = fw.iter().map(|c| c.colour).collect();
                        for w in grid.cells.windows(l) {
                            let fc: Vec<_> = w.iter().map(|c| c.colour).collect();
                            if sc == fc && (fw[0].row != w[0].row || fw[0].col != w[0].col) {
                                let patch = grid.get_patch(w[0].row, w[0].col, m.rows, m.columns);
                                if patch.is_full() && s.same_patch(&patch) {
                                    let sox = if s.orow > 0 { s.orow - 1 } else { s.orow };
                                    let soy = if s.ocol > 0 { s.ocol - 1 } else { s.ocol };

                                    grid.fill_patch_mut(&patch, sox, soy);

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

        if run_experiment(&file, 1, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let func = &|ex: &Example| {
            if ex.input.black.is_empty() {
                return Grid::trivial();
            }

            let mut grid = ex.input.grid.clone();

            grid.fill_border();
            let mut px = 0;
            let mut py = 0;
            let mut lc = 0;     // if all else fails break the loop!

            loop {
                let black = grid.find_black_patches();

                if black.is_empty() || lc >= 10 {
                    break;
                }

                for p in black.shapes.iter() {
                    let xa = if p.ocol == 0 { 
                        Vec::new()
                    } else {
                        if let Ok(xs) = grid.cells.slice(p.orow .. p.orow+p.cells.rows, p.ocol-1 .. p.ocol) {
                            let xa: Vec<_> = xs.values().map(|c| c.colour).collect();
                            if Colour::single_colour_vec(&xa) {
                                Vec::new()
                            } else {
                                xa
                            }

                        } else {
                            Vec::new()
                        }
                    };
                    let ya = if p.orow == 0 {
                        Vec::new()
                    } else {
                        if let Ok(ys) = grid.cells.slice(p.orow-1 .. p.orow, p.ocol .. p.ocol+p.cells.columns) {
                            let ya: Vec<_> = ys.values().map(|c| c.colour).collect();
                            if Colour::single_colour_vec(&ya) {
                                Vec::new()
                            } else {
                                ya
                            }
                        } else {
                            Vec::new()
                        }
                    };
                    if !xa.is_empty() && xa.len() >= ya.len() {
                        let (xo, yo) = grid.find_row_seq(p.orow, p.ocol, &xa, p.cells.columns);
                        if xo == usize::MAX && yo == usize::MAX || px == xo && py == yo {
                            return Grid::trivial();
                        }
                        for x in 0 .. p.cells.rows {
                            for y in 0 .. p.cells.columns {
                                if grid.cells[(p.orow+x,p.ocol+y)].colour == Black && xo+x < grid.cells.rows && yo+1+y < grid.cells.columns {
                                    grid.cells[(p.orow+x,p.ocol+y)] = grid.cells[(xo+x,yo+1+y)].clone();
                                }
                            }
                        }

                        px = xo;
                        py = yo;
                    }
                    else if !ya.is_empty() {
                        let (xo, yo) = grid.find_col_seq(p.orow, p.ocol, &ya, p.cells.rows);
                        if xo == usize::MAX && yo == usize::MAX || px == xo && py == yo {
                            return Grid::trivial();
                        }
                        for x in 0 .. p.cells.rows {
                            for y in 0 .. p.cells.columns {
                                if grid.cells[(p.orow+x,p.ocol+y)].colour == Black && xo+1+x < grid.cells.rows && yo+y < grid.cells.columns {
                                    grid.cells[(p.orow+x,p.ocol+y)] = grid.cells[(xo+1+x,yo+y)].clone();
                                }
                            }
                        }

                        px = xo;
                        py = yo;
                    } else {
                        return Grid::trivial();
                    }
                }
                lc += 1;
            }
//grid.show();
            
            grid
        };

        if run_experiment(&file, 4, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        // Duplicate
        let func = |ex: &Example| {
            let largest = ex.input.coloured_shapes.largest();

            if largest.cells.rows < 3 || largest.cells.columns < 3 {
                return Grid::trivial();
            }

            let shape = largest.subshape(0, 3, 0, 3);
            let mut grid = ex.input.grid.clone();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.size() == 1 {
                    let new_shape = shape.translate_absolute(s.orow,s.ocol);

                    grid.copy_shape_to_grid(&new_shape);
//new_shape.show_summary();
                }
            }
//grid.show();

            grid
        };

        if run_experiment(&file, 2, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            if run_experiment(&file, 3, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = BGGridInBlack;
    if all || cat.contains(&gc) && !cat.contains(&BGGridOutBlack) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if run_experiment(&file, 10, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_min().to_grid(), output) { return; };

        if run_experiment(&file, 20, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_max_colour_count().to_grid(), output) { return; };

        if run_experiment(&file, 30, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.has_mirror_x().to_grid(), output) { return; };

        if run_experiment(&file, 40, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.has_mirror_y().to_grid(), output) { return; };
        //if run_experiment_examples(&file, 1000, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return; };

        //-if run_experiment_examples(&file, 1010, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_shape_fill(&exs), output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }

    let gc = SingleColourOut2xIn;
    if all || cat.contains(&gc) { // 3?
        *cap_cats.entry(gc).or_insert(0) += 1;

        if run_experiment_tries(&file, 800, experiment, trans, is_test, &examples, &targets, done, tries, &|ex, _, n| mirror_only(ex, n), output) { return; };

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

        if run_experiment(&file, 31, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

        if run_experiment(&file, 41, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            if run_experiment(&file, 50, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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
                let y = Colour::to_usize(*col) - 1;

                if *size >= rows || y >= cols {
                    return Grid::trivial();
                }

                for x in 0 .. *size {
                    let x = rows - x - 1;

                    grid.cells[(x,y)].row = x;
                    grid.cells[(x,y)].col = y;
                    grid.cells[(x,y)].colour = *col;
                }
            }

            grid
        };

        if run_experiment(&file, 60, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let func = |gi: &Example| {
//gi.input.shapes.show();
            let mut xy: Vec<(usize, usize, Colour)> = Vec::new();
            let mut colour = NoColour;
            let mut shapes = gi.input.shapes.clone();

            for s in gi.input.shapes.shapes.iter() {
                if colour == NoColour {
                    colour = s.colour;
                }
                if colour != s.colour {
                    //xx = s.cells[(0,0)].x;
                    //yy = s.cells[(0,0)].y;
                    xy.push((s.cells[(0,0)].row, s.cells[(0,0)].col, s.colour));

                    break;
                }
            }
//println!("{xx}/{yy} {:?}", colour);

            for ss in shapes.shapes.iter_mut() {
                for (xx, yy, colour) in xy.iter() {
                    if ss.cells[(0,0)].row == *xx || ss.cells[(0,0)].col == *yy {
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

        if run_experiment(&file, 70, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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
        if run_experiment(&file, 50, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = FullyPopulatedOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if run_experiment(&file, 80, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_max().to_grid(), output) { return; };

        if run_experiment(&file, 90, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.largest().to_grid(), output) { return; };
//target.show();
//ans.show();

        if run_experiment(&file, 100, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_min().to_grid(), output) { return; };

        let common_colours = examples.find_output_colours();

        for colour in common_colours {
            if run_experiment(&file, 110, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.cell_colour_cnts(colour).to_grid(), output) { return; };
        }

        if run_experiment(&file, 120, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_sub_largest_count().to_grid(), output) { return; };

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
            let mut x = 0;

            // Get right ordering by munging x coord then sorting
            for (i, s) in shapes.iter_mut().enumerate() {
                if i > 0 && (i % 3) == 0 {
                    x += 1;
                }

                s.orow = x;
            }

            shapes.sort_by(|a, b| (a.orow, a.ocol).cmp(&(b.orow, b.ocol)));
//println!("{} by {}", ex.input.grid.cells.rows, ex.input.grid.cells.columns);

            let mut i = 0;

            for s in shapes.iter() {
                let x = i / 3;
                let y = i % 3;
//println!("{:?} {i} {}/{} -> {}/{}", s.colour, s.orow, s.ocol, x, y);

                if x >= grid.cells.rows || y >= grid.cells.columns {
                    return Grid::trivial();
                }

                grid.colour = s.colour;

                grid.cells[(x,y)].row = i / 3;
                grid.cells[(x,y)].col = i / height;
                grid.cells[(x,y)].colour = s.colour;

                i += 1;
            }
//grid.show();

            grid
        };

        if run_experiment(&file, 121, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            for ((x, y), c) in ex.input.grid.cells.items() {
                if c.colour == Black {
                    let idx = (x + y) % len;

                    if let Some(colour) = colours.get(&idx) {
                        grid.cells[(x,y)].colour = colour.clone();
                    }
                }
            }
//grid.show();

            grid
        };

        if run_experiment(&file, 122, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let func = |ex: &Example| {
            let in_colour = ex.output.grid.get_diff_colour(&ex.input.grid);
            let out_colour = ex.input.grid.get_diff_colour(&ex.output.grid);

            ex.input.grid.recolour(in_colour, out_colour)
        };

        if run_experiment(&file, 123, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = MirrorXOut;
    if all || cat.contains(&gc) || cat.contains(&MirrorYOut) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        let func = |gi: &Example| {
            if gi.input.coloured_shapes.len() != 1 {
                return Grid::trivial();
            }

            let s = gi.input.grid.as_shape();

            let rows = s.cells.rows;
            let cols = s.cells.columns;
            let mut shapes = Shapes::new_sized(rows * 2, cols * 2);

            shapes.shapes.push(s.mirrored_x().mirrored_y());
            shapes.shapes.push(s.mirrored_y().translate_absolute(rows, 0));
            shapes.shapes.push(s.mirrored_x().translate_absolute(0, cols));
            shapes.shapes.push(s.translate_absolute(rows, cols));

//shapes.to_grid().show();
            shapes.to_grid()
        };

        if run_experiment(&file, 280, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let func = |ex: &Example| {
            let bg = ex.input.grid.max_colour();
            let shapes = ex.input.grid.to_shapes_sq();
            let shapes = shapes.colour_groups_to_shapes(bg);

            shapes.to_grid()
        };

        if run_experiment(&file, 281, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Div9In;
    if all || cat.contains(&gc) && !cat.contains(&Div9Out) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        //-if run_experiment_examples(&file, 1020, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_9_in_out(&exs), output) { return; };

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

        //-if run_experiment_examples(&file, 1030, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return; };

        //-if run_experiment_examples(&file, 1040, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_double(&exs), output) { return; };

        //-if run_experiment_examples(&file, 1050, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_expand_3x3(&exs), output) { return; };

        if examples.examples[0].input.grid.size() * 9 == examples.examples[0].output.grid.size() {
            let func = |gi: &Example| {
                let shape = &gi.input.grid.as_shape();
                let (colour, _) = shape.colour_cnt(false);

                let posns = shape.colour_position(colour);
                let mut shapes = Shapes::new_sized(shape.cells.rows * 3, shape.cells.columns * 3);

                for (x, y) in posns.iter() {
                    let ox = x * 3;
                    let oy = y * 3;

                    shapes.shapes.push(shape.translate_absolute(ox, oy));
                }
//shapes.to_grid().show();
                shapes.to_grid()
            };

            if run_experiment(&file, 130, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = IdenticalNoPixels;
    if all || cat.contains(&gc) { // 164
        *cap_cats.entry(gc).or_insert(0) += 1;

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

            new_shapes.to_grid()
        };

        if run_experiment(&file, 140, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

        if run_experiment(&file, 141, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let func = |ex: &Example| {
            let mut grid = ex.input.grid.clone();

            for (r, c) in ex.input.grid.cells.keys() {
                if ex.input.grid.cells[(r, 0)].colour != Black && grid.cells[(r, c)].colour != Black {
                    grid.cells[(r, c)].colour = ex.input.grid.cells[(r, 0)].colour;
                }
            }
            
            grid
        };

        if run_experiment(&file, 142, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Is3x3In;
    if all || cat.contains(&gc) && cat.contains(&Is3x3Out){ 
        *cap_cats.entry(gc).or_insert(0) += 1;

//println!("#### {file}");
        //-if run_experiment_examples(&file, 1060, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return; };

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
                        for y in 1 .. 3 {
                            m[(0,y)].row = 0;
                            m[(0,y)].col = y;
                        }
                        for x in 1 .. 3 {
                            for y in 0 .. 3 {
                                m[(x,y)].row = x;
                                m[(x,y)].col = y;
                                m[(x,y)].colour = s.cells[(x-1,y)].colour;
                            }
                        }
                        big = Shape::new(0, 0, &m);
                    } else if s.cells.rows == 3 && s.cells.columns == 2 {
                        let mut m = Matrix::new(3, 3, Cell::new(0, 0, 0));
                        for x in 1 .. 3 {
                            m[(x,0)].row = x;
                            m[(x,0)].col = 0;
                        }
                        for x in 0 .. 3 {
                            for y in 1 .. 3 {
                                m[(x,y)].row = x;
                                m[(x,y)].col = y;
                                m[(x,y)].colour = s.cells[(x,y-1)].colour;
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

        if run_experiment(&file, 140, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = InOutSquareSameSize;
    if all || cat.contains(&gc) { // 164
        *cap_cats.entry(gc).or_insert(0) += 1;

        if all || cat.contains(&InSameCountOut) || cat.contains(&InSameCountOutColoured) {
            //if run_experiment_examples(&file, 1070, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_shape_substitute(&exs), output) { return; };

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

            if run_experiment(&file, 141, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

            let func = |ex: &Example| {
                if ex.input.shapes.shapes.len() > 20 {
                    return Grid::trivial();
                }

                let mut shapes = ex.input.shapes.clone();

                shapes.shapes = Vec::new();

                //for s in &ex.input.shapes.consolidate_shapes().shapes {
                for s in &ex.input.shapes.shapes {
                    shapes.shapes.push(s.mirrored_x());
                }
//shapes.to_grid().show();

                shapes.to_grid()
            };

            // evaluation only
            if run_experiment(&file, 142, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        if all || cat.contains(&InLessCountOut) {
            //// see 045e512c.json
//println!("#### {}", examples.examples[0].input.shapes.len());
//examples.examples[0].input.shapes.show();
        }

        // 0ca9ddb6 4258a5f9 913fb3ed 95990924 b60334d2 test
        if run_experiment_tries(&file, 0, experiment, trans, is_test, &examples, &targets, done, tries, &|ex, _, n| transform_only(ex, n), output) { return; };
        //-if run_experiment_examples(&file, 1100, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return; };

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

        if run_experiment(&file, 220, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        // Duplicate!
        let func = &|ex: &Example| {
            let mut grid = ex.input.grid.clone();
            let cnts = ex.input.shapes.shape_colour_cnt_map();

            for a_colour in Colour::all_colours().iter() {
                if let Some(cs) = cnts.get(&a_colour) {
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

        if run_experiment(&file, 51, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

        if run_experiment(&file, 163, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        // Duplicated
        let func = |ex: &Example| {
            let largest = ex.input.coloured_shapes.largest();

            if largest.cells.rows < 3 || largest.cells.columns < 3 {
                return Grid::trivial();
            }

            let shape = largest.subshape(0, 3, 0, 3);
            let mut grid = ex.input.grid.clone();

            for s in ex.input.coloured_shapes.shapes.iter() {
                if s.size() == 1 {
                    let new_shape = shape.translate_absolute(s.orow,s.ocol);

                    grid.copy_shape_to_grid(&new_shape);
//new_shape.show_summary();
                }
            }
//grid.show();

            grid
        };

        if run_experiment(&file, 2, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        // Duplicate
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
                        let (x, y) = s.centre_of();

                        s.flood_fill_mut(x - s.orow, y - s.ocol, NoColour, *colour);
                    }
                }

                shapes.to_grid()
            };

            if run_experiment(&file, 181, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = InOutSquare;
    if all || cat.contains(&gc) { // 27
        *cap_cats.entry(gc).or_insert(0) += 1;

        //if run_experiment_examples(&file, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_shape_fill(&exs), output) { return; };

        //-if run_experiment_examples(&file, 1050, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_expand_3x3(&exs), output) { return; };

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
        if run_experiment(&file, 160, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            let shape = ex.input.grid.as_shape_position(0, width).mirrored_x();
            shapes.shapes.push(shape);

            let shape = ex.input.grid.as_shape_position(height, 0).mirrored_y();
            shapes.shapes.push(shape);

            let shape = ex.input.grid.as_shape().mirrored_x().mirrored_y();
            shapes.shapes.push(shape);

            shapes.to_grid()
        };
        if run_experiment(&file, 162, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let func = &|ex: &Example| {
            if ex.input.coloured_shapes.shapes.len() != 1 || !ex.input.coloured_shapes.shapes[0].is_square() || ex.input.coloured_shapes.shapes[0].size() != 36 {
                return Grid::trivial();
            }

            let shape = &ex.input.coloured_shapes.shapes[0];
            let ss = shape.subshape(0, 3, 0, 3);

            ss.to_grid()
        };

        if run_experiment(&file, 163, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        // Duplicate
        let func = |ex: &Example| {
            let shape = ex.input.grid.as_shape().scale_up(2);

            Shapes::new_shapes(&[shape]).to_grid()
        };

        if run_experiment(&file, 2140, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
//a.show();
    }
    let gc = InOutSameSize;
    if all || cat.contains(&gc) { // 97
        *cap_cats.entry(gc).or_insert(0) += 1;

        /* Causes problems probably not
        //if run_experiment_examples(&file, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_shape_fill(&exs), output) { return; };

        //if run_experiment_examples(&file, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return; };

        // 5
        if all || cat.contains(&InSameCountOut) {
            //if run_experiment_examples(&file, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_move_together(&exs), output) { return; };
        }
        */

        if cat.contains(&IsPanelledXOut) {
            let func = |ex: &Example| {
                let grid_shape = ex.input.grid.as_shape();
                let colour = grid_shape.majority_colour();
                let bg_shapes = ex.input.grid.to_shapes_base_bg(colour);
                let not_empty: Vec<Shape> = bg_shapes.shapes.iter().filter(|s| !s.is_empty()).map(|s| s.clone()).collect();

                if not_empty.len() != 1 {
                    return Grid::trivial();
                }

                let mut shapes = ex.input.shapes.clone();

                shapes.shapes = Vec::new();

                shapes.shapes.push(grid_shape);

                for s in &bg_shapes.shapes {
                    if s.is_empty() {
                        shapes.shapes.push(not_empty[0].copy_into(&s));
                    } else {
                        shapes.shapes.push(s.clone());
                    }
                }
//shapes.to_grid().show();

                shapes.to_grid()
            };

            if run_experiment(&file, 500, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        if all | cat.contains(&InOutSameShapesColoured) {
            // Duplicate
            let func = |ex: &Example| {
                let largest = ex.input.coloured_shapes.largest();

                if largest.cells.rows < 3 || largest.cells.columns < 3 {
                    return Grid::trivial();
                }

                let shape = largest.subshape(0, 3, 0, 3);
                let mut grid = ex.input.grid.clone();

                for s in ex.input.coloured_shapes.shapes.iter() {
                    if s.size() == 1 {
                        let new_shape = shape.translate_absolute(s.orow,s.ocol);

                        grid.copy_shape_to_grid(&new_shape);
//new_shape.show_summary();
                    }
                }
//grid.show();

                grid
            };

            if run_experiment(&file, 2, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        //if run_experiment(&file, 150, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.toddle_colours().to_grid(), output) { return; };

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

            let mut shapes = Shapes::new_sized(gi.input.shapes.nx, gi.input.shapes.ny);

            for s in gi.input.shapes.shapes.iter() {
                if s.size() < size { continue; }
//s.show_summary();
                if let Some(shape) = h.get(&s.colour) {
                    if shape.pixels() >= s.pixels() {
                        let mut new_shape = s.clone();
//shape.show_summary();
                        for ((x, y), c) in shape.cells.items() {
                            if x >= new_shape.cells.rows || y >= new_shape.cells.columns {
                                return Grid::trivial();
                            }
                            new_shape.cells[(x,y)].colour = c.colour;
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

        if run_experiment(&file, 161, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        if !cat.contains(&FullyPopulatedIn) {
            let func = |gi: &Example| {
                let mut si = gi.clone();
                let mut dx = 0;
                let mut dy = 0;

                for s in &gi.input.shapes.shapes {
                    if dx == 0 {
                        dx = s.orow;
                    } else if s.orow % dx != 0 {
                        return Grid::trivial();
                    }
                    if dy == 0 {
                        dy = s.ocol;
                    } else if s.ocol % dy != 0 {
                        return Grid::trivial();
                    }
//println!("{} {}", s.ox, s.oy);
                }

                if dx != 0 && dy != 0 {
                    return Grid::trivial();
                }

                //si.input.shapes.shapes.sort_by(|a, b| b.pixels().cmp(&a.pixels()));
                si.input.shapes.shapes.sort_by_key(|b| std::cmp::Reverse(b.pixels()));

                for (i, s) in si.input.shapes.shapes.iter_mut().enumerate() {
                    s.orow = i * dx;
                    s.ocol = i * dy;

                    for (x, y) in s.cells.keys() {
                        s.cells[(x, y)].row = s.orow + x;
                        s.cells[(x, y)].col = s.ocol + y;
                    }
                }
//si.input.shapes.to_grid().show();

                si.input.shapes.to_grid()
            };

            if run_experiment(&file, 170, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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
            //si.shapes.sort_by(|a, b| b.size().cmp(&a.size()));
            si.shapes.sort_by_key(|b| std::cmp::Reverse(b.size()));

            si.to_grid()
        };

        if run_experiment(&file, 180, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        // Duplicate!
        let func = &|ex: &Example| {
            let mut grid = ex.input.grid.clone();
            let cnts = ex.input.shapes.shape_colour_cnt_map();

            for a_colour in Colour::all_colours().iter() {
                if let Some(cs) = cnts.get(&a_colour) {
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
        if run_experiment(&file, 51, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        // Duplicate
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
                        let (x, y) = s.centre_of();

                        s.flood_fill_mut(x - s.orow, y - s.ocol, NoColour, *colour);
                    }
                }

                shapes.to_grid()
            };

            if run_experiment(&file, 181, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = Double;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        //-if run_experiment_examples(&file, 1110, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_double(&exs), output) { return; };

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

        if run_experiment(&file, 190, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

        if run_experiment(&file, 191, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        */

    let func = |ex: &Example| {
        if in_colours.len() != 1 || colours.len() != 1 {
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

    if run_experiment(&file, 192, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    /*
    } else if all || cat.contains(&InLessThanOut) {
        *cap_todo.entry(gc).or_insert(0) += 1;
    */
    let gc = SingleShapeOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if all || cat.contains(&SingleShapeIn) {
            //if run_experiment_examples(&file, 1120, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return; };

            //if run_experiment_examples(&file, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_move_same_colour(&exs), output) { return; };

            let func = |gi: &Grid, go: &Grid, n: &mut usize| {
                let grid = gi.recolour(gi.colour, go.colour);

                move_only(&grid, n)
            };

            if run_experiment_tries(&file, 310, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

            let func = |gi: &Example| {
                if gi.input.coloured_shapes.shapes.is_empty() {
                    return Grid::trivial();
                }

                gi.input.coloured_shapes.shapes[0].to_origin().to_grid()
            };

            if run_experiment(&file, 200, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }
        if all || !cat.contains(&SingleShapeIn) {
            let x = examples.examples[0].output.grid.cells.rows;
            let y = examples.examples[0].output.grid.cells.columns;

            if x == 1 || y == 1 {
                let colour = examples.examples[0].output.grid.colour;

                let func = |ex: &Example| {
                    let mut i = 0;
                    let mut grid = Grid::new(x, y, Black);

                    for s in ex.input.shapes.shapes.iter() {
                        if i >= y {
                            return Grid::trivial();
                        }
                        if s.colour == colour && s.size() > 1 {
                            if x == 1 {
                                grid.cells[(0, i)].colour = colour;
                            } else {
                                grid.cells[(i, 0)].colour = colour;
                            }
                            i += 1;
                        }
                    }

                    grid
                };

                if run_experiment(&file, 400, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
            }
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = OutLessThanIn;
    if all || cat.contains(&gc) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        if run_experiment(&file, 201, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_unique().to_grid(), output) { return; };

        if run_experiment(&file, 205, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.first().to_grid(), output) { return; };

        if run_experiment(&file, 206, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.last().to_grid(), output) { return; };

        let func = |ex: &Example| {
            for s in &ex.input.shapes.shapes {
                if !s.is_mirror_x() && !s.is_mirror_y() {
                    return s.to_grid();
                }
            }

            Grid::trivial()
        };

        if run_experiment(&file, 207, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let func = |ex: &Example| {
            for s in &ex.input.shapes.shapes {
                if s.is_mirror_x() || s.is_mirror_y() {
                    return s.to_grid();
                }
            }

            Grid::trivial()
        };

        if run_experiment(&file, 208, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let func = |ex: &Example| {
            let bordered = ex.input.shapes.border_only();

            if bordered == Shape::trivial() {
                return Grid::trivial();
            }

            let subg = ex.input.grid.subgrid(bordered.orow + 1, bordered.cells.rows - 2, bordered.ocol + 1, bordered.cells.columns - 2);

            subg
        };

        if run_experiment(&file, 209, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
    }
    let gc = SingleColouredShapeOut;
    if all || cat.contains(&gc) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        //if run_experiment_examples(&file, 1130, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_overlay(&exs), output) { return; };

        // Duplicate
        let func = |ex: &Example| {
            let shape = ex.input.grid.as_shape().scale_up(2);

            Shapes::new_shapes(&[shape]).to_grid()
        };

        if run_experiment(&file, 2140, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        //if run_experiment_examples(&file, 1140, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_mirror(&exs), output) { return; };

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

        if run_experiment(&file, 204, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

        if run_experiment(&file, 205, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            if run_experiment(&file, 600, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = NoShapesIn(1);
    if all || cat.contains(&gc) && cat.contains(&NoColouredShapesOut(1)) && !cat.contains(&NoShapesOut(1)) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if cat.contains(&NoShapesIn(1)) && cat.contains(&NoShapesOut(6)) {
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

            if run_experiment(&file, 610, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

        if run_experiment(&file, 621, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            if run_experiment(&file, 622, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

            if run_experiment(&file, 623, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            if run_experiment(&file, 624, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

                if run_experiment(&file, 635, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

                if run_experiment(&file, 625, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

                if run_experiment(&file, 626, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

            if run_experiment(&file, 627, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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
ex.output.grid.show();
//println!("####");
//shapes.to_grid().show();
//shapes.to_grid().inverse_transform(trans).show();

                    shapes.to_grid()
                };

                if run_experiment(&file, 628, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

            if run_experiment(&file, 630, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

                    rfrom = if rfrom > dist { rfrom - dist } else { 0 };
                    cfrom = if cfrom == 0 { dist } else { 0 };
                    odd = !odd;
                }

                grid = grid.recolour(Black, *fill_colour);

//grid.show();
                grid
            };

            if run_experiment(&file, 631, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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
    let gc = OutLessThanIn;
    if all || cat.contains(&gc) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        if all || cat.contains(&SinglePixelOut) {
            let func = |gi: &Example| {
                if gi.input.shapes.shapes.is_empty() {
                    return Grid::trivial();
                }

                let h = gi.input.shapes.shape_colour_cnt_map();
                let pair: Option<Vec<Shape>> = h.clone().into_values().filter(|p| p.len() == 2 && p[0].size() == p[1].size()).last();
                if let Some(pair) = pair {
                    // TODO fix coloured shape when mixed
                    for cs in gi.input.coloured_shapes.shapes.iter() {
                        if cs.is_contained(&pair[0]) && cs.is_contained(&pair[1]) {
                            // find joining colour
                            let colour = h.keys().filter(|&&c| c != pair[0].colour && c != pair[1].colour).collect::<Vec<_>>();

                            return Grid::new(1, 1, *colour[0]);
                        }
                    }
                }

                Grid::new(1, 1, Black)
            };

            if run_experiment(&file, 210, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

            // Cross example knowledge needed for closure
            let h = examples.bleached_io_map();

            let func = |ex: &Example| {
                if let Some(grid) = h.get(&ex.input.grid.bleach().to_json()) {
                    grid.clone()
                } else {
                    Grid::trivial()
                }
            };

            if run_experiment(&file, 221, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

        if run_experiment(&file, 222, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        //if run_experiment(&file, 230, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_max().to_grid(), output) { return; };
        if run_experiment(&file, 230, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.shapes.find_max().to_grid(), output) { return; };

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

        if run_experiment_tries(&file, 320, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

        if run_experiment_tries(&file, 330, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        // No divider
        let func = |gi: &Grid, _: &Grid, n: &mut usize| {
            let shapes = gi.split_2();
            if shapes.is_empty() || shapes.shapes.len() != 2 { return Grid::trivial(); }
            let diff = shapes.shapes[0].diff(&shapes.shapes[1]);

            if let Some(diff) = diff {
                diff_only(&diff.to_grid(), colour, n)
            } else {
                Grid::trivial()
            }
        };

        if run_experiment_tries(&file, 340, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        let colours = examples.examples[0].input.grid.cell_colour_cnt_map();

        for (&in_colour, _) in colours.iter() {
            let func = |ex: &Example| {
                let colour = ex.output.grid.colour;
                let side = ex.output.grid.cells.rows;
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

            if run_experiment(&file, 240, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

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

            if run_experiment(&file, 250, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = Div9Out;
    if all || cat.contains(&gc) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        //if run_experiment_examples(&file, experiment, trans, is_test, &examples, &targets, done, tries, &|exs| cat_div9(&exs), output) { return; };

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

        if run_experiment(&file, 251, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = GravityDown;
    if all || cat.contains(&gc) || cat.contains(&GravityUp) { 
        *cap_cats.entry(gc).or_insert(0) += 1;

        if run_experiment_tries(&file, 350, experiment, trans, is_test, &examples, &targets, done, tries, &|ex, _, n| gravity_only(ex, n), output) { return; };

        let func = |ex: &Example| {
                let mut shapes = ex.input.shapes.clone();
                let sccm = shapes.shape_colour_cnt_map();
                let mut fill_colour = NoColour;

                shapes.shapes = Vec::new();
                
                for (col, ss) in sccm.iter() {
                    if ss.len() == 1 {
                        shapes.shapes.push(ss[0].clone())
                    } else {
                        fill_colour = col.clone();
                    } 
                }

                if shapes.shapes.is_empty() {
                    return Grid::trivial();
                }

                let grid = shapes.to_grid();

                grid.flood_fill(grid.cells.rows - 1, 0, NoColour, fill_colour)
        };

        if run_experiment(&file, 251, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            if run_experiment_tries(&file, 500, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

                if run_experiment(&file, 270, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

                if run_experiment(&file, 281, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
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

            if run_experiment(&file, 282, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };

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

            if run_experiment(&file, 280, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
            */
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
    let gc = SingleColourOut;
    if all || cat.contains(&gc) && !cat.contains(&SingleColourIn) {
        *cap_cats.entry(gc).or_insert(0) += 1;

        if run_experiment(&file, 290, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_pixels_min().to_grid(), output) { return; };

        if run_experiment(&file, 300, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.find_pixels_max().to_grid(), output) { return; };

        if run_experiment(&file, 301, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_max().to_grid(), output) { return; };

        if run_experiment(&file, 302, experiment, trans, is_test, &examples, &targets, done, tries, &|ex| ex.input.coloured_shapes.hollow_cnt_min().to_grid(), output) { return; };

        let out_colours = examples.find_output_colours();

        if out_colours.len() == 1 {
            let in_colours = examples.find_input_colours().uniq(out_colours);
            if in_colours.len() == 2 {
                let func = |ex: &Example| {
                  let c1s = ex.input.grid.find_colour(in_colours[0]);
                  let c2s = ex.input.grid.find_colour(in_colours[1]);
                  let len = c1s.len();
                  if len == c2s.len() {
                      let y_dim = c1s[0].col == c2s[0].col;
                      let mut grid: Grid;

                      if y_dim {
                          let width = c1s[0].row.min(c2s[0].row) - c1s[0].row.min(c2s[0].row) - 1; // assume all same

                          grid = Grid::new(width, len, Black);

                          for (y, (a, b)) in c1s.iter().zip(c2s.iter()).enumerate() {
                              for x in a.row + 1 .. b.row {
                                  grid.cells[(x - a.row - 1, y)].row = ex.input.grid.cells[(x,a.col)].row;
                                  grid.cells[(x - a.row - 1, y)].col = ex.input.grid.cells[(x,a.col)].col;
                                  grid.cells[(x - a.row - 1, y)].colour = ex.input.grid.cells[(x,a.col)].colour;
                              }
                          }
                      } else {
                          let height = c1s[0].col.max(c2s[0].col) - c1s[0].col.min(c2s[0].col) - 1; // assume all same

                          grid = Grid::new(len, height, Black);

                              for (x, (a, b)) in c1s.iter().zip(c2s.iter()).enumerate() {
                              for y in a.col + 1 .. b.col {
                                  grid.cells[(x, y - a.col - 1)].row = ex.input.grid.cells[(a.row,y)].row;
                                  grid.cells[(x, y - a.col - 1)].col = ex.input.grid.cells[(a.row,y)].col;
                                  grid.cells[(x, y - a.col - 1)].colour = ex.input.grid.cells[(a.row,y)].colour;
                              }
                          }
                      };
                      return grid;
                  }

                  Grid::trivial()
                };

                if run_experiment(&file, 303, experiment, trans, is_test, &examples, &targets, done, tries, &func, output) { return; };
            }
        }

        *cap_todo.entry(gc).or_insert(0) += 1;
    } 
}

fn run_experiment(file: &str, experiment: usize, experiment_todo: &str, trans: Transformation, is_test: bool, examples: &Examples, targets: &[Grid], done: &mut BTreeSet<String>, tries: &mut usize, func: &(dyn Fn(&Example) -> Grid + std::panic::RefUnwindSafe), output: &mut BTreeMap<String, Vec<OutputData>>) -> bool {
    if !experiment_todo.is_empty() {
        if let Ok(ex) = experiment_todo.parse::<usize>() {
            if ex != experiment {
                return false;
            }
        } 
    }
    if done.contains(file) {   // already done???
        return true;
    }

    let ans = panic::catch_unwind(|| experiment_example(examples, file, experiment, trans, func));

    let ans = match ans {
        Ok(ans) => ans,
        Err(e) => {
            eprintln!("{file} / {experiment} Exception: {e:?}");
            vec![Grid::trivial()]
        },
    };

    *tries += 1;

    save(file, experiment, trans, is_test, &ans, targets, done, output)
}

fn run_experiment_tries(file: &str, experiment: usize, experiment_todo: &str, trans: Transformation, is_test: bool, examples: &Examples, targets: &[Grid], done: &mut BTreeSet<String>, tries: &mut usize, func: &(dyn Fn(&Grid, &Grid, &mut usize) -> Grid + RefUnwindSafe), output: &mut BTreeMap<String, Vec<OutputData>>) -> bool {
    if !experiment_todo.is_empty() {
        if let Ok(ex) = experiment_todo.parse::<usize>() {
            if ex != experiment {
                return false;
            }
        } 
    }
    if done.contains(file) {   // already done???
        return true;
    }

    let ans = panic::catch_unwind(|| experiment_grid(examples, file, experiment, trans, func));

    let ans = match ans {
        Ok(ans) => ans,
        Err(e) => {
            eprintln!("{file} / {experiment} Exception: {e:?}");
            vec![Grid::trivial()]
        },
    };

    *tries += 1;

    save(file, experiment, trans, is_test, &ans, targets, done, output)
}

fn save(file: &str, experiment: usize, trans: Transformation, is_test: bool, ans: &[Grid], targets: &[Grid], done: &mut BTreeSet<String>, results: &mut BTreeMap<String, Vec<OutputData>>) -> bool {
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

    if !is_test && same {
        add_real_output(file, ans, results);

        done.insert(file.to_string());
        println!("Success: {experiment:>05} {trans:?} / {file}");

        true
    } else if is_test && ans_size > 0 {
        add_real_output(file, ans, results);

        done.insert(file.to_string());
        println!("Test Success: {experiment:>05} {trans:?} / {file}");

        true
    } else {
        add_dummy_output(file, ans.len(), results);

        false
    }
}
