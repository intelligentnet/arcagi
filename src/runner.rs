use std::panic;
use std::panic::RefUnwindSafe;
use std::collections::{BTreeSet, BTreeMap};
use pathfinding::prelude::Matrix;
use crate::cats::*;
use crate::examples::*;
use crate::experiments::*;
use crate::rules::*;
//use crate::oldrules::*;
use crate::grid::*;
use crate::shape::*;
use crate::cell::*;
use crate::data::*;

pub fn runner(data: &str, catfile: &str, all: bool) {
    let tdata = load_files(data);
    let is_test = data == "test";
    let mut cnt = 0;
    let mut output: BTreeMap<String, Vec<OutputData>> = BTreeMap::new();
    let mut cap_cats: BTreeMap<GridCategory, i32> = BTreeMap::new();
    let mut cap_todo: BTreeMap<GridCategory, i32> = BTreeMap::new();
    let mut done: BTreeSet<String> = BTreeSet::new();
    let mut todo: BTreeSet<(String, GridCategory)> = BTreeSet::new();

    'outer:
    for (file_name, tf) in tdata.iter() {
//println!("Doing: {file}");
        if !catfile.is_empty() && catfile != file_name {
            continue;
        }

        let file = file_name.to_string();
        let example = Examples::new(tf);
        //let cat = categorise_grids(&example.examples);
        let cat = &example.cat;
        println!("{file}: {:?}", cat);
        let targets: Vec<Grid> = example.tests.iter().map(|test| test.output.grid.clone()).collect();

//difference(&example);
        let gc = GridCategory::BlankIn;

        if all || cat.contains(&gc) { // 2 done
            *cap_cats.entry(gc).or_insert(0) += 1;

            let colour = example.examples[0].output.grid.colour;

            let func = &|ex: &Example| {
                let odd = ex.cat.contains(&GridCategory::InOutSquareSameSizeOdd);
                let mut grid;

                if odd {
                    // TODO: Might be improved by using permutation of output grid?
                    grid = ex.input.grid.clone();

                    for (x, y) in grid.cells.keys() {
                        if x == 0 || y == 0  || x == grid.cells.rows - 1 || y == grid.cells.columns - 1 {
                            grid.cells[(x,y)].colour = colour;
                        } else if x % 2 == 0 || y % 2 == 0 {
                            grid.cells[(x,y)].colour = colour;
                        }
                    }
                } else {
                    let sq = ex.cat.contains(&GridCategory::InOutSquareSameSize);

                    grid = ex.input.grid.do_circle(colour, sq);
                }

                grid
            };

            if run_experiment(&file, 0, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::BlackPatches;
        if all || cat.contains(&gc) { // 3?
            *cap_cats.entry(gc).or_insert(0) += 1;
//difference(&example);

            // TODO: 981571dc.json and af22c60d.json mirrored
            // 1e97544e.json e95e3d8e.json e95e3d8e.json 0dfd9992.json + test c3f564a4.json 29ec7d0e.json
            let func = &|ex: &Example| {
//println!("{}", ex.input.black.is_empty());
                if ex.input.black.is_empty() {
                    return Grid::trivial();
                }

                /* TODO: pairs does not work 7c9b52a0
                if ex.input.shapes.len() > 0 && ex.input.shapes.len() == ex.input.black.len() && ex.input.black.shapes[0].contained_by(&ex.input.shapes.shapes[0]) {
                    let mut shape = ex.input.black.shapes[0].clone();
                    let mut shapes = Shapes::new_sized(shape.cells.rows, shape.cells.columns);

                    shape.to_origin_mut();
                    shapes.shapes.push(shape.clone());

let pairs = ex.input.black.pair_shapes(&ex.input.shapes, false);
                    for (b, s, _) in pairs.iter() {
b.show_summary();
s.show_summary();
                    }
println!("{:?}", pairs.len());
                    /*
                    for (b, s) in ex.input.black.shapes.iter().zip(ex.input.shapes.shapes.iter()) {
b.show_summary();
s.show_summary();
                        let ts = s.translate_absolute(s.ox - b.ox, s.oy - b.oy);

                        shapes.shapes.push(ts.clone());
                    }
                    */
//println!("here {}", ex.input.black.shapes[0].contained_by(&ex.input.shapes.shapes[0]));
shapes.to_grid().show();
                    return shapes.to_grid();
                }
            */

                let mut grid = ex.input.grid.clone();

                grid.fill_border();
//grid.show();
                let mut px = 0;
                let mut py = 0;

                loop {
                    let black = grid.find_black_patches();

                    if black.is_empty() {
                        break;
                    }

                    for p in black.shapes.iter() {
                        //println!("{}/{} {} {}", p.ox, p.oy, p.cells.rows, p.cells.columns);
                        let xa = if p.oy == 0 { 
                            Vec::new()
                        } else {
                            if let Ok(xs) = grid.cells.slice(p.ox .. p.ox+p.cells.rows, p.oy-1 .. p.oy) {
                                let xa: Vec<_> = xs.values().map(|c| c.colour).collect();
                                //println!("x {xa:?}");
                                if Colour::single_colour_vec(&xa) {
                                    Vec::new()
                                } else {
                                    xa
                                }

                            } else {
                                Vec::new()
                            }
                        };
                        let ya = if p.ox == 0 {
                            Vec::new()
                        } else {
                            if let Ok(ys) = grid.cells.slice(p.ox-1 .. p.ox, p.oy .. p.oy+p.cells.columns) {
                                let ya: Vec<_> = ys.values().map(|c| c.colour).collect();
                                //println!("y {ya:?}");
                                if Colour::single_colour_vec(&ya) {
                                    Vec::new()
                                } else {
                                    ya
                                }
                            } else {
                                Vec::new()
                            }
                        };
//println!("{}/{} {xa:?} {ya:?}", p.ox, p.oy);
                        if !xa.is_empty() && xa.len() >= ya.len() {
                            //let (xo, yo) = grid.find_x_seq(p.oy+1, &xa, p.cells.columns);
                            let (xo, yo) = grid.find_x_seq(p.ox, p.oy, &xa, p.cells.columns);
                            if xo == usize::MAX && yo == usize::MAX || px == xo && py == yo {
//println!("1");
                                return Grid::trivial();
                            }
//println!("x {}/{}", xo, yo);
                            for x in 0 .. p.cells.rows {
                                for y in 0 .. p.cells.columns {
                                    if grid.cells[(p.ox+x,p.oy+y)].colour == Colour::Black && xo+x < grid.cells.rows && yo+1+y < grid.cells.columns {
                                        grid.cells[(p.ox+x,p.oy+y)] = grid.cells[(xo+x,yo+1+y)].clone();
                                    }
                                }
                            }

                            px = xo;
                            py = yo;
//grid.diff(target).unwrap().show_full();
//grid.show();
                        }
                        else if !ya.is_empty() {
                            let (xo, yo) = grid.find_y_seq(p.ox, p.oy, &ya, p.cells.rows);
                            if xo == usize::MAX && yo == usize::MAX || px == xo && py == yo {
//println!("2");
                                return Grid::trivial();
                            }
//println!("y {}/{}", xo, yo);
                            for x in 0 .. p.cells.rows {
                                for y in 0 .. p.cells.columns {
                                    if grid.cells[(p.ox+x,p.oy+y)].colour == Colour::Black && xo+1+x < grid.cells.rows && yo+y < grid.cells.columns {
                                        grid.cells[(p.ox+x,p.oy+y)] = grid.cells[(xo+1+x,yo+y)].clone();
                                    }
                                }
                            }

                            px = xo;
                            py = yo;
                        } else {
                            return Grid::trivial();
                        }
                    }
                }
//grid.show();
//target.show();
//grid.diff(target).unwrap().show_full();
                
                //Grid::trivial()
                grid
            };

            if run_experiment(&file, 0, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            let func = |ex: &Example| {
                let largest = ex.input.coloured_shapes.largest();

                if largest.cells.rows < 3 || largest.cells.columns < 3 {
                    return Grid::trivial();
                }

                let shape = largest.subshape(0, 3, 0, 3);
                let mut grid = ex.input.grid.clone();

                for s in ex.input.coloured_shapes.shapes.iter() {
                    if s.size() == 1 {
                        let new_shape = shape.translate_absolute(s.ox,s.oy);

                        grid.copy_shape_to_grid(&new_shape);
//new_shape.show_summary();
                    }
                }
//grid.show();

                grid
            };

            if run_experiment(&file, 1, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            if !example.examples[0].input.black.is_empty() && !example.examples[0].output.shapes.shapes.is_empty() {
                let mut ccm = example.examples[0].output.grid.as_shape().cell_colour_cnt_map();
                // assume grid background and border shapes same colour for all tests
                let bg = example.examples[0].output.grid.cells[(0,0)].colour;
                let border = example.examples[0].output.shapes.shapes[0].colour;

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
                        if s.ox == 0 || s.oy == 0 {
                            return Grid::trivial();
                        }
                        if s.ox <= 2 || s.oy <= 2 || s.ox >= grid.cells.rows - side - 2 || s.oy >= grid.cells.columns - side - 2 {
                            grid.flood_fill_in_situ(s.ox, s.oy, Colour::NoColour, border);
                        } else {
                            grid.flood_fill_in_situ(s.ox, s.oy, Colour::NoColour, *inner);
                        }
                    }
//grid.show();

                    grid
                };

                if run_experiment(&file, 2, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            }

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::BGGridInBlack;
        if all || cat.contains(&gc) && !cat.contains(&GridCategory::BGGridOutBlack) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            if run_experiment(&file, 10, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.find_min().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 20, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.find_max_colour_count().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 30, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.has_mirror_x().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 40, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.has_mirror_y().to_grid(), &mut output) { continue; };
            //if run_experiment_examples(&file, 1000, is_test, &example, &targets, &mut done, &|exs| cat_overlay(&exs), &mut output) { continue; };

            //-if run_experiment_examples(&file, 1010, is_test, &example, &targets, &mut done, &|exs| cat_shape_fill(&exs), &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::InOutSameShapes;
        if all || cat.contains(&gc) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            let func = &|ex: &Example| {
                let mut grey = Shape::trivial();
                let mut colours: BTreeSet<Colour> = BTreeSet::new();

                for s in ex.input.shapes.shapes.iter() {
                    if s.colour == Colour::Grey {
                        grey = s.clone();
                    } else {
                        colours.insert(s.colour);
                    }
                }

                if colours.len() != 2 || grey == Shape::trivial() {
                    return Grid::trivial();
                }
                let sg = ex.input.grid.subgrid(grey.ox, grey.cells.rows, grey.oy, grey.cells.columns);
                let colours: Vec<&Colour> = colours.iter().collect();
//sg.toddle_colour(*colours[0], *colours[1]).show();

                let s = sg.toddle_colour(*colours[0], *colours[1]);

                let mut shapes = Shapes::new_sized(ex.input.grid.cells.rows, ex.input.grid.cells.columns);

                shapes.shapes.push(ex.input.grid.as_shape());
                shapes.shapes.push(s.as_shape());
//shapes.to_grid().show();

                shapes.to_grid()
            };

            if run_experiment(&file, 41, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }

        let gc = GridCategory::BGGridOutBlack;
        if all || cat.contains(&gc) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            if all || cat.contains(&GridCategory::IdenticalNoColours) {
                let func = &|ex: &Example| {
                    if ex.input.shapes.shapes.len() % 2 != 0 || ex.input.shapes.shapes.len() == ex.input.coloured_shapes.shapes.len() { //|| ex.input.shapes.shapes.len() > 4 {
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

                if run_experiment(&file, 50, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            }

            let func = &|ex: &Example| {
                let rows = ex.input.grid.cells.rows;
                let cols = ex.input.grid.cells.columns;
                let h = ex.input.grid.cell_colour_cnt_map();
//print!("{h:?} -> ");
//let h2 = ex.input.grid.cell_colour_cnt_map();
//println!("{h2:?}");
                let mut grid = Grid::new(rows, cols, Colour::Black);

                for (col, size) in h.iter() {
                    let y = Colour::to_usize(*col) - 1;

                    if *size >= rows || y >= cols {
                        return Grid::trivial();
                    }

                    for x in 0 .. *size {
                        let x = rows - x - 1;

                        grid.cells[(x,y)].x = x;
                        grid.cells[(x,y)].y = y;
                        grid.cells[(x,y)].colour = *col;
                    }
                }

                grid
            };

            if run_experiment(&file, 60, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            let func = |gi: &Example| {
//gi.input.shapes.show();
                let mut xy: Vec<(usize, usize, Colour)> = Vec::new();
                let mut colour = Colour::NoColour;
                let mut shapes = gi.input.shapes.clone();

                for s in gi.input.shapes.shapes.iter() {
                    if colour == Colour::NoColour {
                        colour = s.colour;
                    }
                    if colour != s.colour {
                        //xx = s.cells[(0,0)].x;
                        //yy = s.cells[(0,0)].y;
                        xy.push((s.cells[(0,0)].x, s.cells[(0,0)].y, s.colour));

                        break;
                    }
                }
//println!("{xx}/{yy} {:?}", colour);

                for ss in shapes.shapes.iter_mut() {
                    for (xx, yy, colour) in xy.iter() {
                        if ss.cells[(0,0)].x == *xx || ss.cells[(0,0)].y == *yy {
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

            if run_experiment(&file, 70, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

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
                    if ans.size() != s.size() || ans.colour == Colour::NoColour  || i / ans.cells.rows >= ans.cells.rows  {
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
            if run_experiment(&file, 50, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::FullyPopulatedOut;
        if all || cat.contains(&gc) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            if run_experiment(&file, 80, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.find_sub_max().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 90, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.largest().to_grid(), &mut output) { continue; };
//target.show();
//ans.show();

            if run_experiment(&file, 100, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.find_sub_min().to_grid(), &mut output) { continue; };

            let common_colours = example.find_output_colours();

            for colour in common_colours {
                if run_experiment(&file, 110, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.cell_colour_cnts(colour).to_grid(), &mut output) { continue 'outer; };
            }

            if run_experiment(&file, 120, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.find_sub_largest_count().to_grid(), &mut output) { continue; };

            let func = |ex: &Example| {
                let mut cnt = 0;
                let mut shapes: Vec<Shape> = Vec::new();

                for s in ex.input.shapes.shapes.iter() {
                    if s.size() >= 9 && s.colour != Colour::Mixed && s.dense() {
                        cnt += 1;

                        shapes.push(s.clone())
                    }
                }

                if cnt % 3 != 0 {
                    return Grid::trivial();
                }

                let height = cnt / 3;
                let mut grid = Grid::new(height, 3, Colour::Black);

                let mut x = 0;

                // Get right ordering by munging x coord then sorting
                for (i, s) in shapes.iter_mut().enumerate() {
                    if i > 0 && (i % 3) == 0 {
                        x += 1;
                    }
                    s.ox = x;
                }

                shapes.sort_by(|a, b| (a.ox, a.oy).cmp(&(b.ox, b.oy)));
//println!("{} by {}", ex.input.grid.cells.rows, ex.input.grid.cells.columns);

                let mut i = 0;

                for s in shapes.iter() {
                    let x = i / 3;
                    let y = i % 3;
//println!("{:?} {i} {}/{} -> {}/{}", s.colour, s.ox, s.oy, x, y);

                    grid.colour = s.colour;

                    grid.cells[(x,y)].x = i / 3;
                    grid.cells[(x,y)].y = i / height;
                    grid.cells[(x,y)].colour = s.colour;

                    i += 1;
                }
//grid.show();

                grid
            };

            if run_experiment(&file, 121, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::MirrorXOut;
        if all || cat.contains(&gc) || cat.contains(&GridCategory::MirrorYOut) {
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

            if run_experiment(&file, 280, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::Div9In;
        if all || cat.contains(&gc) && cat.contains(&GridCategory::Div9Out) { 
            *cap_cats.entry(gc).or_insert(0) += 1;

            //-if run_experiment_examples(&file, 1020, is_test, &example, &targets, &mut done, &|exs| cat_9_in_out(&exs), &mut output) { continue; };


            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::Div9Out;
        if all || cat.contains(&GridCategory::Is3x3In) && cat.contains(&gc) && !cat.contains(&GridCategory::Div9In){ 
            *cap_cats.entry(gc).or_insert(0) += 1;

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::Div9In;
        if all || cat.contains(&gc) && cat.contains(&GridCategory::Is3x3Out) && !cat.contains(&GridCategory::Div9Out) { 
            *cap_cats.entry(gc).or_insert(0) += 1;

            //-if run_experiment_examples(&file, 1030, is_test, &example, &targets, &mut done, &|exs| cat_mirror(&exs), &mut output) { continue; };

            //-if run_experiment_examples(&file, 1040, is_test, &example, &targets, &mut done, &|exs| cat_double(&exs), &mut output) { continue; };

            //-if run_experiment_examples(&file, 1050, is_test, &example, &targets, &mut done, &|exs| cat_expand_3x3(&exs), &mut output) { continue; };

            if example.examples[0].input.grid.size() * 9 == example.examples[0].output.grid.size() {
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
    //println!("{:?}", shapes);
                    shapes.to_grid()
                };

                if run_experiment(&file, 130, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            }


            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::IdenticalNoPixels;
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
                    if s.ox == 0 && s.cells.columns == grid.cells.columns {
                        x = true;
                        border1 = s;
                    } else if s.ox == grid.cells.rows - 1 && s.cells.columns == grid.cells.columns {
                        x = true;
                        border2 = s;
                    } else if s.oy == 0 && s.cells.rows == grid.cells.rows {
                        border1 = s;
                    } else if s.oy == grid.cells.columns - 1 && s.cells.rows == grid.cells.rows {
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
            if run_experiment(&file, 140, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::Is3x3In;
        if all || cat.contains(&gc) && cat.contains(&GridCategory::Is3x3Out){ 
//println!("#### {file}");
            *cap_cats.entry(gc).or_insert(0) += 1;

            //-if run_experiment_examples(&file, 1060, is_test, &example, &targets, &mut done, &|exs| cat_mirror(&exs), &mut output) { continue; };

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
                                m[(0,y)].x = 0;
                                m[(0,y)].y = y;
                            }
                            for x in 1 .. 3 {
                                for y in 0 .. 3 {
                                    m[(x,y)].x = x;
                                    m[(x,y)].y = y;
                                    m[(x,y)].colour = s.cells[(x-1,y)].colour;
                                }
                            }
                            big = Shape::new(0, 0, &m);
                        } else if s.cells.rows == 3 && s.cells.columns == 2 {
                            let mut m = Matrix::new(3, 3, Cell::new(0, 0, 0));
                            for x in 1 .. 3 {
                                m[(x,0)].x = x;
                                m[(x,0)].y = 0;
                            }
                            for x in 0 .. 3 {
                                for y in 1 .. 3 {
                                    m[(x,y)].x = x;
                                    m[(x,y)].y = y;
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

            if run_experiment(&file, 140, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::InOutSquareSameSize;
        if all || cat.contains(&gc) { // 164
            *cap_cats.entry(gc).or_insert(0) += 1;
//difference_shapes(&example, false);

            if all || cat.contains(&GridCategory::InSameCountOut) || cat.contains(&GridCategory::InSameCountOutColoured) {
                //if run_experiment_examples(&file, 1070, is_test, &example, &targets, &mut done, &|exs| cat_shape_substitute(&exs), &mut output) { continue; };

                let func = |ex: &Example| {
                    let biggest = ex.input.coloured_shapes.biggest_shape();
                    if biggest.size() != ex.input.grid.size() {
                        return Grid::trivial();
                    }
                    let (idx, dir) = ex.input.grid.corner_idx();
                    if dir == Direction::Other {
                        return Grid::trivial();
                    }
                    let mut shapes = Shapes::new_from_shape(&biggest);
                    let body = ex.input.grid.corner_body(dir.opposite());
                    let four = body.split_4();

                    if four.is_empty() {
                        return Grid::trivial();
                    }

                    four.iter().zip(idx.cells.values())
                        .for_each(|(s,c)| shapes.add(&s.recolour(s.colour, c.colour).as_shape()));

                    shapes.to_grid()
                };

                if run_experiment(&file, 141, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

                let func = |ex: &Example| {
                    if ex.input.shapes.shapes.len() > 20 {
                        return Grid::trivial();
                    }

                    let mut shapes = ex.input.shapes.clone();

                    shapes.shapes = Vec::new();

                    for s in &ex.input.shapes.consolidate_shapes().shapes {
                        shapes.shapes.push(s.mirrored_x());
                    }
//shapes.to_grid().show();

                    shapes.to_grid()
                };

                if run_experiment(&file, 142, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            }

            if all || cat.contains(&GridCategory::InLessCountOut) {
                //// see 045e512c.json
//println!("#### {}", example.examples[0].input.shapes.len());
//example.examples[0].input.shapes.show();
            }

            //-if run_experiment_examples(&file, 1080, is_test, &example, &targets, &mut done, &|exs| cat_shape_fill(&exs), &mut output) { continue; };
            //if run_experiment_examples(&file, 1090, is_test, &example, &targets, &mut done, &|exs| cat_overlay(&exs), &mut output) { continue; };
            // 0ca9ddb6 4258a5f9 913fb3ed 95990924 b60334d2 test
            if run_experiment_tries(&file, 0, is_test, &example, &targets, &mut done, &|ex, _, n| transform_only(ex, n), &mut output) { continue; };
            //-if run_experiment_examples(&file, 1100, is_test, &example, &targets, &mut done, &|exs| cat_mirror(&exs), &mut output) { continue; };

            let func = |ex: &Example| {
                let mut grid = ex.input.grid.clone();

                for s in ex.input.shapes.shapes.iter() {
                    if s.size() == 1 {
                        grid.cells[(ex.input.grid.cells.rows - 1,s.oy)].colour = s.colour;
                        grid.cells[(s.ox,s.oy)].colour = Colour::Black;
                    }
                }

                grid
            };

            if run_experiment(&file, 220, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

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
            if run_experiment(&file, 51, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            let func = &|ex: &Example| {
                //if ex.input.coloured_shapes.shapes.len() != 1 || !ex.input.coloured_shapes.shapes[0].is_square() || ex.input.coloured_shapes.shapes[0].size() != 36 {
                //    return Grid::trivial();
                //}
                fn do_diag(grid: &mut Grid, s: &Shape) {
                    /* Without rotation
                    grid.diagonal_mut(s.ox, s.oy, Direction::TopLeft, s.cells[(0,0)].colour);
                    grid.diagonal_mut(s.ox, s.oy + s.cells.columns, Direction::TopRight, s.cells[(0,s.cells.columns-1)].colour);
                    grid.diagonal_mut(s.ox + s.cells.rows, s.oy + s.cells.columns, Direction::BottomRight, s.cells[(s.cells.rows-1,s.cells.columns-1)].colour);
                    grid.diagonal_mut(s.ox + s.cells.rows, s.oy, Direction::BottomLeft, s.cells[(s.cells.rows-1,0)].colour);
                    */
                    grid.diagonal_mut(s.ox, s.oy, Direction::TopLeft, s.cells[(s.cells.rows-1,0)].colour);
                    grid.diagonal_mut(s.ox, s.oy + s.cells.columns, Direction::TopRight, s.cells[(0,0)].colour);
                    grid.diagonal_mut(s.ox + s.cells.rows-1, s.oy + s.cells.columns-1, Direction::BottomRight, s.cells[(0,s.cells.columns-1)].colour);
                    grid.diagonal_mut(s.ox + s.cells.rows, s.oy, Direction::BottomLeft, s.cells[(s.cells.rows-1,s.cells.columns-1)].colour);
                }

                let mut grid = ex.input.grid.clone();
//ex.input.shapes.show();

                for s in ex.input.coloured_shapes.shapes.iter() {
                    if s.cells.rows != 8 || s.cells.columns != 6 {
                        return grid;
                    }
//s.show();
                    if s.size() > 16 {
                        // TODO: Parameterise it
                        let ss1 = s.subshape_remain(0, 4, 0, 6).shrink();
                        let ss2 = s.subshape_remain(4, 4, 0, 6).shrink();

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

            if run_experiment(&file, 163, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::InOutSquare;
        if all || cat.contains(&gc) { // 27
            *cap_cats.entry(gc).or_insert(0) += 1;

            //if run_experiment_examples(&file, is_test, &example, &targets, &mut done, &|exs| cat_shape_fill(&exs), &mut output) { continue; };

            //-if run_experiment_examples(&file, 1050, is_test, &example, &targets, &mut done, &|exs| cat_expand_3x3(&exs), &mut output) { continue; };

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
            if run_experiment(&file, 160, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

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
            if run_experiment(&file, 162, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            let func = &|ex: &Example| {
                if ex.input.coloured_shapes.shapes.len() != 1 || !ex.input.coloured_shapes.shapes[0].is_square() || ex.input.coloured_shapes.shapes[0].size() != 36 {
                    return Grid::trivial();
                }

                let shape = &ex.input.coloured_shapes.shapes[0];
                let ss = shape.subshape(0, 3, 0, 3);

                ss.to_grid()
            };

            if run_experiment(&file, 163, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
//a.show();
        }
        let gc = GridCategory::InOutSameSize;
        if all || cat.contains(&gc) { // 97
            *cap_cats.entry(gc).or_insert(0) += 1;

            /* Causes problems probably not
            //if run_experiment_examples(&file, is_test, &example, &targets, &mut done, &|exs| cat_shape_fill(&exs), &mut output) { continue; };

            //if run_experiment_examples(&file, is_test, &example, &targets, &mut done, &|exs| cat_overlay(&exs), &mut output) { continue; };

            // 5
            if all || cat.contains(&GridCategory::InSameCountOut) {
                //if run_experiment_examples(&file, is_test, &example, &targets, &mut done, &|exs| cat_move_together(&exs), &mut output) { continue; };
            }
            */

            if run_experiment(&file, 150, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.toddle_colours().to_grid(), &mut output) { continue; };

            // Clunky and not very generic 42918530.json
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

            if run_experiment(&file, 161, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            if !cat.contains(&GridCategory::FullyPopulatedIn) {
                let func = |gi: &Example| {
                    let mut si = gi.clone();
                    let mut dx = 0;
                    let mut dy = 0;

                    for s in &gi.input.shapes.shapes {
                        if dx == 0 {
                            dx = s.ox;
                        } else if s.ox % dx != 0 {
                            return Grid::trivial();
                        }
                        if dy == 0 {
                            dy = s.oy;
                        } else if s.oy % dy != 0 {
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
                        s.ox = i * dx;
                        s.oy = i * dy;

                        for (x, y) in s.cells.keys() {
                            s.cells[(x, y)].x = s.ox + x;
                            s.cells[(x, y)].y = s.oy + y;
                        }
                    }
//si.input.shapes.to_grid().show();

                    si.input.shapes.to_grid()
                };

                if run_experiment(&file, 170, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
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

            if run_experiment(&file, 180, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

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
            if run_experiment(&file, 51, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            if (all || cat.contains(&GridCategory::InOutSameShapesColoured)) && !example.examples[0].output.shapes.shapes.is_empty() {
                let mut pairs: BTreeMap<Shape, Shape> = BTreeMap::new();

                for exs in example.examples.iter() {
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

                            s.flood_fill_in_situ(x - s.ox, y - s.oy, Colour::NoColour, *colour);
                        }
                    }

                    shapes.to_grid()
                };
                if run_experiment(&file, 181, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            }

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::Double;
        if all || cat.contains(&gc) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            //-if run_experiment_examples(&file, 1110, is_test, &example, &targets, &mut done, &|exs| cat_double(&exs), &mut output) { continue; };

            let func = |gi: &Example| {
                /*
                if gi.input.shapes.shapes.len() != 1 {
                    return Grid::trivial();
                }
                */

                let s = &gi.input.grid.as_shape();
                let s = s.toddle_colour(Colour::Black, s.colour);
                let rows = s.cells.rows;
                let cols = s.cells.columns;
                let mut shapes = Shapes::new_sized(rows * 2, cols * 2);

                shapes.shapes.push(s.clone());
                shapes.shapes.push(s.translate_absolute(rows, 0));
                shapes.shapes.push(s.translate_absolute(0, cols));
                shapes.shapes.push(s.translate_absolute(rows, cols));

                shapes.to_grid()
            };

            if run_experiment(&file, 190, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            // TODO Important 10fcaaa3
            let common_colour = example.find_output_colours();

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

                copy.trim_to_grid()
            };

            if run_experiment(&file, 190, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        /*
        } else if all || cat.contains(&GridCategory::InLessThanOut) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        */
        let gc = GridCategory::SingleShapeOut;
        if all || cat.contains(&gc) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            if all || cat.contains(&GridCategory::SingleShapeIn) {
                //if run_experiment_examples(&file, 1120, is_test, &example, &targets, &mut done, &|exs| cat_overlay(&exs), &mut output) { continue; };

                //if run_experiment_examples(&file, is_test, &example, &targets, &mut done, &|exs| cat_move_same_colour(&exs), &mut output) { continue; };

                let func = |gi: &Grid, go: &Grid, n: &mut usize| {
                    let grid = gi.recolour(gi.colour, go.colour);

                    move_only(&grid, n)
                };

                if run_experiment_tries(&file, 310, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

                let func = |gi: &Example| {
                    if gi.input.coloured_shapes.shapes.is_empty() {
                        return Grid::trivial();
                    }

                    gi.input.coloured_shapes.shapes[0].to_origin().to_grid()
                };

                if run_experiment(&file, 200, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            }
            if all || !cat.contains(&GridCategory::SingleShapeIn) {
                let colour = example.examples[0].output.grid.colour;
                let x = example.examples[0].output.grid.cells.rows;
                let y = example.examples[0].output.grid.cells.columns;

                let func = |ex: &Example| {
                    let mut i = 0;
                    let mut grid = Grid::new(x, y, Colour::Black);

                    for s in ex.input.shapes.shapes.iter() {
                        if i >= y {
                            return Grid::trivial();
                        }
                        if s.colour == colour && s.size() > 1 {
                            grid.cells[(0, i)].colour = colour;
                            i += 1;
                        }
                    }

                    grid
                };

                if run_experiment(&file, 400, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            }

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        } 
        let gc = GridCategory::OutLessThanIn;
        if all || cat.contains(&gc) { 
            *cap_cats.entry(gc).or_insert(0) += 1;

            if run_experiment(&file, 201, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.hollow_cnt_unique().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 205, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.first().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 206, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.last().to_grid(), &mut output) { continue; };

            let func = |ex: &Example| {
                for s in &ex.input.shapes.shapes {
                    if !s.is_mirror_x() && !s.is_mirror_y() {
                        return s.to_grid();
                    }
                }

                Grid::trivial()
            };

            if run_experiment(&file, 207, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            let func = |ex: &Example| {
                for s in &ex.input.shapes.shapes {
                    if s.is_mirror_x() || s.is_mirror_y() {
                        return s.to_grid();
                    }
                }

                Grid::trivial()
            };

            if run_experiment(&file, 208, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        }
        let gc = GridCategory::SingleColouredShapeOut;
        if all || cat.contains(&gc) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            //if run_experiment_examples(&file, 1130, is_test, &example, &targets, &mut done, &|exs| cat_overlay(&exs), &mut output) { continue; };

            let func = |ex: &Example| {
                let shape = ex.input.grid.as_shape().scale_up(2);

                Shapes::new_shapes(&[shape]).to_grid()
            };

            if run_experiment(&file, 2140, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            //if run_experiment_examples(&file, 1140, is_test, &example, &targets, &mut done, &|exs| cat_mirror(&exs), &mut output) { continue; };

            let edge_colour = example.largest_shape_colour();

            let func = |ex: &Example| {
                let h = ex.input.grid.cell_colour_cnt_map();

                if let Some(colour) = h.iter().min_by(|(_, cnt1), (_, cnt2)| cnt1.cmp(cnt2)).map(|(k, _)| k) {
                    let h = ex.input.shapes.shape_colour_cnt_map();

                    if let Some(vc) = h.get(colour) {
                        if vc.len() != 1 || vc[0].size() != 1 || vc[0].ox == 0 || vc[0].oy == 0 || vc[0].ox >= ex.input.grid.cells.rows - 1 || vc[0].oy >= ex.input.grid.cells.columns - 1 {
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

            if run_experiment(&file, 204, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        } 
        let gc = GridCategory::OutLessThanIn;
        if all || cat.contains(&gc) { 
            *cap_cats.entry(gc).or_insert(0) += 1;

            if all || cat.contains(&GridCategory::SinglePixelOut) {
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

                    Grid::new(1, 1, Colour::Black)
                };

                if run_experiment(&file, 210, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

                // Cross example knowledge needed for closure
                let h = example.bleached_io_map();

                let func = |ex: &Example| {
                    if let Some(grid) = h.get(&ex.input.grid.bleach().to_json()) {
                        grid.clone()
                    } else {
                        Grid::trivial()
                    }
                };

                if run_experiment(&file, 221, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
            }

            let func = |ex: &Example| {
                    let consolidated = ex.input.shapes.consolidate_shapes();
                    let scm = consolidated.shape_colour_cnt_map();

                    for v in scm.values() {
                        if v.len() == 1 {
                            return v[0].to_grid();
                        }
                    }

                    Grid::trivial()
            };

            if run_experiment(&file, 222, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            if run_experiment(&file, 230, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.find_max().to_grid(), &mut output) { continue; };

            let colour = example.examples[0].output.grid.colour;

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

            if run_experiment_tries(&file, 320, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

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

            if run_experiment_tries(&file, 330, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

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

            if run_experiment_tries(&file, 340, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            let func = |gi: &Example| {
                let mut m = 0;

                for s in &gi.input.shapes.shapes {
                    if s.size() == 4 && s.colour == Colour::Green {
                        m += 1;
                    }
                }
//println!("{m}");
//Grid::colour_every_nxn_for_m(colour, 4, m).show();

                Grid::colour_every_nxn_for_m(colour, 3, 4, m)
            };

            if run_experiment(&file, 240, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

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

                if run_experiment(&file, 250, is_test, &example, &targets, &mut done, &func, &mut output) { continue 'outer; };
            }
 
            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        } 
        let gc = GridCategory::Div9Out;
        if all || cat.contains(&gc) { 
            *cap_cats.entry(gc).or_insert(0) += 1;

            //if run_experiment_examples(&file, is_test, &example, &targets, &mut done, &|exs| cat_div9(&exs), &mut output) { continue; };

            let colour_map = example.find_colour_io_map();

            let func = |ex: &Example| {
                    let mut shapes = ex.input.shapes.clone();
                    
                    for s in shapes.shapes.iter_mut() {
                        if let Some(colour) = colour_map.get(&s.colour) {
                            s.recolour_mut(s.colour, *colour);
                        }
                    }

                    shapes.to_grid()
            };

            if run_experiment(&file, 251, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        } 
        let gc = GridCategory::GravityDown;
        if all || cat.contains(&gc) || cat.contains(&GridCategory::GravityUp) { 
            *cap_cats.entry(gc).or_insert(0) += 1;

            if run_experiment_tries(&file, 350, is_test, &example, &targets, &mut done, &|ex, _, n| gravity_only(ex, n), &mut output) { continue; };

            let func = |ex: &Example| {
                    let mut shapes = ex.input.shapes.clone();
                    let sccm = shapes.shape_colour_cnt_map();
                    let mut fill_colour = Colour::NoColour;

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

                    grid.flood_fill(grid.cells.rows - 1, 0, Colour::NoColour, fill_colour)
            };

            if run_experiment(&file, 251, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        } 
        /*
        let gc = GridCategory::SingleColouredShapeIn;
        if all || cat.contains(&gc) && cat.contains(&GridCategory::SingleColouredShapeOut) {
            // d631b094.json
            *cap_cats.entry(gc).or_insert(0) += 1;

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        } 
        */
        let gc = GridCategory::SingleColourIn;
        if all || cat.contains(&gc) {
            // ff28f65a.json
            *cap_cats.entry(gc).or_insert(0) += 1;

            if !cat.contains(&GridCategory::SingleColourOut) {
//println!("ffff");
//example.test.input.grid.show();
/*
                let func = |gi: &Grid, _: &Grid, n: &mut usize| {
                    let shapes = gi.to_shapes_base_bg(gi.cells[(1,3)].colour);
                    if shapes.len() < 2 { return Grid::trivial(); }
                    let diff = shapes.shapes[0].diff(&shapes.shapes[1]).unwrap();
                    let colour = example.examples[0].output.grid.colour;

                    diff.to_grid().diff_only(colour, *n)
                };

                if run_experiment_tries(&file, 500, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
*/

                if !example.examples.is_empty() && !example.examples[0].input.shapes.shapes.is_empty() {
                    // Global colour data for two rules
                    let colour = example.examples[0].input.shapes.shapes[0].colour;
                    let mut new_colour = colour;

                    for s in example.examples[0].output.shapes.shapes.iter() {
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

                        si.shapes.sort_by(|a, b| (a.oy, a.ox).cmp(&(b.oy, b.ox)));
                        let mut other = si.shapes.len() % 2 != 0;

                        for s in si.shapes.iter_mut() {
                            if other {
                                s.mut_force_recolour(new_colour);
                            }

                            other = !other;
                        }

                        si.to_grid()
                    };

                    if run_experiment(&file, 270, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

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

                    if run_experiment(&file, 280, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
                }

                let ccm = example.find_hollow_cnt_colour_map();

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

                if run_experiment(&file, 280, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };

                /* TODO 759f3fd3 translate_absolute_clip
                let sq_colour = if !example.examples.is_empty() && !example.examples[0].output.shapes.shapes.is_empty() {
                    example.examples[0].output.shapes.shapes[0].colour
                } else {
                    Colour::NoColour
                };

                let func = |ex: &Example| {
                    if ex.input.shapes.len() != 4 || ex.input.shapes.shapes[0].colour != Colour::NoColour {
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

                if run_experiment(&file, 280, is_test, &example, &targets, &mut done, &func, &mut output) { continue; };
                */
            }

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        } 
        let gc = GridCategory::SingleColourOut;
        if all || cat.contains(&gc) && !cat.contains(&GridCategory::SingleColourIn) {
            *cap_cats.entry(gc).or_insert(0) += 1;

            if run_experiment(&file, 290, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.find_pixels_min().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 300, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.find_pixels_max().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 301, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.hollow_cnt_max().to_grid(), &mut output) { continue; };

            if run_experiment(&file, 302, is_test, &example, &targets, &mut done, &|ex| ex.input.coloured_shapes.hollow_cnt_min().to_grid(), &mut output) { continue; };

            *cap_todo.entry(gc).or_insert(0) += 1;
            todo.insert((file.clone(), gc));
        } 

        cnt += 1;
    }

    let tries: i32 = cap_cats.values().sum();
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
    println!("cnt = {cnt}, tries = {tries}, done = {}", done.len());
    println!("{done:?}");
    println!("{:?}", todo.len());
}

fn run_experiment(file: &str, experiment: usize, is_test: bool, examples: &Examples, targets: &[Grid], done: &mut BTreeSet<String>, func: &(dyn Fn(&Example) -> Grid + std::panic::RefUnwindSafe), output: &mut BTreeMap<String, Vec<OutputData>>) -> bool {
    let ans = panic::catch_unwind(|| experiment_example(examples, file, experiment, func));

    let ans = match ans {
        Ok(ans) => ans,
        Err(e) => {
            eprintln!("{file} / {experiment} Exception: {e:?}");
            vec![Grid::trivial()]
        },
    };

    save(file, experiment, is_test, &ans, targets, done, output)
}

/*
fn run_experiment_examples(file: &str, experiment: usize,  is_test: bool, examples: &Examples, targets: &Vec<Grid>, done: &mut BTreeSet<String>, func: &dyn Fn(&Examples) -> Grid, output: &mut BTreeMap<String, Vec<OutputData>>) -> bool {
    let ans = func(&examples);

    if is_test {
        false 
    } else {
        save(file, experiment, is_test, &vec![ans], targets, done, output)
    }
}
*/

fn run_experiment_tries(file: &str, experiment: usize, is_test: bool, examples: &Examples, targets: &[Grid], done: &mut BTreeSet<String>, func: &(dyn Fn(&Grid, &Grid, &mut usize) -> Grid + RefUnwindSafe), output: &mut BTreeMap<String, Vec<OutputData>>) -> bool {
    let ans = panic::catch_unwind(|| experiment_grid(examples, file, experiment, func));

    let ans = match ans {
        Ok(ans) => ans,
        Err(e) => {
            eprintln!("{file} / {experiment} Exception: {e:?}");
            vec![Grid::trivial()]
        },
    };

    save(file, experiment, is_test, &ans, targets, done, output)
}

fn save(file: &str, experiment: usize, is_test: bool, ans: &[Grid], targets: &[Grid], done: &mut BTreeSet<String>, results: &mut BTreeMap<String, Vec<OutputData>>) -> bool {
    let target_size: usize = targets.iter().map(|target| target.size()).sum();
    let ans_size: usize = ans.iter().map(|ans| ans.size()).sum();
    let same = if target_size > 0 && ans_size > 0 {
        targets.iter()
            .zip(ans.iter())
            .map(|(target, ans)| {
                if ans.equals(target) == Colour::Same { 1 } else { 0 }
            }).sum::<usize>() > 0
    } else {
        false
    };

    if !is_test && same {
        done.insert(file.to_string());
        println!("Success: {file}, Experiment: {experiment}");

        true
    } else if is_test && !same && ans_size > 0 {
        add_real_output(file, ans, results);

        done.insert(file.to_string());
        println!("Success: {file}, Experiment: {experiment}");

        true
    } else {
        add_dummy_output(file, ans.len(), results);

        false
    }
}
