//pub mod queue;
pub mod cats;
pub mod data;
pub mod examples;
pub mod cell;
pub mod grid;
pub mod shape;
pub mod rules;
pub mod utils;
//pub mod oldrules;
pub mod experiments;
pub mod runner;
use crate::runner::runner;
//pub mod summary;

#[unsafe(no_mangle)]
pub extern "C" fn training() {
    runner("training", "", "", "trans", false)
}

#[unsafe(no_mangle)]
pub extern "C" fn training_all() {
    runner("training", "", "", "trans", true)
}

#[unsafe(no_mangle)]
pub extern "C" fn evaluation() {
    runner("evaluation", "", "", "trans", false)
}

#[unsafe(no_mangle)]
pub extern "C" fn evaluation_all() {
    runner("evaluation", "", "", "trans", true)
}

#[unsafe(no_mangle)]
pub extern "C" fn test() {
    runner("test", "", "", "trans", false)
}

#[unsafe(no_mangle)]
pub extern "C" fn test_all() {
    runner("test", "", "", "trans", true)
}

#[unsafe(no_mangle)]
pub extern "C" fn training_trans() {
    runner("training", "", "", "trans", false)
}

#[unsafe(no_mangle)]
pub extern "C" fn training_all_trans() {
    runner("training", "", "", "trans", true)
}

#[unsafe(no_mangle)]
pub extern "C" fn evaluation_trans() {
    runner("evaluation", "", "", "trans", false)
}

#[unsafe(no_mangle)]
pub extern "C" fn evaluation_all_trans() {
    runner("evaluation", "", "", "trans", true)
}

#[unsafe(no_mangle)]
pub extern "C" fn test_trans() {
    runner("test", "", "", "trans", false)
}

#[unsafe(no_mangle)]
pub extern "C" fn test_all_trans() {
    runner("test", "", "", "trans", true)
}
