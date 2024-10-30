pub mod cats;
pub mod data;
pub mod examples;
pub mod cell;
pub mod grid;
pub mod shape;
pub mod rules;
pub mod experiments;
pub mod runner;
use crate::runner::runner;

#[no_mangle]
pub extern "C" fn arcagi(data_type: i32, all_int: i32) {
    let all: bool = all_int != 0;
    match data_type {
        0 => runner("training", "", all),
        1 => runner("evaluation", "", all),
        2 => runner("test", "", all),
        _ => runner("training", "", all),
    }
}

#[no_mangle]
pub extern "C" fn training() {
    runner("training", "", false)
}

#[no_mangle]
pub extern "C" fn training_all() {
    runner("training", "", true)
}

#[no_mangle]
pub extern "C" fn evaluation() {
    runner("evaluation", "", false)
}

#[no_mangle]
pub extern "C" fn evaluation_all() {
    runner("evaluation", "", true)
}

#[no_mangle]
pub extern "C" fn test() {
    runner("test", "", false)
}

#[no_mangle]
pub extern "C" fn test_all() {
    runner("test", "", true)
}
