use std::process::ExitCode;
use arc_agi::runner::runner;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();

    if args.len() == 1 {
        eprintln!("Format: {} <Run type> [task] [Transformation] [test all]", args[0]);
        eprintln!("Where:");
        eprintln!("Run type: training | evaluation | test");
        eprintln!("task: name of task which is 8 hex digits");
        eprintln!("      optionally followed by '/' the experiment to run");
        eprintln!("      experiment is the number of the experiment");
        eprintln!("      task is hex name or can be 'all'");
        eprintln!("Transformation: if 'all' will run all transformations");
        eprintln!("all: if 'all' will run all experiments for all tasks");

        return ExitCode::from(1);
    }

    let data = if args.len() >= 2 { &args[1] } else { "training" };
    let task = if args.len() >= 3 { &args[2] } else { "" };
    let trans = if args.len() >= 4 { &args[3] } else { "NoTrans" };
    let all = args.len() >= 5 && args[4] == "all";

    let parts: Vec<&str> = task.split('/').collect();
    let (task, experiment) = if parts.len() == 1 {
        (task, "")
    } else {
        (parts[0], parts[1])
    };

    runner(data, task, experiment,  trans, all);

    ExitCode::from(0)
}
