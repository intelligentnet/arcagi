use arc_agi::runner::runner;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data = if args.len() >= 2 { &args[1] } else { "training" };
    let catfile = if args.len() >= 3 { &args[2] } else { "" };
    let all = args.len() >= 4 && args[3] == "all";

    runner(data, catfile, all);
}
