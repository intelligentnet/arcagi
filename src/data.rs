use std::fs;
use std::collections::BTreeMap;
use serde_derive::{Deserialize, Serialize};
use crate::grid::Grid;

#[derive(Deserialize, Serialize, Debug)]
pub struct Data {
    pub train: Vec<IO>,
    pub test: Vec<IO>
}

#[derive(Deserialize, Serialize, Debug)]
pub struct IO {
    pub input: Vec<Vec<usize>>,
    pub output: Option<Vec<Vec<usize>>>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Training {
    //#[serde(rename = "file")]
    #[serde(flatten)]
    file: BTreeMap<String, Data>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Solution {
    //#[serde(rename = "file")]
    #[serde(flatten)]
    file: BTreeMap<String, Vec<Vec<Vec<usize>>>>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Input {
    input: Vec<Vec<usize>>,
}

#[derive(Deserialize, Serialize, Debug)]
pub struct Output {
    #[serde(flatten)]
    file: BTreeMap<String, Vec<OutputData>>,
}

#[derive(Deserialize, Serialize, Clone, Debug)]
pub struct OutputData {
    attempt_1: Vec<Vec<usize>>,
    attempt_2: Vec<Vec<usize>>,
}

pub fn add_dummy_output(file: &str, no_answers: usize, results: &mut BTreeMap<String, Vec<OutputData>>) {
    let attempt = vec![vec![0, 0], vec![0, 0]];
    let output_data: Vec<OutputData> = (0 .. no_answers).map(|_| OutputData { attempt_1: attempt.clone(), attempt_2: attempt.clone() }).collect();

    results.insert(file.to_string(), output_data);
    //results.insert(serde_json::to_string_pretty(file).unwrap(), output_data);
}

pub fn add_real_output(file: &str, answers: &[Grid], results: &mut BTreeMap<String, Vec<OutputData>>) {
    let output_data: Vec<OutputData> = answers.iter().map(|ans| OutputData { attempt_1: ans.to_vec(), attempt_2: ans.to_vec() }).collect();

    results.insert(file.to_string(), output_data);
    //results.insert(serde_json::to_string_pretty(file).unwrap(), output_data);
}

pub fn create_output(live: bool, file: &BTreeMap<String, Vec<OutputData>>) {
    let output = Output { file: file.clone() };

    match serde_json::to_string(&output) {
        Ok(data) => {
            //let data = format!("{data}\n");

            fs::write(if live { "submission.json" } else { "/kaggle/working/submission.json" }, data).expect("Unable to write file");
        },
        Err(e) => eprintln!("Failed to write submission.json: {e}"),
    }
}

pub fn dir(dir: &str) -> Vec<String> {
    let paths = fs::read_dir(dir).unwrap();

    paths.map(|p| p.unwrap().path().display().to_string()).collect()
}

pub fn load_files(data: &str) -> BTreeMap<String, Data> {
    /*
    let prefix = if std::path::Path::new("input/arc-prize-2024").is_dir() {
        "input/arc-prize-2024/arc-agi"
    } else if std::path::Path::new("kaggle/input").is_dir() {
        "kaggle/input/arc-prize-2025/arc-agi"
    } else {
        "/kaggle/input/arc-prize-2025/arc-agi"
    */
    /*
    let prefix = if std::path::Path::new("input/arc-prize-2025").is_dir() {
        "input/arc-prize-2025/arc-agi"
    } else if std::path::Path::new("kaggle/working").is_dir() {
        "kaggle/working/arc-agi"
    } else {
        "/kaggle/working/arc-agi"
    };
    */
    let prefix = if std::path::Path::new("input/arc-prize-2025").is_dir() {
        "input/arc-prize-2025/arc-agi"
    } else if std::path::Path::new("/kaggle/input/arc-prize-2025").is_dir() {
        "/kaggle/input/arc-prize-2025/arc-agi"
    } else {
        "/kaggle/input/arc-prize-2025/arc-agi"
    };
//eprintln!("#### {prefix}");
    let training_suffix = "challenges.json";
    let solution_suffix = "solutions.json";
    let training = format!("{}_{}_{}", prefix, data, training_suffix);
    let solution = format!("{}_{}_{}", prefix, data, solution_suffix);

    let mut tdata =
        match std::fs::read_to_string(&training) {
            Ok(cf) => {
                let training: Result<Training, _> = serde_json::from_str(&cf);

                match training {
                    Ok(files) => {
                        files.file
                    },
                    Err(e) => {
                        eprintln!("{e:?}");

                        BTreeMap::new()
                    }
                }
            },
            Err(e) => {
                    eprintln!("{training}: {e}");

                    BTreeMap::new()
                }
        };

    let sdata =
        match std::fs::read_to_string(&solution) {
            Ok(sf) => {
                let solutions: Result<Solution, _> = serde_json::from_str(&sf);

                match solutions {
                    Ok(files) => {
                        files.file
                    },
                    Err(e) => {
                        eprintln!("{e:?}");

                        BTreeMap::new()
                    }
                }
            },
            Err(e) => {
                    eprintln!("{solution}: {e}");

                    BTreeMap::new()
                }
        };

    //println!("{}", cfdata.len());

    // If we have data splice in test answers if any
    if !tdata.is_empty() {
        for (file, data) in tdata.iter_mut() {
            if sdata.contains_key(file) {
                for (i, tests) in data.test.iter_mut().enumerate() {
                    if tests.output.is_none() {
                        tests.output = Some(sdata[file][i].clone());
                    }
                }
            }
        }
    }

    tdata
}
