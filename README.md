# arcagi
A Rust attempt at the ARC-AGI prize 2024

NOTE: This code is messy and imcomplete and is likely to change rapidly.

The Kaggle ARC-AGI challenge 2024 is an attempt to get computers to solve
visual 2D puzzles, similar to those in many IQ test. To solve this challenge
in a principled way requires perception of stucture and reasoning guided by 
comstraints. It currently cannot be solved by LLM/GenAI based methods though
there may be a small place for them (or possibly just simple Neural Networks
for specific recognition tasks). 

This code is a first submission for the prize, it could conceivably solve all
the puzzles in the challenge, but at the moment solves just over 10%. Much
more work is required to reach the 85% or better level. 

This prize in important because it will demonstrate reasoning abilities and
not just the memorisztion with some 'light' generalisation that is currently
demonstrated by all Gen-AI models (in fact all Machine Learning models). 

Abstraction, Generalisation and Functional composition are key concepts.
These work in tandem with constraint recognition and discovery to tame the
size of the search space.

See: https://www.kaggle.com/competitions/arc-prize-2024

See inet submission.

To run this the first parameter is one of 'training', 'evaluation' or 'test'. There is no solutions file for the test run, this is expected so ignore the error message. A second parameter can be supplied which is the test to run, useful
for testing individual cases. A typical call to run the training set would be
'cargo run --release training'. 
