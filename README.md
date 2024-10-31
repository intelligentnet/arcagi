# arcagi
A Rust attempt at the ARC-AGI prize 2024

WARNING:
-----
This code is messy and imcomplete and is likely to change rapidly. At the
moment it is a set of experiments to guage what works well and what does not.
It is perhaps 20% complete and will answer just over 10% of the examples given.

Background
----------

The Kaggle ARC-AGI challenge 2024 is an attempt to get computers to solve
visual 2D puzzles, similar to those in many IQ test. To solve this challenge
in a principled way requires perception of stucture and reasoning guided by 
constraints. It currently cannot be solved by LLM/GenAI based methods though
there may be a small place for them (or possibly just simple Neural Networks
for specific recognition tasks). 

This code is a first submission for the prize, it could conceivably solve all
the puzzles in the challenge, but at the moment solves just over 10%. Much
more work is required to reach the 85% or better level. 

This prize in important because it will demonstrate reasoning abilities and
not just the memoristion with some 'light' generalisation that is currently
demonstrated by all Gen-AI models (in fact all Machine Learning models). 

Abstraction, Generalisation and Functional composition are key concepts.
These work in tandem with constraint recognition and discovery to tame the
size of the search space.

This is a pure Symbolic AI approach and hence is deterministic. There may be a
place for an LLM/Gen-AI in object recognition and/or generating candidate
solutions.
It was hoped that the author's previous experience of LLMs (see LLMClient),
would help. It might eventually, for now there is plenty to explore with the
current Symbolic approach.
Code generation was experimented with, given some data
structures and a textual description of the problem, but the code
generated was poor quality and time consuming to debug (Rust is probably not
the best target for code generation given it's memory management as simple
next token prediction does not understand that).

See: https://www.kaggle.com/competitions/arc-prize-2024

See inet submission.

To run this the first parameter is one of 'training', 'evaluation' or 'test'. There is no solutions file for the test run, this is expected so ignore the error message. A second parameter can be supplied which is the test to run, useful
for testing individual cases. A typical call to run the training set would be
'cargo run --release training'. 

Kaggle notebook
---------------

```
from cffi import FFI
ffi = FFI()
ffi.cdef("int test();")
rust = ffi.dlopen("/kaggle/input/release/libarc_agi.so")
a = rust.test();
print(a)
```

NOTE
----

To compile the above for kaggle a docker image needs to be created and the .so
file uploaded and referenced, somewhat tedious. It will run locally for public
data test with cargo as normal. A release version should run in less that a
second on most machines, this will change as more cases are completed.

Currently there is a problem with the above, with kaggle scoring on the
private dataset. Only run locally on public data for now or join the inet team.
