# arcagi
A Rust attempt at the ARC-AGI prize 2025

WARNING:
-----
This code is messy and imcomplete and is likely to change rapidly. At the
moment it is a set of experiments to guage what works well and what does not.
It is perhaps 30% complete and will answer about 40% of the examples given.

If you use this code as part of a Kaggle submission, or in any other way, then please give credit where credit is due.

New Baseline for v2 of ARC-AGI
------------------------------

This is cleaned up code that does not throw any errors on the new 2025 version.
It will achieve 281 correct training examples, but unfortunately only 2 correct
evaluation answers. There is a lot of work to be done! Identifying and 
composing training set code to be used to solve evaluation set problems is 
going to be the biggest challenge.

Background
----------

The Kaggle ARC-AGI challenge 2024 is an attempt to get computers to solve
visual 2D puzzles, similar to those in many IQ tests. To solve this challenge
in a principled way requires perception of stucture and reasoning guided by 
constraints. It currently cannot be solved by LLM/GenAI based methods though
there may be a small place for them (or possibly just simple Neural Networks
for specific recognition tasks). 

This code is a first submission for the prize, it could conceivably solve all
the puzzles in the challenge, but at the moment solves just over 10% in the
public evaluation set. Much more work is required to reach the 85% or better
level. 

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
would help (for code generation). It might eventually, for now there is
plenty to explore with the current Symbolic approach.
Code generation was experimented with, given some data
structures and a textual description of the problem, but the code
generated was poor quality and time consuming to debug (Rust is probably not
the best target for code generation given it's memory management as simple
next token prediction does not understand that).

See: https://www.kaggle.com/competitions/arc-prize-2025

See inet submission.

To run this the first parameter is one of 'training', 'evaluation' or 'test'. There is no solutions file for the test run, this is expected so ignore the error message. A second parameter can be supplied which is the test to run, useful
for testing individual cases. A typical call to run the training set would be
'cargo run --release training'. 

Kaggle notebook
---------------

For V1 (2024) version) of this challenge this code was used to run Rust from the notebook.

```
from cffi import FFI
ffi = FFI()
ffi.cdef("int test();")
rust = ffi.dlopen("/kaggle/input/release/libarc_agi.so")
a = rust.test();
print(a)
```

This does not seem to work in V2 (2025) so a simple wrapper is now used to
call it from the command line wrapped in C. 

```
!echo 'int main() { test(); }' > /kaggle/working/arcagi.c
!gcc -O -o /kaggle/working/arcagi /kaggle/working/arcagi.c /kaggle/input/release/libarc_agi.so
!LD_LIBRARY_PATH=/kaggle/input/release /kaggle/working/arcagi -larcagi
```
Ignore the warning.

Note 
----

A shared object file call libarcagi.so is required in a 'release' directory
created by uploading a 'model'.

If you wish to use this for a Kaggle submission, then please ask the author
who will be happy to share a Kaggle shared object. Credit is expected of course..

To compile the above for kaggle a docker image needs to be created and the .so
file uploaded and referenced, somewhat tedious. It will run locally for public
data test with cargo as normal. A release version should run in less that a
second on most machines, this will change as more cases are completed.
