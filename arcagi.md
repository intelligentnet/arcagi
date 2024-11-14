# Static Symbolic Function Composition 

**Abstract**

The solution to the ARC-AGI prize given here is to loosely categorise a problem then use functional composition to produce an answer. Constraints are found then used to limit exponential search space growth. Then functional composition of simple but flexible functions composed together, guided by constraints, to produce candidate solutions with a limited amount of search.

**Introduction**

The Kaggle ARC_AGI prize was launched to encourage participation in creating computer systems to solve human Intelligence tests. This challenge is unlikely to be solved by simple memorisation, or by brute force methods. 

This solution is currently deterministic. Each task is categorised then sequences of statically programmed sequences of functions are composed together. There is also a framework for experiments that takes a task, then tries a small set of possible solutions (sequences of functional compositions that are parametrised and data driven, which are function closures in Rust) and tests against the tasks/experiments. 

If all experiments succeed for a task, then the system generates an answer for the submission file (always 2 the same at the moment). Depending on the task it may also try a sequence of similar transformations using higher order functions, which reduces the number of closures required. The constraints imposed by categorisation of problems helps speed up search considerably, though it is not perfect and there may be more than one set of functional compositions that might work, so a few trials may be required.

**Prior Work**

No explicit prior work is used in this submission other than the referenced paper below. The author has spent about 35 years thinking about similar problems and this solution is informed by that thinking and inspired by Francois Chollet [1]. 

This challenge provided a test set to explore a space where constraint finding and functional composition could be used. There is still a lot to learn, but only by experiencing the pain of finding solutions then refining, combining and modelling them well can better solutions be found. This is probably how evolution taught life, what is essentially a tool box, that can then be refined by individuals and passed on through language and culture.

**Approach**

This solution is very literal. It takes a problem, does some analysis of it's inputs and outputs and the mappings between them. It then finds relevant data about the problem, such as dimensions, colours, embedded shapes (and some common types of shapes) and mappings of these between experiment and solution. As few assumptions about the problem are made as reasonably possible. Functional transformations are then tried, depending on the categorisations found, to transform the input into the ouput. Higher order functions may be used to iterate through similar transformations such as And/Or/Xor or rotation and mirroring or overlay order of permutations. 

The system preprocesses each task and creates a set of categories that apply to all the experimental examples for a task. It then parses the input when appropriate into shapes (which might contain shapes too). The shapes are also categorised. The relationship between input and output solution and shapes is also categorised. The intersection of categories across the examples is used to create a generic solution by applying the appropriate low level transformations or previously discovered higher level sub-sequences of transformations.

Over the month of submissions, 1.00 was always achieved, however 47 evaluation tasks were correctly answered. There is likely a problem with submissions.

**Conclusion**

Given the simplicity of this solution it performs well on public datasets. 47 problems were solved in the evaluation set, in about 100ms. It was found that with experience the top down, constraint driven, static function compositions were solved more as more quickly as the set of available functions grew. Occasionally two Rust closures of function compositions could be combined (and categories adjusted). There is no reason to think that this will not continue to be the case. Indeed with more experience, better categorisation, more primitive functions and packaged intermediate collections of function compositions and transformations progress should both accelerate and generalisation improve. The limits of this process are not clear yet. More work is required.

**Analysis**

The biggest concern when starting this challenge was, "Will this approach generalise?". It was found that the more training problems that were solved resulted in more evaluation problems being solved. Though not at the same rate. Often solutions could be made more flexible, for example find a similar pattern in X or Y dimension and be able to solve both, or solve for any colours, having previously discovered the actual colours from examining the experimental pairs, or some common pattern between shapes predicated on some generic criteria. 

There is a limit to the combinations of these categories that 'make sense' to humans and there are also limits to the search space and the number of actual permutations used in particular contexts. Finding these human centric patterns requires experimentation. 'Collections' of common transformations can also be grouped together then used as higher level transformations when appropriate, further reducing search space.

There was a problem with code submissions. It is believed this is now solved. The problem was with the actual data being entered into the submission file. This should now be fixed but no submissions can now be done so this cannot be fully established.

<u>Reference</u>

Francois Chollet, Mike Knoop, Bryan Landers, Greg Kamradt, Hansueli Jud, Walter Reade, Addison Howard. (2024). ARC Prize 2024. Kaggle. https://kaggle.com/competitions/arc-prize-2024

<u>Links</u>

Full Paper: [https://github.com/intelligentnet/arcagi/blob/main/ARC-AGI.pdf](URL)

Final Submission: [https://www.kaggle.com/code/dipplec/inet-rust?scriptVersionId=206363766](URL)

Github: [https://github.com/intelligentnet/arcagi](URL)
