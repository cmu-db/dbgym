## Concepts to Include
I'm not writing the README now because the overall structure depends on what needs to be included, which I don't know yet.
However, it's still useful to note down what needs to be included as it comes up so I don't need to go and remember.
Here is the list:
- The `dbgym_workspace/` directory and its structure
- What `open_and_save()` as well as the distinction between configs, dependencies, and results
We used a conda env with Python 3.10.13

## to include in PR
tell Will I combined train, analyze, and selection into a single step because:
 1. only the final model is important for the user
 2. old run_*/ dirs should be immutable so I'd have to change the code as we currently create stats.txt and analyze.txt directly inside the models
tell Will I combined the "start_epoch" args of eval and analyze
tell Will I tried to leave most of the options intact. this resulted in way too many CLI args, but those can be pruned in the future. it's better to have too many than too little