## Concepts to Include
I'm not writing the README now because the overall structure depends on what needs to be included, which I don't know yet.
However, it's still useful to note down what needs to be included as it comes up so I don't need to go and remember.
Here is the list:
- The `dbgym_workspace/` directory and its structure
- What `open_and_save()` as well as the distinction between configs, dependencies, and results
We used a conda env with Python 3.10.13

## to include in this PR
batch limit and sample limit rename
new way of doing batch limit (now sample limit)
I'm assuming Postgres is already running. a future PR will deal with automatically managing different ports, pgdata dirs, and [benchmark].tgz files