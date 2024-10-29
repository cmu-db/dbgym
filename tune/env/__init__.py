# This folder contains code for managing the environment (aka the DBMS) that is shared across all tuning agents.
# Even though it is a folder in tune/, it in itself is not a tuning agent.
# The difference between this and dbms/ is that dbms/ is the CLI to build the database while this is code to use the database.
# The reason this is not a top-level directory is because the environment in itself is not a CLI command.