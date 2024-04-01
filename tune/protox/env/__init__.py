from gymnasium import register

register(
    id="Postgres-v0",
    entry_point="tune.protox.env.pg_env:PostgresEnv",
)
