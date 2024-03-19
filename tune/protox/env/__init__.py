from gymnasium import register

register(
    id="Postgres-v0",
    entry_point="envs.pg_env:PostgresEnv",
)
