from dbms.cli import dbms_group
from dbms.postgres.cli import postgres_group

dbms_group.add_command(postgres_group)
