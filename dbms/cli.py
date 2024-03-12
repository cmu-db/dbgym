import click


@click.group(name="dbms")
@click.pass_obj
def dbms_group(dbgym_cfg):
    dbgym_cfg.append_group("dbms")
