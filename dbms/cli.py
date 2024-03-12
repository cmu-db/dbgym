import click


@click.group(name="dbms")
@click.pass_obj
def dbms_group(cfg):
    cfg.append_group("dbms")
