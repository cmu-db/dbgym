import click


@click.group(name="dbms")
@click.pass_obj
def dbms_group(config):
    config.append_group("dbms")
