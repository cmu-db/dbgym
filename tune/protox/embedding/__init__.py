from tune.protox.embedding.cli import embedding_group
from tune.protox.embedding.datagen import datagen
from tune.protox.embedding.train import train

embedding_group.add_command(train)
embedding_group.add_command(datagen)
