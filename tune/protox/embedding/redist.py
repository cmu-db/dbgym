import shutil


# TODO(phw2): check what happens if num_parts doesn't evenly divide num_samples

def redist(ctx, num_parts):
    '''
    Redistribute all embeddings_*/ folders inside the run_*/ folder into num_parts subfolders
    '''
    inputs = [f for f in ctx.obj.dbgym_this_run_path.glob("embeddings*")]

    for i in range(num_parts):
        (ctx.obj.dbgym_this_run_path / f"part{i}").mkdir(parents=True, exist_ok=True)

    for i, emb in enumerate(inputs):
        part = f"part{i % num_parts}"
        shutil.move(emb, ctx.obj.dbgym_this_run_path / part)
