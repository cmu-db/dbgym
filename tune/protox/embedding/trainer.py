import tqdm
import random
import itertools
import numpy as np
import torch
from torch.utils.data import Sampler
from typing import Iterator

from pytorch_metric_learning import trainers
from pytorch_metric_learning.utils import common_functions as c_f


class StratifiedRandomSampler(Sampler[int]):
    allow_repeats = True
    labels = None
    max_class = 0
    batch_size = 0
    elem_per_class = 0

    def __init__(self,
                 labels,
                 max_class: int,
                 batch_size: int,
                 allow_repeats=True):
        self.allow_repeats = allow_repeats
        self.labels = labels
        self.max_class = max_class
        self.batch_size = batch_size
        assert self.batch_size > 0

    def compute(self):
        r = {}
        for c in range(self.max_class):
            lc = np.argwhere(self.labels == c)
            lc = lc.reshape(lc.shape[0])
            r[c] = (lc.shape[0], lc)
        elem_per_class = self.batch_size // len([k for k in r if r[k][0] > 0])

        min_size = min([r[k][0] for k in r if r[k][0] > 0])
        min_steps = min_size // elem_per_class
        return r, elem_per_class, min_steps

    def __iter__(self) -> Iterator[int]:
        r, elem_per_class, min_steps = self.compute()
        if self.allow_repeats:
            for _ in range(len(self.labels) // self.batch_size):
                elems = [r[k][1][np.random.randint(0, high=r[k][0], size=(elem_per_class,))].tolist() for k in r if r[k][0] > 0]
                yield from itertools.chain(*elems)
        else:
            for k in r:
                if r[k][0] > 0:
                    random.shuffle(r[k][1])

            for i in range(min_steps):
                elems = [r[k][1][i*elem_per_class:i*elem_per_class + elem_per_class].tolist() for k in r if r[k][0] > 0]
                yield from itertools.chain(*elems)


    def __len__(self) -> int:
        if self.allow_repeats:
            return len(self.labels) // self.batch_size
        else:
            r, elem_per_class, min_steps = self.compute()
            return min_steps


class VAETrainer(trainers.BaseTrainer):
    def __init__(self, disable_tqdm, bias_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.failed = False
        self.fail_msg = None
        self.fail_data = None
        self.disable_tqdm = disable_tqdm
        self.bias_fn = bias_fn
        self.eval = False

        self.last_recon_loss = 0

    def compute(self, base_output):
        assert False

    def maybe_get_metric_loss(self, embeddings, labels, indices_tuple):
        if self.loss_weights.get("metric_loss", 0) > 0:
            return self.loss_funcs["metric_loss"](embeddings, labels, indices_tuple)
        return 0

    def maybe_get_vae_loss(self, preds, data, labels):
        if self.loss_weights.get("vae_loss", 0) > 0:
            return self.loss_funcs["vae_loss"](preds, None, None, (data, labels), is_eval=self.eval)
        return 0

    def calculate_loss(self, curr_batch):
        data, labels = curr_batch
        if labels.shape[1] == 1:
            # Flatten labels if it's a class.
            labels = labels.flatten().long()

        data = data.to(self.data_device)
        labels = labels.to(self.data_device)

        bias = None
        if self.bias_fn is not None:
            bias = self.bias_fn(data, labels)

            # Ensure that the bias is all valid.
            if isinstance(bias, torch.Tensor):
                assert not (bias.isnan() | bias.isinf()).any()
            else:
                assert not (bias[0].isnan() | bias[0].isinf()).any()
                assert not (bias[1].isnan() | bias[1].isinf()).any()

        # Compute latent space.
        embeddings, preds, error = self.models["embedder"](data, bias=bias)

        if error:
            # We've encountered an error.
            self.failed = True
            self.fail_msg = "Latents is undefined."
            # Don't tamper with any losses and just return.
            self.fail_data = (data, labels, embeddings, preds)
            return

        indices_tuple = self.maybe_mine_embeddings(embeddings, labels)
        ml = self.maybe_get_metric_loss(embeddings, labels, indices_tuple)

        self.losses["metric_loss"] = ml
        self.losses["vae_loss"] = self.maybe_get_vae_loss(preds, data, labels)
        self.last_recon_loss = self.loss_funcs["vae_loss"].last_loss_dict["recon_loss"]["losses"].item()

    def backward(self):
        if not self.failed:
            self.losses["total_loss"].backward()

    def train(self, start_epoch=1, num_epochs=1):
        self.initialize_dataloader()
        for self.epoch in range(start_epoch, num_epochs + 1):
            self.set_to_train()
            c_f.LOGGER.info("TRAINING EPOCH %d" % self.epoch)

            if not self.disable_tqdm:
                pbar = tqdm.tqdm(range(self.iterations_per_epoch))
            else:
                pbar = range(self.iterations_per_epoch)

            for self.iteration in pbar:
                self.forward_and_backward()
                self.end_of_iteration_hook(self)
                if self.failed or \
                   np.isnan(self.losses["total_loss"].item()) or np.isinf(self.losses["total_loss"].item()) or \
                   np.isnan(self.losses["vae_loss"].item()) or np.isinf(self.losses["vae_loss"].item()) or \
                   np.isnan(self.last_recon_loss) or np.isinf(self.last_recon_loss):
                    # Abort this particular run in this case.
                    self.failed = True

                    ml = self.losses["metric_loss"]
                    vl = self.losses["vae_loss"]

                    if self.fail_msg is not None:
                        pass
                    elif np.isnan(self.losses["total_loss"].item()) or np.isinf(self.losses["total_loss"].item()):
                        self.fail_msg = f"Total Loss is invalid ({ml}, {vl}, {self.last_recon_loss}"
                    elif np.isnan(self.losses["vae_loss"].item()) or np.isinf(self.losses["vae_loss"].item()):
                        self.fail_msg = "VAE Loss is invalid ({ml}, {vl}, {self.last_recon_loss}"
                    elif np.isnan(self.last_recon_loss) or np.isinf(self.last_recon_loss):
                        self.fail_msg = "Recon Loss is invalid ({ml}, {vl}, {self.last_recon_loss}"

                    print(self.fail_msg)
                    return

                if not self.disable_tqdm:
                    pbar.set_description("total=%.5f recon=%.5f metric=%.5f" % (self.losses["total_loss"], self.last_recon_loss, self.losses["metric_loss"]))
                self.step_lr_schedulers(end_of_epoch=False)
            self.step_lr_schedulers(end_of_epoch=True)
            self.zero_losses()
            if self.end_of_epoch_hook(self) is False:
                break

    def compute_embeddings(self, base_output):
        embeddings, _, _ = self.compute(base_output)
        return embeddings

    def get_batch(self):
        self.dataloader_iter, curr_batch = c_f.try_next_on_generator(
            self.dataloader_iter, self.dataloader
        )
        data, labels = self.data_and_label_getter(curr_batch)
        return data, labels

    def modify_schema(self):
        self.schema["loss_funcs"].keys += ["vae_loss"]

    def switch_eval(self):
        self.eval = True

    def switch_train(self):
        self.eval = False
