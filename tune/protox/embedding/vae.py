import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.utils import common_functions as c_f


def gen_vae_collate(max_categorical, infer=False):
    def vae_collate(batch, device="auto"):
        if infer:
            x = torch.as_tensor(batch).type(torch.int64)
        else:
            assert len(batch) > 0
            x = torch.stack([e[0] for e in batch]).type(torch.int64)

            y_shape = batch[0][1].shape[0]
            y = torch.stack([e[1] for e in batch]).view((x.shape[0], y_shape))

        # One-hot all the X's.
        scatter_dim = len(x.size())
        x_tensor = x.view(*x.size(), -1)
        x = torch.zeros(*x.size(), max_categorical, dtype=x.dtype)
        x = x.scatter_(scatter_dim, x_tensor, 1).view(x.shape[0], -1).type(torch.float32)

        if infer:
            return x
        else:
            return x, y
    return vae_collate


def acquire_loss_function(loss_type, max_attrs, max_categorical):
    def vae_cat_loss(preds, data, labels):
        if len(labels.shape) == 2:
            labels = labels[: , -1].flatten()

        preds = preds.view(preds.shape[0], -1, max_categorical)
        data = data.view(data.shape[0], -1, max_categorical)

        # Shape: <batch size, # categories, max # output per category>
        preds = torch.swapaxes(preds, 1, 2)
        data = torch.argmax(data, dim=2)

        # Pray for ignore_index..?
        data[:, 1:][data[:, 1:] == 0] = -100

        recon_loss = F.cross_entropy(preds, data, weight=None, ignore_index=-100, label_smoothing=1./max_categorical, reduction="none")
        if torch.isnan(recon_loss).any():
            # Dump any found nan in the loss.
            print(preds[torch.isnan(recon_loss)])
            assert False

        recon_loss = recon_loss.sum(dim=(1,))
        return recon_loss

    loss_fn = {
        "vae_cat_loss": vae_cat_loss,
    }[loss_type]
    return loss_fn


class VAEReducer(reducers.MultipleReducers):
    def __init__(self, *args, **kwargs):
        reducer = {
            "recon_loss": reducers.MeanReducer(),
            "elbo": reducers.MeanReducer(),
        }
        super().__init__(reducer, *args, **kwargs)

    def sub_loss_reduction(self, sub_losses, embeddings=None, labels=None):
        assert "elbo" in self.reducers
        for i, k in enumerate(self.reducers.keys()):
            if k == "elbo":
                return sub_losses[i]


class VAELoss(losses.BaseMetricLossFunction):
    def __init__(self, loss_fn, max_attrs, max_categorical, *args, **kwargs):
        super().__init__(reducer=VAEReducer(), *args, **kwargs)
        self.loss_fn = acquire_loss_function(loss_fn, max_attrs, max_categorical)

        eval_loss_fn_name = "vae_cat_loss"
        self.eval_loss_fn = acquire_loss_function(eval_loss_fn_name, max_attrs, max_categorical)


    def forward(
        self, embeddings, labels=None, indices_tuple=None, ref_emb=None, ref_labels=None, is_eval=False
    ):
        """
        Args:
            embeddings: tensor of size (batch_size, embedding_size)
            labels: tensor of size (batch_size)
            indices_tuple: tuple of size 3 for triplets (anchors, positives, negatives)
                            or size 4 for pairs (anchor1, postives, anchor2, negatives)
                            Can also be left as None
        Returns: the loss
        """
        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        ref_emb, ref_labels = c_f.set_ref_emb(embeddings, labels, ref_emb, ref_labels)
        loss_dict = self.compute_loss(
            embeddings, labels, indices_tuple, ref_emb, ref_labels, is_eval=is_eval
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)


    def compute_loss(self, preds, unused0, unused1, data, *args, **kwargs):
        is_eval = kwargs.get("is_eval", False)
        eval_fn = self.eval_loss_fn if is_eval else self.loss_fn

        data, labels = data
        recon_loss = eval_fn(preds, data, labels)

        # ELBO:
        elbo = torch.mean(recon_loss)

        self.last_loss_dict = {
            "recon_loss": {
                "losses": recon_loss.mean(),
                "indices": None,
                "reduction_type": "already_reduced",
            },
            "elbo": {
                "losses": elbo.mean(),
                "indices": None,
                "reduction_type": "already_reduced",
            }
        }
        return self.last_loss_dict

    def _sub_loss_names(self):
        return ["recon_loss", "elbo"]


class Network(nn.Module):
    def __init__(self, input_dim, hidden_sizes, output_dim, act):
        super(Network, self).__init__()

        # Parametrize each standard deviation separately.
        dims = [input_dim] + hidden_sizes + [output_dim]

        layers = []
        for (d1, d2) in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))
            if act is not None:
                layers.append(act())
        if act is not None:
            layers = layers[:-1]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


# Define the encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_sizes, latent_dim, act, mean_output_act=None):
        super(Encoder, self).__init__()

        # Parametrize each standard deviation separately.
        dims = [input_dim] + hidden_sizes + [latent_dim]

        layers = []
        for (d1, d2) in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))
            if act is not None:
                layers.append(act())
        if act is not None:
            layers = layers[:-1]

        self.module = nn.Sequential(*layers)
        if mean_output_act is None:
            self.mean_output_act = None
        else:
            self.mean_output_act = mean_output_act()

    def forward(self, x):
        assert len(x.shape) == 2
        mu = self.module(x)

        # Apply activation function to mean if necessary.
        if self.mean_output_act is not None:
            mu = self.mean_output_act(mu)

        return mu


# Define the decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_sizes, input_dim, act):
        super(Decoder, self).__init__()

        dims = [latent_dim] + [l for l in hidden_sizes] + [input_dim]
        layers = []
        for (d1, d2) in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))
            if act is not None:
                layers.append(act())
        if act is not None:
            layers = layers[:-1]
        self.module = nn.Sequential(*layers)

    def forward(self, z):
        x_hat = self.module(z)
        return x_hat


def init_modules(encoder, decoder, bias_init, weight_init, weight_uniform):
    def init(layer):
        if isinstance(layer, nn.Linear):
            if bias_init == "zeros":
                torch.nn.init.zeros_(layer.bias)
            elif "constant" in bias_init:
                cons = float(bias_init.split("constant")[-1])
                torch.nn.init.constant_(layer.bias, cons)

            if weight_init != "default":
                init_fn = {
                    ("xavier", True): torch.nn.init.xavier_uniform_,
                    ("xavier", False): torch.nn.init.xavier_normal_,
                    ("kaiming", True): torch.nn.init.kaiming_uniform_,
                    ("kaiming", False): torch.nn.init.kaiming_normal_,
                    ("spectral", True): torch.nn.utils.spectral_norm,
                    ("spectral", False): torch.nn.utils.spectral_norm,
                    ("orthogonal", True): torch.nn.init.orthogonal_,
                    ("orthogonal", False): torch.nn.init.orthogonal_,
                }[(weight_init, weight_uniform)]

                if weight_init == "spectral":
                    init_fn(layer)
                else:
                    init_fn(layer.weight)

    modules = [encoder, decoder]
    for module in modules:
        if module is not None:
            module.apply(init)


# Define the model
class VAE(nn.Module):
    def __init__(self,
            max_categorical,
            input_dim,
            hidden_sizes,
            latent_dim,
            act,
            bias_init="default",
            weight_init="default",
            weight_uniform=None,
            mean_output_act=None,
            output_scale=1.,
            ):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_sizes, latent_dim, act, mean_output_act=mean_output_act)
        self.decoder = Decoder(latent_dim, reversed(hidden_sizes), input_dim, act)
        init_modules(self.encoder, self.decoder, bias_init, weight_init, weight_uniform)

        self.input_dim = input_dim
        self.max_categorical = max_categorical
        self._collate = None
        self.output_scale = output_scale

    def get_collate(self):
        if self._collate is None:
            self._collate = gen_vae_collate(self.max_categorical, infer=True)
        return self._collate

    def forward(self, x, bias=None):
        return self.latents(x, bias=bias, require_full=True)

    def latents(self, x, bias=None, require_full=False):
        latents = self.encoder(x)
        latents = latents * self.output_scale

        if bias is not None:
            if isinstance(bias, torch.Tensor):
                assert bias.shape[0] == latents.shape[0]
                assert bias.shape[1] == 1
                latents = latents + bias
            else:
                # Add the bias.
                latents = latents + bias[0]
                if isinstance(bias[1], torch.Tensor):
                    latents = torch.clamp(latents, torch.zeros_like(bias[1]), bias[1])
                else:
                    latents = torch.clamp(latents, 0, bias[1])

        error = (latents.isnan() | latents.isinf()).any()

        if require_full:
            decoded = self.decoder(latents)
            error = error or (decoded.isnan() | decoded.isinf()).any()
            return latents, decoded, error

        return latents, error
