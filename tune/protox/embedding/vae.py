from typing import Any, Callable, Optional, Tuple, Type, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_metric_learning import losses, reducers
from pytorch_metric_learning.utils import common_functions as c_f


def gen_vae_collate(
    max_categorical: int, infer: bool = False
) -> Callable[[list[Any]], Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]]:
    def vae_collate(
        batch: list[Any],
    ) -> Union[tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if infer:
            x = torch.as_tensor(batch).type(torch.int64)
        else:
            assert len(batch) > 0
            x = torch.stack([e[0] for e in batch]).type(torch.int64)

            y_shape = batch[0][1].shape[0]
            ret_y = torch.stack([e[1] for e in batch]).view((x.shape[0], y_shape))

        # One-hot all the X's.
        scatter_dim = len(x.size())
        x_tensor = x.view(*x.size(), -1)
        zero_x = torch.zeros(*x.size(), max_categorical, dtype=x.dtype)
        ret_x: torch.Tensor = (
            zero_x.scatter_(scatter_dim, x_tensor, 1)
            .view(zero_x.shape[0], -1)
            .type(torch.float32)
        )

        if infer:
            return ret_x
        else:
            return ret_x, ret_y

    return vae_collate


def acquire_loss_function(
    loss_type: str, max_attrs: int, max_categorical: int
) -> Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
    def vae_cat_loss(
        preds: torch.Tensor, data: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        if len(labels.shape) == 2:
            labels = labels[:, -1].flatten()

        preds = preds.view(preds.shape[0], -1, max_categorical)
        data = data.view(data.shape[0], -1, max_categorical)

        # Shape: <batch size, # categories, max # output per category>
        preds = torch.swapaxes(preds, 1, 2)
        data = torch.argmax(data, dim=2)

        # Pray for ignore_index..?
        data[:, 1:][data[:, 1:] == 0] = -100

        recon_loss = F.cross_entropy(
            preds,
            data,
            weight=None,
            ignore_index=-100,
            label_smoothing=1.0 / max_categorical,
            reduction="none",
        )
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


class VAEReducer(reducers.MultipleReducers):  # type: ignore
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        reducer = {
            "recon_loss": reducers.MeanReducer(),
            "elbo": reducers.MeanReducer(),
        }
        super().__init__(reducer, *args, **kwargs)

    def sub_loss_reduction(
        self, sub_losses: list[Any], embeddings: Any = None, labels: Any = None
    ) -> Any:
        assert "elbo" in self.reducers
        for i, k in enumerate(self.reducers.keys()):
            if k == "elbo":
                return sub_losses[i]


class VAELoss(losses.BaseMetricLossFunction):  # type: ignore
    def __init__(
        self,
        loss_fn: str,
        max_attrs: int,
        max_categorical: int,
        *args: Any,
        **kwargs: Any
    ):
        super().__init__(reducer=VAEReducer(), *args, **kwargs)
        self.loss_fn = acquire_loss_function(loss_fn, max_attrs, max_categorical)

        eval_loss_fn_name = "vae_cat_loss"
        self.eval_loss_fn = acquire_loss_function(
            eval_loss_fn_name, max_attrs, max_categorical
        )

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: Any = None,
        indices_tuple: Any = None,
        ref_emb: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        ref_labels: Any = None,
        is_eval: bool = False,
    ) -> Any:
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

    def compute_loss(
        self,
        preds: torch.Tensor,
        unused0: Any,
        unused1: Any,
        tdata: Optional[tuple[torch.Tensor, torch.Tensor]],
        *args: Any,
        **kwargs: Any
    ) -> Any:
        is_eval = kwargs.get("is_eval", False)
        eval_fn = self.eval_loss_fn if is_eval else self.loss_fn

        assert tdata
        data, labels = tdata
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
            },
        }
        return self.last_loss_dict

    def _sub_loss_names(self) -> list[str]:
        return ["recon_loss", "elbo"]


class Network(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        output_dim: int,
        act: Callable[[], nn.Module],
    ) -> None:
        super(Network, self).__init__()

        # Parametrize each standard deviation separately.
        dims = [input_dim] + hidden_sizes + [output_dim]

        layers: list[nn.Module] = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))
            if act is not None:
                layers.append(act())
        if act is not None:
            layers = layers[:-1]
        self.module = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.module(x))


# Define the encoder
class Encoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_sizes: list[int],
        latent_dim: int,
        act: Type[nn.Module],
        mean_output_act: Optional[Type[nn.Module]] = None,
    ):
        super(Encoder, self).__init__()

        # Parametrize each standard deviation separately.
        dims = [input_dim] + hidden_sizes + [latent_dim]

        layers: list[nn.Module] = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert len(x.shape) == 2
        mu = self.module(x)

        # Apply activation function to mean if necessary.
        if self.mean_output_act is not None:
            mu = self.mean_output_act(mu)

        return cast(torch.Tensor, mu)


# Define the decoder
class Decoder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_sizes: list[int],
        input_dim: int,
        act: Type[nn.Module],
    ):
        super(Decoder, self).__init__()

        dims = [latent_dim] + [l for l in hidden_sizes] + [input_dim]
        layers: list[nn.Module] = []
        for d1, d2 in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(d1, d2))
            if act is not None:
                layers.append(act())
        if act is not None:
            layers = layers[:-1]
        self.module = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x_hat = self.module(z)
        return cast(torch.Tensor, x_hat)


def init_modules(
    encoder: Encoder,
    decoder: Decoder,
    bias_init: str,
    weight_init: str,
    weight_uniform: bool,
) -> None:
    def init(layer: nn.Module) -> None:
        if isinstance(layer, nn.Linear):
            if bias_init == "zeros":
                torch.nn.init.zeros_(layer.bias)
            elif "constant" in bias_init:
                cons = float(bias_init.split("constant")[-1])
                torch.nn.init.constant_(layer.bias, cons)

            if weight_init != "default":
                init_fn: Callable[[Union[nn.Module, torch.Tensor]], None] = cast(
                    Callable[[Union[nn.Module, torch.Tensor]], None],
                    {
                        ("xavier", True): torch.nn.init.xavier_uniform_,
                        ("xavier", False): torch.nn.init.xavier_normal_,
                        ("kaiming", True): torch.nn.init.kaiming_uniform_,
                        ("kaiming", False): torch.nn.init.kaiming_normal_,
                        ("spectral", True): torch.nn.utils.spectral_norm,
                        ("spectral", False): torch.nn.utils.spectral_norm,
                        ("orthogonal", True): torch.nn.init.orthogonal_,
                        ("orthogonal", False): torch.nn.init.orthogonal_,
                    }[(weight_init, weight_uniform)],
                )

                if weight_init == "spectral":
                    init_fn(layer)
                else:
                    init_fn(layer.weight)

    modules: list[nn.Module] = [encoder, decoder]
    for module in modules:
        if module is not None:
            module.apply(init)


# Define the model
class VAE(nn.Module):
    def __init__(
        self,
        max_categorical: int,
        input_dim: int,
        hidden_sizes: list[int],
        latent_dim: int,
        act: Type[nn.Module],
        bias_init: str = "default",
        weight_init: str = "default",
        weight_uniform: bool = False,
        mean_output_act: Optional[Type[nn.Module]] = None,
        output_scale: float = 1.0,
    ) -> None:
        super(VAE, self).__init__()
        self.encoder = Encoder(
            input_dim, hidden_sizes, latent_dim, act, mean_output_act=mean_output_act
        )
        self.decoder = Decoder(latent_dim, list(reversed(hidden_sizes)), input_dim, act)
        init_modules(self.encoder, self.decoder, bias_init, weight_init, weight_uniform)

        self.input_dim = input_dim
        self.max_categorical = max_categorical
        self._collate: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self.output_scale = output_scale

    def get_collate(self) -> Callable[[torch.Tensor], torch.Tensor]:
        if self._collate is None:
            # In infer mode, we def know it'll only return 1 argument.
            self._collate = cast(
                Callable[[torch.Tensor], torch.Tensor],
                gen_vae_collate(self.max_categorical, infer=True),
            )
        return self._collate

    def forward(
        self,
        x: torch.Tensor,
        bias: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Union[tuple[torch.Tensor, torch.Tensor, bool], tuple[torch.Tensor, bool]]:
        return self._compute(x, bias=bias, require_full=True)

    def latents(
        self,
        x: torch.Tensor,
        bias: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = None,
        require_full: bool = False,
    ) -> tuple[torch.Tensor, bool]:
        rets = self._compute(x, bias=bias, require_full=False)
        assert len(rets) == 2
        return rets[0], rets[1]

    def _compute(
        self,
        x: torch.Tensor,
        bias: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = None,
        require_full: bool = False,
    ) -> Union[tuple[torch.Tensor, torch.Tensor, bool], tuple[torch.Tensor, bool]]:
        latents: torch.Tensor = self.encoder(x)
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

        lerror = bool((latents.isnan() | latents.isinf()).any())

        if require_full:
            decoded: torch.Tensor = self.decoder(latents)
            derror = bool((decoded.isnan() | decoded.isinf()).any())
            return latents, decoded, (lerror or derror)

        return latents, lerror
