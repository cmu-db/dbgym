import math
import torch
import torch.nn as nn
from pytorch_metric_learning import losses # type: ignore
from pytorch_metric_learning.utils import common_functions as c_f # type: ignore
from typing import Union, Callable, Tuple, Any, cast, Optional

COST_COLUMNS = [
    "quant_mult_cost_improvement",
    "quant_rel_cost_improvement",
]


def get_loss(distance_fn: str) -> nn.Module:
    if distance_fn == "l1":
        return nn.L1Loss(reduction="none")
    elif distance_fn == "l2":
        return nn.MSELoss(reduction="none")
    else:
        assert False


def get_bias_fn(config: dict[str, Any]) -> Callable[[torch.Tensor, torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
    def bias_fn(data: torch.Tensor, labels: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        red_index = COST_COLUMNS.index(config["cost_reduction_type"])
        distance_scale = config["distance_scale"]
        if distance_scale == "auto":
            distance_scale = math.sqrt(data.shape[1])
        else:
            distance_scale = float(distance_scale)

        bias_separation = config.get("bias_separation", 0.05)
        addtl_bias_separation = config.get("addtl_bias_separation", 0)

        assert len(labels.shape) == 2
        assert labels.shape[1] == len(COST_COLUMNS) + 1
        target_loc = labels[:, red_index].reshape(-1, 1)
        target_loc = distance_scale * (1 - target_loc)

        # FIXME: in reality, this should not be linear!.
        perf_degrees = (target_loc / bias_separation).floor()

        if config.get("weak_bias", False):
            # Assign the weak bias allocation based on "separation margin".
            percent_sep = (
                target_loc - perf_degrees * bias_separation
            ) / bias_separation
            weak_bias = (bias_separation + addtl_bias_separation) * 0.95 * percent_sep
            top_clamp = (perf_degrees + 1) * (bias_separation + addtl_bias_separation)
            return (
                perf_degrees * (bias_separation + addtl_bias_separation) + weak_bias,
                top_clamp,
            )
        else:
            return cast(torch.Tensor, perf_degrees * (bias_separation + addtl_bias_separation))

    return bias_fn


def _distance_cost(
        distance_fn: str, distance_scale: str, reduction_type: str, preds: torch.Tensor, targets: torch.Tensor,
        bias: Callable[[torch.Tensor, torch.Tensor], Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]], output_scale: float
) -> Any:
    bias_vals = bias(preds, targets)
    assert isinstance(bias_vals, torch.Tensor)
    preds = preds - bias_vals

    assert reduction_type in COST_COLUMNS
    red_index = COST_COLUMNS.index(reduction_type)

    assert len(targets.shape) == 2
    assert targets.shape[1] == len(COST_COLUMNS) + 1
    target_loc = targets[:, red_index].reshape(-1, 1)

    # The better it is, the closer to zero it is.
    # This subtraction should be fine because target_loc should be between 0 and 1 due to quantile.
    # Then scale it to match the gaussian distance (I think).
    if distance_scale == "auto":
        distance_scale_val = math.sqrt(preds.shape[1])
    else:
        distance_scale_val = float(distance_scale_val)

    # FIXME: Overwrite the distance_scale using the output scale.
    distance_scale_val = output_scale

    # Produce the actual target location.
    target_loc = distance_scale_val * (1 - target_loc)
    target_loc = target_loc.expand(-1, preds.shape[1])
    preds_dist = preds

    comps = distance_fn.split(",")
    margin, margin_spec = comps[0], comps[1]
    if margin == "hard":
        # Directly regress against the boundary.
        losses = get_loss(margin_spec)(preds_dist, target_loc)
    else:
        assert margin == "soft"
        # We accept [preds_dist - margin, preds_dist + margin]
        tmargin = float(margin_spec)

        # How far away from the "boundary" that we are...
        dists = torch.abs(preds_dist - target_loc)
        losses = torch.clamp(dists - tmargin, min=0.)

    # Reduce to a per-row sum loss term.
    losses = losses.sum(dim=1)
    return losses


class CostLoss(losses.BaseMetricLossFunction): # type: ignore
    def __init__(self, metric_loss_md: dict[str, Any], *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.spec = metric_loss_md
        self.bias_fn = get_bias_fn(self.spec)

    def compute_loss(self, preds: torch.Tensor, unused0: Any, unused1: Any, data: torch.Tensor, *args: Any) -> dict[str, Any]:
        losses = _distance_cost(
            self.spec["distance_fn"],
            self.spec["distance_scale"],
            self.spec["cost_reduction_type"],
            preds,
            data,
            self.bias_fn,
            self.spec["output_scale"],
        )

        return {
            "loss": {
                "losses": losses.mean(),
                "indices": None,
                "reduction_type": "already_reduced",
            },
        }

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
        indices_tuple:Optional[Any]=None, ref_emb:Optional[Any]=None, ref_labels:Optional[Any]=None
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
        if labels is not None:
            labels = c_f.to_device(labels, embeddings)
        loss_dict = self.compute_loss(embeddings, None, None, labels)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)
