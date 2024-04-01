from typing import cast

import gymnasium as gym
from gymnasium.spaces import Dict

from tune.protox.env.space.primitive.knob import (
    CategoricalKnob,
    KnobMetadata,
    _create_knob,
)
from tune.protox.env.space.primitive.latent_knob import _create_latent_knob
from tune.protox.env.types import KnobMap, QueryTableAliasMap


class QuerySpace(Dict):
    def __init__(
        self,
        tables: list[str],
        quantize: bool,
        quantize_factor: int,
        seed: int,
        per_query_knobs_gen: dict[str, KnobMetadata] = {},
        per_query_parallel: QueryTableAliasMap = QueryTableAliasMap({}),
        per_query_scans: QueryTableAliasMap = QueryTableAliasMap({}),
        query_names: list[str] = [],
        latent: bool = False,
    ) -> None:

        create_fn = _create_latent_knob if latent else _create_knob
        self.knobs: KnobMap = KnobMap({})
        self.tables = tables
        spaces = []
        for qname in query_names:
            for q, kv in per_query_knobs_gen.items():
                knob = create_fn(None, qname, q, kv, quantize, quantize_factor, seed)
                self.knobs[knob.name()] = knob
                spaces.append((knob.name(), knob))

        for q, pqs in per_query_scans.items():
            for _, aliases in pqs.items():
                for v in aliases:
                    md = KnobMetadata(
                        {
                            "type": "scanmethod_enum",
                            "min": 0.0,
                            "max": 1.0,
                            "quantize": False,
                            "log_scale": 0,
                            "unit": 0,
                        }
                    )

                    knob = create_fn(
                        table_name=None,
                        query_name=q,
                        knob_name=v + "_scanmethod",
                        metadata=md,
                        do_quantize=False,
                        default_quantize_factor=quantize_factor,
                        seed=seed,
                    )
                    self.knobs[knob.name()] = knob
                    spaces.append((knob.name(), knob))

        cat_spaces = []
        self.cat_dims = []
        for q, pqp in per_query_parallel.items():
            values = []
            for _, aliases in pqp.items():
                values.extend(aliases)

            if len(values) < 2:
                continue

            md = KnobMetadata(
                {
                    "type": "query_table_enum",
                    "values": values,
                    "default_value": 0,
                }
            )
            knob = create_fn(
                table_name=None,
                query_name=q,
                knob_name=q + "_parallel_rel",
                metadata=md,
                do_quantize=False,
                default_quantize_factor=0,
                seed=seed,
            )
            self.knobs[knob.name()] = knob

            cat_spaces.append((knob.name(), knob))
            self.cat_dims.append(cast(CategoricalKnob, knob).num_elems)

        # Figure out where the categorical inputs begin.
        self.categorical_start = gym.spaces.utils.flatdim(Dict(spaces))
        spaces.extend(cat_spaces)
        super().__init__(spaces, seed=seed)
        self.final_dim = gym.spaces.utils.flatdim(self)
