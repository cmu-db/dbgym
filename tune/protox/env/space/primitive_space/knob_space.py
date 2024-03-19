from typing import Any, Optional
from gymnasium import spaces

from tune.protox.env.space.primitive.knob import (
    KnobMetadata,
    _create_knob,
)
from tune.protox.env.space.primitive.latent_knob import _create_latent_knob

from tune.protox.env.types import KnobMap


class KnobSpace(spaces.Dict):
    def __init__(
        self,
        tables: list[str],
        knobs: dict[str, KnobMetadata],
        quantize: bool,
        quantize_factor: int,
        seed: int,
        table_level_knobs: dict[str, dict[str, KnobMetadata]] = {},
        latent: bool = False,
    ):
        create_fn = _create_latent_knob if latent else _create_knob
        self.knobs: KnobMap = KnobMap({})
        self.tables = tables
        spaces = []
        for k, md in knobs.items():
            knob = create_fn(None, None, k, md, quantize, quantize_factor, seed)
            self.knobs[knob.name()] = knob
            spaces.append((knob.name(), knob))

        for t, kv in table_level_knobs.items():
            for k, md in kv.items():
                knob = create_fn(t, None, k, md, quantize, quantize_factor, seed)
                self.knobs[knob.name()] = knob
                spaces.append((knob.name(), knob))

        super().__init__(spaces, seed=seed)

    def sample(self, mask: Optional[Any] = None) -> Any:
        raise NotImplementedError()
