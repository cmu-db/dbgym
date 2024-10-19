import inspect
import json
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np
from plumbum import local
from torch.utils.tensorboard.writer import SummaryWriter
from typing_extensions import ParamSpec

from misc.utils import DBGymConfig

P = ParamSpec("P")
T = TypeVar("T")


def time_record(key: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def wrap(f: Callable[P, T]) -> Callable[P, T]:
        def wrapped_f(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.time()
            ret = f(*args, **kwargs)

            # TODO(wz2): This is a hack to get a artifact_manager instance.
            first_arg = args[0]  # Ignore the indexing type error
            assert hasattr(
                first_arg, "artifact_manager"
            ), f"{first_arg} {type(first_arg)}"

            if first_arg.artifact_manager is None:
                # If there is no artifact_manager, just return.
                return ret

            assert isinstance(first_arg.artifact_manager, ArtifactManager)
            if first_arg.artifact_manager is not None:
                cls_name = type(first_arg).__name__
                first_arg.artifact_manager.record(
                    f"{cls_name}_{key}", time.time() - start
                )
            return ret

        return wrapped_f

    return wrap


class Encoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(Encoder, self).default(obj)


class ArtifactManager(object):
    """
    This class manages the following artifacts of Proto-X: info for replaying and TensorBoard output.

    Initializing this class sets up the root logger. However, to use the root logger, you should
    directly use the logging library.
    """

    # The output log is the file that the root logger writes to
    OUTPUT_LOG_FNAME = "output.log"
    REPLAY_INFO_LOG_FNAME = "replay_info.log"
    REPLAY_LOGGER_NAME = "replay_logger"

    def __init__(
        self,
        dbgym_cfg: DBGymConfig,
        trace: bool,
    ) -> None:
        self.log_dpath = dbgym_cfg.cur_task_runs_artifacts_path(mkdir=True)
        self.trace = trace
        self.tensorboard_dpath = self.log_dpath / "tboard"
        self.tuning_steps_dpath = self.log_dpath / "tuning_steps"
        self.tuning_steps_dpath.mkdir(parents=True, exist_ok=True)

        # Setup the root and replay loggers
        formatter = "%(levelname)s:%(asctime)s [%(filename)s:%(lineno)s]  %(message)s"
        logging.basicConfig(format=formatter, level=logging.DEBUG, force=True)
        output_log_handler = logging.FileHandler(
            self.log_dpath / ArtifactManager.OUTPUT_LOG_FNAME
        )
        output_log_handler.setFormatter(logging.Formatter(formatter))
        output_log_handler.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(output_log_handler)

        replay_info_log_handler = logging.FileHandler(
            self.tuning_steps_dpath / ArtifactManager.REPLAY_INFO_LOG_FNAME
        )
        replay_info_log_handler.setFormatter(logging.Formatter(formatter))
        replay_info_log_handler.setLevel(logging.INFO)
        logging.getLogger(ArtifactManager.REPLAY_LOGGER_NAME)

        # Setup the writer.
        self.writer: Union[SummaryWriter, None] = None
        if self.trace:
            self.tensorboard_dpath.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(self.tensorboard_dpath)  # type: ignore[no-untyped-call]

        self.iteration = 1
        self.iteration_data: dict[str, Any] = {}

    def log_to_replay_info(self, message: str) -> None:
        logging.getLogger(ArtifactManager.REPLAY_LOGGER_NAME).info(message)

    def stash_results(
        self,
        info_dict: dict[str, Any],
        name_override: Optional[str] = None,
        ray_trial_id: Optional[str] = None,
    ) -> None:
        """
        Stash data about this step of tuning so that it can be replayed.
        """
        dname = (
            name_override
            if name_override
            else datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        if ray_trial_id != None:
            # Orthogonal to whether name_override is used, ray_trial_id disambiguates between folders created
            # by different HPO trials so that the folders don't overwrite each other.
            dname += f"_{ray_trial_id}"

        target_stash_dpath = self.tuning_steps_dpath / dname

        if (
            info_dict["results_dpath"] is not None
            and Path(info_dict["results_dpath"]).exists()
        ):
            local["mv"][info_dict["results_dpath"], str(target_stash_dpath)].run()
            self.log_to_replay_info(
                f"mv src={info_dict['results_dpath']} dst={str(target_stash_dpath)}"
            )
        else:
            target_stash_dpath.mkdir(parents=True, exist_ok=True)

        if info_dict["prior_pgconf"]:
            local["cp"][
                info_dict["prior_pgconf"], str(target_stash_dpath / "old_pg.conf")
            ].run()

        if info_dict["prior_state_container"]:
            with open(target_stash_dpath / "prior_state.pkl", "wb") as f:
                # info_dict["prior_state_container"] is a somewhat complex object so we use pickle over json
                pickle.dump(info_dict["prior_state_container"], f)

        if info_dict["actions_info"]:
            with open(target_stash_dpath / "action.pkl", "wb") as f:
                pickle.dump(info_dict["actions_info"], f)

    def advance(self) -> None:
        if self.writer is None:
            return

        for key, value in self.iteration_data.items():
            if isinstance(value, str):
                # str is considered a np.ScalarType
                self.writer.add_text(key, value, self.iteration)  # type: ignore[no-untyped-call]
            else:
                self.writer.add_scalar(key, value, self.iteration)  # type: ignore[no-untyped-call]

        del self.iteration_data
        self.iteration_data = {}
        self.iteration += 1
        self.writer.flush()  # type: ignore[no-untyped-call]

    def record(self, key: str, value: Any) -> None:
        stack = inspect.stack(context=2)
        caller = stack[1]

        # Accumulate data.
        assert isinstance(value, np.ScalarType)
        key = f"{caller.filename}:{caller.lineno}_{key}"
        if key not in self.iteration_data:
            self.iteration_data[key] = 0.0
        self.iteration_data[key] += value

    def flush(self) -> None:
        if self.trace:
            assert self.writer
            self.advance()
            self.writer.flush()  # type: ignore[no-untyped-call]
