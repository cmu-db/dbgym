import inspect
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union

import numpy as np
from plumbum import local
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def time_record(key: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def wrap(f: Callable[P, T]) -> Callable[P, T]:
        def wrapped_f(*args: P.args, **kwargs: P.kwargs) -> T:
            start = time.time()
            ret = f(*args, **kwargs)

            # TODO(wz2): This is a hack to get a logger instance.
            assert hasattr(args[0], "logger"), print(args[0], type(args[0]))
            assert isinstance(args[0].logger, Logger)
            if args[0].logger is not None:
                cls_name = type(args[0]).__name__
                args[0].logger.record(f"{cls_name}_{key}", time.time() - start)
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


class Logger(object):
    def __init__(
        self,
        trace: bool,
        verbose: bool,
        output_log_path: str,
        repository_path: str,
        tensorboard_path: str,
    ) -> None:
        self.trace = trace
        self.verbose = verbose
        self.repository_path = repository_path
        Path(repository_path).mkdir(parents=True, exist_ok=True)

        level = logging.INFO if not self.verbose else logging.DEBUG
        formatter = "%(levelname)s:%(asctime)s [%(filename)s:%(lineno)s]  %(message)s"
        logging.basicConfig(format=formatter, level=level, force=True)

        # Setup the file logger.
        Path(output_log_path).mkdir(parents=True, exist_ok=True)
        file_logger = logging.FileHandler("{}/output.log".format(output_log_path))
        file_logger.setFormatter(logging.Formatter(formatter))
        file_logger.setLevel(level)
        logging.getLogger().addHandler(file_logger)

        # Setup the writer.
        self.writer: Union[SummaryWriter, None] = None
        if self.trace:
            Path(tensorboard_path).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(tensorboard_path)  # type: ignore

        self.iteration = 1
        self.iteration_data: dict[str, Any] = {}

    def get_logger(self, name: Optional[str]) -> logging.Logger:
        if name is None:
            return logging.getLogger()
        return logging.getLogger(name)

    def stash_results(
        self, info_dict: dict[str, Any], name_override: Optional[str] = None
    ) -> None:
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        time = name_override if name_override else time
        if info_dict["results"] is not None and Path(info_dict["results"]).exists():
            local["mv"][info_dict["results"], f"{self.repository_path}/{time}"].run()
        else:
            Path(f"{self.repository_path}/{time}").mkdir(parents=True, exist_ok=True)

        if info_dict["prior_pgconf"]:
            local["mv"][
                info_dict["prior_pgconf"], f"{self.repository_path}/{time}/old_pg.conf"
            ].run()

        if info_dict["prior_state_container"]:
            with open(f"{self.repository_path}/{time}/prior_state.txt", "w") as f:
                f.write(str(info_dict["prior_state_container"]))

        if info_dict["action_json"]:
            with open(f"{self.repository_path}/{time}/action.txt", "w") as f:
                f.write(info_dict["action_json"])

    def advance(self) -> None:
        if self.writer is None:
            return

        for key, value in self.iteration_data.items():
            if isinstance(value, str):
                # str is considered a np.ScalarType
                self.writer.add_text(key, value, self.iteration)  # type: ignore
            else:
                self.writer.add_scalar(key, value, self.iteration)  # type: ignore

        del self.iteration_data
        self.iteration_data = {}
        self.iteration += 1
        self.writer.flush()  # type: ignore

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
            self.writer.flush()  # type: ignore
