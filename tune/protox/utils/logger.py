import torch
import numpy as np
import logging
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, spec):
        self.trace = not spec.no_trace
        self.verbose = spec.verbose

        level = logging.INFO if not self.verbose else logging.DEBUG
        formatter = "%(levelname)s:%(asctime)s %(message)s"
        logging.basicConfig(format=formatter, level=level, force=True)

        # Setup the file logger.
        Path(spec.output_log_path).mkdir(parents=True, exist_ok=True)
        file_logger = logging.FileHandler("{}/output.log".format(spec.output_log_path))
        file_logger.setFormatter(logging.Formatter(formatter))
        file_logger.setLevel(level)
        logging.getLogger().addHandler(file_logger)

        # Setup the writer.
        self.writer = None
        if self.trace:
            Path(spec.tensorboard_path).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(spec.tensorboard_path)

        self.spec = spec
        self.iteration = 1
        self.iteration_data = {}
        self.iteration_label = {}
        self.iteration_header = {}

    def advance(self):
        for key, value in self.iteration_data.items():
            if isinstance(value, np.ScalarType):
                if isinstance(value, str):
                    # str is considered a np.ScalarType
                    self.writer.add_text(key, value, self.iteration)
                else:
                    self.writer.add_scalar(key, value, self.iteration)
            elif isinstance(value, list):
                if len(value) > 0:
                    assert isinstance(value[0], torch.Tensor) or isinstance(value[0], np.ndarray)
                    write_value = torch.cat(value) if isinstance(value[0], torch.Tensor) else np.concatenate(value, axis=0)
                    self.writer.add_embedding(
                        write_value,
                        metadata=self.iteration_label[key],
                        global_step=self.iteration,
                        tag=key,
                        metadata_header=self.iteration_header[key])
            else:
                assert False, print("Unknown record: ", type(value), key, value)

        del self.iteration_data
        del self.iteration_label
        self.iteration_data = {}
        self.iteration_label = {}
        self.iteration_header = {}
        self.iteration += 1
        self.writer.flush()

    def record(self, key, value, label=None, header=None):
        # Accumulate data.
        if isinstance(value, np.ScalarType):
            if isinstance(value, str):
                self.iteration_data[key] = value
            else:
                if key not in self.iteration_data:
                    self.iteration_data[key] = 0.0
                self.iteration_data[key] += value
        elif isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            if label is not None and len(value.shape) == 2:
                if key not in self.iteration_data:
                    self.iteration_data[key] = []
                if key not in self.iteration_label:
                    self.iteration_label[key] = []
                self.iteration_data[key].append(value)
                self.iteration_label[key].extend(label)
                self.iteration_header[key] = header

    def flush(self):
        if self.trace:
            self.advance()
            self.writer.flush()
