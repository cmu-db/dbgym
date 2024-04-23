import json
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from tune.protox.env.logger import Logger

# Initial penalty to apply to create the "worst" perf from the baseline.
INITIAL_PENALTY_MULTIPLIER = 4.0


class RewardUtility(object):
    def __init__(
        self, target: str, metric: str, reward_scaler: float, logger: Logger
    ) -> None:
        self.reward_scaler = reward_scaler
        self.target = target
        self.metric = metric
        self.maximize = target == "tps"
        self.worst_perf: Optional[float] = None
        self.relative_baseline: Optional[float] = None
        self.previous_result: Optional[float] = None
        self.logger = logger

    def is_perf_better(self, new_perf: float, old_perf: float) -> bool:
        if self.maximize and new_perf > old_perf:
            return True
        elif not self.maximize and new_perf < old_perf:
            return True
        return False

    def set_relative_baseline(
        self, relative_baseline: float, prev_result: Optional[float] = None
    ) -> None:
        self.logger.get_logger(__name__).debug(
            f"[set_relative_baseline]: {relative_baseline}"
        )
        self.relative_baseline = relative_baseline
        self.previous_result = prev_result
        if self.worst_perf is None:
            if self.maximize:
                self.worst_perf = relative_baseline / INITIAL_PENALTY_MULTIPLIER
            else:
                self.worst_perf = relative_baseline * INITIAL_PENALTY_MULTIPLIER
        elif not self.is_perf_better(relative_baseline, self.worst_perf):
            self.worst_perf = relative_baseline

        if self.previous_result is None:
            # Set the previous result to the baseline if not specified.
            self.previous_result = relative_baseline

    def parse_tps_avg_p99_for_metric(
        self, parent: Union[Path, str]
    ) -> Tuple[float, float, float]:
        files = [f for f in Path(parent).rglob("*.summary.json")]
        assert len(files) == 1

        summary = files[0]
        self.logger.get_logger(__name__).debug(
            f"Reading TPS metric from file: {summary}"
        )
        # don't call open_and_save() because summary is generated from this run
        with open(summary, "r") as f:
            s = json.load(f)
            tps = s["Throughput (requests/second)"]
            p99 = s["Latency Distribution"]["99th Percentile Latency (microseconds)"]
            avg = s["Latency Distribution"]["Average Latency (microseconds)"]

        return float(tps), float(p99), float(avg)

    def __parse_tps_for_metric(self, parent: Union[str, Path]) -> float:
        files = [f for f in Path(parent).rglob("*.summary.json")]
        assert len(files) == 1

        summary = files[0]
        self.logger.get_logger(__name__).debug(
            f"Reading TPS metric from file: {summary}"
        )
        # don't call open_and_save() because summary is generated from this run
        with open(summary, "r") as f:
            tps = json.load(f)["Throughput (requests/second)"]
        return float(tps)

    def __parse_runtime_for_metric(self, parent: Union[str, Path]) -> float:
        files = [f for f in Path(parent).rglob("*.raw.csv")]
        assert len(files) > 0

        summary = [f for f in Path(parent).rglob("*.raw.csv")][0]
        data = pd.read_csv(summary)
        assert len(data.columns) == 7
        summed_data = data.sum()
        summed_latency: float = summed_data["Latency (microseconds)"]
        return summed_latency / 1.0e6

    def __call__(
        self,
        result_dir: Union[str, Path, None] = None,
        metric: Optional[float] = None,
        update: bool = True,
        did_error: bool = False,
    ) -> Tuple[float, float]:

        # TODO: we need to get the memory consumption of indexes. if the index usage
        # exceeds the limit, then kill the reward function. may also want to penalize
        # reward based on delta.
        #
        # (param) (new_tps/old_tps) + (1-param) (max(min_mem, new_mem)/min_mem
        #
        # minimum memory before start trading...)
        assert did_error or result_dir is not None or metric is not None
        self.logger.get_logger(__name__).debug(
            f"[reward_calc]: {result_dir} {metric} {update} {did_error}"
        )

        if metric is None:
            # Either it errored or we have a result directory to process.
            assert did_error or result_dir

            # Extract the metric if we're running it manually.
            metric_fn = (
                self.__parse_tps_for_metric
                if self.target == "tps"
                else self.__parse_runtime_for_metric
            )

            if did_error:
                metric = self.worst_perf
            else:
                assert result_dir
                metric = metric_fn(result_dir)
        actual_r = None
        assert metric is not None

        # Note that if we are trying to minimize, the smaller metric is, the better we are.
        # And policy optimization maximizes the rewards.
        #
        # As such, for all relative-ness, we treat maximize 100 -> 1000 with reward 9
        # similarly to the case of minimize 1000 -> 100 with reward 9.
        # This can effectively be done as flipping what is considered baseline and what is not.

        if self.relative_baseline is None:
            # Use the metric directly.
            actual_r = metric
        elif self.metric == "multiplier":
            actual_r = (
                metric / self.relative_baseline
                if self.maximize
                else self.relative_baseline / metric
            )
        elif self.metric == "relative":
            if self.maximize:
                actual_r = (metric - self.relative_baseline) / self.relative_baseline
            else:
                actual_r = (self.relative_baseline - metric) / self.relative_baseline
        elif self.metric == "cdb_delta":
            assert self.previous_result

            # refer to https://dbgroup.cs.tsinghua.edu.cn/ligl/papers/sigmod19-cdbtune.pdf.
            relative_baseline = (
                (metric - self.relative_baseline) / self.relative_baseline
                if self.maximize
                else (self.relative_baseline - metric) / self.relative_baseline
            )
            relative_prev = (
                (metric - self.previous_result) / self.previous_result
                if self.maximize
                else (self.previous_result - metric) / self.previous_result
            )

            if relative_baseline > 0:
                actual_r = (pow(1 + relative_baseline, 2) - 1) * abs(1 + relative_prev)
            else:
                actual_r = -(pow(1 - relative_baseline, 2) - 1) * abs(1 - relative_prev)

            # Apply the truncation step.
            if actual_r > 0 and relative_prev < 0:
                actual_r = 0

        if update:
            # Update worst seen metric.
            if self.worst_perf is None or not self.is_perf_better(
                metric, self.worst_perf
            ):
                self.worst_perf = metric

            self.previous_result = metric

        # Scale the actual reward by the scaler.
        assert actual_r is not None
        return metric, actual_r * self.reward_scaler
