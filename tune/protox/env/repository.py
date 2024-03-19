import json
from datetime import datetime

import numpy as np
from plumbum import local


class Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(Encoder, self).default(obj)


class Repository(object):
    def __init__(self, repository_path, action_space):
        self.repository_path = repository_path
        self.action_space = action_space

    def add(self, action, metric, reward, results, conf_path=None, prior_state=None):
        time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        local["mv"][results, f"{self.repository_path}/{time}"].run()

        if conf_path is not None:
            local["cp"][conf_path, f"{self.repository_path}/{time}/pg.conf"].run()

        if prior_state is not None:
            with open(f"{self.repository_path}/{time}/prior_state.txt", "w") as f:
                f.write(str(prior_state))

        with open(f"{self.repository_path}/{time}/act_sql.txt", "w") as f:
            idx_space = self.action_space.get_index_space()
            sql = ""
            if idx_space is not None:
                sql = idx_space.construct_indexaction(
                    action[self.action_space.index_space_ind]
                ).sql(add=True)
            f.write(sql + "\n\n")

            knobs = []
            if self.action_space.knob_space_ind is not None:
                knobs = sorted(
                    [
                        (act, val)
                        for act, val in action[self.action_space.knob_space_ind].items()
                    ],
                    key=lambda x: x[0],
                )
            for knob, val in knobs:
                f.write(f"{knob} = {val}\n")

        return reward
