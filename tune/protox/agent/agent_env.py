import copy
import inspect
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray


class AgentEnv(gym.Wrapper[Any, Any, Any, Any]):
    def __init__(self, env: gym.Env[Any, Any]):
        super().__init__(env)
        self.class_attributes = dict(inspect.getmembers(self.__class__))

    def reset(self, **kwargs: Any) -> Tuple[Any, dict[str, Any]]:
        observations, info = self.env.reset(**kwargs)
        self._check_val(event="reset", observations=observations)
        self._observations = observations
        return observations, info

    def step(
        self, actions: NDArray[np.float32]
    ) -> Tuple[Any, float, bool, bool, dict[str, Any]]:
        self._actions = actions

        observations, rewards, term, trunc, infos = self.env.step(actions)
        self._check_val(event="step_wait", observations=observations, rewards=[rewards])
        self._observations = observations

        # Automatically reset.
        if term or trunc:
            infos["terminal_observation"] = observations
            observations, _ = self.env.reset()

        return observations, float(rewards), term, trunc, copy.deepcopy(infos)

    def __getattr__(self, name: str) -> Any:
        """Find attribute from wrapped env(s) if this wrapper does not have it.
        Useful for accessing attributes from envs which are wrapped with multiple wrappers
        which have unique attributes of interest.
        """
        blocked_class = self.getattr_depth_check(name, already_found=False)
        if blocked_class is not None:
            own_class = f"{type(self).__module__}.{type(self).__name__}"
            error_str = (
                f"Error: Recursive attribute lookup for {name} from {own_class} is "
                f"ambiguous and hides attribute from {blocked_class}"
            )
            raise AttributeError(error_str)

        return self.getattr_recursive(name)

    def _get_all_attributes(self) -> Dict[str, Any]:
        """Get all (inherited) instance and class attributes

        :return: all_attributes
        """
        all_attributes = self.__dict__.copy()
        all_attributes.update(self.class_attributes)
        return all_attributes

    def getattr_recursive(self, name: str) -> Any:
        """Recursively check wrappers to find attribute.

        :param name: name of attribute to look for
        :return: attribute
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes:  # attribute is present in this wrapper
            attr = getattr(self, name)
        elif hasattr(self.env, "getattr_recursive"):
            # Attribute not present, child is wrapper. Call getattr_recursive rather than getattr
            # to avoid a duplicate call to getattr_depth_check.
            attr = self.env.getattr_recursive(name)
        else:  # attribute not present, child is an unwrapped VecEnv
            attr = getattr(self.env, name)

        return attr

    def getattr_depth_check(self, name: str, already_found: bool) -> str:
        """See base class.

        :return: name of module whose attribute is being shadowed, if any.
        """
        all_attributes = self._get_all_attributes()
        if name in all_attributes and already_found:
            # this env's attribute is being hidden because of a higher env.
            shadowed_wrapper_class = f"{type(self).__module__}.{type(self).__name__}"
        #elif name in all_attributes and not already_found:
        #    # we have found the first reference to the attribute. Now check for duplicates.
        #    shadowed_wrapper_class = self.env.getattr_depth_check(name, True)
        #else:
        #    # this wrapper does not have the attribute. Keep searching.
        #    shadowed_wrapper_class = self.env.getattr_depth_check(name, already_found)

        return shadowed_wrapper_class

    def check_array_value(
        self, name: str, value: NDArray[np.float32]
    ) -> List[Tuple[str, str]]:
        """
        Check for inf and NaN for a single numpy array.

        :param name: Name of the value being check
        :param value: Value (numpy array) to check
        :return: A list of issues found.
        """
        found = []
        has_nan = np.any(np.isnan(value))
        has_inf = np.any(np.isinf(value))
        if has_inf:
            found.append((name, "inf"))
        if has_nan:
            found.append((name, "nan"))
        return found

    def _check_val(self, event: str, **kwargs: Any) -> None:
        found = []
        for name, value in kwargs.items():
            if isinstance(value, (np.ndarray, list)):
                found += self.check_array_value(name, np.asarray(value))
            elif isinstance(value, dict):
                for inner_name, inner_val in value.items():
                    found += self.check_array_value(f"{name}.{inner_name}", inner_val)
            elif isinstance(value, tuple):
                for idx, inner_val in enumerate(value):
                    found += self.check_array_value(f"{name}.{idx}", inner_val)
            else:
                raise TypeError(f"Unsupported observation type {type(value)}.")

        if found:
            msg = ""
            for i, (name, type_val) in enumerate(found):
                msg += f"found {type_val} in {name}"
                if i != len(found) - 1:
                    msg += ", "

            msg += ".\r\nOriginated from the "

            if event == "reset":
                msg += "environment observation (at reset)"
            elif event == "step":
                msg += (
                    f"environment, Last given value was: \r\n\taction={self._actions}"
                )
            elif event == "step_async":
                msg += f"RL model, Last given value was: \r\n\tobservations={self._observations}"
            else:
                raise ValueError("Internal error.")

            raise ValueError(msg)
