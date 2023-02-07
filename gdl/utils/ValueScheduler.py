from typing import Dict
from omegaconf import DictConfig

class ValueScheduler(object): 

    def __call__(self, step):
        raise NotImplementedError()

class StaticValueScheduler(ValueScheduler):
    def __init__(self, value):
        self.value = value

    def __call__(self, step):
        return self.value

class LinearValueScheduler(ValueScheduler):

    def __init__(self, start_value, end_value, start_step, end_step):
        self.start_value = start_value
        self.end_value = end_value
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, step):
        if step is None or step < 0:
            return self.end_value
        if step <= self.start_step:
            return self.start_value
        if step >= self.end_step:
            return self.end_value
        interval = self.end_step - self.start_step
        slope = (self.end_value - self.start_value) / interval
        return self.start_value + slope * (step - self.start_step)


def scheduler_from_dict(cfg):
    if isinstance(cfg, (int, float)):
        return StaticValueScheduler(cfg.tau)
    assert isinstance(cfg, (Dict, dict, DictConfig))
    if cfg.schedule_type == "linear":
        return LinearValueScheduler(cfg.start.value, cfg.end.value, cfg.start.step, cfg.end.step)
    elif cfg.schedule_type == "static":
        return StaticValueScheduler(cfg.value)
    else:
        raise NotImplementedError(f"Unknown schedule type {cfg.schedule_type}.")