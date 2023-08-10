import numpy as np


class LambdaWarmUpCosineScheduler:
    """Cosine scheduler with warm up."""

    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0

    def schedule(self, n):
        """Schedule the lr for input steps."""
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            lr = (
                self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
                if t < 1.0
                else self.lr_min
            )
            self.last_lr = lr
            return lr

    def __call__(self, n):
        """Return the scheduler func output when called."""
        return self.schedule(n)
