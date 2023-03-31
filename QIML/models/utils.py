import numpy as np
import torch


class BetaRateScheduler:
    def __init__(
        self,
        initial_beta: float = 0.0,
        end_beta: float = 4.0,
        cap_steps: int = 4000,
        hold_steps: int = 2000,
    ):
        self._initial_beta = initial_beta
        self._end_beta = end_beta
        self._cap_steps = cap_steps
        self._hold_steps = hold_steps
        self.reset()

    @property
    def current_step(self):
        return self._current_step

    @current_step.setter
    def current_step(self, value: int):
        assert value >= 0
        self._current_step = value

    def reset(self):
        self.current_step = 0

    def __iter__(self):
        return self.beta()

    def beta(self):
        """
        Returns a generator that yields the next value of beta
        according to the scheduler. In the current implementation,
        the scheduler corresponds to a linear ramp up to `cap_steps`
        and subsequently holds the value of `end_beta` for another
        `hold_steps`. Once this is done, the value of `beta` is
        set back to zero, and the cycle begins anew.

        Yields
        -------
        float
            Value of beta at the current global step
        """
        beta_values = np.concatenate(
            [
                np.linspace(self._initial_beta, self._end_beta, self._cap_steps),
                np.array([self._end_beta for _ in range(self._hold_steps)]),
            ]
        )
        while self.current_step < self._cap_steps + self._hold_steps:
            self.current_step = self.current_step + 1
            yield beta_values[self.current_step - 1]
        self.reset()


def get_encoded_size(data, model):
    data.setup()
    # Loop to generate a batch of data taken from dataset
    for i in range(0, 12):
        if i == 0:
            X, _ = data.train_set.__getitem__(0)
            X = X.unsqueeze(0)
        else:
            Xtemp, _ = data.train_set.__getitem__(0)
            Xtemp = Xtemp.unsqueeze(0)
            X = torch.cat((X, Xtemp), dim=0)

    # some shape tests before trying to actually train
    z, res = model.encoder(X.unsqueeze(1))
    return z, res