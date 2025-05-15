###############################################################################
# Copied from Friedemann
###############################################################################


import numpy as np
import torch
from torch.nn import Parameter

from stork import activations
from stork.nodes.base import CellGroup


class EfficientIFGroup(CellGroup):
    def __init__(
        self,
        nb_units,
        decay_mode="exp",
        tau_mem=20e-3,
        bias_current=0.0,
        activation=activations.SuperSpike,
        dropout_p=0.0,
        stateful=False,
        **kwargs
    ):
        """
        Args:
            nb_units: The number of units in this group
            decay_mode: Specifies membrane decay mode (either exp, None, or stochastic).
            tau_mem: The membrane time constant in s
            bias_current: Add constant current input which is adjusted to membrane time constant
            activation: The surrogate derivative enabled activation function
            dropout_p: Dropout probability
            stateful: Make the neuron model stateful
        """

        super().__init__(nb_units, dropout_p=dropout_p, stateful=stateful, **kwargs)

        if decay_mode is not None and decay_mode != "none":
            self.decay_mode = decay_mode.lower()
        else:
            self.decay_mode = None

        self.tau_mem = tau_mem
        self.spk_nl = activation.apply
        self.bias_current = bias_current
        self.mem = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem
        if self.decay_mode == "stochastic":
            self.stochastic_decay_probability = time_step / self.tau_mem
        elif self.decay_mode == "periodic":
            self.period = int(self.tau_mem / time_step)
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.decay_mode == "periodic":
            self.offset = np.random.randint(self.period)
        self.mem = self.get_state_tensor("mem", state=self.mem)
        self.out = self.states["out"] = torch.zeros(
            self.int_shape, device=self.device, dtype=self.dtype
        )

    def get_spike_and_reset(self, mem):
        mthr = mem - 1.0
        out = self.spk_nl(mthr)
        rst = out.detach()  # do not BP through reset
        return out, rst

    def forward(self):
        # spike & reset
        new_out, rst = self.get_spike_and_reset(self.mem)

        # synaptic & membrane dynamics
        if self.decay_mode is None:
            new_mem = self.mem
        elif self.decay_mode == "exp":
            new_mem = self.dcy_mem * self.mem
        elif self.decay_mode == "stochastic":
            reset = (
                torch.rand(self.int_shape, device=self.device)
                > self.stochastic_decay_probability
            )
            new_mem = self.mem * reset
        elif self.decay_mode == "periodic":
            offset = self.offset = (self.offset + 1) % self.period
            decay = torch.ones_like(self.mem)
            decay = decay.reshape((decay.shape[0], -1))
            decay[:, offset :: self.period] = 0.1
            decay = decay.reshape(self.mem.shape)
            new_mem = self.mem * decay
        else:
            raise ValueError(
                f"Unknown decay mode {self.decay_mode}. Supported are None, 'exp', 'stochastic', and 'periodic'."
            )


        new_mem *= 1.0 - rst  # multiplicative reset
        new_mem += self.scl_mem * self.bias_current + self.input

        self.out = self.states["out"] = new_out
        self.mem = self.states["mem"] = new_mem
