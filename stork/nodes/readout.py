import numpy as np
import torch

from stork.nodes.base import CellGroup


class ReadoutGroup(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        tau_syn=5e-3,
        bias_current=0.0,
        weight_scale=1.0,
        initial_state=-1e-3,
        stateful=False,
        name="Readout",
        dropout_p=0.0,
        regularizers=None,
        **kwargs,
    ):
        super().__init__(
            shape,
            dropout_p=dropout_p,
            stateful=stateful,
            name=name,
            regularizers=regularizers,
            **kwargs,
        )

        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.store_output_seq = True
        self.initial_state = initial_state
        self.weight_scale = weight_scale
        self.out = None
        self.syn = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem
        self.dcy_syn = float(np.exp(-time_step / self.tau_syn))
        self.scl_syn = (1.0 - self.dcy_syn) * self.weight_scale

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)
        self.syn = self.get_state_tensor("syn", state=self.syn)

    def forward(self):
        # synaptic & membrane dynamics
        new_syn = self.dcy_syn * self.syn + self.input
        new_mem = self.dcy_mem * self.out + self.scl_mem * self.syn

        self.out = self.states["out"] = new_mem
        self.syn = self.states["syn"] = new_syn
        # self.out_seq.append(self.out)


####################################################################################################
# EXPERIMENTAL NEWLY ADDED GROUPS (TODO: still to be tested)
####################################################################################################
class DeltaSynapseReadoutGroup(CellGroup):
    def __init__(
        self,
        shape,
        tau_mem=10e-3,
        weight_scale=1.0,
        initial_state=-1e-3,
        stateful=False,
    ):
        super().__init__(shape, stateful=stateful, name="Readout")
        self.tau_mem = tau_mem
        self.store_output_seq = True
        self.initial_state = initial_state
        self.weight_scale = weight_scale
        self.out = None
        self.syn = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)

    def forward(self):
        # synaptic & membrane dynamics
        new_mem = self.dcy_mem * self.out + self.scl_mem * self.input
        self.out = self.states["out"] = new_mem


class EfficientReadoutGroup(CellGroup):
    def __init__(
        self,
        shape,
        decay_mode=None,
        tau_mem=10e-3,
        bias_current=0.0,
        weight_scale=1.0,
        initial_state=-1e-3,
        stateful=False,
        name="Readout",
        dropout_p=0.0,
        regularizers=None,
        **kwargs,
    ):
        super().__init__(
            shape,
            dropout_p=dropout_p,
            stateful=stateful,
            name=name,
            regularizers=regularizers,
            **kwargs,
        )

        self.tau_mem = tau_mem
        self.store_output_seq = True
        self.initial_state = initial_state
        self.weight_scale = weight_scale
        self.out = None
        self.syn = None

        self.bias_current = bias_current

        if decay_mode is not None and decay_mode != "none":
            self.decay_mode = decay_mode.lower()
        else:
            self.decay_mode = None

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)
        self.dcy_mem = float(np.exp(-time_step / self.tau_mem))
        self.scl_mem = 1.0 - self.dcy_mem
        if self.decay_mode == "stochastic":
            self.stochastic_decay_probability = time_step / self.tau_mem
        elif self.decay_mode == "periodic":
            self.period = int(self.tau_mem / time_step)

    def reset_state(self, batch_size=None):
        super().reset_state(batch_size)
        if self.decay_mode == "periodic":
            self.offset = np.random.randint(self.period)
        self.out = self.get_state_tensor("out", state=self.out, init=self.initial_state)

    def forward(self):
        # membrane dynamics
        if self.decay_mode is None:
            new_mem = self.out + self.scl_mem * self.input

        elif self.decay_mode == "exp":
            new_mem = self.dcy_mem * self.out
        elif self.decay_mode == "stochastic":
            reset = (
                torch.rand(self.int_shape, device=self.device)
                > self.stochastic_decay_probability
            )
            new_mem = self.out * reset
        elif self.decay_mode == "periodic":
            offset = self.offset = (self.offset + 1) % self.period
            decay = torch.ones_like(self.out)
            decay = decay.reshape((decay.shape[0], -1))
            decay[:, offset :: self.period] = 0.1
            decay = decay.reshape(self.out.shape)
            new_mem = self.out * decay
        else:
            raise ValueError(
                f"Unknown decay mode {self.decay_mode}. Supported are None, 'exp', 'stochastic', and 'periodic'."
            )

        self.out = self.states["out"] = new_mem
        # self.out_seq.append(self.out)
