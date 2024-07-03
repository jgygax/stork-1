import numpy as np
import torch

from stork.nodes.input.base import InputGroup


class PoissonStimulus(InputGroup):
    """Gets a vector of firing rates (a firing rate for each neuron) and creates a Poisson spike train for each neuron"""

    def __init__(self, shape, n_steps, dt=1e-3, name="PoissonStimulus"):
        super(PoissonStimulus, self).__init__(shape, name=name)
        self.dt = dt
        self.n_steps = n_steps

    def feed_data(self, data):
        prob_single_t = data * self.dt
        mask = torch.rand(*data.shape, self.n_steps, device=self.device)
        prob = torch.stack([prob_single_t] * self.n_steps, dim=2).to(self.device)
        spikes = torch.zeros(mask.shape, device=self.device)
        spikes[mask < prob] = 1.0
        self.local_data = spikes.to(self.device)

    def forward(self):
        self.out = self.states["out"] = self.local_data[:, :, self.clk]
