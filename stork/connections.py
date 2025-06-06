import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

import numpy as np

from . import core
from . import constraints as stork_constraints


class BaseConnection(core.NetworkNode):
    def __init__(
        self, src, dst, target=None, name=None, regularizers=None, constraints=None
    ):
        """Abstract base class of Connection objects.

        Args:
            src (CellGroup): The source group
            dst (CellGroup: The destination group
            target (string, optional): The name of the target state tensor.
            name (string, optional): Name of the node
            regularizers (list): List of regularizer objects.
            constraints (list): List of constraints.

        """

        super(BaseConnection, self).__init__(name=name, regularizers=regularizers)
        self.src = src
        self.dst = dst

        if target is None:
            self.target = dst.default_target
        else:
            self.target = target

        if constraints is None:
            self.constraints = []
        elif isinstance(constraints, list):
            self.constraints = constraints
        elif issubclass(type(constraints), stork_constraints.WeightConstraint):
            self.constraints = [constraints]
        else:
            raise ValueError

    def init_parameters(self, initializer):
        """
        Initializes connection weights and biases.
        """
        initializer.initialize(self)
        self.apply_constraints()

    def propagate(self):
        raise NotImplementedError

    def apply_constraints(self):
        raise NotImplementedError

    def reset_state(self, batchsize):
        pass


class Connection(BaseConnection):
    def __init__(
        self,
        src,
        dst,
        operation=nn.Linear,
        target=None,
        bias=False,
        requires_grad=True,
        propagate_gradients=True,
        flatten_input=False,
        name=None,
        regularizers=None,
        constraints=None,
        **kwargs,
    ):
        super(Connection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.requires_grad = requires_grad
        self.propagate_gradients = propagate_gradients
        self.flatten_input = flatten_input

        if flatten_input:
            self.op = operation(src.nb_units, dst.shape[0], bias=bias, **kwargs)
        else:
            self.op = operation(src.shape[0], dst.shape[0], bias=bias, **kwargs)
        for name, param in self.op.named_parameters():
            param.requires_grad = requires_grad

    def configure(self, batch_size, nb_steps, time_step, device, dtype):
        super().configure(batch_size, nb_steps, time_step, device, dtype)

    def get_weights(self):
        # Todo: maybe rather self.op.get_weights()?
        return self.op.weight

    def get_regularizer_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
        return reg_loss

    def forward(self):
        preact = self.src.out
        if not self.propagate_gradients:
            preact = preact.detach()
        if self.flatten_input:
            shp = preact.shape
            preact = preact.reshape(shp[:1] + (-1,))

        out = self.op(preact)
        self.dst.add_to_state(self.target, out)

    def propagate(self):
        self.forward()

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.op.weight)


class ConvConnection(Connection):
    def __init__(self, src, dst, conv=nn.Conv1d, **kwargs):
        super(ConvConnection, self).__init__(src, dst, operation=conv, **kwargs)


class Conv2dConnection(Connection):
    def __init__(self, src, dst, conv=nn.Conv2d, **kwargs):
        super(Conv2dConnection, self).__init__(src, dst, operation=conv, **kwargs)
