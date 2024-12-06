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


class StructuredLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        src_blocks=None,
        dst_blocks=None,
        mask=None,
        requires_grad=True,
        bias=False,
    ):
        super(StructuredLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        if src_blocks is None:
            mask = mask
        else:
            self.src_blocks = src_blocks
            self.dst_blocks = dst_blocks
            mask = self.create_block_diagonal()

        assert mask.shape == (
            out_features,
            in_features,
        ), f"Mask dimensions must match (out_features, in_features), got {mask.shape} and {(out_features, in_features)}"

        self.register_buffer("mask", mask)  # Now register it as a buffer

        # Trainable weights and optional bias
        self.weight = nn.Parameter(
            torch.randn(out_features, in_features, requires_grad=requires_grad)
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=requires_grad)
            if bias
            else None
        )

        # Precompute masked weights initially
        self.register_buffer("masked_weight", self.weight * self.mask)

    def forward(self, x):
        # Update masked_weight only if weights are updated
        if self.training:
            self.masked_weight = self.weight * self.mask
        return nn.functional.linear(x, self.masked_weight, self.bias)

    def create_block_diagonal(self):
        assert (
            self.in_features % self.src_blocks == 0
            and self.out_features % self.dst_blocks == 0
        ), "Source and destination shapes must be divisible by their respective block sizes"

        # Number of blocks
        num_blocks = self.in_features // self.src_blocks

        # Create one block and repeat to form a block diagonal
        block = torch.ones(self.dst_blocks, self.src_blocks)
        matrix = torch.zeros(self.out_features, self.in_features)
        for i in range(num_blocks):
            matrix[
                i * self.dst_blocks : (i + 1) * self.dst_blocks,
                i * self.src_blocks : (i + 1) * self.src_blocks,
            ] = block

        return matrix


class StructuredLinearBlocks(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        src_blocks=None,
        dst_blocks=None,
        requires_grad=True,
        bias=False,
    ):
        super(StructuredLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.src_blocks = src_blocks
        self.dst_blocks = dst_blocks
        self.requires_grad = requires_grad

        # Verify divisibility
        assert (
            self.in_features % self.src_blocks == 0
            and self.out_features % self.dst_blocks == 0
        ), "Source and destination shapes must be divisible by their respective block sizes"

        # Number of blocks
        self.num_blocks = self.in_features // self.src_blocks

        # Create trainable blocks and register them as parameters
        self.blocks = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn(
                        self.dst_blocks,
                        self.src_blocks,
                        requires_grad=self.requires_grad,
                    )
                )
                for _ in range(self.num_blocks)
            ]
        )

        # Optional bias
        self.bias = (
            nn.Parameter(torch.zeros(out_features), requires_grad=requires_grad)
            if bias
            else None
        )

    def forward(self, x):
        # Initialize the weight matrix
        weight = torch.zeros(self.out_features, self.in_features, device=x.device)

        # Populate weight matrix using the block parameters
        for i, block in enumerate(self.blocks):
            weight[
                i * self.dst_blocks : (i + 1) * self.dst_blocks,
                i * self.src_blocks : (i + 1) * self.src_blocks,
            ] = block

        # Perform the linear operation
        return nn.functional.linear(x, weight, self.bias)


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

    def add_diagonal_structure(self, width=1.0, ampl=1.0):
        if not isinstance(self.op, nn.Linear):
            raise ValueError("Expected op to be nn.Linear to add diagonal structure.")
        A = np.zeros(self.op.weight.shape)
        x = np.linspace(0, A.shape[0], A.shape[1])
        for i in range(len(A)):
            A[i] = ampl * np.exp(-((x - i) ** 2) / width**2)
        self.op.weight.data += torch.from_numpy(A)

    def get_weights(self):
        try:
            return self.op.weight
        except:
            weight = torch.zeros(self.op.out_features, self.op.in_features)

            for i, block in enumerate(self.op.blocks):
                weight[
                    i * self.op.dst_blocks : (i + 1) * self.op.dst_blocks,
                    i * self.op.src_blocks : (i + 1) * self.op.src_blocks,
                ] = block
            return weight

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


class IdentityConnection(BaseConnection):
    def __init__(
        self,
        src,
        dst,
        target=None,
        bias=False,
        requires_grad=True,
        name=None,
        regularizers=None,
        constraints=None,
        tie_weights=None,
        weight_scale=1.0,
    ):
        """Initialize IdentityConnection

        Args:
            tie_weights (list of int, optional): Tie weights along dims given in list
            weight_scale (float, optional): Scale everything by this factor. Useful when the connection is used for relaying currents rather than spikes.
        """
        super(IdentityConnection, self).__init__(
            src,
            dst,
            name=name,
            target=target,
            regularizers=regularizers,
            constraints=constraints,
        )

        self.requires_grad = requires_grad
        self.weight_scale = weight_scale
        wshp = src.shape

        # Set weights tensor dimension to 1 along tied dimensions
        if tie_weights is not None:
            wshp = list(wshp)
            for d in tie_weights:
                wshp[d] = 1
            wshp = tuple(wshp)

        self.weights = Parameter(torch.randn(wshp), requires_grad=requires_grad)
        if bias:
            self.bias = Parameter(torch.randn(wshp), requires_grad=requires_grad)

    def get_weights(self):
        return self.weights

    def get_regularizer_loss(self):
        reg_loss = torch.tensor(0.0, device=self.device)
        for reg in self.regularizers:
            reg_loss += reg(self.get_weights())
        return reg_loss

    def apply_constraints(self):
        for const in self.constraints:
            const.apply(self.weights)

    def forward(self):
        preact = self.src.out
        if self.bias is None:
            self.dst.scale_and_add_to_state(
                self.weight_scale, self.target, self.weights * preact
            )
        else:
            self.dst.scale_and_add_to_state(
                self.weight_scale, self.target, self.weights * preact + self.bias
            )

    def propagate(self):
        self.forward()


class ConvConnection(Connection):
    def __init__(self, src, dst, conv=nn.Conv1d, **kwargs):
        super(ConvConnection, self).__init__(src, dst, operation=conv, **kwargs)


class Conv2dConnection(Connection):
    def __init__(self, src, dst, conv=nn.Conv2d, **kwargs):
        super(Conv2dConnection, self).__init__(src, dst, operation=conv, **kwargs)
