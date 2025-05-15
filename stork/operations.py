import torch
import torch.nn as nn

from torch.nn.parameter import Parameter

import numpy as np

from . import core
from . import constraints as stork_constraints

##############################################################################################################################
# TODO: clean up this file
##############################################################################################################################

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

        # Handle mask creation
        if src_blocks is None:
            if mask is None:
                raise ValueError("Either src_blocks or mask must be provided")
        else:
            self.src_blocks = src_blocks
            self.dst_blocks = dst_blocks
            mask = self.create_block_diagonal()

        # Verify mask dimensions
        if mask.shape != (out_features, in_features):
            raise ValueError(
                f"Mask dimensions must match (out_features, in_features), "
                f"got {mask.shape} and {(out_features, in_features)}"
            )

        # self.mask = mask
        # Register mask as a buffer (not a parameter)
        self.register_buffer("mask", mask)

        # Initialize trainable weights and optional bias
        self.weight = nn.Parameter(
            self.mask
            * torch.randn(out_features, in_features)
            / (in_features**0.5),  # Xavier init
            requires_grad=requires_grad,
        )

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features), requires_grad=requires_grad
            )
        else:
            self.register_parameter("bias", None)

    def create_block_diagonal(self):
        """
        Creates a block diagonal mask matrix based on src_blocks and dst_blocks.

        Returns:
            torch.Tensor: A binary mask tensor with 1s in block diagonal positions.
        """
        # Validate block sizes
        if not hasattr(self, "src_blocks") or not hasattr(self, "dst_blocks"):
            raise AttributeError("src_blocks and dst_blocks must be defined")

        if (
            self.in_features % self.src_blocks != 0
            or self.out_features % self.dst_blocks != 0
        ):
            raise ValueError(
                f"Source and destination shapes must be divisible by their respective block sizes. "
                f"Got {self.in_features} in_features with {self.src_blocks} src_blocks, and "
                f"{self.out_features} out_features with {self.dst_blocks} dst_blocks"
            )

        # Calculate number of blocks and create mask more efficiently
        num_blocks = self.in_features // self.src_blocks

        # print(num_blocks, self.dst_blocks, self.src_blocks, self.out_features//self.dst_blocks)

        blocks = [
            torch.ones(self.dst_blocks, self.src_blocks) for _ in range(num_blocks)
        ]
        return torch.block_diag(*blocks)

        # # Fallback implementation
        # mask = torch.zeros(self.out_features, self.in_features)

        # # Vectorized implementation for better performance
        # block_indices = torch.arange(num_blocks)
        # row_indices = block_indices.unsqueeze(1) * self.dst_blocks + torch.arange(self.dst_blocks).unsqueeze(0)
        # col_indices = block_indices.unsqueeze(1) * self.src_blocks + torch.arange(self.src_blocks).unsqueeze(0)

        # for i in range(num_blocks):
        #     rows = row_indices[i].reshape(-1, 1)
        #     cols = col_indices[i].reshape(1, -1)
        #     mask[rows, cols] = 1.0

        # return mask

    def get_weights(self):
        """returns the masked weights"""
        # Create masked weight on-the-fly
        return self.weight * self.mask

    def forward(self, x):
        # Apply mask during forward pass
        masked_weight = self.weight * self.mask
        return nn.functional.linear(x, masked_weight, self.bias)

    def extra_repr(self):
        """Add interpretable representation for print statements."""
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# class StructuredLinearBlocks(nn.Module):
#     def __init__(
#         self,
#         in_features,
#         out_features,
#         src_blocks=None,
#         dst_blocks=None,
#         requires_grad=True,
#         bias=False,
#     ):
#         super(StructuredLinear, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.src_blocks = src_blocks
#         self.dst_blocks = dst_blocks
#         self.requires_grad = requires_grad

#         # Verify divisibility
#         assert (
#             self.in_features % self.src_blocks == 0
#             and self.out_features % self.dst_blocks == 0
#         ), "Source and destination shapes must be divisible by their respective block sizes"

#         # Number of blocks
#         self.num_blocks = self.in_features // self.src_blocks

#         # Create trainable blocks and register them as parameters
#         self.blocks = nn.ParameterList(
#             [
#                 nn.Parameter(
#                     torch.randn(
#                         self.dst_blocks,
#                         self.src_blocks,
#                         requires_grad=self.requires_grad,
#                     )
#                 )
#                 for _ in range(self.num_blocks)
#             ]
#         )

#         # Optional bias
#         self.bias = (
#             nn.Parameter(torch.zeros(out_features), requires_grad=requires_grad)
#             if bias
#             else None
#         )

#     def forward(self, x):
#         # Initialize the weight matrix
#         weight = torch.zeros(self.out_features, self.in_features, device=x.device)

#         # Populate weight matrix using the block parameters
#         for i, block in enumerate(self.blocks):
#             weight[
#                 i * self.dst_blocks : (i + 1) * self.dst_blocks,
#                 i * self.src_blocks : (i + 1) * self.src_blocks,
#             ] = block

#         # Perform the linear operation
#         return nn.functional.linear(x, weight, self.bias)
