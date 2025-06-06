import torch
import torch.nn as nn


class StructuredLinear(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        mask,
        requires_grad=True,
        bias=False,
    ):
        super(StructuredLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Verify mask dimensions
        if mask.shape != (out_features, in_features):
            raise ValueError(
                f"Mask dimensions must match (out_features, in_features), "
                f"got {mask.shape} for mask and {(out_features, in_features)} for (out_features, in_features)"
            )

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

