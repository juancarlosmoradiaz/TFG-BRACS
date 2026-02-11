# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Chowder aggregation algorithm."""

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Owkin, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.



import warnings
from typing import List, Optional, Tuple, Union

import torch
from torch import nn


class MLP(torch.nn.Sequential):
    """MLP Module.
    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int = 1
        Prediction (model output) dimension.
    hidden: Optional[List[int]] = None
        Dimension of hidden layer(s).
    dropout: Optional[List[float]] = None
        Dropout rate(s).
    activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        MLP activation.
    bias: bool = True
        Add bias to MLP hidden layers.
    Raises
    ------
    ValueError
        If ``hidden`` and ``dropout`` do not share the same length.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden: Optional[List[int]] = None,
        dropout: Optional[List[float]] = None,
        activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ):
        if dropout is not None:
            if hidden is not None:
                assert len(hidden) == len(
                    dropout
                ), "hidden and dropout must have the same length"
            else:
                raise ValueError(
                    "hidden must have a value and have the same length as dropout if dropout is given."
                )

        d_model = in_features
        layers = []

        if hidden is not None:
            for i, h in enumerate(hidden):
                seq = [torch.nn.Linear(d_model, h, bias=bias)]
                d_model = h

                if activation is not None:
                    seq.append(activation)

                if dropout is not None:
                    seq.append(torch.nn.Dropout(dropout[i]))

                layers.append(torch.nn.Sequential(*seq))

        layers.append(torch.nn.Linear(d_model, out_features))

        super(MLP, self).__init__(*layers)

class MaskedLinear(torch.nn.Linear):
    """
    Linear layer to be applied tile wise.
    This layer can be used in combination with a mask
    to prevent padding tiles from influencing the values of a subsequent
    activation.
    Example:
        >>> module = Linear(in_features=128, out_features=1) # With Linear
        >>> out = module(slide)
        >>> wrong_value = torch.sigmoid(out) # Value is influenced by padding
        >>> module = MaskedLinear(in_features=128, out_features=1, mask_value='-inf') # With MaskedLinear
        >>> out = module(slide, mask) # Padding now has the '-inf' value
        >>> correct_value = torch.sigmoid(out) # Value is not influenced by padding as sigmoid('-inf') = 0
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    mask_value: Union[str, int]
        value to give to the mask
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        mask_value: Union[str, float],
        bias: bool = True,
    ):
        super(MaskedLinear, self).__init__(
            in_features=in_features, out_features=out_features, bias=bias
        )
        self.mask_value = mask_value

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):  # pylint: disable=arguments-renamed
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, SEQ_LEN, IN_FEATURES).
        mask: Optional[torch.BoolTensor] = None
            True for values that were padded, shape (B, SEQ_LEN, 1),
        Returns
        -------
        x: torch.Tensor
            (B, SEQ_LEN, OUT_FEATURES)
        """
        x = super(MaskedLinear, self).forward(x)
        if mask is not None:
            x = x.masked_fill(mask, float(self.mask_value))
        return x

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"mask_value={self.mask_value}, bias={self.bias is not None}"
        )

class TilesMLP(torch.nn.Module):
    """MLP to be applied to tiles to compute scores.
    This module can be used in combination of a mask
    to prevent padding from influencing the scores values.
    Parameters
    ----------
    in_features: int
        size of each input sample
    out_features: int
        size of each output sample
    hidden: Optional[List[int]] = None
        Number of hidden layers and their respective number of features.
    bias: bool = True
        If set to ``False``, the layer will not learn an additive bias.
    activation: torch.nn.Module = torch.nn.Sigmoid()
        MLP activation function
    dropout: Optional[torch.nn.Module] = None
        Optional dropout module. Will be interlaced with the linear layers.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int = 1,
        hidden: Optional[List[int]] = None,
        bias: bool = True,
        activation: torch.nn.Module = torch.nn.Sigmoid(),
        dropout: Optional[torch.nn.Module] = None,
    ):
        super(TilesMLP, self).__init__()

        self.hidden_layers = torch.nn.ModuleList()
        if hidden is not None:
            for h in hidden:
                self.hidden_layers.append(
                    MaskedLinear(in_features, h, bias=bias, mask_value="-inf")
                )
                self.hidden_layers.append(activation)
                if dropout:
                    self.hidden_layers.append(dropout)
                in_features = h

        self.hidden_layers.append(
            torch.nn.Linear(in_features, out_features, bias=bias)
        )

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ):
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES), True for values that were padded.
        Returns
        -------
        x: torch.Tensor
            (B, N_TILES, OUT_FEATURES)
        """
        for layer in self.hidden_layers:
            if isinstance(layer, MaskedLinear):
                x = layer(x, mask)
            else:
                x = layer(x)
        return x

class ExtremeLayer(torch.nn.Module):
    """Extreme layer.
    Returns concatenation of n_top top tiles and n_bottom bottom tiles
    .. warning::
        If top tiles or bottom tiles is superior to the true number of
        tiles in the input then padded tiles will be selected and their value
        will be 0.
    Parameters
    ----------
    n_top: Optional[int] = None
        Number of top tiles to select
    n_bottom: Optional[int] = None
        Number of bottom tiles to select
    dim: int = 1
        Dimension to select top/bottom tiles from
    return_indices: bool = False
        Whether to return the indices of the extreme tiles
    Raises
    ------
    ValueError
        If ``n_top`` and ``n_bottom`` are set to ``None`` or both are 0.
    """

    def __init__(
        self,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        dim: int = 1,
        return_indices: bool = False,
    ):
        super(ExtremeLayer, self).__init__()

        if not (n_top is not None or n_bottom is not None):
            raise ValueError("one of n_top or n_bottom must have a value.")

        if not (
            (n_top is not None and n_top > 0)
            or (n_bottom is not None and n_bottom > 0)
        ):
            raise ValueError("one of n_top or n_bottom must have a value > 0.")

        self.n_top = n_top
        self.n_bottom = n_bottom
        self.dim = dim
        self.return_indices = return_indices

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass.
        Parameters
        ----------
        x: torch.Tensor
            Input tensor, shape (B, N_TILES, IN_FEATURES).
        mask: Optional[torch.BoolTensor]
            True for values that were padded, shape (B, N_TILES, 1).
        Warnings
        --------
        If top tiles or bottom tiles is superior to the true number of tiles in
        the input then padded tiles will be selected and their value will be 0.
        Returns
        -------
        values: torch.Tensor
            Extreme tiles, shape (B, N_TOP + N_BOTTOM).
        indices: torch.Tensor
            If ``self.return_indices=True``, return extreme tiles' indices.
        """

        if (
            self.n_top
            and self.n_bottom
            and ((self.n_top + self.n_bottom) > x.shape[self.dim])
        ):
            warnings.warn(
                f"Sum of tops is larger than the input tensor shape for dimension {self.dim}: "
                f"{self.n_top + self.n_bottom} > {x.shape[self.dim]}. "
                f"Values will appear twice (in top and in bottom)"
            )

        top, bottom = None, None
        top_idx, bottom_idx = None, None
        if mask is not None:
            if self.n_top:
                top, top_idx = x.masked_fill(mask, float("-inf")).topk(
                    k=self.n_top, sorted=True, dim=self.dim
                )
                top_mask = top.eq(float("-inf"))
                if top_mask.any():
                    warnings.warn(
                        "The top tiles contain masked values, they will be set to zero."
                    )
                    top[top_mask] = 0

            if self.n_bottom:
                bottom, bottom_idx = x.masked_fill(mask, float("inf")).topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )
                bottom_mask = bottom.eq(float("inf"))
                if bottom_mask.any():
                    warnings.warn(
                        "The bottom tiles contain masked values, they will be set to zero."
                    )
                    bottom[bottom_mask] = 0
        else:
            if self.n_top:
                top, top_idx = x.topk(k=self.n_top, sorted=True, dim=self.dim)
            if self.n_bottom:
                bottom, bottom_idx = x.topk(
                    k=self.n_bottom, largest=False, sorted=True, dim=self.dim
                )

        if top is not None and bottom is not None:
            values = torch.cat([top, bottom], dim=self.dim)
            indices = torch.cat([top_idx, bottom_idx], dim=self.dim)
        elif top is not None:
            values = top
            indices = top_idx
        elif bottom is not None:
            values = bottom
            indices = bottom_idx
        else:
            raise ValueError

        if self.return_indices:
            return values, indices
        else:
            return values

    def extra_repr(self) -> str:
        """Format representation."""
        return f"n_top={self.n_top}, n_bottom={self.n_bottom}"
        

class Chowder(nn.Module):
    """Chowder MIL model (See [1]_).
    Example:
        >>> module = Chowder(in_features=128, out_features=1, n_top=5, n_bottom=5)
        >>> logits, extreme_scores = module(slide, mask=mask)
        >>> scores = module.score_model(slide, mask=mask)
    Parameters
    ----------
    in_features: int
        Features (model input) dimension.
    out_features: int
        Controls the number of scores and, by extension, the number of out_features.
    n_top: int
        Number of tiles with hightest scores that are selected and fed to the MLP.
    n_bottom: int
        Number of tiles with lowest scores that are selected and fed to the MLP.
    tiles_mlp_hidden: Optional[List[int]] = None
        Number of units for layers in the first MLP applied tile wise to compute
        a score for each tiles from the tile features.
        If `None`, a linear layer is used to compute tile scores.
        If e.g. `[128, 64]`, the tile scores are computed with a MLP of dimension
        features_dim -> 128 -> 64 -> 1.
    mlp_hidden: Optional[List[int]] = None
        Number of units for layers of the second MLP that combine top and bottom
        scores and outputs a final prediction at the slide-level. If `None`, a
        linear layer is used to compute the prediction from the extreme scores.
        If e.g. `[128, 64]`, the prediction is computed
        with a MLP n_top + n_bottom -> 128 -> 64 -> 1.
    mlp_dropout: Optional[List[float]] = None
        Dropout that is used for each layer of the MLP. If `None`, no dropout
        is used.
    mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid
        Activation that is used after each layer of the MLP.
    bias: bool = True
        Whether to add bias for layers of the tiles MLP.
    References
    ----------
    .. [1] Pierre Courtiol, Eric W. Tramel, Marc Sanselme, and Gilles Wainrib. Classification
    and disease localization in histopathology using only global labels: A weakly-supervised
    approach. CoRR, abs/1802.02212, 2018.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_top: Optional[int] = None,
        n_bottom: Optional[int] = None,
        tiles_mlp_hidden: Optional[List[int]] = None,
        mlp_hidden: Optional[List[int]] = None,
        mlp_dropout: Optional[List[float]] = None,
        mlp_activation: Optional[torch.nn.Module] = torch.nn.Sigmoid(),
        bias: bool = True,
    ) -> None:
        super(Chowder, self).__init__()
        if n_top is None and n_bottom is None:
            raise ValueError(
                "At least one of `n_top` or `n_bottom` must not be None."
            )

        if mlp_dropout is not None:
            if mlp_hidden is not None:
                assert len(mlp_hidden) == len(
                    mlp_dropout
                ), "mlp_hidden and mlp_dropout must have the same length"
            else:
                raise ValueError(
                    "mlp_hidden must have a value and have the same length as mlp_dropout if mlp_dropout is given."
                )

        self.score_model = TilesMLP(
            in_features,
            hidden=tiles_mlp_hidden,
            bias=bias,
            out_features=out_features,
        )
        self.score_model.apply(self.weight_initialization)

        self.extreme_layer = ExtremeLayer(n_top=n_top, n_bottom=n_bottom)

        mlp_in_features = n_top + n_bottom
        self.mlp = MLP(
            mlp_in_features,
            1,
            hidden=mlp_hidden,
            dropout=mlp_dropout,
            activation=mlp_activation,
        )
        self.mlp.apply(self.weight_initialization)

    @staticmethod
    def weight_initialization(module: torch.nn.Module) -> None:
        """Initialize weights for the module using Xavier initialization method,
        "Understanding the difficulty of training deep feedforward neural networks",
        Glorot, X. & Bengio, Y. (2010)."""
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)

            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(
        self, features: torch.Tensor, mask: Optional[torch.BoolTensor] = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        features: torch.Tensor
            (B, N_TILES, IN_FEATURES)
        mask: Optional[torch.BoolTensor] = None
            (B, N_TILES, 1), True for values that were padded.
        Returns
        -------
        logits, extreme_scores: Tuple[torch.Tensor, torch.Tensor]:
            (B, OUT_FEATURES), (B, N_TOP + N_BOTTOM, OUT_FEATURES)
        """
        scores = self.score_model(x=features[..., :], mask=mask)
        extreme_scores = self.extreme_layer(
            x=scores, mask=mask
        )  # (B, N_TOP + N_BOTTOM, OUT_FEATURES)

        # Apply MLP to the N_TOP + N_BOTTOM scores.
        y = self.mlp(extreme_scores.transpose(1, 2))  # (B, OUT_FEATURES, 1)

        return y.squeeze(2)