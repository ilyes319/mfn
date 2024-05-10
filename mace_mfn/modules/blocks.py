###########################################################################################
# Elementary Block for Building O(3) Equivariant Higher Order Message Passing Neural Network
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from abc import abstractmethod
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional
from e3nn import nn, o3
from e3nn.util.jit import compile_mode
import wandb

from mace_mfn.tools.torch_geometric.utils import to_dense_adj, to_dense_batch
from mace_mfn.tools.torch_tools import get_mask
from mace_mfn.tools.scatter import scatter_sum

from .irreps_tools import (
    construct_blocks_diag,
    construct_blocks_off,
    extract_blocks,
    fill_blocks,
    unreshape_irreps,
    linear_out_irreps,
    reshape_irreps,
    tp_out_irreps_with_instructions,
)
from .radial import BesselBasis, PolynomialCutoff
from .symmetric_contraction import SymmetricContraction
import logging

@compile_mode("script")
class LinearNodeEmbeddingBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:  # [n_nodes, irreps]
        return self.linear(node_attrs)


@compile_mode("script")
class LinearReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        super().__init__()
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=o3.Irreps("0e"))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearReadoutBlock(torch.nn.Module):
    def __init__(
        self, irreps_in: o3.Irreps, MLP_irreps: o3.Irreps, gate: Optional[Callable]
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.hidden_irreps)
        self.non_linearity = nn.Activation(irreps_in=self.hidden_irreps, acts=[gate])
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=o3.Irreps("0e")
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.non_linearity(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class LinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, dipole_only: bool = False):
        super().__init__()
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        self.linear = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        return self.linear(x)  # [n_nodes, 1]


@compile_mode("script")
class NonLinearDipoleReadoutBlock(torch.nn.Module):
    def __init__(
        self,
        irreps_in: o3.Irreps,
        MLP_irreps: o3.Irreps,
        gate: Callable,
        dipole_only: bool = False,
    ):
        super().__init__()
        self.hidden_irreps = MLP_irreps
        if dipole_only:
            self.irreps_out = o3.Irreps("1x1o")
        else:
            self.irreps_out = o3.Irreps("1x0e + 1x1o")
        irreps_scalars = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l == 0 and ir in self.irreps_out]
        )
        irreps_gated = o3.Irreps(
            [(mul, ir) for mul, ir in MLP_irreps if ir.l > 0 and ir in self.irreps_out]
        )
        irreps_gates = o3.Irreps([mul, "0e"] for mul, _ in irreps_gated)
        self.equivariant_nonlin = nn.Gate(
            irreps_scalars=irreps_scalars,
            act_scalars=[gate for _, ir in irreps_scalars],
            irreps_gates=irreps_gates,
            act_gates=[gate] * len(irreps_gates),
            irreps_gated=irreps_gated,
        )
        self.irreps_nonlin = self.equivariant_nonlin.irreps_in.simplify()
        self.linear_1 = o3.Linear(irreps_in=irreps_in, irreps_out=self.irreps_nonlin)
        self.linear_2 = o3.Linear(
            irreps_in=self.hidden_irreps, irreps_out=self.irreps_out
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [n_nodes, irreps]  # [..., ]
        x = self.equivariant_nonlin(self.linear_1(x))
        return self.linear_2(x)  # [n_nodes, 1]


@compile_mode("script")
class AtomicEnergiesBlock(torch.nn.Module):
    atomic_energies: torch.Tensor

    def __init__(self, atomic_energies: Union[np.ndarray, torch.Tensor]):
        super().__init__()
        assert len(atomic_energies.shape) == 1

        self.register_buffer(
            "atomic_energies",
            torch.tensor(atomic_energies, dtype=torch.get_default_dtype()),
        )  # [n_elements, ]

    def forward(
        self, x: torch.Tensor  # one-hot of elements [..., n_elements]
    ) -> torch.Tensor:  # [..., ]
        return torch.matmul(x, self.atomic_energies)

    def __repr__(self):
        formatted_energies = ", ".join([f"{x:.4f}" for x in self.atomic_energies])
        return f"{self.__class__.__name__}(energies=[{formatted_energies}])"


@compile_mode("script")
class RadialEmbeddingBlock(torch.nn.Module):
    def __init__(self, r_max: float, num_bessel: int, num_polynomial_cutoff: int):
        super().__init__()
        self.bessel_fn = BesselBasis(r_max=r_max, num_basis=num_bessel)
        self.cutoff_fn = PolynomialCutoff(r_max=r_max, p=num_polynomial_cutoff)
        self.out_dim = num_bessel

    def forward(
        self,
        edge_lengths: torch.Tensor,  # [n_edges, 1]
    ):
        bessel = self.bessel_fn(edge_lengths)  # [n_edges, n_basis]
        cutoff = self.cutoff_fn(edge_lengths)  # [n_edges, 1]
        return bessel * cutoff  # [n_edges, n_basis]


@compile_mode("script")
class EquivariantProductBasisBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        correlation: int,
        use_sc: bool = True,
        num_elements: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.use_sc = use_sc
        self.symmetric_contractions = SymmetricContraction(
            irreps_in=node_feats_irreps,
            irreps_out=target_irreps,
            correlation=correlation,
            num_elements=num_elements,
        )
        # Update linear
        self.linear = o3.Linear(
            target_irreps,
            target_irreps,
            internal_weights=True,
            shared_weights=True,
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        sc: Optional[torch.Tensor],
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        node_feats = self.symmetric_contractions(node_feats, node_attrs)
        if self.use_sc and sc is not None:
            return self.linear(node_feats) + sc

        return self.linear(node_feats)


@compile_mode("script")
class InteractionBlock(torch.nn.Module):
    def __init__(
        self,
        node_attrs_irreps: o3.Irreps,
        node_feats_irreps: o3.Irreps,
        edge_attrs_irreps: o3.Irreps,
        edge_feats_irreps: o3.Irreps,
        target_irreps: o3.Irreps,
        hidden_irreps: o3.Irreps,
        avg_num_neighbors: float,
        radial_MLP: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.node_attrs_irreps = node_attrs_irreps
        self.node_feats_irreps = node_feats_irreps
        self.edge_attrs_irreps = edge_attrs_irreps
        self.edge_feats_irreps = edge_feats_irreps
        self.target_irreps = target_irreps
        self.hidden_irreps = hidden_irreps
        self.avg_num_neighbors = avg_num_neighbors
        if radial_MLP is None:
            radial_MLP = [64, 64, 64]
        self.radial_MLP = radial_MLP

        self._setup()

    @abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


nonlinearities = {1: torch.nn.functional.silu, -1: torch.tanh}


@compile_mode("script")
class TensorProductWeightsBlock(torch.nn.Module):
    def __init__(self, num_elements: int, num_edge_feats: int, num_feats_out: int):
        super().__init__()

        weights = torch.empty(
            (num_elements, num_edge_feats, num_feats_out),
            dtype=torch.get_default_dtype(),
        )
        torch.nn.init.xavier_uniform_(weights)
        self.weights = torch.nn.Parameter(weights)

    def forward(
        self,
        sender_or_receiver_node_attrs: torch.Tensor,  # assumes that the node attributes are one-hot encoded
        edge_feats: torch.Tensor,
    ):
        return torch.einsum(
            "be, ba, aek -> bk", edge_feats, sender_or_receiver_node_attrs, self.weights
        )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(shape=({", ".join(str(s) for s in self.weights.shape)}), '
            f"weights={np.prod(self.weights.shape)})"
        )


@compile_mode("script")
class ResidualElementDependentInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )
        self.conv_tp_weights = TensorProductWeightsBlock(
            num_elements=self.node_attrs_irreps.num_irreps,
            num_edge_feats=self.edge_feats_irreps.num_irreps,
            num_feats_out=self.conv_tp.weight_numel,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(node_attrs[sender], edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return message + sc  # [n_nodes, irreps]


@compile_mode("script")
class AgnosticNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        tp_weights = self.conv_tp_weights(edge_feats)
        node_feats = self.linear_up(node_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return message  # [n_nodes, irreps]


@compile_mode("script")
class AgnosticResidualNonlinearInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps, self.edge_attrs_irreps, self.target_irreps
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = linear_out_irreps(irreps_mid, self.target_irreps)
        self.irreps_out = self.irreps_out.simplify()
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.irreps_out
        )

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = message + sc
        return message  # [n_nodes, irreps]


@compile_mode("script")
class RealAgnosticInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.irreps_out, self.node_attrs_irreps, self.irreps_out
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        message = self.skip_tp(message, node_attrs)
        return (
            self.reshape(message),
            None,
        )  # [n_nodes, channels, (lmax + 1)**2]


@compile_mode("script")
class RealAgnosticResidualInteractionBlock(InteractionBlock):
    def _setup(self) -> None:
        # First linear
        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        # TensorProduct
        irreps_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            self.target_irreps,
        )
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.edge_attrs_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=False,
            internal_weights=False,
        )

        # Convolution weights
        input_dim = self.edge_feats_irreps.num_irreps
        self.conv_tp_weights = nn.FullyConnectedNet(
            [input_dim] + self.radial_MLP + [self.conv_tp.weight_numel],
            torch.nn.functional.silu,
        )

        # Linear
        irreps_mid = irreps_mid.simplify()
        self.irreps_out = self.target_irreps
        self.linear = o3.Linear(
            irreps_mid, self.irreps_out, internal_weights=True, shared_weights=True
        )

        # Selector TensorProduct
        self.skip_tp = o3.FullyConnectedTensorProduct(
            self.node_feats_irreps, self.node_attrs_irreps, self.hidden_irreps
        )
        self.reshape = reshape_irreps(self.irreps_out)

    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = node_feats.shape[0]
        sc = self.skip_tp(node_feats, node_attrs)
        node_feats = self.linear_up(node_feats)
        tp_weights = self.conv_tp_weights(edge_feats)
        mji = self.conv_tp(
            node_feats[sender], edge_attrs, tp_weights
        )  # [n_edges, irreps]
        message = scatter_sum(
            src=mji, index=receiver, dim=0, dim_size=num_nodes
        )  # [n_nodes, irreps]
        message = self.linear(message) / self.avg_num_neighbors
        return (
            self.reshape(message),
            sc,
        )  # [n_nodes, channels, (lmax + 1)**2]


class OffDiagMatrixConstructionBlock(torch.nn.Module):
    def __init__(
            self,
            node_feats_irreps,
            num_features,
            num_basis,
            num_poles,
            sh_irreps,
            avg_num_neighbors,
            num_keep,
    ):
        super().__init__()
        self.node_feats_irreps = node_feats_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.num_poles = num_poles
        self.num_features = num_features

        # Irreps
        irreps_matrix = o3.Irreps(
        (num_features * o3.Irreps.spherical_harmonics(2, p=-1))
        .sort()
        .irreps.simplify()
        )
        num_keep_irreps = o3.Irreps(
            (num_keep * o3.Irreps.spherical_harmonics(2, p=-1))
            .sort()
            .irreps.simplify()
        )

        irreps_mid_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.node_feats_irreps,
            self.node_feats_irreps,
        )

        irreps_mid, instructions_sh = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            sh_irreps,
            irreps_matrix,
        )

        # Tensor Products
        self.conv_tp = o3.TensorProduct(
            self.node_feats_irreps,
            self.node_feats_irreps,
            irreps_mid_mid,
            instructions=instructions,
            shared_weights=True,
            internal_weights=True,
        )

        self.linear_mid = o3.Linear(
            irreps_mid_mid,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        self.conv_tp_sh = o3.TensorProduct(
            self.node_feats_irreps,
            sh_irreps,
            irreps_mid,
            instructions=instructions_sh,
            shared_weights=False,
            internal_weights=False,
        )

        # Edge features
        self.edge_feats_mlp_off = nn.FullyConnectedNet(
            [num_basis] + 3 * [64] + [self.conv_tp_sh.weight_numel],
            torch.nn.functional.silu,
        )

        # Linears
        irreps_mid = irreps_mid.simplify()
        num_keep_matrix_irreps =  (num_keep * irreps_matrix).sort().irreps.simplify()
        self.linear_tp_off_out = o3.Linear(irreps_mid, num_keep_matrix_irreps)
        self.linear_tp_off = o3.Linear(num_keep_irreps, o3.Irreps("2x0e + 2x1o + 1x2e"))
        self.reshape = reshape_irreps(num_keep_matrix_irreps, num_keep=num_keep)


        # Matrix Construction
        self.construct_block_off = construct_blocks_off()
        
    def forward(self, node_feats, edge_feats, edge_attrs, edge_index, batch):
        # Node features
        mask_edge_index = (edge_index[0, :] < edge_index[1, :])
        filtered_edges = edge_index[:,mask_edge_index]
        sender, receiver = filtered_edges[0], filtered_edges[1]
        edge_feats_weights_off = self.edge_feats_mlp_off(edge_feats[mask_edge_index, :])
        symmetric_features = self.conv_tp(
            node_feats[sender], node_feats[receiver]
        )
        symmetric_features = self.linear_mid(symmetric_features)    
        symmetric_features = self.conv_tp_sh(symmetric_features, edge_attrs[mask_edge_index, :], edge_feats_weights_off)
        symmetric_features = self.reshape(self.linear_tp_off_out(symmetric_features))
        H = self.construct_block_off(self.linear_tp_off(symmetric_features))
        H_dense = to_dense_adj(edge_index=filtered_edges, batch=batch, edge_attr=H)
        H_dense = H_dense.permute(0, 3, 1, 4, 2, 5).reshape(
            H_dense.shape[0], H_dense.shape[3], H_dense.shape[1] * H_dense.shape[4], H_dense.shape[2] * H_dense.shape[5]
        ) #[n_edges, n_features,  n_nodes * (lmax + 1)**2, n_nodes * (lmax + 1)**2]
        H_dense = H_dense + H_dense.transpose(-1, -2)
        H_dense = H_dense.repeat(1, self.num_poles, 1, 1)
        return H_dense
    
class DiagMatrixConstructionBlock(torch.nn.Module):
    def __init__(
            self,
            node_feats_irreps,
            num_features,
            num_poles,
            sh_irreps,
            num_keep,
    ):
        super().__init__()
        self.node_feats_irreps = node_feats_irreps
        self.num_poles = num_poles

        self.linear_diag = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )

        # Irreps
        num_keep_irreps = o3.Irreps(
            (num_keep * o3.Irreps.spherical_harmonics(2, p=-1))
            .sort()
            .irreps.simplify()
        )
        irreps_matrix = o3.Irreps(
            (num_features * o3.Irreps.spherical_harmonics(2, p=-1))
            .sort()
            .irreps.simplify()
            )

        #Instructions
        irreps_mid_mid, instructions = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            self.node_feats_irreps,
            self.node_feats_irreps,
        )
        irreps_mid, _ = tp_out_irreps_with_instructions(
            self.node_feats_irreps,
            sh_irreps,
            irreps_matrix,
        )

        # Tensor Product
        self.conv_tp_diag = o3.TensorProduct(
            self.node_feats_irreps,
            self.node_feats_irreps,
            irreps_mid,
            instructions=instructions,
            shared_weights=True,
            internal_weights=True,
        )

        # Linears
        irreps_mid = irreps_mid.simplify()
        num_keep_matrix_irreps =  (num_keep * irreps_matrix).sort().irreps.simplify()
        self.linear_tp_diag_out = o3.Linear(irreps_mid, num_keep_matrix_irreps)
        self.linear_tp_diag = o3.Linear(num_keep_irreps, o3.Irreps("2x0e + 1x1o + 1x2e"))

        self.reshape = reshape_irreps(num_keep_matrix_irreps, num_keep=num_keep)
        # Matrix Construction
        self.construct_block_diag = construct_blocks_diag()

    def forward(self, node_feats, batch):
        node_feats_diag = self.linear_diag(node_feats)
        diag_H = self.conv_tp_diag(node_feats_diag, node_feats_diag)
        diag_H = self.reshape(self.linear_tp_diag_out(diag_H))
        diag_H = self.construct_block_diag(self.linear_tp_diag(diag_H)).repeat(1, self.num_poles, 1, 1)
        diag_H = to_dense_batch(diag_H, batch)[0]
        return diag_H
            

class MatrixFunctionBlock(torch.nn.Module):
    def __init__(
        self,
        node_feats_irreps,
        num_features,
        num_basis,
        num_poles,
        sh_irreps,
        avg_num_neighbors,
        num_keep,
        diagonal="learnable",
        use_equiariant=True,
        learnable_resolvent=False
        ):
        super().__init__()
        # First linear
        self.diagonal = diagonal
        self.node_feats_irreps = node_feats_irreps
        self.avg_num_neighbors = avg_num_neighbors
        self.num_poles = num_poles
        self.learnable_resolvent = learnable_resolvent
        self.use_equiariant = use_equiariant
        self.num_features = num_features
        num_keep = num_keep

        irreps_out = o3.Irreps(
        (num_features * o3.Irreps.spherical_harmonics(self.node_feats_irreps.lmax, p=-1))
        .sort()
        .irreps.simplify()
        )

        self.linear_up = o3.Linear(
            self.node_feats_irreps,
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
        )
        self.off_matrix_construction = OffDiagMatrixConstructionBlock(
            self.node_feats_irreps,
            num_features,
            num_basis,
            num_poles,
            sh_irreps,
            avg_num_neighbors,
            num_keep,
        )
        self.diagonal_matrix_construction = DiagMatrixConstructionBlock(
            self.node_feats_irreps,
            num_features,
            num_poles,
            sh_irreps,
            num_keep,
        )

        self.extract_block = extract_blocks(irreps_out, num_features * num_poles)
        self.unreshape_irreps = unreshape_irreps(irreps_out)
            
        self.fill = fill_blocks(irreps_out)
        norm = 'layer'
        if norm == 'batch':
            self.matrix_norm = EigenvalueBatchNorm(num_features * num_poles)
        elif norm == 'layer':
            self.matrix_norm = EigenvalueLayerNorm(num_features * num_poles)
        logging.info('Using {} norm'.format(norm))

        linear_irreps_in = o3.Irreps(
            (2 * num_poles * irreps_out)
            .sort()
            .irreps.simplify()
        )

        # Linear out
        self.linear_out = o3.Linear(
            linear_irreps_in,  # 2* for real and imaginary
            self.node_feats_irreps,
            internal_weights=True,
            shared_weights=True,
            biases=False,
        )

    def matrix_function(self, H_dense, diag_H_shape, node_feats, ptr):
        D_z = torch.diag_embed(
            torch.view_as_complex(
                torch.stack([torch.zeros_like(H_dense[...,0], dtype=node_feats.dtype), torch.ones_like(H_dense[...,0], dtype=node_feats.dtype)], dim=-1)
            )
        )
        mask = get_mask(ptr[1:] - ptr[:-1]).to(node_feats.device)
        mask_matrix = mask.reshape(
            (ptr[1:] - ptr[:-1]).shape[0],
            (ptr[1:] - ptr[:-1]).max())
        mask_matrix = mask_matrix.unsqueeze(-1).repeat(1,1,diag_H_shape).reshape(mask_matrix.shape[0], -1)
        H_dense_normalized = self.matrix_norm(H_dense, mask_matrix)
        R_dense = D_z - H_dense_normalized
        LUP = torch.linalg.lu_factor_ex(R_dense)
        LU, P = LUP.LU, LUP.pivots
        self.identity = (
            torch.eye(R_dense.shape[-1], dtype=D_z.dtype, device=D_z.device)
            .unsqueeze(0)
            .repeat(R_dense.shape[0], R_dense.shape[1], 1, 1)
        )
        features_full = torch.linalg.lu_solve(LU, P,  self.identity)
        features_full = torch.einsum("bijk,bikl->bijl", H_dense, features_full.real) + 1j * torch.einsum("bijk,bikl->bijl", H_dense, features_full.imag)
        features = self.extract_block(features_full)
        features = features.reshape(features.shape[0]*features.shape[1], features.shape[2], features.shape[3])
        node_features_real = features.real[mask, :, :] / self.avg_num_neighbors
        node_features_imag = features.imag[mask, :, :] / self.avg_num_neighbors
        node_features = torch.cat([node_features_real, node_features_imag], dim=1)
        node_features = self.unreshape_irreps(node_features)
        out = self.linear_out(node_features) # [n_nodes, irreps]
        return out, features_full.real
        
    def forward(self, node_feats, edge_feats, edge_attrs, matrix_feats, edge_index, batch, ptr):
        node_feats = self.linear_up(node_feats)
        H_dense_off = self.off_matrix_construction(node_feats, edge_feats, edge_attrs, edge_index, batch)
        diag_H = self.diagonal_matrix_construction(node_feats, batch)
        H_dense = self.fill(H_dense_off, diag_H)
        if matrix_feats is not None:
            H_dense = H_dense + matrix_feats
        out, features_full = self.matrix_function(H_dense, diag_H.shape[-1], node_feats, ptr)
        return out, features_full

class EigenvalueBatchNorm(torch.nn.Module):
    def __init__(self,num_features, eps=1e-9, momentum=0.997, starting_momentum = 0.8, warmup_steps = 100, using_moving_average=True, random_weights=True):
        super(EigenvalueBatchNorm, self).__init__()
        self.eps = eps  # Small constant for numerical stability
        self.momentum = momentum  # Momentum for moving average
        self.starting_momentum = starting_momentum
        self.warmup_steps = warmup_steps
        self.using_moving_average = using_moving_average  # Whether to use moving average or not
        self.num_features = num_features  # Number of feature channels
        self.random_weights = random_weights
        self.steps = 0
        if random_weights:
            # TODO: decouple number of channels and number of matrix functions
            weight_exp = torch.randn(1, num_features,1,1)
            weight_bias = torch.zeros(1, num_features,1,1)
            weight = torch.ones(1, num_features,1,1)
            bias   = torch.randn(1, num_features,1)
        else:
            weight = torch.ones(1, num_features, 1, 1)
            bias   = torch.zeros(1, num_features, 1)

        self.weight = torch.nn.Parameter(weight)  # Learnable weight for scaling
        self.weight_exp = torch.nn.Parameter(weight_exp)  # Learnable weight for scaling
        self.weight_bias = torch.nn.Parameter(weight_bias)  # Learnable weight for scaling
        self.bias = torch.nn.Parameter(bias)  # Learnable bias for shifting
        self.register_buffer("running_mean", torch.zeros(num_features))  # Running mean of eigenvalues
        self.register_buffer("running_var", torch.ones(num_features))  # Running variance of eigenvalues
        self.reset_parameters()

    # Reset parameters (running mean and variance)
    def reset_parameters(self):
        self.running_mean.zero_()
        self.running_var.fill_(1)
        self.steps = 0

    # Check the dimensions of the input tensor
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected input 4 dimensions (got {}D input)".format(input.dim()))
        if input.shape[1] != self.num_features:
            raise ValueError(f"expected num_features to be {self.num_features}, got {input.shape[1]}")

    # Compute the mean and variance using the trace method
    def _mean_and_variance_trace(self,matrix,mask):
        n = torch.einsum('bfii->bf', mask)
        n2 = torch.clamp(n-1,1)# Clamp at n - 1 = 1 to avoid division by zero
        matrix_squared = torch.linalg.matrix_power(matrix,2)*mask
        trace = torch.einsum('bfii->bf', matrix*mask)
        trace_square = torch.einsum('bfii->bf', matrix_squared*mask)
        mean = trace / n
        variance = trace_square / n2 - trace ** 2/ (n*n2)
        return mean.mean(0), variance.mean(0)

    # Reformat the mask tensor to match the input tensor shape
    def _reformat_mask(self,mask):
        mask = mask[:,None,:,None] * mask[:,None,None,:]
        mask = mask.expand(-1,self.num_features,-1,-1)
        return mask

    # Forward function for the normalization layer
    def forward(self, x,mask = None):
        # x: [batch_size, num_features, N, N]
        # mask: [batch_size, N]
        self._check_input_dim(x)  # Check input dimensions
        if mask is None:
            mask = torch.ones(x.shape, device=x.device)  # Create a mask with all ones if no mask is provided
        else:
            mask = self._reformat_mask(mask)  # Reformat the mask tensor to match the input tensor shape

        if self.training:
            mean, variance = self._mean_and_variance_trace(x, mask)  # Compute mean and variance using the trace method
            if self.using_moving_average:
                if self.steps < self.warmup_steps: #Momentum Warmup
                    beta = self.steps / self.warmup_steps
                    momentum =  self.momentum * beta + self.starting_momentum * (1 - beta) # Interpolating momentum between starting momentum and final momentum
                else:
                    momentum = self.momentum
                self.running_mean.mul_(momentum)  # Update running mean with momentum
                self.running_mean.add_((1 - momentum) * mean.data)  # Update running mean with new mean
                self.running_var.mul_(momentum)  # Update running variance with momentum
                self.running_var.add_((1 - momentum) * variance.data)  # Update running variance with new variance
            else:
                self.running_mean.add_(mean.data)  # Update running mean without momentum
                self.running_var.add_(variance.data)  # Update running variance without momentum
            self.steps += 1
        m = torch.autograd.Variable(self.running_mean)  # Running mean variable
        v = torch.autograd.Variable(self.running_var)  # Running variance variable

        m_t = torch.diag_embed(m[None, :, None].expand_as(x[..., 0]), offset=0, dim1=-2, dim2=-1)  # Diagonal matrix of the running mean
        x_centered = (x - m_t) * mask  # Subtract the running mean from the input tensor
        x_normalized = x_centered / ((v[None, :, None, None].expand_as(x)).sqrt() + self.eps)  # Normalize the centered tensor
        # (1 + self.weight) change to that
        return x_normalized * (self.weight * torch.exp(self.weight_exp) + self.weight_bias) + torch.diag_embed(self.bias.expand_as(x[..., 0]), offset=0, dim1=-2, dim2=-1)  # Return the normalized tensor scaled by the learned weight and shifted by the learned bias

# Eigenvalue layer normalization
class EigenvalueLayerNorm(torch.nn.Module):
    def __init__(self, num_features, eps=1e-9, random_weights=True):
        super(EigenvalueLayerNorm, self).__init__()
        self.eps = eps  # Small constant for numerical stability
        self.num_features = num_features  # Number of feature channels

        if random_weights:
            weight_exp = torch.randn(1, num_features,1,1)
            weight_bias = torch.zeros(1, num_features,1,1)
            weight = torch.ones(1, num_features,1,1)
            bias   = torch.randn(1, num_features,1)

        else:
            weight = torch.ones(1, num_features, 1, 1)
            bias   = torch.zeros(1, num_features, 1)

        self.weight = torch.nn.Parameter(weight)  # Learnable weight for scaling
        self.weight_exp = torch.nn.Parameter(weight_exp)  # Learnable weight for scaling
        self.weight_bias = torch.nn.Parameter(weight_bias)  # Learnable weight for scaling
        self.bias = torch.nn.Parameter(bias)  # Learnable bias for shifting

    # Check the dimensions of the input tensor
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected input 4 dimensions (got {}D input)".format(input.dim()))
        if input.shape[1] != self.num_features:
            raise ValueError(f"expected num_features to be {self.num_features}, got {input.shape[1]}")

    # Compute the mean and variance using the trace method
    def _mean_and_variance_trace(self, matrix, mask):
        n = torch.einsum('bfii->bf', mask)
        n2 = torch.clamp(n - 1, 1)  # Clamp at n - 1 = 1 to avoid division by zero
        matrix_squared = torch.linalg.matrix_power(matrix, 2) * mask
        trace = torch.einsum('bfii->bf', matrix * mask)
        trace_square = torch.einsum('bfii->bf', matrix_squared * mask)
        mean = trace / n
        variance = trace_square / n2 - trace ** 2 / (n * n2)
        return mean.mean(1), variance.mean(1)

    # Reformat the mask tensor to match the input tensor shape
    def _reformat_mask(self, mask):
        mask = mask[:, None, :, None] * mask[:, None, None, :]
        mask = mask.expand(-1, self.num_features, -1, -1)
        return mask

    # Forward function for the normalization layer
    def forward(self, x, mask=None):
        # x: [batch_size, num_features, N, N]
        # mask: [batch_size, N]
        self._check_input_dim(x)  # Check input dimensions
        if mask is None:
            mask = torch.ones(x.shape)  # Create a mask with all ones if no mask is provided
        else:
            mask = self._reformat_mask(mask)  # Reformat the mask tensor to match the input tensor shape

        mean, variance = self._mean_and_variance_trace(x, mask)  # Compute mean and variance using the trace method

        m = mean.unsqueeze(1).unsqueeze(-1)  # Add dimensions to match x
        v = variance.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # Add dimensions to match x

        m_t = torch.diag_embed(m.expand_as(x[..., 0]), offset=0, dim1=-2, dim2=-1)  # Diagonal matrix of the running mean
        x_centered = (x - m_t) * mask  # Subtract the running mean from the input tensor
        x_normalized = x_centered / (v.expand_as(x) + self.eps).sqrt()  # Normalize the centered tensor
        return x_normalized * (self.weight * torch.exp(self.weight_exp) + self.weight_bias) + torch.diag_embed(self.bias.expand_as(x[..., 0]), offset=0, dim1=-2, dim2=-1)

@compile_mode("script")
class ScaleShiftBlock(torch.nn.Module):
    def __init__(self, scale: float, shift: float):
        super().__init__()
        self.register_buffer(
            "scale", torch.tensor(scale, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "shift", torch.tensor(shift, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scale * x + self.shift

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(scale={self.scale:.6f}, shift={self.shift:.6f})"
        )
