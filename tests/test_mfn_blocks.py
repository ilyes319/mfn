import torch
from torch.autograd import gradcheck
import pytest
import numpy as np
import torch
import torch.nn.functional
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R

from mace_mfn import data, modules, tools
from mace_mfn.tools import torch_geometric


@pytest.fixture(scope="module")
def input_tensors():
    R_dense = torch.randn((40,40), dtype=torch.float64, requires_grad=True) + 1j*torch.randn((40,40), dtype=torch.float64, requires_grad=True)
    R_dense = R_dense + R_dense.transpose(-2, -1).conj()  # Ensure the matrix is Hermitian
    return R_dense.to("cuda")

def compute_resolvent(R_dense):
    LUP = torch.linalg.lu_factor_ex(R_dense)
    LU, P = LUP.LU, LUP.pivots
    identity = torch.eye(R_dense.shape[-1], dtype=R_dense.dtype, device=R_dense.device)
    features_full = torch.linalg.lu_solve(LU, P, identity)
    return torch.trace(features_full)

def test_resolvent_gradcheck(input_tensors):
    R_dense = input_tensors
    # Ensure the input tensor requires grad and is of double precision
    R_dense = R_dense.to(dtype=torch.float64)
    R_dense.requires_grad_(True)
    # Call gradcheck
    assert gradcheck(compute_resolvent, (R_dense,), eps=1e-6, atol=1e-4), "Gradcheck failed"

@pytest.fixture(scope="module")
def input_matrix_block():
    config = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, -2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]
        ),
        energy=-1.5,
        charges=np.array([-2.0, 1.0, 1.0]),
        dipole=np.array([-1.5, 1.5, 2.0]),
    )
    # Created the rotated environment
    rot = R.from_euler("z", 60, degrees=True).as_matrix()
    positions_rotated = np.array(rot @ config.positions.T).T
    config_rotated = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=positions_rotated,
        forces=np.array(
            [
                [0.0, -1.3, 0.0],
                [1.0, 0.2, 0.0],
                [0.0, 1.1, 0.3],
            ]
        ),
        energy=-1.5,
        charges=np.array([-2.0, 1.0, 1.0]),
        dipole=np.array([-1.5, 1.5, 2.0]),
    )
    table = tools.AtomicNumberTable([1, 8])
    atomic_energies = np.array([1.0, 3.0], dtype=float)
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=3.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to("cuda")
    num_edge = batch["edge_index"].shape[1]
    num_atoms = batch["node_attrs"].shape[0]
    node_feats = torch.randn(num_atoms, 32*4, requires_grad=True).to("cuda")
    edge_feats = torch.randn(num_edge, 8).to("cuda")
    edge_attrs = torch.randn(num_edge, 16).to("cuda")
    return node_feats, edge_feats, edge_attrs, batch.to_dict(), table, atomic_energies

@pytest.fixture(scope="module")
def create_model(input_matrix_block):
    _, _, _, _, table, atomic_energies = input_matrix_block
    model_config = dict(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps("32x0e + 32x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        num_poles=4,
        num_features_matrix=8,
        num_keep=8,
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
    )
    model = modules.MFN_MACE(**model_config)
    return model

def test_diag(input_matrix_block, create_model):
    node_feats, edge_feats, edge_attrs, batch, table, atomic_energies = input_matrix_block
    model = create_model
    model.to("cuda")
    off_diag_function = model.bond_interactions[0].diagonal_matrix_construction
    def diag_construction(node_feats, batch):
        return off_diag_function(node_feats, batch)[0]
    assert gradcheck(diag_construction, (node_feats, batch["batch"]), eps=1e-3, atol=1e-3), "Gradcheck failed"

def test_off_diag(input_matrix_block, create_model):
    node_feats, edge_feats, edge_attrs, batch, table, atomic_energies = input_matrix_block
    node_feats = node_feats.detach().clone().requires_grad_()
    model = create_model
    model.to("cuda")
    off_diag_function = model.bond_interactions[0].off_matrix_construction
    print(off_diag_function(node_feats, edge_feats, edge_attrs, batch["edge_index"], batch["batch"]))
    def off_diag_construction(node_feats, edge_feats, edge_attrs, edge_index, batch):
        return off_diag_function(node_feats, edge_feats, edge_attrs, edge_index, batch)
    assert gradcheck(off_diag_construction, (node_feats, edge_feats, edge_attrs, batch["edge_index"], batch["batch"]), eps=1e-3, atol=1e-3), "Gradcheck failed"


def test_matrix_block(input_matrix_block):
    node_feats, edge_feats, edge_attrs, batch, table, atomic_energies = input_matrix_block
    model = create_model
    model.to("cuda")
    def forward_bond(node_feats, edge_feats, edge_attrs, edge_index, ptr):
        print(model.bond_interactions[0](
            node_feats=node_feats,
            edge_feats=edge_feats,
            edge_attrs=edge_attrs,
            edge_index=edge_index,
            batch=batch["batch"],
            ptr=ptr,
            matrix_feats=None,
        )[0].shape)
        return model.bond_interactions[0](
            node_feats=node_feats,
            edge_feats=edge_feats,
            edge_attrs=edge_attrs,
            edge_index=edge_index,
            batch=batch["batch"],
            ptr=ptr,
            matrix_feats=None,
        )[0]
    assert gradcheck(forward_bond, (node_feats, edge_feats, edge_attrs, batch["edge_index"], batch["ptr"]), eps=1e-6, atol=1e-4), "Gradcheck failed"

