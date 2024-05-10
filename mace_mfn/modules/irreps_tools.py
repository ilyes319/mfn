###########################################################################################
# Elementary tools for handling irreducible representations
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import List, Tuple
from math import sqrt

import torch
from e3nn import o3
from mace.tools import cg
from e3nn.util.jit import compile_mode


# Based on mir-group/nequip
def tp_out_irreps_with_instructions(
    irreps1: o3.Irreps, irreps2: o3.Irreps, target_irreps: o3.Irreps
) -> Tuple[o3.Irreps, List]:
    trainable = True

    # Collect possible irreps and their instructions
    irreps_out_list: List[Tuple[int, o3.Irreps]] = []
    instructions = []
    for i, (mul, ir_in) in enumerate(irreps1):
        for j, (_, ir_edge) in enumerate(irreps2):
            for ir_out in ir_in * ir_edge:  # | l1 - l2 | <= l <= l1 + l2
                if ir_out in target_irreps:
                    k = len(irreps_out_list)  # instruction index
                    irreps_out_list.append((mul, ir_out))
                    instructions.append((i, j, k, "uvu", trainable))

    # We sort the output irreps of the tensor product so that we can simplify them
    # when they are provided to the second o3.Linear
    irreps_out = o3.Irreps(irreps_out_list)
    irreps_out, permut, _ = irreps_out.sort()

    # Permute the output indexes of the instructions to match the sorted irreps:
    instructions = [
        (i_in1, i_in2, permut[i_out], mode, train)
        for i_in1, i_in2, i_out, mode, train in instructions
    ]

    instructions = sorted(instructions, key=lambda x: x[2])

    return irreps_out, instructions


def linear_out_irreps(irreps: o3.Irreps, target_irreps: o3.Irreps) -> o3.Irreps:
    # Assuming simplified irreps
    irreps_mid = []
    for _, ir_in in irreps:
        found = False

        for mul, ir_out in target_irreps:
            if ir_in == ir_out:
                irreps_mid.append((mul, ir_out))
                found = True
                break

        if not found:
            raise RuntimeError(f"{ir_in} not in {target_irreps}")

    return o3.Irreps(irreps_mid)


@compile_mode("script")
class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps, num_keep: int = 1) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(num_keep * d)
            self.muls.append(mul // num_keep)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)

class unreshape_irreps(torch.nn.Module):
    # This is the inverse of reshape_irreps
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, :, ix : ix + d]
            ix += d
            field = field.reshape(batch, -1)
            out.append(field)
        return torch.cat(out, dim=-1)


@compile_mode("script")
class extract_blocks(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps, num_channels: int) -> None:
        super().__init__()
        self.max_L = irreps.lmax
        self.num_channels = num_channels
        self.linear_real = torch.nn.Linear(2 * self.num_channels, self.num_channels, bias=False)
        self.linear_imag = torch.nn.Linear(2 * self.num_channels, self.num_channels, bias=False)
    def forward(self, H: torch.Tensor) -> torch.Tensor:
        n = H.shape[-1] // (self.max_L + 1)**2
        out = []
        for i in range(n):
            extract = H[:, :, i*(self.max_L + 1)**2:(i+1)*(self.max_L + 1)**2, i*(self.max_L + 1)**2]
            block = H[:, :, i*(self.max_L + 1)**2 + 1:(i+1)*(self.max_L + 1)**2, i*(self.max_L + 1)**2 + 1:(i+1)*(self.max_L + 1)**2]
            trace = block.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
            invariants_extract = torch.cat([extract[:,:,0], trace], dim=-1)
            invariants_extract = self.linear_real(invariants_extract.real) + 1j * self.linear_imag(invariants_extract.imag)
            extract_out = torch.cat([invariants_extract.unsqueeze(-1), extract[:,:,1:]], dim=-1)
            out.append(extract_out)
        tensor_out = torch.stack(out, dim=2)
        return tensor_out.permute(0,2,1,3) # [batch, n_channels, n_atoms, (max_L + 1)**2]

@compile_mode("script")
class fill_blocks(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.max_L = irreps.lmax
    def forward(self, H: torch.Tensor, diag: torch.Tensor) -> torch.Tensor:
        n = H.shape[-1] // (self.max_L + 1)**2
        for i in range(n):
            H[:, :, i*(self.max_L + 1)**2:(i+1)*(self.max_L + 1)**2, i*(self.max_L + 1)**2:(i+1)*(self.max_L + 1)**2] = diag[:, i, :, :,:]
        return H
    
class construct_blocks_diag(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        u_matrix = cg.U_matrix_real(
            irreps_in=o3.Irreps("1x1o"), irreps_out=o3.Irreps("1x2e"), correlation=2
        )[-1].squeeze(-1)
        self.register_buffer("u_matrix", u_matrix)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], x.shape[1], 4, 4).to(x.device)
        x_pp_ = torch.einsum("bck, kie -> bcei",x[:,:,-5:], self.u_matrix)
        x_trace = (1/sqrt(3)) * torch.einsum("bci,ier->bcer" , x[:,:,1].unsqueeze(-1), torch.eye(3 , device=x.device).unsqueeze(0))
        x_pp = x_pp_ + x_trace
        out[:,:,1:,1:] = x_pp # pp
        out[:,:,0,0] = x[:,:,0] # ss
        out[:,:,0,1:] = x[:,:,4:7] # sp
        out[:,:,1:,0] = x[:,:,4:7] # ps
        return out
    
class construct_blocks_off(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        u_matrix = cg.U_matrix_real(
            irreps_in=o3.Irreps("1x1e"), irreps_out=o3.Irreps("1x2e"), correlation=2
        )[-1].squeeze(-1)
        self.register_buffer("u_matrix", u_matrix)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros(x.shape[0], x.shape[1], 4, 4).to(x.device)
        x_pp_ = torch.einsum("bck, kie -> bcei",x[:,:,-5:], self.u_matrix)
        x_trace = (1/sqrt(3)) * torch.einsum("bci,ier->bcer" , x[:,:,1].unsqueeze(-1), torch.eye(3 , device=x.device).unsqueeze(0))
        x_pp = x_pp_ + x_trace
        out[:,:,1:,1:] = x_pp # pp
        out[:,:,0,0] = x[:,:,0] # ss
        out[:,:,0,1:] = x[:,:,2:5] # sp
        out[:,:,1:,0] = x[:,:,5:8] # ps
        return out