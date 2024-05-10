# Matrix Function Networks

This repository contains the Matrix Function Network (MFN) implementation developed by
Ilyes Batatia, Lars Schaaf, and Felix Faber. Most of this repo reproduces the MACE repo (https://github.com/ACEsuit/mace). Please refer to the original repo for more information.

## Installation

Requirements:
* Python >= 3.7
* [PyTorch](https://pytorch.org/) >= 1.8

### conda installation

If you do not have CUDA pre-installed, it is **recommended** to follow the conda installation process:
```sh
# Create a virtual environment and activate it
conda create mfn_env
conda activate mfn_env

# Install PyTorch
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch-lts -c conda-forge

# Clone and install MACE (and all required packages), use token if still private repo
git clone git@github.com:ilyes319/mfn.git
pip install ./mfn
```

### pip installation

To install via `pip`, follow the steps below:
```sh
# Create a virtual environment and activate it
python -m venv mfn-venv
source mfn-venv/bin/activate

# Install PyTorch (for example, for CUDA 10.2 [cu102])
pip install torch==1.8.2 --extra-index-url "https://download.pytorch.org/whl/lts/1.8/cu102"

# Clone and install MACE (and all required packages)
git clone git@github.com:ilyes319/mfn.git
pip install ./mfn
```

**Note:** The homonymous package on [PyPI](https://pypi.org/project/MACE/) has nothing to do with this one.

## Usage

### Training 

To reproduce the MFN model on the cumulene data, you can use the `run_train.py` script:

```sh
python ./mfn/scripts/run_train.py \
    --train_file="GNL-v0.2/gnl-v0.2-train.xyz" \
    --valid_file="GNL-v0.2/gnl-v0.2-val.xyz" \
    --test_file="GNL-v0.2/gnl-v0.2-test.xyz" \
    --energy_weight=10.0 \
    --forces_weight=100.0 \
    --config_type_weights="{'Default':1.0}" \
    --E0s="average" \
    --model="ScaleShiftMFN_MACE" \
    --num_features_matrix=16 \
    --num_poles=16 \
    --num_cutoff_basis=5 \
    --num_radial_basis=8 \
    --lr=0.01 \
    --interaction_first="RealAgnosticResidualInteractionBlock" \
    --interaction="RealAgnosticResidualInteractionBlock" \
    --num_interactions=2 \
    --max_ell=3 \
    --hidden_irreps="64x0e + 64x1o" \
    --correlation=3 \
    --r_max=3.0 \
    --scaling="rms_forces_scaling" \
    --swa \
    --swa_forces_weight=1000.0 \
    --swa_energy_weight=1000.0 \
    --batch_size=4 \
    --max_num_epochs=800 \
    --start_swa=400 \
    --patience=256 \
    --ema \
    --ema_decay=0.99 \
    --amsgrad \
    --default_dtype="float64" \
    --clip_grad=10 \
    --device=cuda \
    --seed=4 \
    --restart_latest \
```

You can download the datasets here: https://github.com/LarsSchaaf/Guaranteed-Non-Local-Molecular-Dataset
## References

If you use this code, please cite our papers:
```text
@misc{batatia2024equivariant,
      title={Equivariant Matrix Function Neural Networks}, 
      author={Ilyes Batatia and Lars L. Schaaf and Huajie Chen and Gábor Csányi and Christoph Ortner and Felix A. Faber},
      year={2024},
      eprint={2310.10434},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
@misc{Batatia2022MACE,
  title = {MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author = {Batatia, Ilyes and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Simm, Gregor N. C. and Ortner, Christoph and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2206.07697},
  eprint = {2206.07697},
  eprinttype = {arxiv},
  doi = {10.48550/ARXIV.2206.07697},
  archiveprefix = {arXiv}
}
```

## Contact

If you have any questions, please contact us at ib467@cam.ac.uk.

## License

MFN is published and distributed under the MIT license.
