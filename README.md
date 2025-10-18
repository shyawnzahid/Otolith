# Otolith age determination of Antarctic toothfish using computer vision

## Installation

Running this project locally requires downloading files from multiple locations. The project code is available in this repository, data & trained models are available on HuggingFace, and running the [`remove_background.py`](./remove_background.py) script requires the [SAM 2](https://github.com/facebookresearch/sam2) repository as a dependency. Installation instructions are given below for all of these locations.

### Project Code

This repository contains all of the project code, which can be downloaded by cloning it.

```shell
git clone https://github.com/shyawnzahid/Otolith.git && cd Otolith
```

Install [`conda`](https://www.anaconda.com/docs/getting-started/miniconda/install) if you have not already, and create a new `conda` environment.

```shell
conda create -n <environment name here> python=3.12.11
conda activate <environment name here>
```

Install dependencies:

```shell
pip install -r requirements.txt
```

### Data & Trained Models

Data and trained models are available on HuggingFace.

- [Data](https://huggingface.co/datasets/shyawnzahid/data)
- [Trained models](https://huggingface.co/datasets/shyawnzahid/saved_models) (ResNet regression, ResNet classification, ViT regression, ViT classification)

To download these files from their HuggingFace repositories, Git Large File Storage (LFS) must first be installed.

```shell
git lfs install
```

Then run the following commands from the project base directory.

```shell
git clone git@hf.co:datasets/shyawnzahid/data
git clone git@hf.co:shyawnzahid/saved_models
```

These commands will create and populate new directories `Otolith/data` and `Otolith/saved_models` with the appropriate files.

### SAM 2

The `remove_background.py` script included in the project code requires installation of the [SAM 2](https://github.com/facebookresearch/sam2) repository. Do not follow the installation instructions in the SAM 2 repository's `README.md`, as they are not necessary for running the `remove_background.py` script. It only requires that the repository is cloned into the folder `Otolith/sam2`. Run the following command to do this.

```shell
git clone https://github.com/facebookresearch/sam2.git
```
