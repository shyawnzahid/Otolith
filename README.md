# Otolith age determination of Antarctic toothfish using computer vision

## Set Up & Installation

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

Run the [setup.sh](./setup.sh) script to set up an environment variable `BASE_DIRECTORY` containing the absolute file path of the `Otolith` directory. This will create a file `Otolith/.env`.

```shell
./setup.sh
```

### Data & Trained Models

Data and trained models are available on HuggingFace.

- [Data](https://huggingface.co/datasets/shyawnzahid/data)
- [Trained models](https://huggingface.co/datasets/shyawnzahid/saved_models) (ResNet regression, ResNet classification, ViT regression, ViT classification)

To download these files from their HuggingFace repositories, Git Large File Storage (LFS) must first be downloaded and installed following these [instructions](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage). Then run the following commands from the project base directory. They will take a while, as they download more than 2 GB of files.

```shell
git clone https://huggingface.co/datasets/shyawnzahid/data
git clone https://huggingface.co/shyawnzahid/saved_models
```

These commands will create and populate new directories `Otolith/data` and `Otolith/saved_models` with the appropriate files.

### SAM 2

The `remove_background.py` script included in the project code requires installation of the [SAM 2](https://github.com/facebookresearch/sam2) repository. Do not follow the installation instructions in the SAM 2 repository's `README.md`, as they are not necessary for running the `remove_background.py` script. It only requires that the repository is cloned into the folder `Otolith/sam2`. Run the following command to do this.

```shell
git clone https://github.com/facebookresearch/sam2.git
```

Then move the `remove_background.py` script into the `Otolith/sam2` folder.

```shell
mv ./remove_background.py ./sam2/remove_background.py
```

## Code Structure

### `Otolith/src`

### `Otolith/data`

### `Otolith/saved_models`

### `Otolith/saved_labels`

This directory contains the [`OA_labels.pkl`](./saved_labels/OA_labels.pkl) file. It can be loaded into a `python` `dict`.

```python
with open(label_dir, 'rb') as f:
        ages = pickle.load(f)
```

It contains the age labels used during training for the images in `Otolith/data/OA`. In general, `ages[x]` will contain the age label for the image `data/OA/x.png`.

### `Otolith/saved_model_predictions`

The files [`res_clf.pkl`](./saved_model_predictions/res_clf.pkl), [`res_reg.pkl`](./saved_model_predictions/res_reg.pkl), [`vit_clf.pkl`](./saved_model_predictions/vit_clf.pkl), [`vit_clf.pkl`](./saved_model_predictions/vit_reg.pkl) in this directory are all formatted similar to [`OA_labels.pkl`](./saved_labels/OA_labels.pkl), and contain the model predictions for the ResNet classification, ResNet regression, ViT classification, and ViT regression models, respectively.

### [`Otolith/sam2/remove_background.py`](./Otolith/sam2/remove_background.py)
