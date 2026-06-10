# Weighted PaDiM

Implementation of Weighted PaDiM, an unsupervised anomaly detection method for identifying volcanic deformation patterns in InSAR imagery. The approach extends PaDiM by applying layer-specific weights during feature extraction.

## Installation

```bash
conda env create -f environment.yml
conda activate volcanic-anomaly-detection
````

## Data Structure

The model can be applied to any dataset following the directory structure below:

```text
main.py
res/
npz/
models/
tools/
datasets/
└── dataset_name/
    ├── train/
    ├── test_normal/
    ├── test_abnormal/
        ├── dataset_format/
            ├── image_001.png
            ├── image_002.png
            ├── image_003.png
            └── ...
```

The input directory should contain image files in PNG, JPG, or TIFF format. The inference script will process all images found in the specified input folder.
In this directory structure, the convention is that `dataset_name` is the name of the volcano (e.g. `taal`), while `dataset_format` refers to the preprocessing variant or image representation used by the model (e.g. `unw_png`).


## Preprocessing

The MATLAB script `preprocess_dataset.m` was used to generate the PNG images used by the model.

To apply the preprocessing pipeline to a new dataset, the following variables should be updated:

```matlab
folder = 'datasets/my_dataset/';
subfolders = ["train"; "test_normal"; "test_abnormal"];
sea = imread([folder, 'my_water_mask.tif']) == 1;
```

### Required inputs

* `folder`: path to the dataset directory.
* `subfolders`: dataset partitions to process (e.g., `train`, `test_normal`, `test_abnormal`).
* `all/`: directory containing the original unwrapped interferograms.
* `sea`: binary water mask with the same dimensions as the interferograms.

### Directory structure

```text
datasets/
└── my_dataset/
    ├── all/
    │   ├── *.unw
    │   └── ...
    ├── train/
    │   └── unw/
    ├── test_normal/
    │   └── unw/
    ├── test_abnormal/
    │   └── unw/
    └── my_water_mask.tif
```

The script will generate processed PNG images in:

```text
train/unw_remade_water/
test_normal/unw_remade_water/
test_abnormal/unw_remade_water/
```

These generated PNG files can then be used directly with the Python code provided in this repository.

### Filename conventions

The code assumes filenames like this: `taal_20141019_20141112.geo.unw.tif`.
If the source interferograms use a different filename convention, the `file_type` variable and filename matching logic may need to be adjusted.



## Example

The following command trains and evaluates the model on the specified dataset:

```bash
python main.py \
    --dataset 'taal' \
    --subfolder 'unw_png' \
    --epochs 50 \
    --num-channels 1 \
    --batch-size 32 \
    --image-size 128 \
    --result-dir res/unw_png_128 \
    --model-type 'padim_weights_pdf'
```

### Using Your Own Data

To run inference on your own images, place them in a directory following the structure above and provide the directory path via the `--dataset`  and `--subfolder` arguments.


The repository includes a small set of example preprocessed images in `datasets/taal/` to verify that the model pipeline executes correctly. These images are intended for demonstration purposes only and are not used for model evaluation.


## Notes

The code requires `torchmetrics==0.10.1`. Newer versions introduce API changes that may cause import errors. Please use the provided `environment.yml`.


## Citation

This repository accompanies the paper:

**Unsupervised Anomaly Detection for Volcanic Deformation in InSAR Imagery**  
Robert Popescu, Nantheera Anantrasirichai, Juliet Biggs  
*Earth and Space Science*, 2025  
DOI: 10.1029/2024EA003892

If you use this code in your research, please cite:

```bibtex
@article{Popescu2025,
  author = {Popescu, Robert and Anantrasirichai, Nantheera and Biggs, Juliet},
  title = {Unsupervised Anomaly Detection for Volcanic Deformation in InSAR Imagery},
  journal = {Earth and Space Science},
  volume = {12},
  number = {6},
  pages = {e2024EA003892},
  year = {2025},
  doi = {10.1029/2024EA003892}
}
```