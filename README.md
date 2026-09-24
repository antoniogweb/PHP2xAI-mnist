# PHP2xAI MNIST

MNIST classification exercise built on PHP2xAI. The project demonstrates dataset preprocessing in PHP, configuration of the computational graph, and training orchestration, while model training and inference are executed by the C++ runtime.

## Installation

```bash
composer create-project antoniogweb/php2xai-mnist
```

## Training

The current training and validation scripts use HDF5 datasets. First, put the PNG images in `src/images/<digit>/` (one directory per digit, `0` through `9`), then create the HDF5 files from `src/`:

```bash
cd src
php create_data_hdf5.php
```

The script splits each digit's images into training (80%) and test (20%), shuffles both sets, converts each 28×28 image to a 784-value vector, and writes integer labels. It creates `src/DataLabelInt/Training/train.h5` and `src/DataLabelInt/Training/test.h5`; each file has an `x` field (`FLOAT32`, 784 values per sample) and a `y` field (`INT64`, one label per sample).

Then run training from `src/`:

```bash
php train.php
```

What it does:
- Reads `train.h5` and `test.h5` from `src/DataLabelInt/Training/` in batches of 300 samples.
- Trains the model and saves weights to `src/weights.json`.

PHP or C++ runtime:
- In `src/train.php` you will find `setRuntime("CPP")`: it uses the C++ runtime.
- In `src/train.php` you will find `setProvider("EIGEN")`: it uses the C++ EIGEN library. Comment or remove the call to use naive/standard C++ runtime
- To use the PHP runtime, replace it with `setRuntime("PHP")` or remove the call.

Batch size:
- The second argument of `HDF5Dataset` controls the batch size. In `src/train.php`, change the `300` passed to each dataset to adjust it.

## Validation

Run the script from `src/`:

```bash
cd src
php validate.php
```

What it does:
- Reads `test.h5` in batches of 300, loads the model from `src/model.json` and weights from `src/weights.json`.
- Computes accuracy and inference time on the test set.

PHP or C++ runtime:
- In `src/validate.php` you will find `setRuntime("CPP")`: it uses the C++ runtime.
- In `src/validate.php` you will find `setProvider("EIGEN")`: it uses the C++ EIGEN library. Comment or remove the call to use naive/standard C++ runtime
- To use the PHP runtime, replace it with `setRuntime("PHP")` or remove the call.

## Notes

- HDF5 generation and loading require the HDF5 support provided by the installed PHP2xAI runtime.
- A text-file alternative is also available: run `php create_data_one_file.php` from `src/` to generate `train.txt` and `test.txt` from `src/images/`. To train or validate with those files, switch the `HDF5Dataset` lines in `src/train.php` and `src/validate.php` to the corresponding `StreamFileDataset` lines (currently commented out).
