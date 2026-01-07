# PHP2xAI MNIST

MNIST classification exercise built on PHP2xAI. The project demonstrates dataset preprocessing in PHP, configuration of the computational graph, and training orchestration, while model training and inference are executed by the C++ runtime.

## Installation

```bash
composer create-project antoniogweb/php2xai-mnist
```

## Training

Run the script from `src/`:

```bash
cd src
php train.php
```

What it does:
- Reads data from `src/DataLabelInt/Training/train.txt` and `src/DataLabelInt/Training/test.txt`.
- Trains the model and saves weights to `src/Output/weights.json`.

PHP or C++ runtime:
- In `src/train.php` you will find `setRuntime("CPP")`: it uses the C++ runtime.
- To use the PHP runtime, replace it with `setRuntime("PHP")` or remove the call.

Batch size:
- The second argument of `StreamFileDataset` controls the batch size.
- In `src/train.php`, edit `new StreamFileDataset($path."/train.txt", 1024)` (and the validation dataset if needed).

## Validation

Run the script from `src/`:

```bash
cd src
php validate.php
```

What it does:
- Loads the model from `src/model.json` and weights from `src/Output/weights.json`.
- Computes accuracy and inference time on the test set.

PHP or C++ runtime:
- In `src/validate.php` you will find `setRuntime("CPP")`: it uses the C++ runtime.
- To use the PHP runtime, replace it with `setRuntime("PHP")` or remove the call.

## Notes

- If you don't have `train.txt` and `test.txt` yet, generate them with `src/create_data_one_file.php` (requires images in `src/images/`).
