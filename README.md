# PEADProject

This repo is a small machine learning demo for class.

It compares two ways a simple model can learn from fruit examples:

- `HebbianNetwork.py`: learns by strengthening connections between features and the correct class.
- `BackdropNetwork.py`: learns by using prediction error to adjust the weights.

The shared fruit data is in `_ExampleData.py`.

## Setup

Install the only Python package used by the experiments:

```bash
python3 -m pip install numpy
```

## Run the Hebbian experiment

```bash
python3 HebbianNetwork.py
```

This prints the learned weights and then predicts the fruit class for the test examples.

## Run the backpropagation experiment

```bash
python3 BackdropNetwork.py
```

This prints the loss during training, the final weights, and the predictions for the test examples.

## Choose simple or complex data

Each experiment file has this line near the top:

```python
run_complex_example = True
```

Use `True` for the bigger banana/apple/pear example.
Use `False` for the smaller banana/apple example.
