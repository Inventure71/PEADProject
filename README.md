# PEADProject

This repo is a small machine learning demo for class.

It compares two ways a simple model can learn from fruit examples:

- `HebbianNetwork.py`: learns by strengthening connections between features and the correct class.
- `BackdropNetwork.py`: learns by using prediction error to adjust the weights.
- `web-demo/`: runs a browser visualization where the backend calculates each training step and the frontend shows the neurons, weights, vectors, and accuracies changing.

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

## Run the web visualization

```bash
cd web-demo/backend
python3 server.py
```

Then open:

```text
http://127.0.0.1:5173
```

The web demo has three sections:

- Fruit Weights: the easy fruit example with neurons and changing feature-to-class weights.
- Oja / PCA: one Oja neuron learns the first principal component of a 2D dataset.
- Forgetting: a two-hidden-layer ReLU network learns Task A, then Task B, and shows that EWC reduces catastrophic forgetting compared with standard backpropagation.

If port `5173` is busy, run with another port:

```bash
PORT=5174 python3 server.py
```
