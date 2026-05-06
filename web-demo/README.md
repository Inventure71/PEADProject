# PEAD Web Demo

This folder contains the browser visualization for the project.

The backend does the real calculations. The frontend only displays the returned training steps.

## Run

```bash
cd web-demo/backend
python3 server.py
```

Open:

```text
http://127.0.0.1:5173
```

Use another port if needed:

```bash
PORT=5174 python3 server.py
```

## Sections

- Fruit Weights: shows the existing fruit example as neurons and changing connections.
- Oja / PCA: shows Oja's rule rotating a weight vector toward the PCA direction.
- Forgetting: shows a two-hidden-layer ReLU network learning Task A, then Task B, with EWC reducing forgetting compared against standard backpropagation.

## Backend endpoints

```text
/api/fruit
/api/oja
/api/forgetting
```

Each endpoint returns JSON with step-by-step values used by the visualization.
