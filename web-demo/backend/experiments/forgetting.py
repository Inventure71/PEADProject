from __future__ import annotations

import numpy as np


PARAMETER_KEYS = ("w1", "b1", "w2", "b2", "w3", "b3")


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def _make_dataset(seed: int = 9) -> dict:
    rng = np.random.default_rng(seed)
    separation = 0.6
    spread = 0.65
    covariance = np.eye(2) * spread**2

    train_a_x = rng.multivariate_normal([-separation, 0.0], covariance, size=360)
    test_a_x = rng.multivariate_normal([-separation, 0.0], covariance, size=240)
    train_b_x = rng.multivariate_normal([separation, 0.0], covariance, size=360)
    test_b_x = rng.multivariate_normal([separation, 0.0], covariance, size=240)

    def task_a_labels(points: np.ndarray) -> np.ndarray:
        return (points[:, 1] > 0).astype(int)

    def task_b_labels(points: np.ndarray) -> np.ndarray:
        return (points[:, 1] <= 0).astype(int)

    return {
        "train_a_x": train_a_x,
        "test_a_x": test_a_x,
        "train_a": task_a_labels(train_a_x),
        "test_a": task_a_labels(test_a_x),
        "train_b_x": train_b_x,
        "test_b_x": test_b_x,
        "train_b": task_b_labels(train_b_x),
        "test_b": task_b_labels(test_b_x),
    }


def _init_model(seed: int = 13, hidden_units: int = 16) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "w1": rng.normal(scale=0.35, size=(2, hidden_units)),
        "b1": np.zeros(hidden_units),
        "w2": rng.normal(scale=0.28, size=(hidden_units, hidden_units)),
        "b2": np.zeros(hidden_units),
        "w3": rng.normal(scale=0.28, size=(hidden_units, 2)),
        "b3": np.zeros(2),
    }


def _copy_model(model: dict) -> dict:
    return {key: value.copy() for key, value in model.items()}


def _forward(model: dict, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hidden_1_raw = np.dot(inputs, model["w1"]) + model["b1"]
    hidden_1 = np.maximum(0.0, hidden_1_raw)
    hidden_2_raw = np.dot(hidden_1, model["w2"]) + model["b2"]
    hidden_2 = np.maximum(0.0, hidden_2_raw)
    logits = np.dot(hidden_2, model["w3"]) + model["b3"]
    probabilities = _softmax(logits)
    return hidden_1_raw, hidden_1, hidden_2_raw, hidden_2, probabilities


def _loss_and_gradients(model: dict, inputs: np.ndarray, labels: np.ndarray) -> tuple[float, dict]:
    hidden_1_raw, hidden_1, hidden_2_raw, hidden_2, probabilities = _forward(model, inputs)
    batch_size = len(labels)
    loss = -np.mean(np.log(probabilities[np.arange(batch_size), labels] + 1e-9))

    dlogits = probabilities.copy()
    dlogits[np.arange(batch_size), labels] -= 1.0
    dlogits /= batch_size

    gradients = {}
    gradients["w3"] = np.dot(hidden_2.T, dlogits)
    gradients["b3"] = np.sum(dlogits, axis=0)

    dhidden_2 = np.dot(dlogits, model["w3"].T)
    dhidden_2_raw = dhidden_2 * (hidden_2_raw > 0)
    gradients["w2"] = np.dot(hidden_1.T, dhidden_2_raw)
    gradients["b2"] = np.sum(dhidden_2_raw, axis=0)

    dhidden_1 = np.dot(dhidden_2_raw, model["w2"].T)
    dhidden_1_raw = dhidden_1 * (hidden_1_raw > 0)
    gradients["w1"] = np.dot(inputs.T, dhidden_1_raw)
    gradients["b1"] = np.sum(dhidden_1_raw, axis=0)

    return float(loss), gradients


def _empirical_fisher(model: dict, inputs: np.ndarray, labels: np.ndarray) -> dict:
    fisher = {key: np.zeros_like(model[key]) for key in PARAMETER_KEYS}

    for index in range(len(labels)):
        _loss, gradients = _loss_and_gradients(model, inputs[index : index + 1], labels[index : index + 1])
        for key in PARAMETER_KEYS:
            fisher[key] += gradients[key] ** 2

    for key in PARAMETER_KEYS:
        fisher[key] /= len(labels)

    max_value = max(float(np.max(value)) for value in fisher.values())
    if max_value > 0:
        for key in PARAMETER_KEYS:
            fisher[key] = fisher[key] / max_value

    return fisher


def _ewc_penalty(model: dict, consolidated_model: dict, fisher: dict) -> float:
    penalty = 0.0
    for key in PARAMETER_KEYS:
        difference = model[key] - consolidated_model[key]
        penalty += float(np.sum(fisher[key] * difference**2))
    return penalty


def _train_epoch(
    model: dict,
    inputs: np.ndarray,
    labels: np.ndarray,
    learning_rate: float,
    consolidated_model: dict | None = None,
    fisher: dict | None = None,
    ewc_lambda: float = 0.0,
) -> tuple[float, float]:
    loss, gradients = _loss_and_gradients(model, inputs, labels)
    penalty = 0.0

    if consolidated_model is not None and fisher is not None:
        penalty = _ewc_penalty(model, consolidated_model, fisher)
        for key in PARAMETER_KEYS:
            gradients[key] += ewc_lambda * fisher[key] * (model[key] - consolidated_model[key])
        loss += 0.5 * ewc_lambda * penalty

    for key in PARAMETER_KEYS:
        model[key] -= learning_rate * gradients[key]

    return loss, penalty


def _accuracy(model: dict, inputs: np.ndarray, labels: np.ndarray) -> float:
    _hidden_1_raw, _hidden_1, _hidden_2_raw, _hidden_2, probabilities = _forward(model, inputs)
    predictions = np.argmax(probabilities, axis=1)
    return float(np.mean(predictions == labels))


def _snapshot_weights(model: dict) -> dict:
    return {
        "input_hidden_1": np.round(model["w1"], 4).tolist(),
        "hidden_1_hidden_2": np.round(model["w2"], 4).tolist(),
        "hidden_2_output": np.round(model["w3"], 4).tolist(),
        "hidden_1_bias": np.round(model["b1"], 4).tolist(),
        "hidden_2_bias": np.round(model["b2"], 4).tolist(),
        "output_bias": np.round(model["b3"], 4).tolist(),
    }


def _record_step(
    model: dict,
    method: str,
    phase: str,
    epoch: int,
    loss: float,
    penalty: float,
    dataset: dict,
) -> dict:
    return {
        "method": method,
        "phase": phase,
        "epoch": epoch,
        "loss": round(float(loss), 4),
        "ewc_penalty": round(float(penalty), 4),
        "task_a_accuracy": round(_accuracy(model, dataset["test_a_x"], dataset["test_a"]), 4),
        "task_b_accuracy": round(_accuracy(model, dataset["test_b_x"], dataset["test_b"]), 4),
        "weights": _snapshot_weights(model),
    }


def _train_task_a(model: dict, dataset: dict, learning_rate: float, epochs: int, method: str) -> list[dict]:
    steps = []
    for epoch in range(1, epochs + 1):
        loss, penalty = _train_epoch(model, dataset["train_a_x"], dataset["train_a"], learning_rate)
        steps.append(_record_step(model, method, "Train Task A", epoch, loss, penalty, dataset))
    return steps


def _train_task_b_standard(model: dict, dataset: dict, learning_rate: float, epochs: int) -> list[dict]:
    steps = []
    for epoch in range(1, epochs + 1):
        loss, penalty = _train_epoch(model, dataset["train_b_x"], dataset["train_b"], learning_rate)
        steps.append(_record_step(model, "standard", "Train Task B", epoch, loss, penalty, dataset))
    return steps


def _train_task_b_ewc(
    model: dict,
    dataset: dict,
    learning_rate: float,
    epochs: int,
    consolidated_model: dict,
    fisher: dict,
    ewc_lambda: float,
) -> list[dict]:
    steps = []
    for epoch in range(1, epochs + 1):
        loss, penalty = _train_epoch(
            model,
            dataset["train_b_x"],
            dataset["train_b"],
            learning_rate,
            consolidated_model=consolidated_model,
            fisher=fisher,
            ewc_lambda=ewc_lambda,
        )
        steps.append(_record_step(model, "ewc", "Train Task B", epoch, loss, penalty, dataset))
    return steps


def _summary_from_steps(steps: list[dict], task_a_epochs: int) -> dict:
    after_a = steps[task_a_epochs - 1]
    after_b = steps[-1]
    return {
        "task_a_after_a": after_a["task_a_accuracy"],
        "task_b_after_a": after_a["task_b_accuracy"],
        "task_a_after_b": after_b["task_a_accuracy"],
        "task_b_after_b": after_b["task_b_accuracy"],
    }


def build_forgetting_demo() -> dict:
    dataset = _make_dataset()
    task_a_epochs = 160
    task_b_epochs = 140
    learning_rate = 0.03
    ewc_lambda = 30.0

    task_a_model = _init_model()
    standard_steps = _train_task_a(task_a_model, dataset, learning_rate, task_a_epochs, "standard")
    consolidated_model = _copy_model(task_a_model)
    fisher = _empirical_fisher(task_a_model, dataset["train_a_x"], dataset["train_a"])

    standard_model = _copy_model(consolidated_model)
    standard_steps.extend(_train_task_b_standard(standard_model, dataset, learning_rate, task_b_epochs))

    ewc_model = _copy_model(consolidated_model)
    ewc_steps = [
        {**step, "method": "ewc", "ewc_penalty": 0.0}
        for step in standard_steps[:task_a_epochs]
    ]
    ewc_steps.extend(
        _train_task_b_ewc(
            ewc_model,
            dataset,
            learning_rate,
            task_b_epochs,
            consolidated_model,
            fisher,
            ewc_lambda,
        )
    )

    sample_points = []
    for point, label_a in zip(dataset["test_a_x"][:60], dataset["test_a"][:60]):
        sample_points.append(
            {
                "task": "A",
                "x": round(float(point[0]), 4),
                "y": round(float(point[1]), 4),
                "label": int(label_a),
            }
        )
    for point, label_b in zip(dataset["test_b_x"][:60], dataset["test_b"][:60]):
        sample_points.append(
            {
                "task": "B",
                "x": round(float(point[0]), 4),
                "y": round(float(point[1]), 4),
                "label": int(label_b),
            }
        )

    summary = {
        "standard": _summary_from_steps(standard_steps, task_a_epochs),
        "ewc": _summary_from_steps(ewc_steps, task_a_epochs),
    }

    return {
        "title": "Catastrophic Forgetting",
        "description": (
            "A two-hidden-layer ReLU network first learns Task A, then learns Task B from a distinct input "
            "distribution. Standard backprop overwrites much of Task A behavior; EWC adds a Fisher-weighted "
            "penalty that reduces forgetting by keeping important Task A weights closer to their consolidated values."
        ),
        "architecture": {
            "input_units": 2,
            "hidden_layers": [16, 16],
            "activation": "ReLU",
            "output_units": 2,
        },
        "task_relationship": "distinct input distributions",
        "task_a_rule": "Task A distribution is centered left; class 1 if y > 0",
        "task_b_rule": "Task B distribution is centered right; class 1 if y <= 0",
        "learning_rate": learning_rate,
        "ewc_lambda": ewc_lambda,
        "summary": summary,
        "sample_points": sample_points,
        "series": {
            "standard": standard_steps,
            "ewc": ewc_steps,
        },
        "steps": standard_steps,
    }
