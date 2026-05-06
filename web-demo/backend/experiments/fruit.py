from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from _ExampleData import get_complex_example  # noqa: E402


def _as_list(matrix: np.ndarray) -> list[list[float]]:
    return np.round(matrix.astype(float), 4).tolist()


def _active_features(features: list[str], inputs: np.ndarray) -> list[str]:
    return [feature for feature, value in zip(features, inputs) if value == 1]


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


def _prediction(classes: list[str], weights: np.ndarray, inputs: np.ndarray) -> str:
    scores = inputs @ weights
    return classes[int(np.argmax(scores))]


def _score_model(
    classes: list[str],
    weights: np.ndarray,
    raw_inputs: list[int],
    include_probabilities: bool = False,
) -> dict:
    inputs = np.array(raw_inputs, dtype=float)
    scores = inputs @ weights
    prediction_index = int(np.argmax(scores))
    payload = {
        "prediction": classes[prediction_index],
        "scores": np.round(scores, 4).tolist(),
    }

    if include_probabilities:
        probabilities = _softmax(scores)
        payload["probabilities"] = np.round(probabilities, 4).tolist()
        payload["confidence"] = round(float(probabilities[prediction_index]), 4)
    else:
        positive_scores = np.maximum(scores, 0)
        total = float(np.sum(positive_scores))
        strengths = positive_scores / total if total > 0 else np.zeros_like(scores)
        payload["strengths"] = np.round(strengths, 4).tolist()
        payload["confidence"] = round(float(strengths[prediction_index]), 4) if total > 0 else 0.0

    return payload


def _verification_predictions(
    classes: list[str],
    weights: np.ndarray,
    test_inputs: list[tuple[list[int], str]],
    include_probabilities: bool = False,
) -> list[dict]:
    predictions = []
    for raw_inputs, expected in test_inputs:
        scored = _score_model(classes, weights, raw_inputs, include_probabilities)
        predictions.append(
            {
                "input": raw_inputs,
                "expected": expected,
                **scored,
                "correct": scored["prediction"] == expected,
            }
        )
    return predictions


def _accuracy(predictions: list[dict]) -> float:
    if not predictions:
        return 0.0
    correct = sum(1 for prediction in predictions if prediction["correct"])
    return round(correct / len(predictions), 4)


def _build_hebbian_steps(
    features: list[str],
    classes: list[str],
    training_data: list[tuple[list[int], str]],
    epochs: int,
    learning_rate: float,
) -> dict:
    weights = np.zeros((len(features), len(classes)))
    steps = []

    for epoch in range(1, epochs + 1):
        for sample_index, (raw_inputs, label) in enumerate(training_data, start=1):
            inputs = np.array(raw_inputs, dtype=float)
            target = np.zeros(len(classes))
            class_index = classes.index(label)
            target[class_index] = 1.0

            old_weights = weights.copy()
            weight_delta = learning_rate * np.outer(inputs, target)
            weights = weights + weight_delta

            scores = inputs @ weights
            steps.append(
                {
                    "mode": "hebbian",
                    "epoch": epoch,
                    "sample_index": sample_index,
                    "input": raw_inputs,
                    "active_features": _active_features(features, inputs),
                    "label": label,
                    "target": target.astype(int).tolist(),
                    "old_weights": _as_list(old_weights),
                    "weight_delta": _as_list(weight_delta),
                    "new_weights": _as_list(weights),
                    "scores": np.round(scores, 4).tolist(),
                    "prediction": classes[int(np.argmax(scores))],
                    "explanation": "Active input features strengthen only the connection to the observed fruit class.",
                }
            )

    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "final_weights": _as_list(weights),
        "steps": steps,
    }


def _build_backprop_steps(
    features: list[str],
    classes: list[str],
    training_data: list[tuple[list[int], str]],
    epochs: int,
    learning_rate: float,
) -> dict:
    weights = np.zeros((len(features), len(classes)))
    steps = []

    for epoch in range(1, epochs + 1):
        for sample_index, (raw_inputs, label) in enumerate(training_data, start=1):
            inputs = np.array(raw_inputs, dtype=float)
            target = np.zeros(len(classes))
            class_index = classes.index(label)
            target[class_index] = 1.0

            old_weights = weights.copy()
            scores = inputs @ weights
            probabilities = _softmax(scores)
            error = probabilities - target
            weight_delta = -learning_rate * np.outer(inputs, error)
            weights = weights + weight_delta
            new_scores = inputs @ weights
            loss = -np.log(probabilities[class_index] + 1e-9)

            steps.append(
                {
                    "mode": "backprop",
                    "epoch": epoch,
                    "sample_index": sample_index,
                    "input": raw_inputs,
                    "active_features": _active_features(features, inputs),
                    "label": label,
                    "target": target.astype(int).tolist(),
                    "old_weights": _as_list(old_weights),
                    "weight_delta": _as_list(weight_delta),
                    "new_weights": _as_list(weights),
                    "scores": np.round(scores, 4).tolist(),
                    "probabilities": np.round(probabilities, 4).tolist(),
                    "error": np.round(error, 4).tolist(),
                    "loss": round(float(loss), 4),
                    "prediction_before_update": classes[int(np.argmax(scores))],
                    "prediction": classes[int(np.argmax(new_scores))],
                    "explanation": "The model compares prediction probabilities with the target, then increases correct-class weights and decreases competing-class weights.",
                }
            )

    return {
        "learning_rate": learning_rate,
        "epochs": epochs,
        "final_weights": _as_list(weights),
        "steps": steps,
    }


def build_fruit_demo() -> dict:
    features, classes, training_data, test_inputs = get_complex_example()

    hebbian = _build_hebbian_steps(
        features=features,
        classes=classes,
        training_data=training_data,
        epochs=3,
        learning_rate=0.1,
    )
    backprop = _build_backprop_steps(
        features=features,
        classes=classes,
        training_data=training_data,
        epochs=8,
        learning_rate=0.1,
    )

    final_predictions = {
        "hebbian": [
            {
                "input": inputs,
                "expected": expected,
                "prediction": _prediction(classes, np.array(hebbian["final_weights"]), np.array(inputs)),
            }
            for inputs, expected in test_inputs
        ],
        "backprop": [
            {
                "input": inputs,
                "expected": expected,
                "prediction": _prediction(classes, np.array(backprop["final_weights"]), np.array(inputs)),
            }
            for inputs, expected in test_inputs
        ],
    }

    return {
        "title": "Fruit Weights",
        "features": features,
        "classes": classes,
        "training_data": [
            {"input": inputs, "label": label, "active_features": _active_features(features, np.array(inputs))}
            for inputs, label in training_data
        ],
        "test_inputs": [
            {"input": inputs, "expected": expected, "active_features": _active_features(features, np.array(inputs))}
            for inputs, expected in test_inputs
        ],
        "hebbian": hebbian,
        "backprop": backprop,
        "final_predictions": final_predictions,
    }


def build_playground_demo() -> dict:
    fruit_demo = build_fruit_demo()
    features = fruit_demo["features"]
    classes = fruit_demo["classes"]
    test_inputs = [(item["input"], item["expected"]) for item in fruit_demo["test_inputs"]]

    hebbian_weights = np.array(fruit_demo["hebbian"]["final_weights"], dtype=float)
    backprop_weights = np.array(fruit_demo["backprop"]["final_weights"], dtype=float)

    hebbian_predictions = _verification_predictions(classes, hebbian_weights, test_inputs)
    backprop_predictions = _verification_predictions(
        classes,
        backprop_weights,
        test_inputs,
        include_probabilities=True,
    )

    presets = [
        {
            "name": "Banana check",
            "input": [1, 0, 0, 0, 1, 1, 0, 1],
            "expected": "banana",
            "note": "The exact held-out banana pattern: yellow, long, sweet, soft.",
        },
        {
            "name": "Apple check",
            "input": [0, 1, 0, 1, 0, 1, 1, 0],
            "expected": "apple",
            "note": "The exact held-out apple pattern: red, round, sweet, crunchy.",
        },
        {
            "name": "Pear check",
            "input": [0, 0, 1, 0, 1, 1, 0, 1],
            "expected": "pear",
            "note": "The exact held-out pear pattern: green, long, sweet, soft.",
        },
        {
            "name": "Green round fruit",
            "input": [0, 0, 1, 1, 0, 1, 1, 0],
            "expected": "apple",
            "note": "A training-style apple variant that shares green with pear but keeps round and crunchy.",
        },
        {
            "name": "Yellow pear hybrid",
            "input": [1, 0, 1, 0, 1, 1, 0, 1],
            "expected": None,
            "note": "A deliberately mixed fruit: banana color plus pear-like green, long, sweet, soft.",
        },
        {
            "name": "Red soft hybrid",
            "input": [0, 1, 0, 0, 1, 1, 0, 1],
            "expected": None,
            "note": "A custom ambiguous case that tests whether red outweighs long and soft.",
        },
    ]

    scored_presets = []
    for preset in presets:
        scored_presets.append(
            {
                **preset,
                "active_features": _active_features(features, np.array(preset["input"])),
                "predictions": {
                    "hebbian": _score_model(classes, hebbian_weights, preset["input"]),
                    "backprop": _score_model(
                        classes,
                        backprop_weights,
                        preset["input"],
                        include_probabilities=True,
                    ),
                },
            }
        )

    return {
        "title": "Model Playground",
        "features": features,
        "classes": classes,
        "presets": scored_presets,
        "models": {
            "hebbian": {
                "name": "Hebbian associative model",
                "description": "Scores come directly from learned feature-to-class associations.",
                "weights": _as_list(hebbian_weights),
                "verification_accuracy": _accuracy(hebbian_predictions),
                "verification_predictions": hebbian_predictions,
            },
            "backprop": {
                "name": "Backprop softmax model",
                "description": "Scores are converted to probabilities after supervised error-based training.",
                "weights": _as_list(backprop_weights),
                "verification_accuracy": _accuracy(backprop_predictions),
                "verification_predictions": backprop_predictions,
            },
        },
    }
