from __future__ import annotations

import math

import numpy as np


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    a_unit = _unit(a)
    b_unit = _unit(b)
    cosine = float(np.clip(abs(np.dot(a_unit, b_unit)), -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _generate_points(seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    theta = math.radians(45)
    rotation = np.array(
        [
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta), math.cos(theta)],
        ]
    )
    scale = np.diag([2.8, 0.45])
    points = rng.normal(size=(180, 2)) @ scale @ rotation.T
    points = points - points.mean(axis=0)
    return points


def _first_principal_component(points: np.ndarray) -> np.ndarray:
    covariance = np.cov(points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    principal = eigenvectors[:, int(np.argmax(eigenvalues))]
    if principal[0] < 0:
        principal = -principal
    return _unit(principal)


def build_oja_pca_demo() -> dict:
    points = _generate_points()
    pca_vector = _first_principal_component(points)

    rng = np.random.default_rng(7)
    weight = _unit(rng.normal(size=2))
    if weight[0] < 0:
        weight = -weight

    learning_rate = 0.003
    epochs = 12
    steps = []
    ordered_indices = np.arange(len(points))

    step_number = 0
    for epoch in range(1, epochs + 1):
        rng.shuffle(ordered_indices)
        for sample_index in ordered_indices:
            x = points[int(sample_index)]
            old_weight = weight.copy()
            output = float(np.dot(weight, x))
            weight_delta = learning_rate * (output * x - (output**2) * weight)
            weight = weight + weight_delta

            if np.dot(weight, pca_vector) < 0:
                weight = -weight

            step_number += 1
            steps.append(
                {
                    "step": step_number,
                    "epoch": epoch,
                    "sample_index": int(sample_index),
                    "input": np.round(x, 4).tolist(),
                    "old_weight": np.round(old_weight, 4).tolist(),
                    "output": round(output, 4),
                    "weight_delta": np.round(weight_delta, 4).tolist(),
                    "new_weight": np.round(weight, 4).tolist(),
                    "new_weight_unit": np.round(_unit(weight), 4).tolist(),
                    "weight_norm": round(float(np.linalg.norm(weight)), 4),
                    "angle_degrees": round(_angle_degrees(weight, pca_vector), 4),
                }
            )

    final_angle = _angle_degrees(weight, pca_vector)

    return {
        "title": "Oja's Rule as Online PCA",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "points": np.round(points, 4).tolist(),
        "pca_vector": np.round(pca_vector, 4).tolist(),
        "initial_weight": steps[0]["old_weight"],
        "final_weight": np.round(weight, 4).tolist(),
        "final_weight_unit": np.round(_unit(weight), 4).tolist(),
        "final_angle_degrees": round(final_angle, 4),
        "steps": steps,
    }
