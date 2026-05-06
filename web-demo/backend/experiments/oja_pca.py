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


def _display_unit(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    unit = _unit(vector)
    if np.dot(unit, reference) < 0:
        unit = -unit
    return unit


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
    initial_weight = _unit(rng.normal(size=2))
    if initial_weight[0] < 0:
        initial_weight = -initial_weight

    learning_rate = 0.003
    epochs = 12
    steps = []
    ordered_indices = np.arange(len(points))
    oja_weight = initial_weight.copy()
    pure_weight = initial_weight.copy()

    step_number = 0
    for epoch in range(1, epochs + 1):
        rng.shuffle(ordered_indices)
        for sample_index in ordered_indices:
            x = points[int(sample_index)]
            old_weight = oja_weight.copy()
            output = float(np.dot(oja_weight, x))
            weight_delta = learning_rate * (output * x - (output**2) * oja_weight)
            oja_weight = oja_weight + weight_delta

            pure_output = float(np.dot(pure_weight, x))
            pure_weight_delta = learning_rate * pure_output * x
            pure_weight = pure_weight + pure_weight_delta

            if np.dot(oja_weight, pca_vector) < 0:
                oja_weight = -oja_weight

            pure_norm = float(np.linalg.norm(pure_weight))

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
                    "new_weight": np.round(oja_weight, 4).tolist(),
                    "new_weight_unit": np.round(_unit(oja_weight), 4).tolist(),
                    "weight_norm": round(float(np.linalg.norm(oja_weight)), 4),
                    "angle_degrees": round(_angle_degrees(oja_weight, pca_vector), 4),
                    "pure_output": round(pure_output, 4),
                    "pure_weight_delta": np.round(pure_weight_delta, 4).tolist(),
                    "pure_weight_unit": np.round(_display_unit(pure_weight, pca_vector), 4).tolist(),
                    "pure_weight_norm": round(pure_norm, 4),
                    "pure_weight_log10_norm": round(math.log10(max(pure_norm, 1e-12)), 4),
                    "pure_angle_degrees": round(_angle_degrees(pure_weight, pca_vector), 4),
                }
            )

    final_angle = _angle_degrees(oja_weight, pca_vector)
    pure_final_norm = float(np.linalg.norm(pure_weight))

    return {
        "title": "Oja's Rule as Online PCA",
        "learning_rate": learning_rate,
        "epochs": epochs,
        "points": np.round(points, 4).tolist(),
        "pca_vector": np.round(pca_vector, 4).tolist(),
        "initial_weight": steps[0]["old_weight"],
        "final_weight": np.round(oja_weight, 4).tolist(),
        "final_weight_unit": np.round(_unit(oja_weight), 4).tolist(),
        "final_weight_norm": round(float(np.linalg.norm(oja_weight)), 4),
        "final_angle_degrees": round(final_angle, 4),
        "pure_hebbian_final_weight_unit": np.round(_display_unit(pure_weight, pca_vector), 4).tolist(),
        "pure_hebbian_final_norm": round(pure_final_norm, 4),
        "pure_hebbian_final_log10_norm": round(math.log10(max(pure_final_norm, 1e-12)), 4),
        "pure_hebbian_final_angle_degrees": round(_angle_degrees(pure_weight, pca_vector), 4),
        "steps": steps,
    }
