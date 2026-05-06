import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPO_ROOT / "web-demo" / "backend"
sys.path.insert(0, str(BACKEND_ROOT))


class WebDemoExperimentTests(unittest.TestCase):
    def test_fruit_payload_contains_real_training_steps(self):
        from experiments.fruit import build_fruit_demo

        payload = build_fruit_demo()

        self.assertIn("features", payload)
        self.assertIn("classes", payload)
        self.assertIn("hebbian", payload)
        self.assertIn("backprop", payload)
        self.assertGreater(len(payload["hebbian"]["steps"]), 0)
        self.assertGreater(len(payload["backprop"]["steps"]), 0)

        first_hebbian_step = payload["hebbian"]["steps"][0]
        self.assertEqual(first_hebbian_step["label"], "banana")
        self.assertIn("old_weights", first_hebbian_step)
        self.assertIn("weight_delta", first_hebbian_step)
        self.assertIn("new_weights", first_hebbian_step)

    def test_playground_payload_compares_trained_fruit_models(self):
        from experiments.fruit import build_playground_demo

        payload = build_playground_demo()

        self.assertEqual(payload["features"], ["yellow", "red", "green", "round", "long", "sweet", "crunchy", "soft"])
        self.assertEqual(payload["classes"], ["banana", "apple", "pear"])
        self.assertGreaterEqual(len(payload["presets"]), 5)

        models = payload["models"]
        self.assertEqual(set(models), {"hebbian", "backprop"})
        for model in models.values():
            self.assertEqual(len(model["weights"]), len(payload["features"]))
            self.assertEqual(len(model["weights"][0]), len(payload["classes"]))
            self.assertGreaterEqual(model["verification_accuracy"], 0.0)
            self.assertLessEqual(model["verification_accuracy"], 1.0)
            self.assertEqual(len(model["verification_predictions"]), 3)

        banana = payload["presets"][0]
        self.assertEqual(banana["expected"], "banana")
        self.assertEqual(banana["predictions"]["hebbian"]["prediction"], "banana")
        self.assertEqual(banana["predictions"]["backprop"]["prediction"], "banana")

    def test_oja_payload_converges_near_first_principal_component(self):
        from experiments.oja_pca import build_oja_pca_demo

        payload = build_oja_pca_demo()

        self.assertGreater(len(payload["points"]), 50)
        self.assertGreater(len(payload["steps"]), 100)
        self.assertIn("pca_vector", payload)
        self.assertIn("final_angle_degrees", payload)
        self.assertLess(payload["final_angle_degrees"], 8.0)
        self.assertLess(payload["final_weight_norm"], 1.2)
        self.assertGreater(payload["pure_hebbian_final_norm"], 1_000_000)
        self.assertGreater(payload["pure_hebbian_final_norm"], payload["final_weight_norm"] * 1_000_000)

        final_step = payload["steps"][-1]
        self.assertIn("old_weight", final_step)
        self.assertIn("weight_delta", final_step)
        self.assertIn("new_weight", final_step)
        self.assertIn("angle_degrees", final_step)
        self.assertIn("pure_weight_norm", final_step)
        self.assertIn("pure_weight_unit", final_step)
        self.assertIn("pure_angle_degrees", final_step)

    def test_forgetting_payload_matches_paper_demo_claims(self):
        from experiments.forgetting import build_forgetting_demo

        payload = build_forgetting_demo()

        self.assertEqual(
            payload["architecture"],
            {
                "input_units": 2,
                "hidden_layers": [16, 16],
                "activation": "ReLU",
                "output_units": 2,
            },
        )
        self.assertEqual(payload["task_relationship"], "distinct input distributions")
        self.assertIn("standard", payload["series"])
        self.assertIn("ewc", payload["series"])
        self.assertGreater(len(payload["series"]["standard"]), 10)
        self.assertEqual(len(payload["series"]["standard"]), len(payload["series"]["ewc"]))

        standard = payload["summary"]["standard"]
        self.assertGreaterEqual(standard["task_a_after_a"], 0.93)
        self.assertGreaterEqual(standard["task_a_after_b"], 0.20)
        self.assertLessEqual(standard["task_a_after_b"], 0.35)
        self.assertGreaterEqual(standard["task_b_after_b"], 0.90)

        ewc = payload["summary"]["ewc"]
        self.assertGreaterEqual(ewc["task_a_after_b"], standard["task_a_after_b"] + 0.30)
        self.assertGreaterEqual(ewc["task_a_after_b"], 0.60)
        self.assertGreaterEqual(ewc["task_b_after_b"], 0.84)

        final_step = payload["series"]["ewc"][-1]
        self.assertEqual(final_step["method"], "ewc")
        self.assertIn("ewc_penalty", final_step)
        self.assertIn("input_hidden_1", final_step["weights"])
        self.assertIn("hidden_1_hidden_2", final_step["weights"])
        self.assertIn("hidden_2_output", final_step["weights"])


if __name__ == "__main__":
    unittest.main()
