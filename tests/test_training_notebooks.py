"""Catch notebook metadata that prevents a private Kaggle worker from starting."""

import json
from pathlib import Path
import unittest


class KaggleTrainingSourceTest(unittest.TestCase):
    def test_training_notebooks_declare_python_kernel(self):
        root = Path(__file__).resolve().parents[1] / "notebooks"
        for path in root.rglob("kernel-metadata.json"):
            descriptor = json.loads(path.read_text())
            if descriptor.get("kernel_type") != "notebook":
                continue
            with self.subTest(path=path):
                notebook = json.loads(
                    (path.parent / descriptor["code_file"]).read_text()
                )
                self.assertEqual(notebook["metadata"]["kernelspec"]["name"], "python3")
