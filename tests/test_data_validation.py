import csv
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from Brain_Tumor_CNN import (
    collect_image_samples, collect_nifti_samples, is_nifti_file,
    load_nifti_slice, predict_image, read_patient_ids, validate_samples,
)


class DataValidationTests(unittest.TestCase):
    def test_image_collection_uses_class_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "healthy").mkdir(); (root / "tumour").mkdir()
            (root / "healthy" / "a.png").touch(); (root / "tumour" / "b.jpg").touch()
            paths, labels = collect_image_samples(root)
            self.assertEqual([path.name for path in paths], ["a.png", "b.jpg"])
            self.assertEqual(labels, ["healthy", "tumour"])

    def test_nifti_manifest_requires_existing_nifti(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); (root / "scan.nii.gz").touch()
            manifest = root / "labels.csv"
            with manifest.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["path", "label"])
                writer.writeheader(); writer.writerow({"path": "scan.nii.gz", "label": "positive"})
            paths, labels = collect_nifti_samples(root, manifest)
            self.assertEqual(paths, [root / "scan.nii.gz"]); self.assertEqual(labels, ["positive"])

    def test_validation_rejects_a_one_sample_class(self):
        with self.assertRaises(ValueError):
            validate_samples([Path("a"), Path("b"), Path("c")], ["a", "a", "b"], 0.5)

    def test_validation_rejects_an_impossible_stratified_split(self):
        with self.assertRaises(ValueError):
            validate_samples([Path(str(i)) for i in range(4)], ["a", "a", "b", "b"], 0.2)

    def test_validation_rejects_mismatched_path_and_label_counts(self):
        with self.assertRaisesRegex(ValueError, "exactly one label"):
            validate_samples([Path("a"), Path("b")], ["a"], 0.5)

    def test_patient_manifest_maps_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample = root / "scan.png"
            sample.touch()
            manifest = root / "patients.csv"
            with manifest.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["path", "patient_id"])
                writer.writeheader()
                writer.writerow({"path": "scan.png", "patient_id": "patient-1"})
            self.assertEqual(read_patient_ids(manifest, [sample], root), ["patient-1"])

    def test_patient_manifest_rejects_missing_sample(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "patients.csv"
            with manifest.open("w", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=["path", "patient_id"])
                writer.writeheader()
                writer.writerow({"path": "other.png", "patient_id": "patient-1"})
            with self.assertRaisesRegex(ValueError, "missing 1 sample"):
                read_patient_ids(manifest, [root / "scan.png"], root)

    def test_nifti_loader_rejects_singleton_fourth_dimension(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is needed for NIfTI preprocessing checks")

        class FakeNib:
            @staticmethod
            def load(_path):
                return types.SimpleNamespace(get_fdata=lambda dtype: np.zeros((8, 8, 4, 1), dtype=dtype))

        with self.assertRaisesRegex(ValueError, "Expected a 3D"):
            load_nifti_slice(Path("scan.nii.gz"), 16, FakeNib, None, np)

    def test_nifti_loader_rejects_empty_volume(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is needed for NIfTI preprocessing checks")

        class FakeNib:
            @staticmethod
            def load(_path):
                return types.SimpleNamespace(get_fdata=lambda dtype: np.zeros((8, 8, 4), dtype=dtype))

        class FakeCV2:
            INTER_AREA = 0

        with self.assertRaisesRegex(ValueError, "empty or nearly constant"):
            load_nifti_slice(Path("scan.nii.gz"), 16, FakeNib, FakeCV2, np)

    def test_prediction_returns_scores_using_label_map_indices(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("NumPy is needed for prediction checks")

        class FakeModel:
            @staticmethod
            def predict(_batch, verbose=0):
                return np.asarray([[0.2, 0.8]])

        fake_tf = types.SimpleNamespace(
            keras=types.SimpleNamespace(models=types.SimpleNamespace(load_model=lambda _path: FakeModel()))
        )
        fake_np = np
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels = root / "label_map.json"
            labels.write_text(json.dumps({"class_to_index": {"healthy": 0, "tumour": 1}}))
            with patch.dict(sys.modules, {"tensorflow": fake_tf, "numpy": fake_np}), \
                 patch("Brain_Tumor_CNN.load_image", return_value=np.zeros((4, 4, 3), dtype=np.float32)):
                result = predict_image(Path("image.png"), Path("model.keras"), labels, image_size=4)
        self.assertEqual(result["prediction"], "tumour")
        self.assertAlmostEqual(result["confidence"], 0.8)
        self.assertEqual(set(result["scores"]), {"healthy", "tumour"})

    def test_nifti_extension_detection(self):
        self.assertTrue(is_nifti_file(Path("x.nii.gz")))
        self.assertFalse(is_nifti_file(Path("x.png")))


if __name__ == "__main__":
    unittest.main()
