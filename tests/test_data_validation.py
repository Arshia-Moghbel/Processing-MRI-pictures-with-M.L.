import csv
import tempfile
import unittest
from pathlib import Path

from Brain_Tumor_CNN import collect_image_samples, collect_nifti_samples, is_nifti_file, validate_samples


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

    def test_nifti_extension_detection(self):
        self.assertTrue(is_nifti_file(Path("x.nii.gz")))
        self.assertFalse(is_nifti_file(Path("x.png")))


if __name__ == "__main__":
    unittest.main()
