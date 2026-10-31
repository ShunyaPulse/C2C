import os
import tempfile
import unittest
import subprocess
import pandas as pd
import numpy as np

class TestSmoke(unittest.TestCase):
    def test_smoke_predict_main(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dict_path = os.path.join(base_dir, "hack-day-lucknow-datasets", "p1-clean-cooking", "data_dictionary.csv")
        dd = pd.read_csv(dict_path)
        
        sample_data = {"household_id": [112481, 102237, 103792, 111046, 108833]}
        for col in dd["column"]:
            if col not in ["household_id", "target_solid_fuel"]:
                sample_data[col] = [np.nan] * 5
        sample_data["s_name"] = ["Uttar Pradesh", "Bihar", "Odisha", "Maharashtra", "Rajasthan"]
        sample_data["land_area_sqfeet"] = [1200.0, 850.0, 2400.0, 500.0, 1600.0]
        sample_data["avg_monthly_bill"] = [350.0, 150.0, 600.0, 1200.0, 400.0]
        sample_data["q216_house_pucca_kachha"] = [2, 2, 1, 1, 3]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_csv = os.path.join(tmpdir, "sample_input.csv")
            output_csv = os.path.join(tmpdir, "sample_output.csv")
            pd.DataFrame(sample_data).to_csv(sample_csv, index=False)
            
            cmd = [
                "python", os.path.join(base_dir, "predict.py"),
                "--input", sample_csv,
                "--output", output_csv
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"predict.py error: {res.stderr}")
            self.assertTrue(os.path.exists(output_csv), "Output CSV does not exist")
            
            out_df = pd.read_csv(output_csv)
            self.assertEqual(list(out_df.columns), ["household_id", "prob_solid_fuel"])
            self.assertEqual(len(out_df), 5)
            self.assertFalse(out_df["prob_solid_fuel"].isna().any())
            self.assertTrue((out_df["prob_solid_fuel"] >= 0.0).all() and (out_df["prob_solid_fuel"] <= 1.0).all())

    def test_smoke_predict_short(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        sample_data = {
            "household_id": [112481, 102237],
            "s_name": ["Uttar Pradesh", "Bihar"],
            "land_area_sqfeet": [1200.0, 850.0],
            "avg_monthly_bill": [350.0, 150.0]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_csv = os.path.join(tmpdir, "short_input.csv")
            output_csv = os.path.join(tmpdir, "short_output.csv")
            pd.DataFrame(sample_data).to_csv(sample_csv, index=False)
            
            cmd = [
                "python", os.path.join(base_dir, "predict_short.py"),
                "--input", sample_csv,
                "--output", output_csv
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"predict_short.py error: {res.stderr}")
            self.assertTrue(os.path.exists(output_csv))
            out_df = pd.read_csv(output_csv)
            self.assertEqual(list(out_df.columns), ["household_id", "prob_solid_fuel"])
            self.assertEqual(len(out_df), 2)

if __name__ == "__main__":
    unittest.main()
