import unittest

from data_pipeline.mlb_park_factors import park_hr_factor, park_run_factor


class MLBParkFactorTests(unittest.TestCase):
    def test_coors_like_distance_boosts_run_and_hr_factors(self) -> None:
        self.assertGreater(park_run_factor(18.5), 1.0)
        self.assertGreater(park_hr_factor(18.5), park_run_factor(18.5))

    def test_negative_distance_suppresses_run_environment(self) -> None:
        self.assertLess(park_run_factor(-6.0), 1.0)
        self.assertLess(park_hr_factor(-6.0), 1.0)


if __name__ == "__main__":
    unittest.main()
