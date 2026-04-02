from __future__ import annotations

import argparse

from basketball_calibration import assert_calibration_targets, run_calibration_suite, suite_as_json


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert-targets", action="store_true")
    args = parser.parse_args()

    report = run_calibration_suite()
    print(suite_as_json())
    if args.assert_targets:
        assert_calibration_targets(report)
