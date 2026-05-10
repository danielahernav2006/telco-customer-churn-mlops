from pathlib import Path

import joblib
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


def main():
    x_train = joblib.load("data/X_train.joblib")
    x_test = joblib.load("data/X_test.joblib")

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=x_train, current_data=x_test)

    output_path = Path("data/drift_report.html")
    report.save_html(str(output_path))

    print(f"Reporte de deriva generado en: {output_path}")


if __name__ == "__main__":
    main()