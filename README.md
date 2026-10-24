# SolarSense — Operation Daybreak

**Project:** Solar radiation analysis, visualization, daily solar summaries, and machine-learning prediction.

## Repository contents

- `SolarSense_Analysis.py` — the single main analysis script.
- `data/SolarPrediction.csv` — input dataset used by the analysis (share only if permitted).
- `solar_analysis_output/graphs/` — generated plots.
- `solar_analysis_output/data/` — analysis tables and model outputs.
- `solar_analysis_output/reports/` — generated text and PDF reports.
- `requirements.txt` — Python dependencies.

## Run locally

Python 3.10+ is recommended. From this repository directory, run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python SolarSense_Analysis.py
```

The script saves plots and tables without opening interactive graph windows. By default, the CSV is read from `data/SolarPrediction.csv`, and outputs are written to `solar_analysis_output/`. You can override these paths with `SOLARSENSE_CSV` and `SOLARSENSE_OUTPUT` environment variables.

## Reproducibility notes

The PDF and output files in this repository are the generated results from the included dataset. Running the script regenerates outputs and may replace files in `solar_analysis_output/`. Review the assignment brief and dataset license before redistribution. Model results are descriptive of this dataset and split design; they do not establish causation or guarantee future solar output. Temperature-based predictions are nowcasts rather than pure forecasts.

## Suggested citation

SolarSense — Operation Daybreak, course project for solar radiation data analysis and prediction.
