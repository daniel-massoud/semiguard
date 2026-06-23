# SemiGuard

A trustworthy ML framework for semiconductor test data: anomaly detection,
out-of-distribution detection, and distribution-shift monitoring for
production ML pipelines.

## Motivation

In semiconductor manufacturing, ML models are increasingly used to predict
yield, detect defects, and guide process decisions. But models silently fail
when input distributions shift, when sensors drift, or when novel failure
modes emerge. SemiGuard provides a layered defense: before trusting any
prediction, score the input for distributional anomalies, score the model''s
own confidence, and flag cases that warrant human review — all calibrated
against a fixed review budget.

## Dataset

[SECOM](https://archive.ics.uci.edu/dataset/179/secom): 1567 wafer fabrication
runs, 590 sensor features, pass/fail labels with timestamps spanning
July-October 2008. Real production data from a semiconductor manufacturer.

## Quick start

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/download_data.py
```

## Status

Block 1 complete: data acquisition.  
In active development.
