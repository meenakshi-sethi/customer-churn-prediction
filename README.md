# Customer Churn Prediction

Explore customer churn patterns in a Telco dataset, train classification models,
and try churn predictions through a Flask form or FastAPI endpoint.

## Findings

The dataset contains 7,043 customers, with an overall churn rate of 26.5%.
Churn varies across customer segments: month-to-month contracts show a 42.7%
churn rate, compared with 2.8% for two-year contracts. These are descriptive
associations, not evidence that a contract type causes churn. The interactive
report at `/report` lets you compare churn rates across several segments.

## Inspiration

This project was inspired by a customer churn prediction video by Mishu Dhar
Chando. This repository presents my portfolio implementation, analysis, and
interactive report.

## Environment setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run this
from the project directory. The project requires Python 3.12; uv creates `.venv`
and installs the exact versions recorded in `uv.lock`.

```bash
uv sync
```

In VS Code, select `.venv` as the Python interpreter and notebook kernel. When
changing dependencies, edit `pyproject.toml`, then run `uv lock` and `uv sync`.

## Run the project

1. Download `WA_Fn-UseC_-Telco-Customer-Churn.csv` from the
	[Kaggle Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
	and place it in the `data/` directory. The dataset is not included here.
2. Run every cell in `Customer Churn Prediction .ipynb` to train the model and create
	`best_model.pkl`, `encoder.pkl`, and `scaler.pkl` in the project directory.
3. Start the Flask web app with `uv run python app.py`, then visit
	`http://127.0.0.1:5000/` for predictions or
	`http://127.0.0.1:5000/report` for the interactive report.
4. To run the API instead, use `uv run uvicorn fastapi_app:app --reload` and open
	`http://127.0.0.1:8000/docs` for its interactive API documentation.

The dataset and generated model files are excluded from Git; download the dataset
and rerun the notebook to recreate the model artifacts locally. The Flask and
FastAPI apps load those artifacts at startup.

## Share the report

Generate a static HTML snapshot after setting up the dataset and model artifacts:

```bash
uv run python -m scripts.build_static_report
```

Commit the generated `docs/report.html`, then in GitHub open **Settings → Pages**
and select **Deploy from a branch**, `main`, and `/docs`. The shareable page will
be `https://meenakshi-sethi.github.io/customer-churn-prediction/report.html`.
Rebuild the snapshot and commit it again whenever the dataset or report changes.
GitHub Pages hosts this report only; the Flask prediction tool still needs a
Python app host.