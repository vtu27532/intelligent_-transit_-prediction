# TransitPredict Workspace Guide

## Project
- Python 3.11 backend: FastAPI, scikit-learn, pandas, joblib, and SQLite.
- Frontend: HTML, CSS, vanilla JavaScript, Chart.js, and Leaflet.
- Keep the demo runnable without credentials or external data feeds; the routes and generated trip history are fictional.

## Commands
- Install: `python -m pip install -r requirements.txt`
- Generate data: `python generate_data.py`
- Train models: `python train.py`
- Run the app: `uvicorn backend.main:app --reload`

## Conventions
- Keep route and stop definitions in `backend/transit_data.py` so data generation, validation, and the map share the same network.
- Keep model inputs and saved-artifact paths consistent through `backend/model.py`.
- The browser pages are served by FastAPI. Keep API requests same-origin and preserve form validation and loading/error states.
- Generated CSV, model, metrics, and SQLite files belong in `data/` and are ignored by Git.
- Explain ML behavior in beginner-friendly terms and avoid claiming synthetic results are real-world service guarantees.
