# House Price Predictor (Nepal)

Implementation of the *House Price Predictor* project report: a Flask web app and REST API
that converts Nepalese land units (Aana, Ropani, Paisa, Daam, Katha, Bigha, Dhur, sq ft, sq m)
to square feet, predicts a house price in NPR with a valuation range, logs queries, and lets
admins upload a dataset and retrain the model.

> **Data note.** No verified real Nepalese dataset exists in this project, so `gen_data.py`
> simulates listings (as in report Section 4.2.3). All accuracy figures describe the
> *pipeline*, not real-world pricing accuracy. To use real data, replace `data/listings.csv`
> (same columns) and run `python3 train.py`, or upload it from the Admin page.

## Quick start
```bash
pip install -r requirements.txt
python3 gen_data.py        # simulate data/listings.csv (skip if you have real data)
python3 train.py           # clean, train 4 models, select best, save models/
python3 evaluate.py        # figures for the report -> reports/
python3 -m unittest discover -s tests -v   # 48 automated tests
python3 manage.py create-admin you@example.com "Your Name"
python3 app.py             # http://127.0.0.1:5000
```

## Layout
| File | Purpose |
|---|---|
| `unit_conversion.py` | Land-unit constants and `to_sqft()` (report Table 4.2) |
| `gen_data.py` | Simulated listings with deliberate duplicates / gaps / bad prices |
| `train.py` | Cleaning, pipeline (one-hot district, log+scale area, scaled numerics, log-price target), 4 models, 5-fold CV selection, metadata |
| `predict_service.py` | `predict_price()`, range, Lakh/Crore formatting, out-of-range warnings |
| `app.py` | Routes: `/`, `/predict`, `/api/predict`, auth, `/history`, `/admin*`, `/model` |
| `schema.sql` | USER, QUERY_LOG, PROPERTY_FEATURE, PREDICTION_RESULT (report Section 3.5.7) |
| `tests/test_all.py` | Unit, integration, black-box and security tests |

## API
```bash
curl -X POST localhost:5000/api/predict -H 'Content-Type: application/json' -d '{
  "district":"Kathmandu","area_value":4,"area_unit":"aana","road_width_ft":13,
  "bedrooms":4,"bathrooms":3,"floors":2.5,"building_age":5}'
```

## Security
Server-side validation, parameterised SQL, Jinja2 auto-escaping, salted password hashes
(Werkzeug), signed HttpOnly session cookies, CSRF tokens on forms, role-based admin routes.
Set `HPP_HTTPS=1` behind HTTPS to mark cookies Secure, and `HPP_SECRET_KEY` in production.
For production use Gunicorn (`gunicorn "app:create_app()"`) and MySQL as the report describes.
