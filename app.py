"""app.py - Flask web application and REST API for the House Price Predictor."""
import os
import secrets
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from functools import wraps

import pandas as pd
from flask import (Flask, abort, flash, g, jsonify, redirect, render_template,
                   request, session, url_for)
from werkzeug.security import check_password_hash, generate_password_hash

import predict_service as ps
from unit_conversion import UNIT_LABELS, SQFT_PER_UNIT

BASE = os.path.dirname(os.path.abspath(__file__))
REQUIRED_COLS = ["district", "area_value", "area_unit", "road_width_ft", "bedrooms",
                 "bathrooms", "floors", "building_age", "price_npr"]


def create_app(db_path=None, testing=False):
    app = Flask(__name__)
    os.makedirs(os.path.join(BASE, "instance"), exist_ok=True)
    app.config["DATABASE"] = db_path or os.path.join(BASE, "instance", "hpp.db")
    app.config["TESTING"] = testing
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get("HPP_HTTPS", "0") == "1"
    app.secret_key = _secret_key()

    # ---------------------------------------------------------------- database
    def get_db():
        if "db" not in g:
            g.db = sqlite3.connect(app.config["DATABASE"])
            g.db.row_factory = sqlite3.Row
            g.db.execute("PRAGMA foreign_keys = ON")
        return g.db

    @app.teardown_appcontext
    def close_db(_):
        db = g.pop("db", None)
        if db is not None:
            db.close()

    def init_db():
        db = sqlite3.connect(app.config["DATABASE"])
        with open(os.path.join(BASE, "schema.sql")) as f:
            db.executescript(f.read())
        db.commit(); db.close()

    init_db()
    app.init_db = init_db

    # ------------------------------------------------------------- auth helpers
    @app.before_request
    def load_user():
        g.user = None
        uid = session.get("user_id")
        if uid is not None:
            g.user = get_db().execute("SELECT * FROM user WHERE user_id = ?", (uid,)).fetchone()

    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    @app.before_request
    def check_csrf():
        if request.method == "POST" and not request.path.startswith("/api/") and not app.config["TESTING"]:
            if not secrets.compare_digest(request.form.get("csrf", ""), session.get("csrf", "x")):
                abort(400, "Invalid or missing CSRF token")

    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.context_processor
    def inject():
        return {"user": g.get("user"), "unit_labels": UNIT_LABELS, "districts": ps.META["districts"],
                "meta": ps.META}

    def login_required(f):
        @wraps(f)
        def wrapper(*a, **kw):
            if g.user is None:
                return redirect(url_for("login", next=request.path))
            return f(*a, **kw)
        return wrapper

    def admin_required(f):
        @wraps(f)
        def wrapper(*a, **kw):
            if g.user is None:
                return redirect(url_for("login", next=request.path))
            if g.user["role"] != "admin":
                abort(403, "Access denied")
            return f(*a, **kw)
        return wrapper

    # -------------------------------------------------------------- validation
    def parse_property(data):
        """Validate raw input (form or JSON). Returns (values, errors)."""
        errors = []
        fields = ["district", "area_value", "area_unit", "road_width_ft",
                  "bedrooms", "bathrooms", "floors", "building_age"]
        if any(str(data.get(k, "")).strip() == "" for k in fields):
            return None, ["All fields are required"]
        v = {}
        v["district"] = str(data["district"]).strip()
        if v["district"] not in ps.META["districts"]:
            errors.append("Unknown district")
        v["area_unit"] = str(data["area_unit"]).strip().lower()
        if v["area_unit"] not in SQFT_PER_UNIT:
            errors.append("Unsupported area unit")
        try:
            v["area_value"] = float(data["area_value"])
            v["road_width_ft"] = float(data["road_width_ft"])
            v["floors"] = float(data["floors"])
            v["bedrooms"] = int(float(data["bedrooms"]))
            v["bathrooms"] = int(float(data["bathrooms"]))
            v["building_age"] = int(float(data["building_age"]))
        except (ValueError, TypeError, OverflowError):
            return None, ["All numeric fields must be valid numbers"]
        if not all(x == x and abs(x) != float("inf") for x in
                   (v["area_value"], v["road_width_ft"], v["floors"])):
            return None, ["All numeric fields must be valid numbers"]
        if v["area_value"] <= 0:
            errors.append("Area must be greater than zero")
        elif v["area_value"] * SQFT_PER_UNIT.get(v["area_unit"], 1) > 1_000_000:
            errors.append("Area is unrealistically large")
        if not 1 <= v["road_width_ft"] <= 200:
            errors.append("Road width must be between 1 and 200 feet")
        if not 0 <= v["bedrooms"] <= 20:
            errors.append("Bedrooms must be between 0 and 20")
        if not 0 <= v["bathrooms"] <= 20:
            errors.append("Bathrooms must be between 0 and 20")
        if not 1 <= v["floors"] <= 10:
            errors.append("Floors must be between 1 and 10")
        if not 0 <= v["building_age"] <= 150:
            errors.append("Building age must be between 0 and 150 years")
        return (v, errors) if not errors else (None, errors)

    def run_prediction(v):
        res = ps.predict_price(v["district"], v["area_value"], v["area_unit"], v["road_width_ft"],
                               v["bedrooms"], v["bathrooms"], v["floors"], v["building_age"])
        db = get_db()
        cur = db.execute("INSERT INTO query_log (user_id, timestamp) VALUES (?, ?)",
                         (g.user["user_id"] if g.user else None,
                          datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
        qid = cur.lastrowid
        db.execute("""INSERT INTO property_feature (query_id, district, unit_type, raw_area_val, area_sqft,
                      road_width_ft, bedrooms, bathrooms, floors, building_age)
                      VALUES (?,?,?,?,?,?,?,?,?,?)""",
                   (qid, v["district"], v["area_unit"], v["area_value"], res["area_sqft"],
                    v["road_width_ft"], v["bedrooms"], v["bathrooms"], v["floors"], v["building_age"]))
        db.execute("""INSERT INTO prediction_result (query_id, predicted_npr, price_min_npr, price_max_npr)
                      VALUES (?,?,?,?)""",
                   (qid, res["predicted_npr"], res["price_min_npr"], res["price_max_npr"]))
        db.commit()
        return res

    # ------------------------------------------------------------------ routes
    @app.route("/")
    def index():
        return render_template("index.html", form={}, errors=[])

    @app.route("/predict", methods=["POST"])
    def predict():
        v, errors = parse_property(request.form)
        if errors:
            return render_template("index.html", form=request.form, errors=errors), 400
        return render_template("result.html", res=run_prediction(v), v=v)

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify(error="Request body must be a JSON object"), 400
        v, errors = parse_property(data)
        if errors:
            return jsonify(error=errors[0], errors=errors), 400
        return jsonify(run_prediction(v))

    @app.route("/register", methods=["GET", "POST"])
    def register():
        if request.method == "POST":
            name = request.form.get("full_name", "").strip()
            email = request.form.get("email", "").strip().lower()
            pw = request.form.get("password", "")
            role = request.form.get("role", "buyer")
            role = role if role in ("buyer", "agent") else "buyer"   # admin cannot be self-assigned
            err = None
            if not name or not email or not pw:
                err = "All fields are required"
            elif "@" not in email or len(email) > 150:
                err = "Enter a valid email address"
            elif len(pw) < 8:
                err = "Password must be at least 8 characters"
            elif get_db().execute("SELECT 1 FROM user WHERE email = ?", (email,)).fetchone():
                err = "Email already registered"
            if err:
                return render_template("register.html", error=err, form=request.form), 400
            get_db().execute("INSERT INTO user (full_name,email,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                             (name[:100], email, generate_password_hash(pw), role,
                              datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
            get_db().commit()
            flash("Account created. Please log in.", "ok")
            return redirect(url_for("login"))
        return render_template("register.html", error=None, form={})

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            pw = request.form.get("password", "")
            row = get_db().execute("SELECT * FROM user WHERE email = ?", (email,)).fetchone()
            if row is None or not check_password_hash(row["password_hash"], pw):
                return render_template("login.html", error="Invalid email or password"), 401
            session.clear()
            session["user_id"] = row["user_id"]
            nxt = request.args.get("next", "")
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("index"))
        return render_template("login.html", error=None)

    @app.route("/logout", methods=["POST"])
    def logout():
        session.clear()
        return redirect(url_for("index"))

    @app.route("/history")
    @login_required
    def history():
        rows = get_db().execute("""
            SELECT q.query_id, q.timestamp, f.district, f.raw_area_val, f.unit_type, f.area_sqft,
                   f.road_width_ft, f.bedrooms, f.bathrooms, f.floors, f.building_age,
                   r.predicted_npr, r.price_min_npr, r.price_max_npr
            FROM query_log q JOIN property_feature f ON f.query_id = q.query_id
            JOIN prediction_result r ON r.query_id = q.query_id
            WHERE q.user_id = ? ORDER BY q.query_id DESC""", (g.user["user_id"],)).fetchall()
        return render_template("history.html", rows=rows, fmt=ps.format_npr)

    # ------------------------------------------------------------------- admin
    @app.route("/admin")
    @admin_required
    def admin():
        db = get_db()
        users = db.execute("SELECT user_id, full_name, email, role, created_at FROM user ORDER BY user_id").fetchall()
        n_queries = db.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
        return render_template("admin.html", users=users, n_queries=n_queries)

    @app.route("/admin/upload", methods=["POST"])
    @admin_required
    def admin_upload():
        f = request.files.get("dataset")
        if not f or not f.filename.lower().endswith(".csv"):
            flash("Choose a .csv file.", "err")
            return redirect(url_for("admin"))
        try:
            df = pd.read_csv(f)
        except Exception:
            flash("Could not read that file as CSV.", "err")
            return redirect(url_for("admin"))
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        if missing:
            flash("Missing columns: " + ", ".join(missing), "err")
            return redirect(url_for("admin"))
        if len(df) < 200:
            flash("At least 200 rows are needed to train a reliable model.", "err")
            return redirect(url_for("admin"))
        path = os.path.join(BASE, "data", "listings.csv")
        if os.path.exists(path):
            os.replace(path, path + ".bak")
        df.to_csv(path, index=False)
        flash(f"Dataset uploaded ({len(df)} rows). Now retrain the model.", "ok")
        return redirect(url_for("admin"))

    @app.route("/admin/retrain", methods=["POST"])
    @admin_required
    def admin_retrain():
        proc = subprocess.run([sys.executable, "train.py"], cwd=BASE, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            flash("Retraining failed: " + proc.stderr.strip().splitlines()[-1][:200], "err")
        else:
            ps.load()          # hot-reload the new model without a restart
            m = ps.META
            flash(f"Retrained. Selected {m['selected_model']} "
                  f"(test R² {m['results'][m['selected_model']]['test_r2']:.3f}).", "ok")
        return redirect(url_for("admin"))

    @app.route("/admin/role", methods=["POST"])
    @admin_required
    def admin_role():
        uid = request.form.get("user_id", type=int)
        role = request.form.get("role")
        if role in ("buyer", "agent", "admin") and uid and uid != g.user["user_id"]:
            get_db().execute("UPDATE user SET role = ? WHERE user_id = ?", (role, uid))
            get_db().commit()
        return redirect(url_for("admin"))

    @app.route("/admin/delete-user", methods=["POST"])
    @admin_required
    def admin_delete_user():
        uid = request.form.get("user_id", type=int)
        if uid and uid != g.user["user_id"]:
            db = get_db()
            db.execute("UPDATE query_log SET user_id = NULL WHERE user_id = ?", (uid,))
            db.execute("DELETE FROM user WHERE user_id = ?", (uid,))
            db.commit()
        return redirect(url_for("admin"))

    @app.route("/model")
    def model_info():
        return render_template("model.html")

    @app.errorhandler(403)
    @app.errorhandler(400)
    @app.errorhandler(404)
    @app.errorhandler(413)
    def http_error(e):
        if request.path.startswith("/api/"):
            return jsonify(error=e.description), e.code
        return render_template("error.html", code=e.code, message=e.description), e.code

    return app


def _secret_key():
    if os.environ.get("HPP_SECRET_KEY"):
        return os.environ["HPP_SECRET_KEY"]
    path = os.path.join(BASE, "instance", "secret_key")
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
        os.chmod(path, 0o600)
    with open(path) as f:
        return f.read().strip()


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)), threaded=True)
