"""Automated tests: unit, integration and black-box (Report 3.7 / 4.3)."""
import os
import sqlite3
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import predict_service as ps
from app import create_app
from unit_conversion import SQFT_PER_UNIT, to_sqft
from werkzeug.security import generate_password_hash

VALID = dict(district="Kathmandu", area_value="4", area_unit="aana", road_width_ft="13",
             bedrooms="4", bathrooms="3", floors="2.5", building_age="5")


class TestUnitConversion(unittest.TestCase):
    def test_aana(self): self.assertEqual(to_sqft(1, "aana"), 342.25)
    def test_ropani(self): self.assertEqual(to_sqft(1, "ropani"), 5476)
    def test_katha(self): self.assertEqual(to_sqft(1, "katha"), 3645)
    def test_bigha(self): self.assertEqual(to_sqft(1, "bigha"), 72900)
    def test_dhur(self): self.assertEqual(to_sqft(1, "dhur"), 182.25)
    def test_paisa_daam(self):
        self.assertEqual(to_sqft(1, "paisa"), 85.56)
        self.assertEqual(to_sqft(1, "daam"), 21.39)
    def test_sqm(self): self.assertEqual(to_sqft(1, "sqm"), 10.76)
    def test_relationships(self):
        S = SQFT_PER_UNIT
        self.assertEqual(S["ropani"], 16 * S["aana"]); self.assertEqual(S["aana"], 4 * S["paisa"])
        self.assertEqual(S["paisa"], 4 * S["daam"]); self.assertEqual(S["bigha"], 20 * S["katha"])
        self.assertEqual(S["katha"], 20 * S["dhur"])
    def test_case_and_space(self): self.assertEqual(to_sqft(2, " Aana "), 684.5)
    def test_unknown_unit(self):
        with self.assertRaises(ValueError): to_sqft(1, "acre")
    def test_non_positive(self):
        for bad in (0, -5, None):
            with self.assertRaises(ValueError): to_sqft(bad, "aana")


class TestPrediction(unittest.TestCase):
    args = ("Kathmandu", 4, "aana", 13, 4, 3, 2.5, 5)

    def test_returns_positive_range(self):
        r = ps.predict_price(*self.args)
        self.assertGreater(r["predicted_npr"], 0)
        self.assertLess(r["price_min_npr"], r["predicted_npr"]); self.assertGreater(r["price_max_npr"], r["predicted_npr"])
    def test_deterministic(self):
        self.assertEqual(ps.predict_price(*self.args), ps.predict_price(*self.args))
    def test_ropani_equals_16_aana(self):
        a = ps.predict_price("Lalitpur", 1, "ropani", 13, 4, 3, 2, 5)["predicted_npr"]
        b = ps.predict_price("Lalitpur", 16, "aana", 13, 4, 3, 2, 5)["predicted_npr"]
        self.assertEqual(a, b)
    def test_bigger_plot_costs_more(self):
        small = ps.predict_price("Pokhara", 3, "aana", 13, 3, 2, 2, 5)["predicted_npr"]
        big = ps.predict_price("Pokhara", 10, "aana", 13, 3, 2, 2, 5)["predicted_npr"]
        self.assertGreater(big, small)
    def test_kathmandu_dearer_than_chitwan(self):
        ktm = ps.predict_price("Kathmandu", 5, "aana", 13, 3, 2, 2, 5)["predicted_npr"]
        cht = ps.predict_price("Chitwan", 5, "aana", 13, 3, 2, 2, 5)["predicted_npr"]
        self.assertGreater(ktm, cht)
    def test_format_npr(self):
        self.assertEqual(ps.format_npr(22_800_000), "Rs. 2.28 Crore")
        self.assertEqual(ps.format_npr(9_500_000), "Rs. 95.00 Lakh")
    def test_extreme_input_flagged(self):
        r = ps.predict_price("Kathmandu", 100, "ropani", 4, 4, 3, 2, 5)
        self.assertIn("area_sqft", r["outside_range"])
    def test_feature_order_matches_meta(self):
        self.assertEqual(ps.META["features"][0], "district")


class AppCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False); self.tmp.close()
        self.app = create_app(db_path=self.tmp.name, testing=True)
        self.c = self.app.test_client()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def register(self, email="a@example.com", pw="password123", client=None):
        return (client or self.c).post("/register", data=dict(full_name="Test User", email=email, password=pw, role="buyer"))

    def login(self, email="a@example.com", pw="password123", client=None):
        return (client or self.c).post("/login", data=dict(email=email, password=pw))

    def make_admin(self, email="admin@example.com"):
        db = sqlite3.connect(self.tmp.name)
        db.execute("INSERT INTO user (full_name,email,password_hash,role,created_at) VALUES (?,?,?,?,datetime('now'))",
                   ("Admin", email, generate_password_hash("adminpass1"), "admin"))
        db.commit(); db.close()


class TestValidationAndPredict(AppCase):
    def test_valid_form(self):
        r = self.c.post("/predict", data=VALID)
        self.assertEqual(r.status_code, 200); self.assertIn(b"Estimated price", r.data)
    def test_negative_area(self):
        r = self.c.post("/predict", data={**VALID, "area_value": "-3"})
        self.assertEqual(r.status_code, 400); self.assertIn(b"Area must be greater than zero", r.data)
    def test_zero_area(self):
        r = self.c.post("/predict", data={**VALID, "area_value": "0"})
        self.assertEqual(r.status_code, 400); self.assertIn(b"Area must be greater than zero", r.data)
    def test_blank_field(self):
        r = self.c.post("/predict", data={**VALID, "bedrooms": ""})
        self.assertEqual(r.status_code, 400); self.assertIn(b"All fields are required", r.data)
    def test_non_numeric(self):
        r = self.c.post("/predict", data={**VALID, "road_width_ft": "abc"})
        self.assertEqual(r.status_code, 400)
    def test_nan_inf_rejected(self):
        for bad in ("nan", "inf", "-inf"):
            self.assertEqual(self.c.post("/predict", data={**VALID, "area_value": bad}).status_code, 400)
    def test_bad_road_width(self):
        self.assertEqual(self.c.post("/predict", data={**VALID, "road_width_ft": "500"}).status_code, 400)
    def test_unknown_district_and_unit(self):
        self.assertEqual(self.c.post("/predict", data={**VALID, "district": "Mars"}).status_code, 400)
        self.assertEqual(self.c.post("/predict", data={**VALID, "area_unit": "acre"}).status_code, 400)
    def test_extreme_input_warns_not_errors(self):
        r = self.c.post("/predict", data={**VALID, "area_value": "100", "area_unit": "ropani", "road_width_ft": "4"})
        self.assertEqual(r.status_code, 200); self.assertIn(b"Warning", r.data)
    def test_identical_inputs_identical_outputs(self):
        a = self.c.post("/api/predict", json=VALID).get_json(); b = self.c.post("/api/predict", json=VALID).get_json()
        self.assertEqual(a["predicted_npr"], b["predicted_npr"])
    def test_api_ok_and_errors(self):
        r = self.c.post("/api/predict", json=VALID); self.assertEqual(r.status_code, 200)
        self.assertIn("range_display", r.get_json())
        self.assertEqual(self.c.post("/api/predict", json={**VALID, "area_value": -1}).status_code, 400)
        self.assertEqual(self.c.post("/api/predict", data="not json").status_code, 400)
        self.assertEqual(self.c.post("/api/predict", json=[1, 2]).status_code, 400)
    def test_xss_is_escaped(self):
        self.c.post("/register", data=dict(full_name="<script>alert(1)</script>", email="x@example.com", password="password123"))
        self.login("x@example.com")
        self.assertNotIn(b"<script>alert(1)</script>", self.c.get("/").data)


class TestAuth(AppCase):
    def test_register_and_login(self):
        self.assertEqual(self.register().status_code, 302)
        self.assertEqual(self.login().status_code, 302)
    def test_duplicate_email(self):
        self.register(); r = self.register()
        self.assertEqual(r.status_code, 400); self.assertIn(b"Email already registered", r.data)
    def test_wrong_password(self):
        self.register(); r = self.login(pw="wrongwrong")
        self.assertEqual(r.status_code, 401); self.assertIn(b"Invalid email or password", r.data)
        with self.c.session_transaction() as s: self.assertNotIn("user_id", s)
    def test_short_password(self):
        self.assertEqual(self.register(pw="short").status_code, 400)
    def test_password_is_hashed(self):
        self.register()
        h = sqlite3.connect(self.tmp.name).execute("SELECT password_hash FROM user").fetchone()[0]
        self.assertNotIn("password123", h)
    def test_sql_injection_login(self):
        r = self.c.post("/login", data=dict(email="' OR '1'='1' --", password="x"))
        self.assertEqual(r.status_code, 401); self.assertIn(b"Invalid email or password", r.data)
    def test_cannot_self_register_as_admin(self):
        self.c.post("/register", data=dict(full_name="Eve", email="eve@example.com", password="password123", role="admin"))
        role = sqlite3.connect(self.tmp.name).execute("SELECT role FROM user").fetchone()[0]
        self.assertNotEqual(role, "admin")
    def test_open_redirect_blocked(self):
        self.register()
        r = self.c.post("/login?next=//evil.com", data=dict(email="a@example.com", password="password123"))
        self.assertNotIn("evil.com", r.headers["Location"])
    def test_logout(self):
        self.register(); self.login(); self.c.post("/logout")
        self.assertEqual(self.c.get("/history").status_code, 302)


class TestHistoryAndAdmin(AppCase):
    def test_history_requires_login(self):
        self.assertEqual(self.c.get("/history").status_code, 302)
    def test_history_lists_user_queries_only(self):
        self.register(); self.login()
        self.c.post("/predict", data=VALID); self.c.post("/predict", data={**VALID, "district": "Pokhara"})
        other = self.app.test_client(); self.register("b@example.com", client=other); self.login("b@example.com", client=other)
        other.post("/predict", data={**VALID, "district": "Butwal"})
        body = self.c.get("/history").data
        self.assertIn(b"Kathmandu", body); self.assertIn(b"Pokhara", body); self.assertNotIn(b"Butwal", body)
    def test_query_rows_written_to_all_three_tables(self):
        self.c.post("/predict", data=VALID)
        db = sqlite3.connect(self.tmp.name)
        for t in ("query_log", "property_feature", "prediction_result"):
            self.assertEqual(db.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0], 1)
    def test_admin_forbidden_for_buyer(self):
        self.register(); self.login()
        self.assertEqual(self.c.get("/admin").status_code, 403)
        self.assertEqual(self.c.post("/admin/retrain").status_code, 403)
    def test_admin_anonymous_redirected(self):
        self.assertEqual(self.c.get("/admin").status_code, 302)
    def test_admin_can_open_dashboard(self):
        self.make_admin(); self.login("admin@example.com", "adminpass1")
        self.assertEqual(self.c.get("/admin").status_code, 200)
    def test_upload_rejects_bad_columns(self):
        import io
        self.make_admin(); self.login("admin@example.com", "adminpass1")
        r = self.c.post("/admin/upload", data={"dataset": (io.BytesIO(b"a,b\n1,2\n"), "x.csv")},
                        content_type="multipart/form-data", follow_redirects=True)
        self.assertIn(b"Missing columns", r.data)


class TestPerformance(AppCase):
    def test_response_time_under_two_seconds(self):
        t = time.perf_counter(); self.c.post("/api/predict", json=VALID)
        self.assertLess(time.perf_counter() - t, 2.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
