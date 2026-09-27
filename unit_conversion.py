"""unit_conversion.py - Nepalese land unit to square feet converter (Report Table 4.2)."""

SQFT_PER_UNIT = {
    "sqft": 1.0,
    "sqm": 10.7639,
    # Hill / Kathmandu Valley system
    "ropani": 5476.0,      # 1 Ropani = 16 Aana
    "aana": 342.25,        # 1 Aana = 4 Paisa
    "paisa": 85.5625,      # 1 Paisa = 4 Daam
    "daam": 21.390625,
    # Terai system
    "bigha": 72900.0,      # 1 Bigha = 20 Katha
    "katha": 3645.0,       # 1 Katha = 20 Dhur
    "dhur": 182.25,
}

UNIT_LABELS = {
    "aana": "Aana", "ropani": "Ropani", "paisa": "Paisa", "daam": "Daam",
    "katha": "Katha", "bigha": "Bigha", "dhur": "Dhur",
    "sqft": "Square feet", "sqm": "Square metres",
}


def to_sqft(value, unit):
    unit = str(unit).strip().lower()
    if unit not in SQFT_PER_UNIT:
        raise ValueError(f"Unsupported unit: {unit}")
    if value is None or value <= 0:
        raise ValueError("Area must be a positive number")
    return round(value * SQFT_PER_UNIT[unit], 2)
