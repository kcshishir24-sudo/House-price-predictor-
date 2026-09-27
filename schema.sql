-- Schema from Report Section 3.5.7 (SQLite dialect; MySQL uses the same tables).
CREATE TABLE IF NOT EXISTS user (
    user_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name     VARCHAR(100) NOT NULL,
    email         VARCHAR(150) NOT NULL UNIQUE,
    password_hash VARCHAR(255) NOT NULL,
    role          VARCHAR(20)  NOT NULL DEFAULT 'buyer' CHECK (role IN ('buyer','agent','admin')),
    created_at    DATETIME     NOT NULL
);
CREATE TABLE IF NOT EXISTS query_log (
    query_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   INTEGER REFERENCES user(user_id),   -- NULL for anonymous estimates
    timestamp DATETIME NOT NULL
);
CREATE TABLE IF NOT EXISTS property_feature (
    feature_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id      INTEGER NOT NULL REFERENCES query_log(query_id),
    district      VARCHAR(50) NOT NULL,
    unit_type     VARCHAR(20) NOT NULL,
    raw_area_val  FLOAT NOT NULL,
    area_sqft     FLOAT NOT NULL,
    road_width_ft FLOAT NOT NULL,
    bedrooms      INT   NOT NULL,
    bathrooms     INT   NOT NULL,
    floors        FLOAT NOT NULL,
    building_age  INT   NOT NULL
);
CREATE TABLE IF NOT EXISTS prediction_result (
    result_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    query_id      INTEGER NOT NULL REFERENCES query_log(query_id),
    predicted_npr DOUBLE NOT NULL,
    price_min_npr DOUBLE NOT NULL,
    price_max_npr DOUBLE NOT NULL
);
