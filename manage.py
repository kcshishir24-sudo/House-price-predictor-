"""manage.py - create an administrator account.  Usage: python3 manage.py create-admin EMAIL 'Full Name'"""
import getpass
import sqlite3
import sys
from datetime import datetime, timezone

from werkzeug.security import generate_password_hash

from app import create_app

if len(sys.argv) >= 4 and sys.argv[1] == "create-admin":
    email, name = sys.argv[2].lower(), sys.argv[3]
    pw = getpass.getpass("Password (min 8 chars): ")
    if len(pw) < 8:
        sys.exit("Password too short")
    app = create_app()
    db = sqlite3.connect(app.config["DATABASE"])
    try:
        db.execute("INSERT INTO user (full_name,email,password_hash,role,created_at) VALUES (?,?,?,?,?)",
                   (name, email, generate_password_hash(pw), "admin",
                    datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")))
        db.commit()
        print("Admin created:", email)
    except sqlite3.IntegrityError:
        sys.exit("That email already exists")
else:
    sys.exit(__doc__)
