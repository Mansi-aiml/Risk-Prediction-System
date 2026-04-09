import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# DB credentials (can come from environment variables)
DB_HOST = os.getenv("DB_HOST", "49.50.69.230")
DB_USER = os.getenv("DB_USER", "phs_demo")
DB_PASSWORD = os.getenv("DB_PASSWORD", "K1P5m3sCv!%")
DB_NAME = os.getenv("DB_NAME", "pehs_ai")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_TABLE = os.getenv("DB_TABLE", "incident_prediction_data")  # fallback table

# ------------------------
# SQLAlchemy connection
# ------------------------
try:
    engine = create_engine(
        f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    # Read table into pandas DataFrame
    df = pd.read_sql(f"SELECT * FROM {DB_TABLE}", engine)

    print(df.head())
    print(f"\nTotal Records : {len(df)}")
    print(f"Columns       : {list(df.columns)}")

except Exception as e:
    print(f"Error connecting to database: {e}")