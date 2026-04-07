import os
import pandas as pd
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


conn = mysql.connector.connect(
    host=os.getenv("DB_HOST", "localhost"),
    user=os.getenv("DB_USER", "root"),
    password=os.getenv("DB_PASSWORD", ""),
    database=os.getenv("DB_NAME", "wps_erp"),
)

query = f"SELECT * FROM {os.getenv('DB_TABLE', 'incident_prediction_data_new2')}"

df = pd.read_sql(query, conn)
conn.close()

print(df.head())
print(f"\nTotal Records : {len(df)}")
print(f"Columns       : {list(df.columns)}")