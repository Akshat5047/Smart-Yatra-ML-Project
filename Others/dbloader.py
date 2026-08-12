import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from pathlib import Path
from dotenv import load_dotenv

def migrate():
    BASE_DIR = Path(__file__).resolve().parent.parent
    load_dotenv(BASE_DIR / "App" / "Backend" / ".env")

    db_url = os.getenv("SUPABASE_DB_URL")

    if not db_url:
        print("Error: SUPABASE_DB_URL not found in .env!")
        return

    print("Loaded .env successfully!")
    print("DB URL:", db_url[:30] + "...")
    
    db_url = os.getenv("SUPABASE_DB_URL")
    
    if not db_url:
        print("Error: SUPABASE_DB_URL not found in .env!")
        return

    print("Connecting to Supabase PostgreSQL...")
    engine = create_engine(db_url)
    
    with engine.connect() as conn:
        print("Connected successfully! Testing query:", conn.execute(text("SELECT 1")).fetchall())

    # Map of tables to upload
    datasets = {
    "climate_dataset": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\Climate_Dataset_Final.csv",

    "crowd_data": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\crowd_data.csv",

    "trip_budget_prediction": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\trip_budget_prediction_dataset.csv",
    
    "other_spots": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\other spots.csv",
    
    "spot_visitors": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\spot_visitors.csv",
    
    "transport_mode_dataset": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\transport_mode_dataset.csv",
    
    "accommodations": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\accommodations.csv",
    
    "nearby_amenities": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\nearby_amenities.csv",
    
    "festivals_geocoded": r"C:\Users\AKSHAT\OneDrive\Desktop\ML Project Batch-1\Data\etl\load\festivals_geocoded.csv"
}

    print("\nStarting automated migration to Supabase...\n")

    for table_name, csv_path in datasets.items():
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            # Sanitize column names
            df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
            
            # Ensure id column exists for primary key
            if "id" not in df.columns:
                df.insert(0, "id", range(1, len(df) + 1))
            
            df.to_sql(table_name, engine, if_exists="replace", index=False)
            
            # Add Primary Key constraint in PostgreSQL
            with engine.connect() as conn:
                trans = conn.begin()
                try:
                    conn.execute(text(f'ALTER TABLE "{table_name}" ADD PRIMARY KEY (id);'))
                    trans.commit()
                except Exception as e:
                    trans.rollback()
            
            print(f"  OK: Uploaded table '{table_name}' ({len(df)} rows)")
        else:
            print(f"  Warning: File not found {csv_path}")


    # Update .streamlit/secrets.toml
    streamlit_dir = os.path.join("App", "frontend", ".streamlit")
    os.makedirs(streamlit_dir, exist_ok=True)
    secrets_path = os.path.join(streamlit_dir, "secrets.toml")
    
    secrets_content = f"""[connections.db]
url = "{db_url}"
"""
    with open(secrets_path, "w", encoding="utf-8") as f:
        f.write(secrets_content)
    print(f"\nOK: Updated Streamlit database secrets in {secrets_path}")

    print("\nMigration completed successfully! All 9 tables are live in Supabase PostgreSQL.")

if __name__ == "__main__":
    migrate()