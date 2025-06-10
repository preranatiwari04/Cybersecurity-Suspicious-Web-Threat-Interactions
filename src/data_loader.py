import pandas as pd
import requests
from pathlib import Path
import sqlite3
from config.settings import DATA_DIR, DATASET_URLS

class DataLoader:
    def __init__(self):
        self.data_dir = Path(DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)
    
    def download_dataset(self, dataset_name, url=None):
        """Download dataset from URL"""
        if url is None:
            url = DATASET_URLS.get(dataset_name)
        
        if not url:
            raise ValueError(f"No URL found for dataset: {dataset_name}")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            file_path = self.data_dir / f"{dataset_name}.csv"
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            print(f"Dataset downloaded: {file_path}")
            return file_path
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    
    def load_csv(self, file_path):
        """Load CSV file"""
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded {len(df)} records from {file_path}")
            return df
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def load_from_database(self, table_name, db_path=None):
        """Load data from SQLite database"""
        if db_path is None:
            db_path = self.data_dir / "cybersecurity.db"
        
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            conn.close()
            return df
        except Exception as e:
            print(f"Error loading from database: {e}")
            return None