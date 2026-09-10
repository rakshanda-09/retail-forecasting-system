from sqlalchemy import create_engine, text
import pandas as pd
import streamlit as st
import urllib.parse

class DatabaseManager:
    def __init__(self):
        # FIXED Connection string
        password = urllib.parse.quote_plus("rakshanda@111")
        self.connection_string = f'mysql+mysqlconnector://root:{password}@localhost:3306/retail_forecasting'
        
        try:
            self.engine = create_engine(
                self.connection_string,
                pool_pre_ping=True,
                pool_recycle=3600,
                echo=False
            )
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            print("✅ MySQL Connected Successfully!")
            st.success("✅ Database Connected!")
        except Exception as e:
            print(f"❌ Database Connection Failed: {e}")
            st.error(f"❌ Database Error: {e}")
            st.info("💡 Using local storage fallback")
            self.engine = None
 
    def ensure_database_exists(self):
        if self.engine:
            try:
                with self.engine.connect() as conn:
                    conn.execute(text("CREATE DATABASE IF NOT EXISTS retail_forecasting"))
                    conn.execute(text("USE retail_forecasting"))
                print("✅ Database ready!")
            except Exception as e:
                print(f"❌ Database setup error: {e}")
 
    def create_tables(self):
        if self.engine:
            self.ensure_database_exists()
            print("✅ Tables ready!")
 
    def save_dataframe(self, df, table_name='sales_data'):
        if self.engine and len(df) > 0:
            print(f"💾 Saving {len(df)} rows to MySQL...")
            try:
                # ✅ FIXED: ONLY clean problematic column names, preserve originals
                df_copy = df.copy()
                protected_columns = ['Units Sold', 'Store ID', 'Product ID', 'Date']  # Keep these exact
                
                # Only clean columns that NEED cleaning (spaces, slashes)
                for col in df_copy.columns:
                    if col not in protected_columns and (' ' in col or '/' in col):
                        clean_name = col.replace(' ', '_').replace('/', '_')
                        df_copy.rename(columns={col: clean_name}, inplace=True)
                
                df_copy.to_sql(
                    table_name, 
                    self.engine, 
                    if_exists='replace',
                    index=False, 
                    chunksize=1000,
                    method='multi'
                )
                print("✅ Database Save Complete!")
                st.success(f"💾 Saved {len(df)} rows to MySQL!")
                return True
            except Exception as e:
                print(f"❌ Save Error: {e}")
                st.error(f"❌ Save failed: {e}")
                return False
        else:
            st.warning("⚠️ Using local storage")
            return False
