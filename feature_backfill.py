import hopsworks
import pandas as pd
import re
from feature_transform import FeatureConfig, build_feature_table, select_feature_store_columns
import os
from dotenv import load_dotenv

if __name__ == "__main__":
    load_dotenv()
    mode = os.getenv("RUNNING_MODE")
    
    df = pd.read_csv('./historical_data.csv',
                        dtype={
            "Trend": "string",
            "source_file": "string"
        },
                    parse_dates=["time_stamp"]
    )
    
    english_pattern = re.compile(r'^#[A-Za-z0-9_]+$')
    df_en = df[df["Trend"].str.match(english_pattern)]
    df_en.head()

    cfg = FeatureConfig(bucket_minutes=5, rolling_windows=(3, 12))
    features_df = build_feature_table(df_en, cfg)
    features_df = select_feature_store_columns(features_df)
    
    if mode == "HOPSWORKS":
        project = hopsworks.login(api_key_value="CNgidWirCRs6p66s.GjTJhC5kmU5qZnGvt4QR5VjDiwU5XgKZeGtjvPojyxFhkAzxgOlEtDxCaYFnh0Ge")
        fs = project.get_feature_store() 
        feature_group = fs.get_or_create_feature_group(
            name="trends_feature_store",
            version=2,
            description="Feature store for trends prediction project",
            event_time ="ts",
            primary_key=["ts", "trend"]
        )
        
        feature_group.insert(features_df, wait=True)
    elif mode == "LOCAL":
        os.makedirs('feature_store', exist_ok=True)
        features_df.to_csv('feature_store/trends_feature_store_v2.csv', index=False)