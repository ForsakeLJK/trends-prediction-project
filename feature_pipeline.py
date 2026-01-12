import hopsworks
import pandas as pd
from scraper import FirehoseScraper
from feature_transform import FeatureConfig, build_feature_table, select_feature_store_columns, calculate_trend_features
from scrape_data_transform import build_csv_data
import os
from datetime import datetime
import re

if __name__ == "__main__":
    time_var = datetime.now().strftime('%Y%m%d_%H%M%S')

    jsonl_folder_path = "data"
    os.makedirs(jsonl_folder_path, exist_ok=True)
    output_file_name = f"{jsonl_folder_path}/bluesky_posts_{time_var}.jsonl"
    print("starting data scrape...")
    # scrape data for 5 minutes
    archiver = FirehoseScraper(output_file=output_file_name, verbose=False, num_workers=4)
    archiver.start_collection(duration_seconds=300, post_limit=None)

    print("transforming scraped data into feature csv...")
    build_csv_data(input_path = jsonl_folder_path, output_csv = "feature_data.csv")

    df = pd.read_csv('train_data/feature_data.csv',
                        dtype={
            "Trend": "string",
            "source_file": "string"
        },
                    parse_dates=["time_stamp"]
    )
    
    print("building feature table...")
    english_pattern = re.compile(r'^#[A-Za-z0-9_]+$')
    df_en = df[df["Trend"].str.match(english_pattern)]
    df_en.head()
    
    cfg = FeatureConfig(bucket_minutes=5, rolling_windows=(3, 12))
    features_df = build_feature_table(df_en, cfg)
    features_df = select_feature_store_columns(features_df)
    print(features_df.head())
    print(features_df.info())
    
    project = hopsworks.login(api_key_value="CNgidWirCRs6p66s.GjTJhC5kmU5qZnGvt4QR5VjDiwU5XgKZeGtjvPojyxFhkAzxgOlEtDxCaYFnh0Ge")
    fs = project.get_feature_store()
    trends_fg = fs.get_feature_group("trends_feature_store", version=2)
    
    trends_df = trends_fg.read()
    trends_df["trend"] = trends_df["trend"].astype("string")
    features_df["ts"] = (
        features_df["ts"]
            .dt.tz_localize("UTC")      # make it timezone-aware
            .dt.tz_convert("Etc/UTC")   # normalize to Etc/UTC (same offset, different name)
            .dt.as_unit("us")          # convert ns → µs
    )
    
    combined_df = pd.concat([features_df, trends_df]).reset_index(drop=True)
    print(combined_df.info())

    final_df = calculate_trend_features(combined_df)
    print(final_df.info())
    print(final_df.head())
    
    print("writing features to feature store...")
    trends_fg.insert(final_df, operation="insert")