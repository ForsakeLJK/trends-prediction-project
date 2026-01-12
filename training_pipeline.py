import hopsworks
from train import train_ridge_next_count_model
import os
from dotenv import load_dotenv
import pandas as pd

SELECT_FEATURES = [
    "trend",
    "ts",
    "post_count",
    "count_lag1",
    "delta_count",
    "share_of_attention",
    "rank_in_snapshot",
    "tokens_per_post",
    "hour_sin",
    "hour_cos",
    "label_count_next",
]

if __name__ == "__main__":
    load_dotenv()
    mode = os.getenv("RUNNING_MODE")
    
    if mode == "HOPSWORKS":
        project = hopsworks.login(api_key_value="CNgidWirCRs6p66s.GjTJhC5kmU5qZnGvt4QR5VjDiwU5XgKZeGtjvPojyxFhkAzxgOlEtDxCaYFnh0Ge")
        fs = project.get_feature_store()
        mr = project.get_model_registry()
        
        trends_fg = fs.get_feature_group("trends_feature_store", version=2)
        selected_features = trends_fg.select(SELECT_FEATURES)

        feature_view = fs.get_or_create_feature_view(
            name="trends_feature_view",
            version=1,
            description="Feature view for trends prediction project",
            query=selected_features,
        )

        features_df, _ = feature_view.training_data()
    elif mode == "LOCAL":
        features_df = pd.read_csv('feature_store/trends_feature_store_v2.csv',
                        dtype={
            "trend": "string",
            # "post_count": "Int64",
            # "count_lag1": "Int64",
            # "delta_count": "Int64",
            # "share_of_attention": "float64",
            # "rank_in_snapshot": "Int64",
            # "tokens_per_post": "float64",
            # "hour_sin": "float64",
            # "hour_cos": "float64",
            # "label_count_next": "Int64"
        },
                    parse_dates=["ts"]
        )
    print(features_df.info())

    model_dir = "trends_model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + "/trend_nextcount_ridge.joblib"
    
    result = train_ridge_next_count_model(features_df, model_path=model_path)

    print(result["metrics"])
    print(result["preview"])
    
    if mode == "HOPSWORKS":
        tr_model = mr.python.create_model(
            name="trend_nextcount_ridge",
            metrics=result["metrics"],
            description="Ridge regression model to predict next count of trends",
            feature_view=feature_view,
        )
        
        tr_model.save(model_dir)