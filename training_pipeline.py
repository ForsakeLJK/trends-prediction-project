import hopsworks
from train import train_ridge_next_count_model
import os

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
    print(features_df.info())

    model_dir = "trends_model"
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    model_path = model_dir + "/trend_nextcount_ridge.joblib"
    
    result = train_ridge_next_count_model(features_df, model_path=model_path)

    print(result["metrics"])
    print(result["preview"])
    
    tr_model = mr.python.create_model(
        name="trend_nextcount_ridge",
        metrics=result["metrics"],
        description="Ridge regression model to predict next count of trends",
        feature_view=feature_view,
    )
    
    tr_model.save(model_dir)