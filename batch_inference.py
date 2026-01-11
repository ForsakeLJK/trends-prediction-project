import hopsworks
import joblib
from sklearn.pipeline import Pipeline
import train
import numpy as np
import json
import pandas as pd

BASELINE_8_FEATURES = [
    "post_count",
    "count_lag1",
    "delta_count",
    "share_of_attention",
    "rank_in_snapshot",
    "tokens_per_post",
    "hour_sin",
    "hour_cos",
]

if __name__ == "__main__":
    project = hopsworks.login(api_key_value="CNgidWirCRs6p66s.GjTJhC5kmU5qZnGvt4QR5VjDiwU5XgKZeGtjvPojyxFhkAzxgOlEtDxCaYFnh0Ge")
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    
    trends_model = mr.get_model(
        name="trend_nextcount_ridge",
        version=3,
    )
    
    fv = trends_model.get_feature_view()
    
    saved_model_dir = trends_model.download()
    print(f"Model downloaded to: {saved_model_dir}")
    
    model = joblib.load(f"{saved_model_dir}/trend_nextcount_ridge.joblib")
    pipeline = model['pipeline']
    
    
    # get the latest 5 minutes of data for inference   
    trends_fg = fs.get_feature_group("trends_feature_store", version=2)
    trends_df = trends_fg.read()
    latest_ts = trends_df['ts'].max()
    inference_df = trends_df[trends_df['ts'] == latest_ts]
    print(f"Inference data timestamp: {latest_ts}")
    print(inference_df.info())
     
    # select features to predict
    X_inference = inference_df[BASELINE_8_FEATURES] 
     
    # predict
    predictions = pipeline.predict(X_inference)
    predictions = train._inverse_transform_target(predictions, mode="log1p")
    predictions = np.maximum(predictions, 0.0)
    predictions = np.rint(predictions)

    inference_df['predicted_count_next'] = predictions
    print(inference_df[['trend', 'post_count', 'predicted_count_next']])
    # rank top 5
    topk = inference_df.sort_values(by='predicted_count_next', ascending=False).head(5)
    print("Top 5 predicted trends for next time window:")
    print(topk[['trend', 'post_count', 'predicted_count_next']])
    
    # backfill predicted data for monitoring  (save predicted data to a new feature group, e.g., trend_nextcount_predictions)
    pred_fg = fs.get_or_create_feature_group(
        name="trends_predictions_store",
        version=1,
        description="Feature store for monitoring trends prediction project",
        event_time ="ts",
        primary_key=["ts", "trend"]
    )
    
    # insert predicted data for monitoring
    monitor_df = inference_df[['ts', 'trend', 'predicted_count_next']]
    monitor_df = monitor_df.rename(columns={'predicted_count_next': 'predicted_post_count'})
    pred_fg.insert(monitor_df, wait=True)
    
    pred_hist_df = pred_fg.read()
    # entry in pred_hits_df, the predicted_post_count should be for the next time window after ts
    # so we can compare with the actual post_count when it becomes available.
    # so we join pred_hist_df with trends_df on ts + 5 minutes and trend to get actual post_count for evaluation later.
    # 1. Ensure both dataframes are sorted chronologically
    pred_hist_df = pred_hist_df.sort_values(['trend', 'ts'])
    trends_df = trends_df.sort_values(['trend', 'ts'])

    # 2. In the actual data, create a 'match_ts' column 
    # This represents the timestamp when the prediction for THIS row would have been made.
    # shift(1) takes the previous timestamp in the sequence for that specific trend.
    trends_df['match_ts'] = trends_df.groupby('trend')['ts'].shift(1)

    # 3. Perform an inner join
    # We match pred_hist_df's 'ts' with trends_df's 'match_ts'
    result_df = pd.merge(
        pred_hist_df,
        trends_df,
        left_on=['trend', 'ts'],
        right_on=['trend', 'match_ts'],
        suffixes=('_pred_origin', '') # 'ts' will now be the actual observation time
    )

    # 4. Cleanup: Drop the helper column and the original prediction timestamp
    # result_df now contains the actual 'ts', the trend, the prediction, and the actuals.
    result_df = result_df.drop(columns=['match_ts', 'ts_pred_origin'])
    
    print("Joined predictions with actuals for evaluation:")
    print(result_df[['ts', 'trend', 'predicted_post_count', 'post_count']].head())
    # update the topk figure with new predictions
    
    # save the figure to directory for github page publishing
    
    # save data for the current real topk trends, and its predicted topk trends for comparison, and the next topk predictions
    # to a json file for github page publishing