import hopsworks
import joblib
from sklearn.pipeline import Pipeline
import train
import numpy as np
import json
import pandas as pd
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt


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

def calculate_top_k_hit_rate(df, k=5):
    df = df.copy()

    rows = []
    for ts, g in df.groupby('ts', sort=True):
        actual_top_k = set(g.nlargest(k, 'post_count')['trend'])
        predicted_top_k = set(g.nlargest(k, 'predicted_post_count')['trend'])
        hit_rate = len(actual_top_k & predicted_top_k) / k
        rows.append((ts, hit_rate))

    return pd.DataFrame(rows, columns=['ts', 'hit_rate'])


if __name__ == "__main__":
    load_dotenv()
    mode = os.getenv("RUNNING_MODE")
    
    if mode == "HOPSWORKS":
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
    elif mode == "LOCAL":
        model_dir = "trends_model"
        model = joblib.load(f"{model_dir}/trend_nextcount_ridge.joblib")
        print("Model loaded from local directory.")
    
    pipeline = model['pipeline']
    
    # get the latest 5 minutes of data for inference   
    if mode == "HOPSWORKS":
        trends_fg = fs.get_feature_group("trends_feature_store", version=2)
        trends_df = trends_fg.read()
    elif mode == "LOCAL":
        trends_df = pd.read_csv('feature_store/trends_feature_store_v2.csv',
                        dtype={
            "trend": "string"
        },
                    parse_dates=["ts"]
        )    
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
    pred_topk = inference_df.sort_values(by='predicted_count_next', ascending=False).head(5)
    print("Top 5 predicted trends for next time window:")
    print(pred_topk[['trend', 'post_count', 'predicted_count_next']])
    
    # insert predicted data for monitoring
    monitor_df = inference_df[['ts', 'trend', 'predicted_count_next']]
    monitor_df = monitor_df.rename(columns={'predicted_count_next': 'predicted_post_count'})
    
    if mode == "hopsworks":
        # backfill predicted data for monitoring  (save predicted data to a new feature group, e.g., trend_nextcount_predictions)
        pred_fg = fs.get_or_create_feature_group(
            name="trends_predictions_store",
            version=1,
            description="Feature store for monitoring trends prediction project",
            event_time ="ts",
            primary_key=["ts", "trend"]
        )
        pred_fg.insert(monitor_df, wait=True)
        pred_hist_df = pred_fg.read()
    elif mode == "LOCAL":
        os.makedirs("feature_store", exist_ok=True)
        # if not exist, create the csv file
        if not os.path.exists('feature_store/trends_predictions_store_v1.csv'):
            monitor_df.to_csv('feature_store/trends_predictions_store_v1.csv', index=False)
        else:
            monitor_df.to_csv('feature_store/trends_predictions_store_v1.csv', index=False, mode='a', header=False)
        
        pred_hist_df = pd.read_csv('feature_store/trends_predictions_store_v1.csv',
                        dtype={
            "trend": "string"
        },
                    parse_dates=["ts"]
        )
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
    # use result_df to draw topk hit rate graph
    
    # Calculate the hit rate
    hit_rate_df = calculate_top_k_hit_rate(result_df, k=5)

    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(hit_rate_df['ts'], hit_rate_df['hit_rate'], marker='o', linestyle='-', color='#2ca02c')

    # Formatting
    plt.title('Top 5 Hit Rate Over Time', fontsize=14)
    plt.xlabel('Timestamp', fontsize=12)
    plt.ylabel('Hit Rate (Overlap %)', fontsize=12)
    plt.ylim(0, 1.1)  # Hit rate is between 0 and 1
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    output_dir = 'hindcast'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    plt.savefig(f'{output_dir}/top_5_hit_rate.png', dpi=300, bbox_inches='tight', transparent=False)
    # save data for the current real topk trends, and its predicted topk trends for comparison, and the next topk predictions
    # A. Current Real Top 5 (Actual leaders right now)
    latest_ts = result_df['ts'].max()
    current_actual_top5 = result_df[result_df['ts'] == latest_ts].nlargest(5, 'post_count')

    # B. Current Predicted Top 5 (What the model thought would be top 5 right now)
    current_predicted_top5 = result_df[result_df['ts'] == latest_ts].nlargest(5, 'predicted_post_count')

    # C. Next Window Top 5 (Future predictions)
    next_predicted_top5 = inference_df.nlargest(5, 'predicted_count_next')
    
    # 1. Prepare and standardize the dataframes (keep only 'trend' and 'count')
    data_to_save = {
        "current_actual_top5": (
            current_actual_top5[['trend', 'post_count']]
            .rename(columns={'post_count': 'count'})
            .to_dict(orient='records')
        ),
        "current_predicted_top5": (
            current_predicted_top5[['trend', 'predicted_post_count']]
            .rename(columns={'predicted_post_count': 'count'})
            .to_dict(orient='records')
        ),
        "next_predicted_top5": (
            next_predicted_top5[['trend', 'predicted_count_next']]
            .rename(columns={'predicted_count_next': 'count'})
            .to_dict(orient='records')
        )
    }
    
    print(data_to_save)

    # 2. Save to a JSON file
    with open('hindcast/trend_report.json', 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)

    print("Data successfully saved to trend_report.json")