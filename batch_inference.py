import hopsworks
import joblib
from sklearn.pipeline import Pipeline


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
    # predict
    # rank top 5
    
    
    
    # backfill predicted data for hindcast  (save predicted data to a new feature group, e.g., trend_nextcount_predictions)
    
    # plot hindcast of topk hit rate over time
    # UI: output 1. current true topk trends  2. predict topk trends for this time window
    # 3. predicted topk trends for next time window 4. hindcast topk hit rate over time

    