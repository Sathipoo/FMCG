from data_ingestion import load_sample_data
from feature_engineering import add_time_features, encode_categoricals
from forecast_model import train_forecast_model
from replenishment import calculate_replenishment

def main():
    print(">>> Loading data...")
    df = load_sample_data()

    print(">>> Performing feature engineering...")
    df = add_time_features(df)
    df = encode_categoricals(df)

    print(">>> Training forecast model...")
    model = train_forecast_model(df)

    print(">>> Calculating replenishment suggestions...")
    replenishment_df = calculate_replenishment(df, model)

    print(">>> Sample output:")
    print(replenishment_df.head())

    replenishment_df.to_csv("output/replenishment_plan.csv", index=False)
    print(">>> Replenishment plan saved to output/replenishment_plan.csv")

if __name__ == "__main__":
    main()