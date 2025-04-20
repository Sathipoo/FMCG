from flask import Flask, render_template, request, send_file
import pandas as pd
import os
from io import BytesIO

from data_ingestion import load_sample_data
from feature_engineering import add_time_features, encode_categoricals
from forecast_model import train_forecast_model
from replenishment import calculate_replenishment

app = Flask(__name__)

@app.route('/')
def home():
    df = load_sample_data()
    df = add_time_features(df)
    df = encode_categoricals(df)
    model = train_forecast_model(df)
    results_df = calculate_replenishment(df, model)

    # Store for download
    results_df.to_csv("output/replenishment_plan.csv", index=False)

    table_html = results_df.to_html(classes='table table-striped', index=False)
    return render_template('dashboard.html', table=table_html)

@app.route('/download')
def download():
    file_path = "output/replenishment_plan.csv"
    return send_file(
        file_path,
        as_attachment=True,
        mimetype='text/csv',
        download_name='replenishment_plan.csv'
    )

if __name__ == '__main__':
    os.makedirs("output", exist_ok=True)
    app.run(debug=True)