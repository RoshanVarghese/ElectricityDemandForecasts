import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from tensorflow.keras.models import load_model
import pickle
from datetime import timedelta
from matplotlib.dates import DateFormatter

# --- 1. DATA LOADING AND PREPROCESSING ---
def load_and_prepare_data(filepath='Dataset.csv'):
    """
    Loads and prepares the data, ensuring consistency with the training process.
    """
    try:
        df = pd.read_csv(filepath)
        print(f"Successfully loaded '{filepath}'. Shape: {df.shape}, Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        print(f"Warning: '{filepath}' not found. Using dummy data.")
        dates = pd.to_datetime(pd.date_range(start='2019-01-01', periods=1000, freq='D'))
        data = {
            'Time': dates, 'Electric_demand': np.random.uniform(20000, 35000, 1000),
            'Temperature': np.random.uniform(10, 35, 1000), 'Humidity': np.random.uniform(30, 90, 1000),
            'Wind_speed': np.random.uniform(0, 10, 1000), 'DHI': np.random.uniform(0, 100, 1000),
            'DNI': np.random.uniform(0, 500, 1000), 'GHI': np.random.uniform(0, 600, 1000)
        }
        df = pd.DataFrame(data)

    # --- FIX: Keep original 'Electric_demand' name for scaler compatibility ---
    df.rename(columns={'Time': 'DateTime'}, inplace=True)
    if 'DateTime' not in df.columns:
        print("Warning: 'DateTime' column not found. Using the first column.")
        df.rename(columns={df.columns[0]: 'DateTime'}, inplace=True)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)

    # Ensure data is daily. If not, resample.
    if len(df) > 1 and (df.index[1] - df.index[0]) < pd.Timedelta(days=1):
        print("Sub-daily data detected. Resampling to daily sums for Electric_demand and means for others.")
        agg_rules = {col: 'mean' for col in df.columns if col != 'Electric_demand'}
        agg_rules['Electric_demand'] = 'sum'
        daily_df = df.resample('D').agg(agg_rules)
    else:
        daily_df = df

    daily_df.interpolate(method='linear', inplace=True)
    daily_df.dropna(inplace=True)
    print(f"Data preparation complete. Final shape: {daily_df.shape}")
    return daily_df

# --- 2. LOAD PRE-TRAINED MODELS ---
def load_models():
    """
    Loads the SARIMAX, LSTM, and scaler objects.
    """
    try:
        sarimax_model = SARIMAXResults.load('sarimax_model.pkl')
        print("SARIMAX model loaded successfully.")
    except Exception as e:
        print(f"Warning: 'sarimax_model.pkl' not found or failed to load. {e}")
        sarimax_model = None

    try:
        lstm_model = load_model('lstm_model.h5', compile=False)
        print("LSTM model loaded successfully.")
    except Exception as e:
        print(f"Warning: 'lstm_model.h5' not found or failed to load. {e}")
        lstm_model = None

    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Scaler loaded successfully.")
    except Exception as e:
        print(f"Warning: 'scaler.pkl' not found or failed to load. {e}")
        scaler = None

    return sarimax_model, lstm_model, scaler

# --- 3. MODEL PREDICTION FUNCTIONS ---
def get_predictions(daily_df, sarimax_model, lstm_model, scaler):
    """
    Generates predictions from all models, ensuring methods match the training notebook.
    """
    if daily_df.empty:
        return pd.Series(), pd.Series(), pd.Series(), pd.Series()

    # Use the exact same 80/20 split as in training
    train_size = int(len(daily_df) * 0.8)
    test_df = daily_df.iloc[train_size:]
    actuals = test_df['Electric_demand']

    # --- Naive Model (Baseline) ---
    naive_preds = actuals.shift(1).bfill()

    # --- SARIMAX Predictions ---
    if sarimax_model:
        try:
            exog_vars = ['Temperature', 'Humidity'] # As used in training
            test_exog = test_df[exog_vars]
            start = len(daily_df) - len(test_df)
            end = len(daily_df) - 1
            sarimax_preds = sarimax_model.predict(start=start, end=end, exog=test_exog)
            sarimax_preds.index = actuals.index # Ensure index alignment
        except Exception as e:
            print(f"SARIMAX prediction failed: {e}")
            sarimax_preds = pd.Series(index=actuals.index)
    else:
        sarimax_preds = pd.Series(index=actuals.index)

    # --- LSTM Predictions ---
    if lstm_model and scaler:
        try:
            # --- FIX: Use the original feature names the scaler was trained on ---
            feature_cols = ['Electric_demand', 'Temperature', 'Humidity', 'Wind_speed', 'DHI', 'DNI', 'GHI']
            
            full_data_for_scaling = daily_df[feature_cols]
            scaled_data = scaler.transform(full_data_for_scaling)
            
            look_back = 7
            X, y = [], []
            for i in range(len(scaled_data) - look_back):
                X.append(scaled_data[i:(i + look_back), 1:]) # Features are all columns except Electric_demand
                y.append(scaled_data[i + look_back, 0])      # Target is Electric_demand (first column)
            X, y = np.array(X), np.array(y)

            seq_train_size = int(len(X) * 0.8)
            X_test = X[seq_train_size:]
            
            lstm_scaled_preds = lstm_model.predict(X_test).flatten()

            dummy_array = np.zeros((len(lstm_scaled_preds), len(feature_cols)))
            dummy_array[:, 0] = lstm_scaled_preds
            lstm_preds_unscaled = scaler.inverse_transform(dummy_array)[:, 0]

            pred_start_index = actuals.index[look_back + (len(y) - len(X_test) - seq_train_size)]
            pred_dates = pd.date_range(start=pred_start_index, periods=len(lstm_preds_unscaled))
            lstm_preds = pd.Series(lstm_preds_unscaled, index=pred_dates)

        except Exception as e:
            print(f"LSTM prediction failed: {e}")
            import traceback
            traceback.print_exc()
            lstm_preds = pd.Series(index=actuals.index)
    else:
        lstm_preds = pd.Series(index=actuals.index)

    return actuals, naive_preds, sarimax_preds, lstm_preds

def calculate_metrics(actual, predicted):
    """Calculates MAE and RMSE after aligning and dropping NaNs."""
    combined = pd.concat([actual, predicted], axis=1).dropna()
    if combined.empty:
        return 0, 0
    mae = np.mean(np.abs(combined.iloc[:, 1] - combined.iloc[:, 0]))
    rmse = np.sqrt(np.mean((combined.iloc[:, 1] - combined.iloc[:, 0])**2))
    return mae, rmse

# --- 4. GRADIO INTERFACE FUNCTIONS ---
def plot_model_comparison(actual, naive, sarimax, lstm):
    """Creates a plot comparing all model forecasts against the actual demand."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(actual.index, actual, label='Actual Demand', color='black', linewidth=2)
    ax.plot(naive.index, naive, label='Naive Forecast', color='orange', linestyle='--')
    ax.plot(sarimax.index, sarimax, label='SARIMAX Forecast', color='blue', linestyle='--')
    ax.plot(lstm.index, lstm, label='LSTM Forecast', color='green', linestyle='--')
    ax.set_title('Model Comparison: Actual vs. Forecasted Demand', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Electricity Demand', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    fig.tight_layout()
    return fig

def get_metrics_df(actual, naive, sarimax, lstm):
    """Creates a DataFrame and summary text for model performance metrics."""
    models = {'Naive': naive, 'SARIMAX': sarimax, 'LSTM': lstm}
    metrics_data = []
    for name, preds in models.items():
        mae, rmse = calculate_metrics(actual, preds)
        metrics_data.append({'Model': name, 'MAE': f'{mae:,.2f}', 'RMSE': f'{rmse:,.2f}'})
    
    metrics_df = pd.DataFrame(metrics_data)
    best_model = metrics_df.loc[metrics_df['RMSE'].astype(str).str.replace(',', '').astype(float).idxmin()]

    summary = f"""
    ### Model Performance Summary
    - **MAE**: The average absolute difference between the forecast and the actual value.
    - **RMSE**: Penalizes larger errors more heavily. Lower is better.

    **🏆 Best Performing Model:** **{best_model['Model']}**
    """
    return metrics_df, summary

def predict_future_demand(start_date_str, num_days, avg_temp, avg_humidity, daily_df, sarimax_model):
    """Predicts future demand using the SARIMAX model."""
    if sarimax_model is None:
        return None, "SARIMAX model is not loaded. Cannot make future predictions."
    
    try:
        start_date = pd.to_datetime(start_date_str)
        last_date = daily_df.index.max()
        
        future_dates = pd.date_range(start=start_date, periods=num_days, freq='D')
        future_exog = pd.DataFrame({'Temperature': avg_temp, 'Humidity': avg_humidity}, index=future_dates)

        start_idx = len(daily_df)
        end_idx = start_idx + num_days - 1
        
        forecast = sarimax_model.predict(start=start_idx, end=end_idx, exog=future_exog)
        forecast.index = future_dates

        fig, ax = plt.subplots(figsize=(12, 6))
        # --- FIX: Use 'Electric_demand' for historical plot ---
        historical_context = daily_df['Electric_demand'].last('60D')
        ax.plot(historical_context.index, historical_context, label='Historical Demand (Last 60 Days)', color='blue')
        ax.plot(forecast.index, forecast, label='Forecasted Demand', color='red', marker='o', linestyle='--')
        
        ax.set_title(f'Forecast for {num_days} Days', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Electricity Demand', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.5)
        ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        
        summary = f"""
        **Forecast Summary:**
        - **Period:** {forecast.index.min().strftime('%Y-%m-%d')} to {forecast.index.max().strftime('%Y-%m-%d')}
        - **Average Predicted Demand:** {forecast.mean():,.0f}
        """
        return fig, summary

    except Exception as e:
        return None, f"An error occurred: {e}"

# --- 5. MAIN APPLICATION EXECUTION ---
if __name__ == "__main__":
    daily_df = load_and_prepare_data('Dataset.csv')
    sarimax_model, lstm_model, scaler = load_models()
    actual, naive, sarimax, lstm = get_predictions(daily_df, sarimax_model, lstm_model, scaler)
    metrics_df, summary_text = get_metrics_df(actual, naive, sarimax, lstm)

    with gr.Blocks(theme=gr.themes.Soft(), title="Electricity Demand Forecaster") as demo:
        gr.Markdown("# ⚡ Electricity Demand Forecasting Dashboard")
        
        with gr.Tabs():
            with gr.TabItem("📊 Model Comparison"):
                gr.Markdown("## Performance Evaluation on Test Data")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Label("Performance Metrics")
                        gr.DataFrame(value=metrics_df, interactive=False)
                        gr.Markdown(summary_text)
                    with gr.Column(scale=2):
                        gr.Label("Forecast Visualization")
                        gr.Plot(value=plot_model_comparison(actual, naive, sarimax, lstm))

            with gr.TabItem("🔮 Future Demand Prediction"):
                gr.Markdown("## Generate a New Forecast")
                gr.Markdown("Use the SARIMAX model to forecast demand for a future period.")
                with gr.Row():
                    with gr.Column(scale=1):
                        start_date_input = gr.Textbox(
                            label="Start Date (YYYY-MM-DD)",
                            value=(daily_df.index.max() + timedelta(days=1)).strftime('%Y-%m-%d')
                        )
                        days_input = gr.Slider(1, 90, value=14, step=1, label="Days to Forecast")
                        temp_input = gr.Slider(-10, 50, value=25, label="Avg. Temperature (°C)")
                        humidity_input = gr.Slider(0, 100, value=60, label="Avg. Humidity (%)")
                        predict_btn = gr.Button("Generate Forecast", variant="primary")
                    
                    with gr.Column(scale=2):
                        output_plot = gr.Plot()
                        output_summary = gr.Markdown()
                
                # --- FIX: Use the single, correct way to wire the button ---
                def wrapped_predict(start_date, num_days, avg_temp, avg_humidity):
                    return predict_future_demand(start_date, num_days, avg_temp, avg_humidity, daily_df, sarimax_model)

                predict_btn.click(
                    fn=wrapped_predict,
                    inputs=[start_date_input, days_input, temp_input, humidity_input],
                    outputs=[output_plot, output_summary]
                )

    demo.launch(share=True, debug=True)
