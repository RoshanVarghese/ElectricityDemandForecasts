import gradio as gr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAXResults
from tensorflow.keras.models import load_model
import pickle
from datetime import timedelta
import functools

# --- 1. DATA LOADING AND PREPROCESSING ---
# This section contains functions to load and prepare the data, similar to your notebook.

def load_and_prepare_data(filepath='Dataset.csv'):
    """
    Loads the electricity data and preprocesses it to a daily format.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        # Create a dummy dataframe if the file is not found, so the app can still launch.
        print("Warning: 'Electricity.csv' not found. Creating a dummy dataframe.")
        dates = pd.date_range(start='2017-01-01', periods=1000, freq='5T')
        data = {
            'Time': dates,
            'Electric_demand': np.random.uniform(2000, 5000, 1000),
            'Temperature': np.random.uniform(10, 35, 1000),
            'Humidity': np.random.uniform(30, 90, 1000)
        }
        df = pd.DataFrame(data)

    # --- FIX: Use the correct column names from the user's new dataset ---
    # Rename columns for consistency within the script
    df.rename(columns={
        'Time': 'DateTime',
        'Electric_demand': 'DEMAND',
        'Temperature': 'TEMPERATURE',
        'Humidity': 'HUMIDITY' # Assuming 'Humidity' is the correct name
    }, inplace=True)

    if 'DateTime' not in df.columns:
         print("Warning: 'DateTime' column not found after renaming. Assuming the first column is the datetime column.")
         df.rename(columns={df.columns[0]: 'DateTime'}, inplace=True)

    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df.set_index('DateTime', inplace=True)

    # Aggregate to daily data
    daily_df = df.resample('D').agg({
        'DEMAND': 'sum',
        'TEMPERATURE': 'mean',
        'HUMIDITY': 'mean'
    })
    daily_df.interpolate(method='linear', inplace=True)
    return daily_df

# --- 2. LOAD PRE-TRAINED MODELS ---
# We load the models and the scaler you saved from your notebook.

def load_models():
    """
    Loads the pre-trained SARIMAX model, LSTM model, and the scaler.
    """
    try:
        sarimax_model = SARIMAXResults.load('sarimax_model.pkl')
    except FileNotFoundError:
        print("Warning: 'sarimax_model.pkl' not found. SARIMAX predictions will not be available.")
        sarimax_model = None

    try:
        lstm_model = load_model('lstm_model.h5')
    except (FileNotFoundError, IOError):
        print("Warning: 'lstm_model.h5' not found. LSTM predictions will not be available.")
        lstm_model = None

    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
    except FileNotFoundError:
        print("Warning: 'scaler.pkl' not found. LSTM predictions may be inaccurate.")
        scaler = None

    return sarimax_model, lstm_model, scaler

# --- 3. MODEL PREDICTION FUNCTIONS ---
# These functions will generate predictions from the loaded models.

def create_lstm_dataset(dataset, look_back=7):
    """
    Creates the feature/label dataset for the LSTM model.
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def get_predictions(daily_df, sarimax_model, lstm_model, scaler):
    """
    Generates predictions for all models on the test set.
    """
    # Split data (using the last 100 days as a test set for demonstration)
    train_size = len(daily_df) - 100
    train, test = daily_df[0:train_size], daily_df[train_size:]

    # --- Naive Model (Baseline) ---
    # The prediction for today is simply yesterday's value.
    naive_preds = test['DEMAND'].shift(1).fillna(method='bfill')

    # --- SARIMAX Predictions ---
    if sarimax_model:
        # --- FIX: Ensure correct column names are used for exogenous variables ---
        sarimax_preds = sarimax_model.get_forecast(steps=len(test), exog=test[['TEMPERATURE', 'HUMIDITY']]).predicted_mean
    else:
        sarimax_preds = pd.Series(np.zeros(len(test)), index=test.index)


    # --- LSTM Predictions ---
    if lstm_model and scaler:
        # The scaler expects columns in the order: DEMAND, TEMPERATURE, HUMIDITY
        # Ensure the dataframe passed to the scaler has this order
        scaled_data = scaler.transform(daily_df[['DEMAND', 'TEMPERATURE', 'HUMIDITY']])
        scaled_test = scaled_data[train_size:]

        look_back = 7 # Should be the same as used in training
        X_test, y_test = create_lstm_dataset(scaled_test, look_back)

        if len(X_test) > 0:  # Only predict if we have data
            # Make predictions
            lstm_scaled_preds = lstm_model.predict(X_test)

            # Inverse transform predictions
            pad_preds = np.zeros((lstm_scaled_preds.shape[0], scaled_data.shape[1]))
            pad_preds[:, 0] = lstm_scaled_preds.flatten()
            lstm_preds_unscaled = scaler.inverse_transform(pad_preds)[:, 0]
            
            # Align predictions with the test set index
            lstm_preds = pd.Series(lstm_preds_unscaled, index=test.index[look_back:])
        else:
            lstm_preds = pd.Series([], dtype=float)
    else:
        lstm_preds = pd.Series(np.zeros(len(test)), index=test.index)

    return test['DEMAND'], naive_preds, sarimax_preds, lstm_preds

def calculate_metrics(actual, predicted):
    """
    Calculates MAE and RMSE metrics.
    """
    # Ensure indices are aligned and drop NaNs
    combined = pd.concat([actual, predicted], axis=1).dropna()
    
    if len(combined) == 0:
        return 0, 0
        
    actual_aligned = combined.iloc[:, 0]
    predicted_aligned = combined.iloc[:, 1]
    
    mae = np.mean(np.abs(predicted_aligned - actual_aligned))
    rmse = np.sqrt(np.mean((predicted_aligned - actual_aligned)**2))
    return mae, rmse

# --- 4. GRADIO INTERFACE FUNCTIONS ---
# These functions will be called by the Gradio components.

def plot_model_comparison(actual, naive, sarimax, lstm):
    """
    Creates a plot comparing the actual demand vs. model predictions.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(actual.index, actual, label='Actual Demand', color='black', linewidth=2)
    
    if len(naive) > 0:
        ax.plot(naive.index, naive, label='Naive Forecast', color='orange', linestyle='--')
    if len(sarimax) > 0:
        ax.plot(sarimax.index, sarimax, label='SARIMAX Forecast', color='blue', linestyle='--')
    if len(lstm) > 0:
        ax.plot(lstm.index, lstm, label='LSTM Forecast', color='green', linestyle='--')
        
    ax.set_title('Model Comparison: Actual vs. Forecasted Demand', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Electricity Demand', fontsize=12)
    ax.legend()
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    return fig

def get_metrics_df(actual, naive, sarimax, lstm):
    """
    Creates a DataFrame with the performance metrics for each model.
    """
    models = {'Naive': naive, 'SARIMAX': sarimax, 'LSTM': lstm}
    metrics_data = []
    
    for name, preds in models.items():
        mae, rmse = calculate_metrics(actual, preds)
        metrics_data.append({'Model': name, 'MAE': f'{mae:,.2f}', 'RMSE': f'{rmse:,.2f}'})

    metrics_df = pd.DataFrame(metrics_data)
    
    # Find best model (lowest RMSE)
    rmse_values = [float(row['RMSE'].replace(',', '')) for row in metrics_data]
    best_idx = np.argmin(rmse_values)
    best_model = metrics_data[best_idx]['Model']

    summary = f"""
    ### Model Performance Summary
    The table below shows the Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for each model on the test data.
    - **MAE**: The average absolute difference between the forecast and the actual value.
    - **RMSE**: The square root of the average of squared differences, which penalizes larger errors more.

    **🏆 Best Performing Model:** **{best_model}**
    
    This model achieved the lowest RMSE, indicating it had the highest overall accuracy in predicting electricity demand during the test period.
    """
    return metrics_df, summary

def predict_future_demand(start_date, num_days, avg_temp, avg_humidity, daily_df, sarimax_model, lstm_model, scaler):
    """
    Predicts future demand using the best model (SARIMAX for simplicity in long-range forecasting).
    """
    if sarimax_model is None:
        return None, "SARIMAX model is not loaded. Cannot make future predictions."
        
    try:
        start_date = pd.to_datetime(start_date)
        
        # Create future exogenous variables
        future_dates = pd.date_range(start=start_date, periods=num_days, freq='D')
        future_exog = pd.DataFrame({
            'TEMPERATURE': [avg_temp] * num_days,
            'HUMIDITY': [avg_humidity] * num_days
        }, index=future_dates)

        # Generate forecast
        forecast = sarimax_model.get_forecast(steps=num_days, exog=future_exog).predicted_mean

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast.index, forecast, marker='o', linestyle='-', label='Forecasted Demand')
        
        # Plot historical data for context
        historical_start = start_date - timedelta(days=30)
        historical_end = start_date - timedelta(days=1)
        
        if historical_start in daily_df.index or any(daily_df.index <= historical_end):
            historical_context = daily_df['DEMAND'].loc[historical_start:historical_end]
            if len(historical_context) > 0:
                ax.plot(historical_context.index, historical_context, color='gray', linestyle='--', label='Historical (30 days)')
        
        ax.set_title(f'Forecasted Electricity Demand for {num_days} Days', fontsize=16)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Predicted Electricity Demand', fontsize=12)
        ax.legend()
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        return fig, ""
    
    except Exception as e:
        return None, f"Error generating forecast: {str(e)}"


# --- 5. MAIN APPLICATION EXECUTION ---
if __name__ == "__main__":
    # Load data and models once at the start
    daily_df = load_and_prepare_data()
    sarimax_model, lstm_model, scaler = load_models()
    actual, naive, sarimax, lstm = get_predictions(daily_df, sarimax_model, lstm_model, scaler)
    metrics_df, summary_text = get_metrics_df(actual, naive, sarimax, lstm)

    # Create a wrapper function that has access to the loaded models and data
    def wrapped_predict_future_demand(start_date, num_days, avg_temp, avg_humidity):
        fig, err = predict_future_demand(start_date, num_days, avg_temp, avg_humidity, 
                                       daily_df, sarimax_model, lstm_model, scaler)
        if err:
            return None, err
        return fig, ""

    # --- Build Gradio Interface ---
    with gr.Blocks(theme=gr.themes.Soft(), title="Electricity Demand Forecaster") as demo:
        gr.Markdown("# ⚡ Electricity Demand Forecasting Dashboard")
        gr.Markdown("An interactive tool to compare forecasting models and predict future electricity demand based on weather conditions.")

        with gr.Tabs():
            with gr.TabItem("📊 Model Comparison"):
                gr.Markdown("## Performance Evaluation")
                gr.Markdown("This tab compares three different forecasting models against the actual historical data.")
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Label("Performance Metrics")
                        gr.DataFrame(metrics_df, interactive=False)
                        gr.Markdown(summary_text)
                    with gr.Column(scale=2):
                        gr.Label("Forecast Visualization")
                        gr.Plot(value=plot_model_comparison(actual, naive, sarimax, lstm))

            with gr.TabItem("🔮 Future Demand Prediction"):
                gr.Markdown("## Predict Future Energy Consumption")
                gr.Markdown("Use the best performing model (SARIMAX) to forecast demand for a future period. Enter the required information below.")
                
                with gr.Row():
                    with gr.Column():
                        start_date_input = gr.Date(label="Forecasting Start Date", 
                                                 value=str(daily_df.index.max() + timedelta(days=1)))
                        days_input = gr.Slider(minimum=1, maximum=90, value=14, step=1, 
                                             label="Number of Days to Forecast")
                        temp_input = gr.Slider(minimum=-10, maximum=50, value=25, 
                                             label="Expected Average Temperature (°C)")
                        humidity_input = gr.Slider(minimum=0, maximum=100, value=60, 
                                                 label="Expected Average Humidity (%)")
                        predict_btn = gr.Button("Generate Forecast", variant="primary")
                    
                    with gr.Column():
                        output_plot = gr.Plot()
                        error_message = gr.Markdown(visible=False)

                # Clean event handler
                predict_btn.click(
                    fn=wrapped_predict_future_demand,
                    inputs=[start_date_input, days_input, temp_input, humidity_input],
                    outputs=[output_plot, error_message]
                )

    demo.launch(debug=True)
