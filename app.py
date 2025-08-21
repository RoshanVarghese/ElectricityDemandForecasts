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
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    except FileNotFoundError:
        # Create a dummy dataframe if the file is not found, so the app can still launch.
        print("Warning: 'Dataset.csv' not found. Creating a dummy dataframe.")
        dates = pd.date_range(start='2017-01-01', periods=1000, freq='D')
        data = {
            'DateTime': dates,
            'Electric_demand': np.random.uniform(2000, 5000, 1000),
            'Temperature': np.random.uniform(10, 35, 1000),
            'Humidity': np.random.uniform(30, 90, 1000),
            'Wind_speed': np.random.uniform(0, 10, 1000),
            'DHI': np.random.uniform(0, 800, 1000),
            'DNI': np.random.uniform(0, 900, 1000),
            'GHI': np.random.uniform(0, 1000, 1000)
        }
        df = pd.DataFrame(data)

    # Handle the Time column correctly
    if 'Time' in df.columns:
        print("Using 'Time' column as datetime index")
        df['DateTime'] = pd.to_datetime(df['Time'])
    elif 'DateTime' not in df.columns and len(df.columns) > 0:
        print("Warning: No 'Time' or 'DateTime' column found. Using first column as datetime.")
        df['DateTime'] = pd.to_datetime(df.iloc[:, 0])
    
    # Drop the original Time column and any unnamed columns
    if 'Time' in df.columns:
        df = df.drop('Time', axis=1)
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    df.set_index('DateTime', inplace=True)

    print(f"Data after datetime processing: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Check if we need to resample to daily data
    time_diff = df.index[1] - df.index[0] if len(df) > 1 else pd.Timedelta(days=1)
    
    if time_diff < pd.Timedelta(hours=23):  # Sub-daily data
        print(f"Detected sub-daily data (interval: {time_diff}). Resampling to daily...")
        # Aggregate to daily data using the exact column names from training
        agg_dict = {}
        for col in df.columns:
            if col == 'Electric_demand':
                agg_dict[col] = 'sum'  # Sum for demand
            else:
                agg_dict[col] = 'mean'  # Mean for other variables
        
        daily_df = df.resample('D').agg(agg_dict)
    else:
        print("Data appears to be daily or less frequent. No resampling needed.")
        daily_df = df.copy()
    
    daily_df.interpolate(method='linear', inplace=True)
    daily_df = daily_df.dropna()  # Remove any remaining NaN values
    
    print(f"Final daily data shape: {daily_df.shape}")
    print(f"Final date range: {daily_df.index.min()} to {daily_df.index.max()}")
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
        # Try loading with compile=False to avoid loss function deserialization issues
        lstm_model = load_model('lstm_model.h5', compile=False)
        print("LSTM model loaded successfully (without compilation)")
    except Exception as e:
        print(f"Warning: 'lstm_model.h5' could not be loaded with compile=False. Error: {str(e)}")
        try:
            # Alternative approach: try with custom_objects
            import tensorflow.keras.losses as losses
            custom_objects = {'mse': losses.MeanSquaredError()}
            lstm_model = load_model('lstm_model.h5', custom_objects=custom_objects)
            print("LSTM model loaded successfully with custom_objects")
        except Exception as e2:
            print(f"Warning: Alternative loading method also failed. Error: {str(e2)}")
            print("LSTM predictions will not be available.")
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
    Based on your training setup: features exclude Electric_demand, target is Electric_demand
    """
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        # Features: all columns except the first (Electric_demand)
        a = dataset[i:(i + look_back), 1:]  
        dataX.append(a)
        # Target: Electric_demand (first column)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

def get_predictions(daily_df, sarimax_model, lstm_model, scaler):
    """
    Generates predictions for all models on the test set.
    """
    # Split data (using 80/20 split like in training)
    train_size = int(len(daily_df) * 0.8)
    train, test = daily_df[:train_size], daily_df[train_size:]

    # --- Naive Model (Baseline) ---
    naive_preds = test['Electric_demand'].shift(1).bfill()

    # --- SARIMAX Predictions ---
    if sarimax_model:
        # Use the same exog variables as in training: Temperature, Humidity
        exog_vars = ['Temperature', 'Humidity']
        try:
            sarimax_preds = sarimax_model.predict(
                start=len(train), 
                end=len(train) + len(test) - 1, 
                exog=test[exog_vars]
            )
        except Exception as e:
            print(f"SARIMAX prediction error: {e}")
            sarimax_preds = pd.Series(np.zeros(len(test)), index=test.index)
    else:
        sarimax_preds = pd.Series(np.zeros(len(test)), index=test.index)

    # --- LSTM Predictions ---
    if lstm_model and scaler:
        try:
            # Use the same features as in training
            features = ['Electric_demand', 'Temperature', 'Humidity', 'Wind_speed', 'DHI', 'DNI', 'GHI']
            
            # Check which features are available
            available_features = [f for f in features if f in daily_df.columns]
            print(f"Available features for LSTM: {available_features}")
            
            if len(available_features) < 3:  # Need at least Electric_demand + 2 others
                print("Insufficient features for LSTM prediction")
                lstm_preds = pd.Series(np.zeros(len(test)), index=test.index)
            else:
                # Scale the data using available features
                scaled_data = scaler.transform(daily_df[available_features])
                scaled_test = scaled_data[train_size:]

                look_back = 7
                X_test, y_test = create_lstm_dataset(scaled_test, look_back)

                if len(X_test) > 0:
                    # Make predictions
                    lstm_scaled_preds = lstm_model.predict(X_test)

                    # Inverse transform predictions
                    pad_preds = np.zeros((lstm_scaled_preds.shape[0], len(available_features)))
                    pad_preds[:, 0] = lstm_scaled_preds.flatten()
                    lstm_preds_unscaled = scaler.inverse_transform(pad_preds)[:, 0]
                    
                    # Align predictions with the test set index
                    lstm_preds = pd.Series(lstm_preds_unscaled, index=test.index[look_back:])
                else:
                    lstm_preds = pd.Series([], dtype=float)
        except Exception as e:
            print(f"LSTM prediction error: {e}")
            lstm_preds = pd.Series(np.zeros(len(test)), index=test.index)
    else:
        lstm_preds = pd.Series(np.zeros(len(test)), index=test.index)

    return test['Electric_demand'], naive_preds, sarimax_preds, lstm_preds

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
    Predicts future demand using the SARIMAX model (matches training setup).
    """
    if sarimax_model is None:
        return None, "SARIMAX model is not loaded. Cannot make future predictions."
        
    try:
        start_date = pd.to_datetime(start_date)
        
        # Create future exogenous variables using the same format as training
        future_dates = pd.date_range(start=start_date, periods=num_days, freq='D')
        future_exog = pd.DataFrame({
            'Temperature': [avg_temp] * num_days,
            'Humidity': [avg_humidity] * num_days
        }, index=future_dates)

        # Generate forecast using predict method (same as training)
        last_train_idx = len(daily_df) - 1
        forecast = sarimax_model.predict(
            start=last_train_idx + 1, 
            end=last_train_idx + num_days, 
            exog=future_exog
        )

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(forecast.index, forecast, marker='o', linestyle='-', label='Forecasted Demand', color='blue')
        
        # Plot historical data for context (last 30 days)
        historical_start = start_date - timedelta(days=30)
        historical_end = start_date - timedelta(days=1)
        
        if historical_start in daily_df.index or any(daily_df.index <= historical_end):
            historical_context = daily_df['Electric_demand'].loc[historical_start:historical_end]
            if len(historical_context) > 0:
                ax.plot(historical_context.index, historical_context, color='gray', 
                       linestyle='--', label='Historical (30 days)', alpha=0.7)
        
        ax.set_title(f'Forecasted Electricity Demand for {num_days} Days\n'
                    f'Temperature: {avg_temp}°C, Humidity: {avg_humidity}%', fontsize=14)
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Predicted Electricity Demand', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig, ""
    
    except Exception as e:
        return None, f"Error generating forecast: {str(e)}"


# --- 5. MAIN APPLICATION EXECUTION ---
if __name__ == "__main__":
    # Load data and models once at the start
    daily_df = load_and_prepare_data('Dataset.csv')  # Use correct filename
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
                        start_date_input = gr.Textbox(label="Forecasting Start Date (YYYY-MM-DD)", 
                                                    value=str(daily_df.index.max() + timedelta(days=1)).split()[0])
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
