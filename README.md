# Electricity Demand Forecasting Dashboard

This project provides an interactive web application to forecast electricity demand by leveraging climatic data. It compares the performance of three different forecasting models (Naive, SARIMAX, and LSTM) and allows users to predict future energy consumption based on expected weather conditions.

## The Dataset

The data used in this project is publicly available and was sourced from Mendeley Data.

* Title: Electricity demand and weather data for Texas

* Date: 27 April 2023

* Source: Mendeley Data, V1, doi: 10.17632/fdfftr3tc2.1

* Description: The dataset contains electricity demand, temperature, and humidity data for Texas, recorded at 5-minute intervals from January 1, 2019, to December 31, 2021. For this project, the data was aggregated to a daily timescale to analyze and forecast overall daily demand.

* Citation:
Rojas Ortega, Sebastian; Castro-Correa, Paola; Sepúlveda-Mora, Sergio; Castro-Correa, Jhon (2023), “Renewable Energy and Electricity Demand Time Series Dataset with Exogenous Variables at 5-minute Interval”, Mendeley Data, V1, doi: 10.17632/fdfftr3tc2.1

## How to Run the Application

You can run this Gradio application either locally on your machine or directly in your browser using Google Colab.

1. Run in Colab (Recommended)

The easiest way to get started is by using Google Colab, which runs everything in the cloud.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RoshanVarghese/TextToImage/blob/main/GenAI_TextToImage_GitHub.ipynb)

2. Running Locally

To run the application on your own computer, follow these steps:

Step A: Clone the Repository
Open your terminal and clone this repository to your local machine.

git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME

Step B: Install Dependencies
Install all the required Python libraries using pip.

pip install gradio pandas numpy matplotlib statsmodels tensorflow scikit-learn

Step C: Launch the App
Run the main application script from your terminal.

python app.py

This will start a local web server, and you can access the application by navigating to the provided URL (usually http://127.0.0.1:7860) in your web browser.

## Project Files

app.py: The main Python script that runs the Gradio web application.

Dataset.csv: The dataset file.

sarimax_model.pkl: The pre-trained SARIMAX model.

lstm_model.h5: The pre-trained LSTM model.

scaler.pkl: The scaler object used for data normalization.

README.md: This file.
