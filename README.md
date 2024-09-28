# Stock Price Predictor App

## Overview

The **Stock Price Predictor App** is a web application designed to predict stock prices using historical data and a machine learning model. Built with Streamlit and Keras, this application allows users to input a stock symbol, view historical stock data, and visualize predictions against actual prices.

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Visualizations](#visualizations)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **User Input:** Enter a stock symbol to filter and retrieve specific stock data.
- **Historical Data Visualization:** Display historical stock data with moving averages to understand trends.
- **Predictions:** Make future price predictions based on a trained machine learning model.
- **Interactive Plots:** Visualize original versus predicted prices in an interactive and user-friendly manner.

## Technologies Used

- **Streamlit:** A fast way to build and share data apps for machine learning and data science.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computing.
- **Keras:** A high-level neural networks API for building and training models.
- **Matplotlib:** For creating static, animated, and interactive visualizations in Python.
- **Scikit-learn:** For data preprocessing and machine learning tasks.

## Dataset

The application relies on a dataset in CSV format named `Data.csv`. This dataset should include the following columns:

- **`Symbol`:** The stock symbol (ticker) of the company (e.g., TCS).
- **`Date`:** The date of the stock price (formatted as `YYYY-MM-DD`).
- **`Close`:** The closing price of the stock on that date.

**Note:** Ensure the dataset is properly formatted to avoid issues during data processing.

## Installation

To get started with the Stock Price Predictor App, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/Parthchauh/Stock-Price-Prediction-Web-App-using-streamlit.git
   ```
   
Navigate to the Project Directory:

```bash
cd Stock-Price-Prediction-Web-App-using-streamlit 
```

## Install Required Packages:

Create a virtual environment (optional but recommended) and install the necessary Python packages using pip:

```bash pip install streamlit pandas numpy keras matplotlib scikit-learn
```

Usage
Run the Application:

Start the Streamlit server to run the app:

```bash 
streamlit run app.py
```

Access the App:

**Open your web browser and go to http://localhost:8501 to view the app.**

Input Stock Symbol:

Enter a valid stock symbol in the input box (default: TCS) to retrieve and predict stock prices.

Model
The application uses a pre-trained Keras model for stock price prediction.

Ensure the model file is named Latest_stock_price_model.keras and is located in the project directory.
The model is designed to accept sequences of stock prices and return predictions based on previous trends.
Visualizations
The app generates several visualizations to help users understand the stock data and predictions:

Stock Data Table: Displays historical stock prices for the selected symbol.
Moving Averages: Plots moving averages for 100, 200, and 250 days, providing insights into trends.
Original vs. Predicted Prices: A line chart comparing actual closing prices with predicted prices.
Contributing
Contributions are welcome! If you'd like to improve this project, please fork the repository and create a pull request. You can also report issues or suggest features.

License
This project is licensed under the MIT License. See the LICENSE file for more information.

Acknowledgments
Streamlit Documentation: Streamlit Documentation
Keras Documentation: Keras Documentation
Pandas Documentation: Pandas Documentation
NumPy Documentation: NumPy Documentation
Matplotlib Documentation: Matplotlib Documentation
Scikit-learn Documentation: Scikit-learn Documentation
Feel free to reach out if you have any questions or need further assistance!

### Instructions for Use:
1. Copy and paste the above content into a new file named `README.md` in your project directory.
2. Customize any sections as needed to fit your project specifics.
3. Save the file, and it will be ready to provide detailed information about your project!
