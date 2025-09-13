# Stock Price Prediction Implementation using FinRL

This implementation provides a comprehensive stock price prediction system built on top of the FinRL framework. The solution includes data simulation, feature engineering, machine learning models, evaluation metrics, and comparison with baseline models.

## 🚀 Features

- **Data Simulation**: Realistic stock price data generation when external data sources are unavailable
- **Technical Indicators**: RSI, Moving Averages, Volatility calculations
- **Machine Learning Model**: Simple linear regression with gradient descent
- **Evaluation Metrics**: RMSE, MAE, MAPE, Directional Accuracy, Correlation
- **Baseline Comparisons**: Naive, Moving Average, and Linear Trend models
- **Model Persistence**: Save and load trained models
- **Comprehensive Testing**: Full unit test suite
- **Visualization**: Text-based result visualization

## 📁 Files Structure

```
/home/runner/work/FinRL/FinRL/
├── stock_prediction_demo.py      # Main prediction implementation
├── evaluation_suite.py           # Comprehensive evaluation system
├── test_stock_prediction.py      # Unit tests
└── README_STOCK_PREDICTION.md    # This documentation
```

## 🏃‍♂️ Quick Start

### 1. Basic Stock Prediction Demo

```bash
python stock_prediction_demo.py
```

This will:
- Fetch simulated data for AAPL from 2023-01-01 to 2024-01-01
- Engineer features with technical indicators
- Train a linear regression model
- Evaluate performance and show results
- Save the trained model

### 2. Comprehensive Evaluation

```bash
python evaluation_suite.py
```

This will:
- Evaluate multiple stocks (AAPL, GOOGL, MSFT)
- Compare ML models with baseline models
- Generate detailed performance comparisons
- Save results to evaluation_results.json

### 3. Run Unit Tests

```bash
python test_stock_prediction.py
```

This will run 22 unit tests covering all components of the system.

## 📊 Sample Output

### Basic Demo Results
```
============================================================
EVALUATION RESULTS
============================================================
Ticker: AAPL
Test Samples: 71
Root Mean Square Error (RMSE): $2.19
Mean Absolute Error (MAE): $1.74
Mean Absolute Percentage Error (MAPE): 1.51%
Directional Accuracy: 52.9%

Sample Predictions vs Actuals:
Predicted | Actual    | Difference
-----------------------------------
$  123.97 | $  128.43 | $   -4.46
$  126.62 | $  127.16 | $   -0.54
$  126.45 | $  131.97 | $   -5.52
...
```

## 🔧 Technical Implementation

### Data Generation
- Simulates realistic stock price movements with trends and volatility
- Generates OHLC (Open, High, Low, Close) data with proper constraints
- Includes volume data for additional features

### Feature Engineering
- **Price Features**: Historical prices with configurable lookback window
- **Technical Indicators**: 
  - Simple Moving Averages (5-day, 20-day)
  - Relative Strength Index (RSI)
  - Price volatility
  - Volume indicators
- **Return Features**: Price change percentages

### Machine Learning Model
- Linear regression trained with gradient descent
- Feature normalization (mean=0, std=1)
- Configurable learning rate and epochs
- Weight initialization and bias terms

### Evaluation Metrics
- **RMSE**: Root Mean Square Error for prediction accuracy
- **MAE**: Mean Absolute Error for average prediction error
- **MAPE**: Mean Absolute Percentage Error for relative accuracy
- **Directional Accuracy**: Percentage of correct trend predictions
- **Correlation**: Linear correlation between predictions and actual values

### Baseline Models
- **Naive Model**: Next price = current price
- **Moving Average**: Uses simple moving average for prediction
- **Linear Trend**: Extrapolates linear trend from historical data

## 🧪 Testing

The implementation includes comprehensive unit tests:

- **Data Generation Tests**: Validate OHLC data consistency
- **Technical Indicator Tests**: Verify calculation accuracy
- **Model Tests**: Test training, prediction, and serialization
- **Integration Tests**: End-to-end workflow validation
- **Baseline Model Tests**: Verify alternative approaches

Run tests with: `python test_stock_prediction.py`

## 📈 Performance Interpretation

### Good Performance Indicators
- **MAPE < 5%**: Low percentage error indicates accurate predictions
- **Directional Accuracy > 55%**: Better than random for trend prediction
- **High Correlation**: Strong linear relationship between predictions and actuals

### Areas for Improvement
- **Add more features**: Include market sentiment, news data, macro indicators
- **Advanced models**: Try neural networks, ensemble methods, or time series models
- **Real data integration**: Connect to actual market data APIs
- **Risk management**: Include volatility-based position sizing

## 🔄 Usage Examples

### Custom Stock Prediction

```python
from stock_prediction_demo import StockPredictor

# Create predictor for custom ticker
predictor = StockPredictor("CUSTOM_TICKER")

# Fetch data and train
data = predictor.fetch_data("2023-01-01", "2023-12-31")
features, targets = predictor.engineer_features(lookback_days=15)
split_index = predictor.train_model(train_ratio=0.8)

# Evaluate
metrics = predictor.evaluate_model(split_index)
print(f"RMSE: ${metrics['rmse']:.2f}")
print(f"Directional Accuracy: {metrics['directional_accuracy']:.1%}")

# Save model
predictor.save_model("my_model.json")
```

### Custom Evaluation

```python
from evaluation_suite import ModelEvaluator

evaluator = ModelEvaluator()

# Add your predictions and actuals
metrics = evaluator.calculate_metrics(
    predictions=[100, 102, 101], 
    actuals=[99, 103, 100], 
    model_name="My Model"
)

# Print results
evaluator.print_comparison_table()
```

## 🚀 Extensions and Improvements

### For Production Use
1. **Real Data Integration**: Connect to Yahoo Finance, Alpha Vantage, or Bloomberg APIs
2. **Database Storage**: Store historical data and model performance
3. **API Endpoint**: Create REST API for prediction requests
4. **Monitoring**: Add model performance monitoring and retraining triggers
5. **Risk Management**: Implement position sizing and stop-loss mechanisms

### Advanced Features
1. **Deep Learning**: Implement LSTM or GRU models for time series
2. **Ensemble Methods**: Combine multiple models for better performance
3. **Feature Selection**: Automatic feature selection and engineering
4. **Hyperparameter Tuning**: Grid search or Bayesian optimization
5. **Multi-timeframe**: Support for different prediction horizons

## 📞 Support and Contributions

This implementation serves as a foundation for stock price prediction research and development. It demonstrates key concepts in:

- Financial time series analysis
- Feature engineering for stock data
- Machine learning model evaluation
- Baseline model comparison
- Software testing for financial applications

For questions or contributions, please refer to the main FinRL project documentation.

## ⚠️ Disclaimer

This implementation is for educational and research purposes only. It uses simulated data and simplified models. Do not use for actual trading without proper validation, risk management, and compliance with financial regulations.

Past performance does not guarantee future results. Always consult with qualified financial advisors before making investment decisions.