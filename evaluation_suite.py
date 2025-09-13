#!/usr/bin/env python3
"""
Stock Prediction Evaluation Suite
=================================

This script provides comprehensive evaluation of stock prediction models
including multiple metrics, visualization (text-based), and comparison
with baseline models.
"""

import json
import math
import random
from typing import Dict, List, Tuple
from stock_prediction_demo import StockPredictor, TechnicalIndicators


class BaselineModels:
    """Baseline models for comparison"""
    
    @staticmethod
    def naive_prediction(prices: List[float]) -> List[float]:
        """Naive model: next price = current price"""
        return prices[:-1]  # Use current price as next day prediction
    
    @staticmethod
    def moving_average_prediction(prices: List[float], window: int = 5) -> List[float]:
        """Moving average model"""
        predictions = []
        for i in range(len(prices) - 1):
            if i < window:
                predictions.append(prices[i])
            else:
                ma = sum(prices[i-window+1:i+1]) / window
                predictions.append(ma)
        return predictions
    
    @staticmethod
    def linear_trend_prediction(prices: List[float], window: int = 10) -> List[float]:
        """Linear trend extrapolation"""
        predictions = []
        for i in range(len(prices) - 1):
            if i < window:
                predictions.append(prices[i])
            else:
                # Calculate trend over window
                x_vals = list(range(window))
                y_vals = prices[i-window+1:i+1]
                
                # Simple linear regression
                n = len(x_vals)
                sum_x = sum(x_vals)
                sum_y = sum(y_vals)
                sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
                sum_x2 = sum(x * x for x in x_vals)
                
                if n * sum_x2 - sum_x * sum_x != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
                    intercept = (sum_y - slope * sum_x) / n
                    next_pred = slope * window + intercept
                    predictions.append(next_pred)
                else:
                    predictions.append(prices[i])
        
        return predictions


class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, predictions: List[float], actuals: List[float], model_name: str) -> Dict:
        """Calculate comprehensive metrics"""
        if len(predictions) != len(actuals):
            raise ValueError("Predictions and actuals must have same length")
        
        n = len(predictions)
        if n == 0:
            return {}
        
        # Basic metrics
        mse = sum((p - a)**2 for p, a in zip(predictions, actuals)) / n
        rmse = math.sqrt(mse)
        mae = sum(abs(p - a) for p, a in zip(predictions, actuals)) / n
        
        # Percentage errors
        mean_actual = sum(actuals) / n
        mape = (mae / mean_actual * 100) if mean_actual != 0 else float('inf')
        
        # Directional accuracy
        correct_directions = 0
        for i in range(1, n):
            pred_dir = 1 if predictions[i] > predictions[i-1] else -1
            actual_dir = 1 if actuals[i] > actuals[i-1] else -1
            if pred_dir == actual_dir:
                correct_directions += 1
        
        directional_accuracy = correct_directions / (n - 1) if n > 1 else 0
        
        # Correlation coefficient
        pred_mean = sum(predictions) / n
        actual_mean = sum(actuals) / n
        
        numerator = sum((p - pred_mean) * (a - actual_mean) for p, a in zip(predictions, actuals))
        pred_var = sum((p - pred_mean)**2 for p in predictions)
        actual_var = sum((a - actual_mean)**2 for a in actuals)
        
        correlation = numerator / math.sqrt(pred_var * actual_var) if pred_var > 0 and actual_var > 0 else 0
        
        # Maximum and minimum errors
        errors = [abs(p - a) for p, a in zip(predictions, actuals)]
        max_error = max(errors)
        min_error = min(errors)
        
        # Percentage of predictions within certain thresholds
        within_1pct = sum(1 for e, a in zip(errors, actuals) if e / a <= 0.01 if a != 0) / n
        within_5pct = sum(1 for e, a in zip(errors, actuals) if e / a <= 0.05 if a != 0) / n
        within_10pct = sum(1 for e, a in zip(errors, actuals) if e / a <= 0.10 if a != 0) / n
        
        metrics = {
            'model_name': model_name,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'directional_accuracy': directional_accuracy,
            'correlation': correlation,
            'max_error': max_error,
            'min_error': min_error,
            'within_1pct': within_1pct,
            'within_5pct': within_5pct,
            'within_10pct': within_10pct,
            'samples': n
        }
        
        self.metrics[model_name] = metrics
        return metrics
    
    def compare_models(self) -> Dict:
        """Compare all evaluated models"""
        if not self.metrics:
            return {}
        
        comparison = {
            'best_rmse': min(self.metrics.keys(), key=lambda k: self.metrics[k]['rmse']),
            'best_mae': min(self.metrics.keys(), key=lambda k: self.metrics[k]['mae']),
            'best_directional': max(self.metrics.keys(), key=lambda k: self.metrics[k]['directional_accuracy']),
            'best_correlation': max(self.metrics.keys(), key=lambda k: self.metrics[k]['correlation']),
        }
        
        return comparison
    
    def print_comparison_table(self):
        """Print a formatted comparison table"""
        if not self.metrics:
            print("No metrics available for comparison")
            return
        
        print("\n" + "=" * 120)
        print("MODEL COMPARISON TABLE")
        print("=" * 120)
        
        # Header
        header = f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'MAPE':<8} {'Dir.Acc':<8} {'Corr':<8} {'Within 5%':<10} {'Samples':<8}"
        print(header)
        print("-" * 120)
        
        # Sort models by RMSE
        sorted_models = sorted(self.metrics.keys(), key=lambda k: self.metrics[k]['rmse'])
        
        for model in sorted_models:
            m = self.metrics[model]
            row = f"{model:<20} {m['rmse']:<8.2f} {m['mae']:<8.2f} {m['mape']:<8.1f}% {m['directional_accuracy']:<8.1%} {m['correlation']:<8.3f} {m['within_5pct']:<10.1%} {m['samples']:<8}"
            print(row)
        
        print("\n" + "=" * 120)
        
        # Best models summary
        comparison = self.compare_models()
        print("BEST PERFORMING MODELS:")
        print(f"• Best RMSE: {comparison['best_rmse']}")
        print(f"• Best MAE: {comparison['best_mae']}")
        print(f"• Best Directional Accuracy: {comparison['best_directional']}")
        print(f"• Best Correlation: {comparison['best_correlation']}")


def create_visualization(predictions: List[float], actuals: List[float], title: str = "Predictions vs Actuals"):
    """Create text-based visualization"""
    print(f"\n{title}")
    print("=" * len(title))
    
    # Show first and last 10 points
    n = len(predictions)
    show_points = min(10, n // 2)
    
    print("First 10 predictions:")
    print("Index | Predicted | Actual    | Error     | Error%")
    print("-" * 50)
    for i in range(min(10, n)):
        error = predictions[i] - actuals[i]
        error_pct = (error / actuals[i] * 100) if actuals[i] != 0 else 0
        print(f"{i:5} | {predictions[i]:9.2f} | {actuals[i]:9.2f} | {error:+9.2f} | {error_pct:+6.1f}%")
    
    if n > 20:
        print("...")
        print("Last 10 predictions:")
        print("Index | Predicted | Actual    | Error     | Error%")
        print("-" * 50)
        for i in range(max(10, n-10), n):
            error = predictions[i] - actuals[i]
            error_pct = (error / actuals[i] * 100) if actuals[i] != 0 else 0
            print(f"{i:5} | {predictions[i]:9.2f} | {actuals[i]:9.2f} | {error:+9.2f} | {error_pct:+6.1f}%")


def evaluate_comprehensive_prediction():
    """Run comprehensive evaluation"""
    print("=" * 80)
    print("COMPREHENSIVE STOCK PREDICTION EVALUATION")
    print("=" * 80)
    
    # Configuration
    TICKERS = ["AAPL", "GOOGL", "MSFT"]
    START_DATE = "2023-01-01"
    END_DATE = "2024-01-01"
    
    evaluator = ModelEvaluator()
    
    for ticker in TICKERS:
        print(f"\n🔍 Evaluating {ticker}...")
        
        # Initialize predictor
        predictor = StockPredictor(ticker)
        
        # Fetch and prepare data
        data = predictor.fetch_data(START_DATE, END_DATE)
        features, targets = predictor.engineer_features(lookback_days=10)
        
        # Train model
        split_index = predictor.train_model(train_ratio=0.8)
        
        # Get test data
        X_test = predictor.features[split_index:]
        y_test = predictor.targets[split_index:]
        
        if not X_test:
            print(f"No test data for {ticker}, skipping...")
            continue
        
        # ML Model predictions
        ml_predictions = predictor.model.predict(X_test)
        evaluator.calculate_metrics(ml_predictions, y_test, f"ML Model ({ticker})")
        
        # Baseline model predictions
        test_prices = [predictor.targets[i] for i in range(split_index-len(y_test), split_index)]
        all_test_prices = test_prices + y_test
        
        # Naive baseline
        naive_pred = BaselineModels.naive_prediction(all_test_prices)
        if len(naive_pred) == len(y_test):
            evaluator.calculate_metrics(naive_pred, y_test, f"Naive ({ticker})")
        
        # Moving average baseline
        ma_pred = BaselineModels.moving_average_prediction(all_test_prices, window=5)
        if len(ma_pred) == len(y_test):
            evaluator.calculate_metrics(ma_pred, y_test, f"MA-5 ({ticker})")
        
        # Linear trend baseline
        trend_pred = BaselineModels.linear_trend_prediction(all_test_prices, window=10)
        if len(trend_pred) == len(y_test):
            evaluator.calculate_metrics(trend_pred, y_test, f"Trend ({ticker})")
        
        # Show visualization for first ticker
        if ticker == TICKERS[0]:
            create_visualization(ml_predictions, y_test, f"ML Model Predictions for {ticker}")
    
    # Print comprehensive comparison
    evaluator.print_comparison_table()
    
    # Save evaluation results
    results_file = "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(evaluator.metrics, f, indent=2)
    
    print(f"\n📊 Detailed results saved to {results_file}")
    
    return evaluator.metrics


def main():
    """Main evaluation function"""
    try:
        results = evaluate_comprehensive_prediction()
        
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        # Count total models evaluated
        ml_models = [k for k in results.keys() if "ML Model" in k]
        baseline_models = [k for k in results.keys() if "ML Model" not in k]
        
        print(f"✓ Total models evaluated: {len(results)}")
        print(f"  • ML Models: {len(ml_models)}")
        print(f"  • Baseline Models: {len(baseline_models)}")
        
        if ml_models and baseline_models:
            avg_ml_rmse = sum(results[k]['rmse'] for k in ml_models) / len(ml_models)
            avg_baseline_rmse = sum(results[k]['rmse'] for k in baseline_models) / len(baseline_models)
            
            improvement = ((avg_baseline_rmse - avg_ml_rmse) / avg_baseline_rmse * 100)
            
            print(f"\n📈 Average ML Model RMSE: ${avg_ml_rmse:.2f}")
            print(f"📊 Average Baseline RMSE: ${avg_baseline_rmse:.2f}")
            print(f"🎯 Improvement: {improvement:+.1f}%")
            
            if improvement > 0:
                print("✓ ML models outperform baselines on average")
            else:
                print("⚠ ML models need improvement to beat baselines")
        
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)