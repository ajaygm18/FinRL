#!/usr/bin/env python3
"""
Unit Tests for Stock Prediction System
======================================

Test suite for the stock prediction functionality to ensure reliability
and correctness of the implementation.
"""

import unittest
import tempfile
import os
import json
from typing import List

from stock_prediction_demo import (
    StockDataFetcher, TechnicalIndicators, SimpleMLModel, StockPredictor
)
from evaluation_suite import BaselineModels, ModelEvaluator


class TestStockDataFetcher(unittest.TestCase):
    """Test stock data fetching"""
    
    def setUp(self):
        self.fetcher = StockDataFetcher("TEST", "2023-01-01", "2023-01-10")
    
    def test_data_generation(self):
        """Test that data is generated correctly"""
        data = self.fetcher.generate_realistic_data()
        
        # Check data structure
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        
        # Check data format
        for item in data[:5]:  # Check first 5 items
            self.assertIn('date', item)
            self.assertIn('ticker', item)
            self.assertIn('open', item)
            self.assertIn('high', item)
            self.assertIn('low', item)
            self.assertIn('close', item)
            self.assertIn('volume', item)
            
            # Basic validation
            self.assertGreaterEqual(item['high'], item['low'])
            self.assertGreaterEqual(item['high'], item['open'])
            self.assertGreaterEqual(item['high'], item['close'])
            self.assertGreaterEqual(item['open'], item['low'])
            self.assertGreaterEqual(item['close'], item['low'])
            self.assertGreater(item['volume'], 0)
    
    def test_date_range(self):
        """Test that data covers the correct date range"""
        data = self.fetcher.generate_realistic_data()
        
        first_date = data[0]['date']
        last_date = data[-1]['date']
        
        self.assertEqual(first_date, "2023-01-01")
        self.assertEqual(last_date, "2023-01-10")


class TestTechnicalIndicators(unittest.TestCase):
    """Test technical indicators calculation"""
    
    def setUp(self):
        self.prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    
    def test_moving_average(self):
        """Test moving average calculation"""
        ma = TechnicalIndicators.moving_average(self.prices, 3)
        
        self.assertEqual(len(ma), len(self.prices))
        self.assertEqual(ma[0], self.prices[0])  # First value should be price itself
        self.assertEqual(ma[1], self.prices[1])  # Second value should be price itself
        
        # Third value should be average of first three
        expected_third = (self.prices[0] + self.prices[1] + self.prices[2]) / 3
        self.assertAlmostEqual(ma[2], expected_third, places=2)
    
    def test_rsi_calculation(self):
        """Test RSI calculation"""
        rsi = TechnicalIndicators.rsi(self.prices, window=5)
        
        self.assertEqual(len(rsi), len(self.prices))
        
        # RSI should be between 0 and 100
        for value in rsi:
            self.assertGreaterEqual(value, 0)
            self.assertLessEqual(value, 100)
    
    def test_volatility_calculation(self):
        """Test volatility calculation"""
        vol = TechnicalIndicators.volatility(self.prices, window=5)
        
        self.assertEqual(len(vol), len(self.prices))
        
        # Volatility should be non-negative
        for value in vol:
            self.assertGreaterEqual(value, 0)


class TestSimpleMLModel(unittest.TestCase):
    """Test machine learning model"""
    
    def setUp(self):
        self.model = SimpleMLModel()
        # Simple test data: y = 2*x1 + 3*x2 + 1
        self.X = [[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]]
        self.y = [8, 14, 20, 26, 32]  # 2*x1 + 3*x2 + 1 + noise
    
    def test_normalization(self):
        """Test feature normalization"""
        X_norm = self.model.normalize_features(self.X, fit=True)
        
        self.assertEqual(len(X_norm), len(self.X))
        self.assertEqual(len(X_norm[0]), len(self.X[0]))
        
        # Check that means and stds are calculated
        self.assertIsNotNone(self.model.feature_means)
        self.assertIsNotNone(self.model.feature_stds)
    
    def test_training(self):
        """Test model training"""
        self.model.train(self.X, self.y, epochs=100, learning_rate=0.1)
        
        # Check that weights and bias are set
        self.assertIsNotNone(self.model.weights)
        self.assertIsNotNone(self.model.bias)
        self.assertEqual(len(self.model.weights), len(self.X[0]))
    
    def test_prediction(self):
        """Test model prediction"""
        self.model.train(self.X, self.y, epochs=100, learning_rate=0.1)
        
        predictions = self.model.predict(self.X)
        
        self.assertEqual(len(predictions), len(self.X))
        
        # Predictions should be reasonable (not testing exact accuracy due to randomness)
        for pred in predictions:
            self.assertIsInstance(pred, (int, float))
    
    def test_prediction_without_training(self):
        """Test that prediction fails without training"""
        with self.assertRaises(ValueError):
            self.model.predict(self.X)


class TestStockPredictor(unittest.TestCase):
    """Test stock predictor functionality"""
    
    def setUp(self):
        self.predictor = StockPredictor("TEST")
    
    def test_fetch_data(self):
        """Test data fetching"""
        data = self.predictor.fetch_data("2023-01-01", "2023-01-10")
        
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)
        self.assertEqual(self.predictor.data, data)
    
    def test_feature_engineering(self):
        """Test feature engineering"""
        self.predictor.fetch_data("2023-01-01", "2023-01-31")
        features, targets = self.predictor.engineer_features(lookback_days=5)
        
        self.assertIsInstance(features, list)
        self.assertIsInstance(targets, list)
        self.assertEqual(len(features), len(targets))
        self.assertGreater(len(features), 0)
    
    def test_model_training(self):
        """Test model training"""
        self.predictor.fetch_data("2023-01-01", "2023-01-31")
        self.predictor.engineer_features(lookback_days=5)
        
        split_index = self.predictor.train_model(train_ratio=0.8)
        
        self.assertIsInstance(split_index, int)
        self.assertGreater(split_index, 0)
        self.assertLess(split_index, len(self.predictor.features))
    
    def test_model_evaluation(self):
        """Test model evaluation"""
        self.predictor.fetch_data("2023-01-01", "2023-01-31")
        self.predictor.engineer_features(lookback_days=5)
        split_index = self.predictor.train_model(train_ratio=0.8)
        
        metrics = self.predictor.evaluate_model(split_index)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('rmse', metrics)
        self.assertIn('mae', metrics)
        self.assertIn('mape', metrics)
        self.assertIn('directional_accuracy', metrics)
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        self.predictor.fetch_data("2023-01-01", "2023-01-31")
        self.predictor.engineer_features(lookback_days=5)
        self.predictor.train_model(train_ratio=0.8)
        
        # Save model
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        try:
            self.predictor.save_model(temp_path)
            self.assertTrue(os.path.exists(temp_path))
            
            # Load model
            new_predictor = StockPredictor("DUMMY")
            new_predictor.load_model(temp_path)
            
            self.assertEqual(new_predictor.ticker, "TEST")
            self.assertIsNotNone(new_predictor.model.weights)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestBaselineModels(unittest.TestCase):
    """Test baseline models"""
    
    def setUp(self):
        self.prices = [100, 102, 101, 103, 105, 104, 106, 108, 107, 109]
    
    def test_naive_prediction(self):
        """Test naive prediction model"""
        predictions = BaselineModels.naive_prediction(self.prices)
        
        self.assertEqual(len(predictions), len(self.prices) - 1)
        
        # Each prediction should be the previous price
        for i, pred in enumerate(predictions):
            self.assertEqual(pred, self.prices[i])
    
    def test_moving_average_prediction(self):
        """Test moving average prediction model"""
        predictions = BaselineModels.moving_average_prediction(self.prices, window=3)
        
        self.assertEqual(len(predictions), len(self.prices) - 1)
        self.assertIsInstance(predictions[0], (int, float))
    
    def test_linear_trend_prediction(self):
        """Test linear trend prediction model"""
        predictions = BaselineModels.linear_trend_prediction(self.prices, window=5)
        
        self.assertEqual(len(predictions), len(self.prices) - 1)
        self.assertIsInstance(predictions[0], (int, float))


class TestModelEvaluator(unittest.TestCase):
    """Test model evaluator"""
    
    def setUp(self):
        self.evaluator = ModelEvaluator()
        self.predictions = [100, 102, 101, 103, 105]
        self.actuals = [99, 103, 100, 104, 106]
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        metrics = self.evaluator.calculate_metrics(self.predictions, self.actuals, "Test Model")
        
        # Check required metrics
        required_metrics = ['rmse', 'mae', 'mape', 'directional_accuracy', 'correlation']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
        
        # Check ranges
        self.assertGreaterEqual(metrics['directional_accuracy'], 0)
        self.assertLessEqual(metrics['directional_accuracy'], 1)
        self.assertGreaterEqual(metrics['correlation'], -1)
        self.assertLessEqual(metrics['correlation'], 1)
    
    def test_mismatched_lengths(self):
        """Test handling of mismatched prediction/actual lengths"""
        with self.assertRaises(ValueError):
            self.evaluator.calculate_metrics([1, 2, 3], [1, 2], "Test")
    
    def test_compare_models(self):
        """Test model comparison"""
        self.evaluator.calculate_metrics(self.predictions, self.actuals, "Model A")
        self.evaluator.calculate_metrics([99, 103, 100, 104, 106], self.actuals, "Model B")
        
        comparison = self.evaluator.compare_models()
        
        self.assertIn('best_rmse', comparison)
        self.assertIn('best_mae', comparison)
        self.assertIn('best_directional', comparison)
        self.assertIn('best_correlation', comparison)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system"""
    
    def test_end_to_end_prediction(self):
        """Test complete prediction workflow"""
        predictor = StockPredictor("INTEGRATION_TEST")
        
        # Complete workflow
        data = predictor.fetch_data("2023-01-01", "2023-02-01")
        features, targets = predictor.engineer_features(lookback_days=5)
        split_index = predictor.train_model(train_ratio=0.8)
        metrics = predictor.evaluate_model(split_index)
        
        # Verify results
        self.assertGreater(len(data), 20)  # Should have at least 20 days of data
        self.assertGreater(len(features), 10)  # Should have features
        self.assertIsInstance(metrics['rmse'], (int, float))
        self.assertGreater(metrics['rmse'], 0)
    
    def test_multiple_tickers(self):
        """Test prediction with multiple tickers"""
        tickers = ["TEST1", "TEST2"]
        evaluator = ModelEvaluator()
        
        for ticker in tickers:
            predictor = StockPredictor(ticker)
            data = predictor.fetch_data("2023-01-01", "2023-01-31")
            features, targets = predictor.engineer_features(lookback_days=5)
            split_index = predictor.train_model(train_ratio=0.8)
            
            X_test = predictor.features[split_index:]
            y_test = predictor.targets[split_index:]
            
            if X_test and y_test:
                predictions = predictor.model.predict(X_test)
                metrics = evaluator.calculate_metrics(predictions, y_test, f"Model ({ticker})")
                
                self.assertIn(f"Model ({ticker})", evaluator.metrics)


def run_tests():
    """Run all tests with detailed output"""
    print("=" * 80)
    print("RUNNING STOCK PREDICTION UNIT TESTS")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromModule(__import__(__name__))
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2, stream=None)
    result = runner.run(test_suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    success = len(result.failures) == 0 and len(result.errors) == 0
    
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
    
    return success


if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)