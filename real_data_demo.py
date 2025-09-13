#!/usr/bin/env python3
"""
Real Data Stock Prediction Demo
==============================

This script demonstrates stock price prediction using REAL market data
from Yahoo Finance, as requested.
"""

from stock_prediction_demo import StockPredictor

def main():
    print("🎯 REAL DATA STOCK PREDICTION DEMO")
    print("=" * 50)
    print("This demo uses REAL market data from Yahoo Finance!")
    print()
    
    # Test with a popular stock
    ticker = "AAPL"
    print(f"📈 Analyzing {ticker} with real market data...")
    
    # Initialize predictor
    predictor = StockPredictor(ticker)
    
    # Fetch REAL data
    print("\n1. Fetching real market data...")
    data = predictor.fetch_data("2024-01-01", "2024-12-01", use_real_data=True)
    print(f"   ✓ Fetched {len(data)} days of real market data")
    
    # Show sample real data
    print("\n📊 Sample real market data:")
    for i in range(min(3, len(data))):
        d = data[i]
        print(f"   {d['date']}: Open=${d['open']}, High=${d['high']}, Low=${d['low']}, Close=${d['close']}")
    
    # Engineer features
    print("\n2. Engineering features...")
    features, targets = predictor.engineer_features()
    print(f"   ✓ Created {len(features)} feature vectors with {len(features[0])} features each")
    
    # Train model
    print("\n3. Training ML model...")
    split_index = predictor.train_model()
    print(f"   ✓ Trained with {split_index} samples")
    
    # Evaluate
    print("\n4. Evaluating predictions...")
    metrics = predictor.evaluate_model(split_index)
    
    print(f"\n🎯 RESULTS ON REAL DATA:")
    print(f"   💰 RMSE: ${metrics['rmse']:.2f}")
    print(f"   📊 MAPE: {metrics['mape']:.1f}%") 
    print(f"   🎯 Directional Accuracy: {metrics['directional_accuracy']:.1f}%")
    
    print(f"\n✅ Successfully predicted {ticker} prices using REAL market data!")
    print("   This demonstrates that the system works with actual stock market data,")
    print("   not just simulated data.")

if __name__ == "__main__":
    main()