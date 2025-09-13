#!/usr/bin/env python3
"""
Complete Stock Prediction Demo
==============================

This script demonstrates the complete stock prediction pipeline
including multiple models, comprehensive evaluation, and results summary.
"""

import sys
import os
from stock_prediction_demo import StockPredictor
from evaluation_suite import ModelEvaluator, BaselineModels


def run_complete_demo():
    """Run the complete demonstration"""
    print("=" * 80)
    print("COMPLETE STOCK PREDICTION DEMONSTRATION")
    print("Built on FinRL Framework")
    print("=" * 80)
    
    print("\n🎯 This demo will:")
    print("1. Train ML models for multiple stock tickers")
    print("2. Compare with baseline prediction models")
    print("3. Evaluate accuracy using multiple metrics")
    print("4. Provide interpretations and recommendations")
    
    # Configuration
    TICKERS = ["AAPL", "GOOGL", "TSLA"]
    START_DATE = "2023-01-01"
    END_DATE = "2024-01-01"
    
    evaluator = ModelEvaluator()
    all_results = {}
    
    print(f"\n📊 Analyzing {len(TICKERS)} stocks: {', '.join(TICKERS)}")
    print(f"📅 Period: {START_DATE} to {END_DATE}")
    
    for i, ticker in enumerate(TICKERS, 1):
        print(f"\n{'='*60}")
        print(f"🔍 ANALYZING STOCK {i}/{len(TICKERS)}: {ticker}")
        print(f"{'='*60}")
        
        try:
            # Initialize predictor
            predictor = StockPredictor(ticker)
            
            # Step 1: Data preparation
            print("📈 Fetching and preparing data...")
            data = predictor.fetch_data(START_DATE, END_DATE)
            features, targets = predictor.engineer_features(lookback_days=10)
            
            print(f"   • Data points: {len(data)}")
            print(f"   • Feature vectors: {len(features)}")
            print(f"   • Features per vector: {len(features[0]) if features else 0}")
            
            # Step 2: Train ML model
            print("🤖 Training ML model...")
            split_index = predictor.train_model(train_ratio=0.8)
            
            # Step 3: Prepare test data
            X_test = predictor.features[split_index:]
            y_test = predictor.targets[split_index:]
            
            if not X_test or not y_test:
                print(f"   ⚠️ Insufficient test data for {ticker}, skipping...")
                continue
            
            print(f"   • Training samples: {split_index}")
            print(f"   • Test samples: {len(y_test)}")
            
            # Step 4: ML model evaluation
            print("📊 Evaluating ML model...")
            ml_predictions = predictor.model.predict(X_test)
            ml_metrics = evaluator.calculate_metrics(ml_predictions, y_test, f"ML-{ticker}")
            
            # Step 5: Baseline models
            print("🔄 Testing baseline models...")
            
            # Get price history for baseline models
            lookback_length = len(y_test)
            historical_prices = [predictor.targets[i] for i in range(split_index-lookback_length, split_index)]
            full_test_prices = historical_prices + y_test
            
            # Naive baseline
            naive_pred = BaselineModels.naive_prediction(full_test_prices)
            if len(naive_pred) >= len(y_test):
                naive_pred = naive_pred[-len(y_test):]
                evaluator.calculate_metrics(naive_pred, y_test, f"Naive-{ticker}")
            
            # Moving average baseline
            ma_pred = BaselineModels.moving_average_prediction(full_test_prices, window=5)
            if len(ma_pred) >= len(y_test):
                ma_pred = ma_pred[-len(y_test):]
                evaluator.calculate_metrics(ma_pred, y_test, f"MA5-{ticker}")
            
            # Store results for this ticker
            all_results[ticker] = {
                'ml_metrics': ml_metrics,
                'data_points': len(data),
                'test_samples': len(y_test),
                'features': len(features[0]) if features else 0
            }
            
            # Quick summary for this ticker
            print(f"   ✓ ML RMSE: ${ml_metrics['rmse']:.2f}")
            print(f"   ✓ ML MAPE: {ml_metrics['mape']:.1f}%")
            print(f"   ✓ Directional Accuracy: {ml_metrics['directional_accuracy']:.1%}")
            
        except Exception as e:
            print(f"   ❌ Error processing {ticker}: {e}")
            continue
    
    # Overall results
    print(f"\n{'='*80}")
    print("📊 COMPREHENSIVE RESULTS COMPARISON")
    print(f"{'='*80}")
    
    if evaluator.metrics:
        evaluator.print_comparison_table()
        
        # Analysis by model type
        ml_models = {k: v for k, v in evaluator.metrics.items() if k.startswith('ML-')}
        baseline_models = {k: v for k, v in evaluator.metrics.items() if not k.startswith('ML-')}
        
        if ml_models and baseline_models:
            print(f"\n{'='*80}")
            print("🎯 MODEL TYPE COMPARISON")
            print(f"{'='*80}")
            
            # Calculate averages
            avg_ml_rmse = sum(m['rmse'] for m in ml_models.values()) / len(ml_models)
            avg_ml_mape = sum(m['mape'] for m in ml_models.values()) / len(ml_models)
            avg_ml_dir = sum(m['directional_accuracy'] for m in ml_models.values()) / len(ml_models)
            
            avg_baseline_rmse = sum(m['rmse'] for m in baseline_models.values()) / len(baseline_models)
            avg_baseline_mape = sum(m['mape'] for m in baseline_models.values()) / len(baseline_models)
            avg_baseline_dir = sum(m['directional_accuracy'] for m in baseline_models.values()) / len(baseline_models)
            
            print(f"📈 ML Models Average Performance:")
            print(f"   • RMSE: ${avg_ml_rmse:.2f}")
            print(f"   • MAPE: {avg_ml_mape:.1f}%")
            print(f"   • Directional Accuracy: {avg_ml_dir:.1%}")
            
            print(f"\n📊 Baseline Models Average Performance:")
            print(f"   • RMSE: ${avg_baseline_rmse:.2f}")
            print(f"   • MAPE: {avg_baseline_mape:.1f}%")
            print(f"   • Directional Accuracy: {avg_baseline_dir:.1%}")
            
            # Improvement analysis
            rmse_improvement = ((avg_baseline_rmse - avg_ml_rmse) / avg_baseline_rmse) * 100
            mape_improvement = ((avg_baseline_mape - avg_ml_mape) / avg_baseline_mape) * 100
            dir_improvement = ((avg_ml_dir - avg_baseline_dir) / avg_baseline_dir) * 100
            
            print(f"\n🎯 ML Model Improvements:")
            print(f"   • RMSE: {rmse_improvement:+.1f}%")
            print(f"   • MAPE: {mape_improvement:+.1f}%")
            print(f"   • Directional Accuracy: {dir_improvement:+.1f}%")
    
    # Final recommendations
    print(f"\n{'='*80}")
    print("💡 INSIGHTS AND RECOMMENDATIONS")
    print(f"{'='*80}")
    
    if all_results:
        best_ticker = min(all_results.keys(), key=lambda t: all_results[t]['ml_metrics']['rmse'])
        best_metrics = all_results[best_ticker]['ml_metrics']
        
        print(f"🏆 Best performing stock: {best_ticker}")
        print(f"   • RMSE: ${best_metrics['rmse']:.2f}")
        print(f"   • MAPE: {best_metrics['mape']:.1f}%")
        print(f"   • Directional Accuracy: {best_metrics['directional_accuracy']:.1%}")
        
        avg_mape = sum(r['ml_metrics']['mape'] for r in all_results.values()) / len(all_results)
        avg_dir_acc = sum(r['ml_metrics']['directional_accuracy'] for r in all_results.values()) / len(all_results)
        
        print(f"\n📈 Overall Model Performance:")
        if avg_mape < 5:
            print("   ✅ Excellent: Low prediction error (MAPE < 5%)")
        elif avg_mape < 10:
            print("   ✅ Good: Moderate prediction error (MAPE < 10%)")
        else:
            print("   ⚠️ Fair: High prediction error - consider model improvements")
        
        if avg_dir_acc > 0.55:
            print("   ✅ Good trend prediction capability")
        else:
            print("   ⚠️ Trend prediction needs improvement")
        
        print(f"\n🔮 For Production Use:")
        print("   • Integrate real market data APIs (Yahoo Finance, Alpha Vantage)")
        print("   • Add more features (sentiment, volume, macro indicators)")
        print("   • Consider ensemble methods or deep learning models")
        print("   • Implement proper risk management and position sizing")
        print("   • Add real-time monitoring and model retraining")
        
        print(f"\n⚠️ Important Disclaimers:")
        print("   • This demo uses simulated data for demonstration purposes")
        print("   • Past performance does not guarantee future results")
        print("   • Always validate models with real data before trading")
        print("   • Consider transaction costs and market conditions")
    
    else:
        print("❌ No successful predictions completed")
        return False
    
    # Save summary
    print(f"\n💾 Saving detailed results...")
    summary = {
        'tickers_analyzed': list(all_results.keys()),
        'total_models': len(evaluator.metrics),
        'ml_models': len([k for k in evaluator.metrics.keys() if k.startswith('ML-')]),
        'baseline_models': len([k for k in evaluator.metrics.keys() if not k.startswith('ML-')]),
        'detailed_metrics': evaluator.metrics,
        'summary_by_ticker': all_results
    }
    
    with open('complete_demo_results.json', 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    print("   ✓ Results saved to: complete_demo_results.json")
    print("   ✓ Model comparison saved to: evaluation_results.json")
    
    return True


if __name__ == "__main__":
    print("Starting Complete Stock Prediction Demo...")
    success = run_complete_demo()
    
    if success:
        print(f"\n{'='*80}")
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        print("\nFiles created:")
        print("• stock_prediction_demo.py - Main prediction implementation")
        print("• evaluation_suite.py - Comprehensive evaluation tools")
        print("• test_stock_prediction.py - Unit test suite")
        print("• README_STOCK_PREDICTION.md - Complete documentation")
        print("• complete_demo_results.json - Demo results summary")
        
        print(f"\nNext steps:")
        print("• Run individual demos: python stock_prediction_demo.py")
        print("• Run evaluations: python evaluation_suite.py")
        print("• Run tests: python test_stock_prediction.py")
        print("• Read documentation: README_STOCK_PREDICTION.md")
    else:
        print("\n❌ Demo failed. Check error messages above.")
    
    sys.exit(0 if success else 1)