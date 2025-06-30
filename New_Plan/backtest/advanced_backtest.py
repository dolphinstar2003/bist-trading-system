"""
Advanced Backtest with ML Model and Portfolio Management
Uses trained GRU model for signal generation
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import torch
from typing import Dict, List, Tuple, Optional
from loguru import logger
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
import asyncio

from core.csv_data_manager import CSVDataManager
from core.feature_engineering import FeatureEngineering
from indicators.indicator_calculator import IndicatorCalculator
from models.simple_gru_model import SimpleMultiTimeframeGRU
from core.portfolio_manager import PortfolioManager


class MLBacktestEngine:
    """Advanced backtest engine with ML predictions and portfolio management"""
    
    def __init__(self, config_path: str = 'config.json'):
        # Load config
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.csv_manager = CSVDataManager()
        self.indicator_calc = IndicatorCalculator()
        self.feature_engineer = FeatureEngineering(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        
        # Load ML model
        self.model = self._load_model()
        
        # Backtest parameters
        self.initial_capital = self.config['backtest']['initial_capital']
        self.commission = self.config['backtest']['commission']
        self.slippage = self.config['backtest']['slippage']
        
        # Signal storage
        self.daily_signals = {}  # date -> list of signals
        self.execution_history = []
        
        logger.info("ML Backtest engine initialized")
    
    def _load_model(self) -> SimpleMultiTimeframeGRU:
        """Load trained GRU model"""
        model_path = Path('models/saved/gru_multi_timeframe.pth')
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Please train the model first.")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model
        model = SimpleMultiTimeframeGRU(
            input_size=checkpoint['config']['input_size'],
            hidden_size=checkpoint['config']['hidden_size'],
            num_layers=checkpoint['config']['num_layers']
        )
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Store scalers
        self.scalers = checkpoint['scalers']
        
        logger.info("ML model loaded successfully")
        return model
    
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str):
        """Run advanced backtest with ML predictions"""
        logger.info(f"Starting ML backtest from {start_date} to {end_date}")
        logger.info(f"Processing {len(symbols)} symbols")
        
        # Convert dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        # Get trading days
        all_dates = pd.date_range(start, end, freq='D')
        
        # Process each day
        for current_date in tqdm(all_dates, desc="Backtesting"):
            # Skip weekends
            if current_date.weekday() >= 5:
                continue
            
            # Update existing positions
            self._update_positions(current_date)
            
            # Generate signals for all symbols
            daily_predictions = self._generate_daily_predictions(symbols, current_date)
            
            # Rank and select best signals
            selected_signals = self._select_best_signals(daily_predictions)
            
            # Execute trades through portfolio manager
            self._execute_signals(selected_signals, current_date)
            
            # Record daily state
            self._record_daily_state(current_date)
        
        # Generate final report
        results = self._generate_report()
        
        return results
    
    def _generate_daily_predictions(self, symbols: List[str], current_date: pd.Timestamp) -> List[Dict]:
        """Generate ML predictions for all symbols"""
        predictions = []
        
        for symbol in symbols:
            try:
                # Skip if already in position
                if symbol in self.portfolio_manager.positions:
                    continue
                
                # Get multi-timeframe data
                data = self._prepare_symbol_data(symbol, current_date)
                if not data:
                    continue
                
                # Generate features
                features = self.feature_engineer.create_features(data, symbol)
                if not features:
                    continue
                
                # Get ML prediction
                prediction = self._get_ml_prediction(features)
                if prediction is None:
                    continue
                
                # Get current price and indicators
                current_price = data['indicators']['1h'].iloc[-1]['close']
                atr = data['indicators']['1h'].iloc[-1].get('atr', current_price * 0.02)
                
                # Calculate MACD confirmation
                macd_signal = self._check_macd_signal(data)
                
                # Create signal if conditions met
                if prediction['probability'] > 0.65 and macd_signal:
                    signal = {
                        'symbol': symbol,
                        'date': current_date,
                        'ml_probability': prediction['probability'],
                        'ml_confidence': prediction['confidence'],
                        'direction': 'buy',
                        'entry_price': current_price,
                        'stop_loss': current_price - 2 * atr,
                        'target_1': current_price + 2 * atr,
                        'target_2': current_price + 3 * atr,
                        'atr': atr,
                        'macd_confirmed': macd_signal,
                        'signal_strength': prediction['probability'] * (1 if macd_signal else 0.8)
                    }
                    
                    predictions.append(signal)
                    
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        return predictions
    
    def _prepare_symbol_data(self, symbol: str, current_date: pd.Timestamp) -> Optional[Dict]:
        """Prepare multi-timeframe data for a symbol"""
        data = {'indicators': {}}
        
        for tf in self.config['timeframes']['analysis']:
            df = self.csv_manager.get_raw_data(symbol, tf)
            if df is None:
                return None
            
            # Get data up to current date
            df = df[df.index <= current_date]
            if len(df) < 100:  # Need sufficient history
                return None
            
            # Calculate indicators if not available
            indicators = self.indicator_calc.calculate_all_indicators(symbol, tf, save=False)
            if not indicators.empty:
                df = pd.concat([df, indicators], axis=1)
            
            data['indicators'][tf] = df
        
        # Add dummy macro/sentiment for now
        data['macro'] = {'vix': 20, 'usdtry': 30}
        data['sentiment'] = {'score': 0, 'count': 0}
        
        return data
    
    def _get_ml_prediction(self, features: Dict[str, pd.DataFrame]) -> Optional[Dict]:
        """Get ML model prediction"""
        try:
            with torch.no_grad():
                # Prepare sequences for each timeframe
                sequences = {}
                
                for tf in ['1h', '4h', '1d']:
                    if tf not in features:
                        continue
                    
                    # Get last sequence_length rows
                    seq_length = self.config['model']['sequence_length']
                    df = features[tf]
                    
                    if len(df) < seq_length:
                        # Pad if necessary
                        pad_length = seq_length - len(df)
                        padding = pd.DataFrame(
                            np.zeros((pad_length, df.shape[1])),
                            columns=df.columns
                        )
                        df = pd.concat([padding, df])
                    else:
                        df = df.tail(seq_length)
                    
                    # Normalize using saved scalers
                    if tf in self.scalers:
                        normalized = self.scalers[tf].transform(df)
                    else:
                        # Fallback normalization
                        normalized = (df - df.mean()) / (df.std() + 1e-8)
                    
                    sequences[tf] = torch.FloatTensor(normalized).unsqueeze(0)
                
                # Model prediction
                x_15m = None  # Not used
                x_1h = sequences.get('1h', None)
                x_4h = sequences.get('4h', None)
                x_1d = sequences.get('1d', None)
                x_1w = None  # Not used
                
                output, attention_weights = self.model(x_15m, x_1h, x_4h, x_1d, x_1w)
                
                # Convert to probability
                probability = torch.sigmoid(output).item()
                
                # Calculate confidence based on attention weights
                confidence = 1.0  # Default
                if attention_weights is not None:
                    # Higher attention variance = lower confidence
                    for tf, weights in attention_weights.items():
                        if weights is not None:
                            variance = weights.var().item()
                            confidence *= (1 - variance)
                
                return {
                    'probability': probability,
                    'confidence': confidence,
                    'raw_output': output.item()
                }
                
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None
    
    def _check_macd_signal(self, data: Dict) -> bool:
        """Check MACD confirmation across timeframes"""
        confirmations = 0
        required_confirmations = 2
        
        for tf in ['1h', '4h', '1d']:
            if tf not in data['indicators']:
                continue
            
            df = data['indicators'][tf]
            if 'macd' not in df.columns or 'macd_signal' not in df.columns:
                continue
            
            # Check last two bars for crossover
            if len(df) >= 2:
                current = df.iloc[-1]
                prev = df.iloc[-2]
                
                # Bullish crossover or already above
                if current['macd'] > current['macd_signal']:
                    confirmations += 1
                    
                    # Extra point for fresh crossover
                    if prev['macd'] <= prev['macd_signal']:
                        confirmations += 0.5
        
        return confirmations >= required_confirmations
    
    def _select_best_signals(self, predictions: List[Dict]) -> List[Dict]:
        """Select best signals based on ML confidence and portfolio constraints"""
        if not predictions:
            return []
        
        # Sort by signal strength
        sorted_signals = sorted(predictions, key=lambda x: x['signal_strength'], reverse=True)
        
        # Get portfolio status
        portfolio_status = self.portfolio_manager.get_portfolio_status()
        risk_status = self.portfolio_manager.risk_check()
        
        # Check if we can open new positions
        if not risk_status['can_open_positions']:
            logger.debug("Cannot open new positions due to risk limits")
            return []
        
        # Calculate available slots
        available_slots = self.config['portfolio']['max_positions'] - portfolio_status['open_positions']
        
        # Select top signals
        selected = []
        for signal in sorted_signals[:available_slots]:
            # Additional filters
            if signal['ml_probability'] < self.config['signals']['confidence_threshold']:
                continue
            
            # Risk/reward check
            risk = abs(signal['entry_price'] - signal['stop_loss'])
            reward = signal['target_1'] - signal['entry_price']
            
            if reward / risk < 2.0:  # Minimum 2:1 R/R
                continue
            
            selected.append(signal)
        
        logger.info(f"Selected {len(selected)} signals from {len(predictions)} predictions")
        return selected
    
    def _execute_signals(self, signals: List[Dict], current_date: pd.Timestamp):
        """Execute selected signals through portfolio manager"""
        for signal in signals:
            try:
                # Create order through portfolio manager (synchronous call)
                order = asyncio.run(self.portfolio_manager.process_signal(signal))
                
                if order:
                    # Simulate order execution (in real trading, this would go to broker)
                    execution_price = signal['entry_price'] * (1 + self.slippage)
                    
                    # Execute order
                    success = self.portfolio_manager.execute_order(order, execution_price)
                    
                    if success:
                        self.execution_history.append({
                            'date': current_date,
                            'symbol': signal['symbol'],
                            'action': 'BUY',
                            'price': execution_price,
                            'ml_probability': signal['ml_probability'],
                            'signal_strength': signal['signal_strength']
                        })
                        
                        logger.info(f"Executed BUY {signal['symbol']} @ {execution_price:.2f}")
                
            except Exception as e:
                logger.error(f"Error executing signal for {signal['symbol']}: {e}")
    
    def _update_positions(self, current_date: pd.Timestamp):
        """Update all open positions"""
        for symbol in list(self.portfolio_manager.positions.keys()):
            try:
                # Get current price
                df = self.csv_manager.get_raw_data(symbol, '1h')
                if df is None or current_date not in df.index:
                    continue
                
                current_price = df.loc[current_date, 'close']
                
                # Update position
                result = self.portfolio_manager.update_position(symbol, current_price)
                
                # Check if position was closed
                if result and result.get('status') == 'closed':
                    self.execution_history.append({
                        'date': current_date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': current_price,
                        'reason': result.get('close_reason', 'unknown')
                    })
                    
                    logger.info(f"Closed {symbol} @ {current_price:.2f} - {result['close_reason']}")
                
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
    
    def _record_daily_state(self, date: pd.Timestamp):
        """Record daily portfolio state"""
        # This can be extended to save daily snapshots
        pass
    
    def _generate_report(self) -> Dict:
        """Generate comprehensive backtest report"""
        # Get final portfolio status
        final_status = self.portfolio_manager.get_portfolio_status()
        
        # Calculate additional metrics
        execution_df = pd.DataFrame(self.execution_history)
        
        # Monthly returns calculation
        if len(self.portfolio_manager.equity_curve) > 30:
            equity_series = pd.Series(self.portfolio_manager.equity_curve)
            monthly_returns = []
            
            for i in range(30, len(equity_series), 30):
                month_return = (equity_series.iloc[i] - equity_series.iloc[i-30]) / equity_series.iloc[i-30]
                monthly_returns.append(month_return)
            
            avg_monthly_return = np.mean(monthly_returns) * 100 if monthly_returns else 0
        else:
            avg_monthly_return = 0
        
        # Generate plots
        self._plot_results(final_status, execution_df)
        
        # Compile results
        results = {
            'portfolio_status': final_status,
            'total_trades': self.portfolio_manager.performance['total_trades'],
            'win_rate': final_status['win_rate'],
            'total_return': final_status['total_return'],
            'avg_monthly_return': avg_monthly_return,
            'sharpe_ratio': final_status['sharpe_ratio'],
            'max_drawdown': final_status['max_drawdown'],
            'profit_factor': final_status['profit_factor'],
            'execution_history': self.execution_history,
            'final_capital': final_status['total_equity']
        }
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("ML BACKTEST RESULTS")
        logger.info("="*60)
        logger.info(f"Total Return: {results['total_return']:.2f}%")
        logger.info(f"Avg Monthly Return: {results['avg_monthly_return']:.2f}%")
        logger.info(f"Total Trades: {results['total_trades']}")
        logger.info(f"Win Rate: {results['win_rate']:.1f}%")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.2f}%")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        logger.info(f"Final Capital: {results['final_capital']:,.0f} TRY")
        logger.info("="*60)
        
        # Save results
        with open('ml_backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _plot_results(self, portfolio_status: Dict, execution_df: pd.DataFrame):
        """Plot backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Equity Curve
        ax = axes[0, 0]
        equity_curve = self.portfolio_manager.equity_curve
        ax.plot(equity_curve, label='Portfolio Value')
        ax.axhline(y=self.initial_capital, color='r', linestyle='--', label='Initial Capital')
        ax.set_title('Equity Curve')
        ax.set_ylabel('Portfolio Value (TRY)')
        ax.legend()
        ax.grid(True)
        
        # 2. Drawdown
        ax = axes[0, 1]
        equity_series = pd.Series(equity_curve)
        rolling_max = equity_series.expanding().max()
        drawdown = (equity_series - rolling_max) / rolling_max * 100
        ax.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True)
        
        # 3. Win/Loss Distribution
        ax = axes[1, 0]
        if not execution_df.empty:
            buys = execution_df[execution_df['action'] == 'BUY']
            sells = execution_df[execution_df['action'] == 'SELL']
            
            # Match trades (simplified)
            trades_pnl = []
            for _, sell in sells.iterrows():
                symbol = sell['symbol']
                buy = buys[buys['symbol'] == symbol].iloc[-1] if len(buys[buys['symbol'] == symbol]) > 0 else None
                if buy is not None:
                    pnl_pct = (sell['price'] - buy['price']) / buy['price'] * 100
                    trades_pnl.append(pnl_pct)
            
            if trades_pnl:
                ax.hist(trades_pnl, bins=30, alpha=0.7, edgecolor='black')
                ax.axvline(x=0, color='r', linestyle='--')
                ax.set_xlabel('P&L (%)')
                ax.set_ylabel('Frequency')
                ax.set_title('Trade P&L Distribution')
        ax.grid(True)
        
        # 4. ML Probability Distribution
        ax = axes[1, 1]
        if 'ml_probability' in execution_df.columns:
            ml_probs = execution_df[execution_df['action'] == 'BUY']['ml_probability'].dropna()
            if not ml_probs.empty:
                ax.hist(ml_probs, bins=20, alpha=0.7, edgecolor='black')
                ax.axvline(x=0.65, color='r', linestyle='--', label='Threshold')
                ax.set_xlabel('ML Probability')
                ax.set_ylabel('Frequency')
                ax.set_title('ML Signal Distribution')
                ax.legend()
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('ml_backtest_results.png', dpi=300)
        logger.info("Plots saved to ml_backtest_results.png")


def main():
    """Run ML backtest"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run ML backtest for hybrid trading system')
    parser.add_argument('--symbols', nargs='+', help='Symbols to backtest (default: all available)')
    parser.add_argument('--start', default='2025-01-01', help='Start date')
    parser.add_argument('--end', default='2025-06-30', help='End date')
    parser.add_argument('--config', default='config.json', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize backtest engine
    engine = MLBacktestEngine(args.config)
    
    # Get symbols
    if args.symbols:
        symbols = args.symbols
    else:
        # Get all available symbols
        symbols = [s for s in engine.csv_manager.get_available_symbols() if not s.endswith('.IS')]
        logger.info(f"Using all {len(symbols)} available symbols")
    
    # Filter symbols with sufficient data
    valid_symbols = []
    for symbol in symbols:
        df = engine.csv_manager.get_raw_data(symbol, '1h')
        if df is not None and len(df) > 100:
            valid_symbols.append(symbol)
    
    logger.info(f"Found {len(valid_symbols)} symbols with sufficient data")
    
    # Run backtest
    results = engine.run_backtest(valid_symbols, args.start, args.end)
    
    logger.info("\nML Backtest completed successfully!")
    
    # Save detailed report
    report = {
        'config': engine.config,
        'symbols': valid_symbols,
        'period': f"{args.start} to {args.end}",
        'results': results
    }
    
    with open('ml_backtest_report.json', 'w') as f:
        json.dump(report, f, indent=2, default=str)


if __name__ == "__main__":
    main()