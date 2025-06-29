"""
Deep Learning Backtesting Framework

This module provides a comprehensive backtesting framework specifically designed
for deep learning trading models with proper time series validation.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from datetime import datetime, timedelta
import json
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeepLearningBacktester:
    """Comprehensive backtesting framework for DL models"""
    
    def __init__(self, initial_capital: float = 100000, commission: float = 0.001,
                 slippage: float = 0.0005, min_position_size: float = 1000,
                 max_positions: int = 10, risk_per_trade: float = 0.02):
        """
        Initialize the backtesting framework
        
        Args:
            initial_capital: Starting capital
            commission: Trading commission rate
            slippage: Slippage rate
            min_position_size: Minimum position size
            max_positions: Maximum number of positions
            risk_per_trade: Risk per trade as fraction of capital
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.min_position_size = min_position_size
        self.max_positions = max_positions
        self.risk_per_trade = risk_per_trade
        
        # Portfolio state
        self.reset_portfolio()
        
        # Results storage
        self.results = {}
        self.trade_history = []
        self.portfolio_history = []
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_capital
        self.positions = {}  # {symbol: {'quantity': float, 'avg_price': float, 'entry_date': datetime}}
        self.portfolio_value = self.initial_capital
        self.trade_count = 0
        self.winning_trades = 0
        self.losing_trades = 0
    
    def backtest_model(self, model: object, df: pd.DataFrame, 
                      signal_generator: Callable, model_type: str = 'dl',
                      walk_forward: bool = True, window_size: int = 252,
                      step_size: int = 21) -> Dict:
        """
        Backtest a deep learning model
        
        Args:
            model: Trained DL model with predict method
            df: DataFrame with OHLCV data
            signal_generator: Function to generate signals from model predictions
            model_type: Type of model ('lstm', 'cnn', 'transformer', 'rl')
            walk_forward: Whether to use walk-forward analysis
            window_size: Training window size (days)
            step_size: Step size for walk-forward (days)
        
        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest for {model_type} model...")
        
        self.reset_portfolio()
        self.trade_history = []
        self.portfolio_history = []
        
        if walk_forward:
            results = self._walk_forward_backtest(
                model, df, signal_generator, model_type, window_size, step_size
            )
        else:
            results = self._simple_backtest(model, df, signal_generator, model_type)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()
        results.update(metrics)
        
        # Generate report
        self._generate_backtest_report(results, model_type)
        
        return results
    
    def _walk_forward_backtest(self, model: object, df: pd.DataFrame,
                              signal_generator: Callable, model_type: str,
                              window_size: int, step_size: int) -> Dict:
        """Perform walk-forward backtesting"""
        results = {
            'dates': [],
            'portfolio_values': [],
            'returns': [],
            'positions': []
        }
        
        # Start from minimum required data
        start_idx = window_size
        
        while start_idx + step_size <= len(df):
            # Get training and testing data
            train_end_idx = start_idx
            test_end_idx = min(start_idx + step_size, len(df))
            
            train_data = df.iloc[:train_end_idx]
            test_data = df.iloc[train_end_idx:test_end_idx]
            
            if len(test_data) == 0:
                break
            
            try:
                # Retrain model if needed (for online learning)
                if hasattr(model, 'partial_fit'):
                    model.partial_fit(train_data)
                
                # Generate signals for test period
                signals = signal_generator(model, test_data, model_type)
                
                # Execute trades
                for idx, (date, signal_row) in enumerate(signals.iterrows()):
                    current_price = test_data.loc[date, 'Close']
                    
                    # Update portfolio value
                    self._update_portfolio_value(current_price, date)
                    
                    # Process signal
                    if 'signal' in signal_row and signal_row['signal'] != 0:
                        confidence = signal_row.get('confidence', 0.5)
                        self._process_signal(
                            signal_row['signal'],
                            current_price,
                            date,
                            confidence,
                            'TestSymbol'  # In real implementation, this would be the actual symbol
                        )
                    
                    # Record state
                    results['dates'].append(date)
                    results['portfolio_values'].append(self.portfolio_value)
                    results['returns'].append(
                        (self.portfolio_value - self.initial_capital) / self.initial_capital
                    )
                    results['positions'].append(len(self.positions))
                    
                    # Save to history
                    self.portfolio_history.append({
                        'date': date,
                        'portfolio_value': self.portfolio_value,
                        'cash': self.cash,
                        'positions': len(self.positions),
                        'return': results['returns'][-1]
                    })
                
            except Exception as e:
                logger.error(f"Error in walk-forward step: {str(e)}")
                continue
            
            # Move to next window
            start_idx += step_size
        
        return results
    
    def _simple_backtest(self, model: object, df: pd.DataFrame,
                        signal_generator: Callable, model_type: str) -> Dict:
        """Perform simple backtesting without walk-forward"""
        results = {
            'dates': [],
            'portfolio_values': [],
            'returns': [],
            'positions': []
        }
        
        # Generate all signals
        signals = signal_generator(model, df, model_type)
        
        # Execute trades
        for date, signal_row in signals.iterrows():
            if date not in df.index:
                continue
                
            current_price = df.loc[date, 'Close']
            
            # Update portfolio value
            self._update_portfolio_value(current_price, date)
            
            # Process signal
            if 'signal' in signal_row and signal_row['signal'] != 0:
                confidence = signal_row.get('confidence', 0.5)
                self._process_signal(
                    signal_row['signal'],
                    current_price,
                    date,
                    confidence,
                    'TestSymbol'
                )
            
            # Record state
            results['dates'].append(date)
            results['portfolio_values'].append(self.portfolio_value)
            results['returns'].append(
                (self.portfolio_value - self.initial_capital) / self.initial_capital
            )
            results['positions'].append(len(self.positions))
            
            # Save to history
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions': len(self.positions),
                'return': results['returns'][-1]
            })
        
        return results
    
    def _process_signal(self, signal: int, price: float, date: datetime,
                       confidence: float, symbol: str):
        """Process trading signal"""
        if signal > 0:  # Buy signal
            self._execute_buy(symbol, price, date, confidence)
        elif signal < 0:  # Sell signal
            self._execute_sell(symbol, price, date)
    
    def _execute_buy(self, symbol: str, price: float, date: datetime, confidence: float):
        """Execute buy order"""
        if len(self.positions) >= self.max_positions:
            return
        
        if symbol in self.positions:
            return  # Already have position
        
        # Calculate position size based on confidence and risk
        available_capital = self.cash * 0.95  # Keep 5% cash buffer
        position_size = min(
            available_capital * self.risk_per_trade * confidence,
            available_capital / (self.max_positions - len(self.positions))
        )
        
        if position_size < self.min_position_size:
            return
        
        # Apply slippage and commission
        execution_price = price * (1 + self.slippage)
        quantity = position_size / execution_price
        cost = quantity * execution_price * (1 + self.commission)
        
        if cost > self.cash:
            return
        
        # Execute trade
        self.cash -= cost
        self.positions[symbol] = {
            'quantity': quantity,
            'avg_price': execution_price,
            'entry_date': date,
            'entry_value': quantity * execution_price
        }
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'buy',
            'price': execution_price,
            'quantity': quantity,
            'value': cost,
            'confidence': confidence
        })
        
        self.trade_count += 1
    
    def _execute_sell(self, symbol: str, price: float, date: datetime):
        """Execute sell order"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        
        # Apply slippage and commission
        execution_price = price * (1 - self.slippage)
        proceeds = position['quantity'] * execution_price * (1 - self.commission)
        
        # Calculate profit/loss
        entry_value = position['entry_value']
        pnl = proceeds - entry_value
        pnl_pct = pnl / entry_value
        
        # Update win/loss statistics
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Execute trade
        self.cash += proceeds
        del self.positions[symbol]
        
        # Record trade
        self.trade_history.append({
            'date': date,
            'symbol': symbol,
            'action': 'sell',
            'price': execution_price,
            'quantity': position['quantity'],
            'value': proceeds,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'holding_period': (date - position['entry_date']).days
        })
    
    def _update_portfolio_value(self, current_price: float, date: datetime):
        """Update portfolio value"""
        positions_value = sum(
            pos['quantity'] * current_price for pos in self.positions.values()
        )
        self.portfolio_value = self.cash + positions_value
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.portfolio_history:
            return {}
        
        # Convert history to DataFrame
        df_history = pd.DataFrame(self.portfolio_history)
        df_trades = pd.DataFrame(self.trade_history)
        
        # Basic metrics
        total_return = (self.portfolio_value - self.initial_capital) / self.initial_capital
        
        # Calculate daily returns
        df_history['daily_return'] = df_history['portfolio_value'].pct_change().fillna(0)
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe_ratio(df_history['daily_return'])
        sortino_ratio = self._calculate_sortino_ratio(df_history['daily_return'])
        max_drawdown = self._calculate_max_drawdown(df_history['portfolio_value'])
        
        # Trade metrics
        win_rate = self.winning_trades / max(self.trade_count, 1)
        
        # Calculate average win/loss
        if not df_trades.empty and 'pnl_pct' in df_trades.columns:
            winning_trades = df_trades[df_trades['pnl_pct'] > 0]
            losing_trades = df_trades[df_trades['pnl_pct'] <= 0]
            
            avg_win = winning_trades['pnl_pct'].mean() if not winning_trades.empty else 0
            avg_loss = abs(losing_trades['pnl_pct'].mean()) if not losing_trades.empty else 0
            profit_factor = (avg_win * win_rate) / (avg_loss * (1 - win_rate)) if avg_loss > 0 else 0
        else:
            avg_win = avg_loss = profit_factor = 0
        
        metrics = {
            'total_return': total_return,
            'annualized_return': self._annualize_return(total_return, len(df_history)),
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': self.trade_count,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'final_portfolio_value': self.portfolio_value,
            'final_cash': self.cash,
            'final_positions': len(self.positions)
        }
        
        return metrics
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / excess_returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Sortino ratio"""
        downside_returns = returns[returns < 0]
        if downside_returns.empty or downside_returns.std() == 0:
            return 0
        
        excess_returns = returns - risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + portfolio_values.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _annualize_return(self, total_return: float, num_days: int) -> float:
        """Annualize return"""
        if num_days == 0:
            return 0
        years = num_days / 252
        return (1 + total_return) ** (1 / years) - 1
    
    def _generate_backtest_report(self, results: Dict, model_type: str):
        """Generate comprehensive backtest report with visualizations"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'{model_type.upper()} Model Backtest Results', fontsize=16)
        
        # 1. Portfolio value over time
        ax = axes[0, 0]
        ax.plot(results['dates'], results['portfolio_values'], 'b-', linewidth=2)
        ax.axhline(y=self.initial_capital, color='gray', linestyle='--', alpha=0.7)
        ax.set_title('Portfolio Value Over Time')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        
        # 2. Returns over time
        ax = axes[0, 1]
        returns_pct = [r * 100 for r in results['returns']]
        ax.plot(results['dates'], returns_pct, 'g-', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax.set_title('Cumulative Returns')
        ax.set_ylabel('Return (%)')
        ax.grid(True, alpha=0.3)
        
        # 3. Drawdown
        ax = axes[1, 0]
        portfolio_series = pd.Series(results['portfolio_values'])
        cumulative = (1 + portfolio_series.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = ((cumulative - running_max) / running_max) * 100
        ax.fill_between(range(len(drawdown)), 0, drawdown, color='red', alpha=0.3)
        ax.plot(drawdown, 'r-', linewidth=1)
        ax.set_title('Drawdown')
        ax.set_ylabel('Drawdown (%)')
        ax.grid(True, alpha=0.3)
        
        # 4. Number of positions
        ax = axes[1, 1]
        ax.plot(results['dates'], results['positions'], 'purple', linewidth=2)
        ax.set_title('Number of Positions Over Time')
        ax.set_ylabel('Active Positions')
        ax.set_ylim(0, self.max_positions + 1)
        ax.grid(True, alpha=0.3)
        
        # 5. Trade distribution
        ax = axes[2, 0]
        if self.trade_history:
            df_trades = pd.DataFrame(self.trade_history)
            if 'pnl_pct' in df_trades.columns:
                pnl_pcts = df_trades['pnl_pct'].dropna() * 100
                if not pnl_pcts.empty:
                    ax.hist(pnl_pcts, bins=30, color='blue', alpha=0.7, edgecolor='black')
                    ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
                    ax.set_title('Trade P&L Distribution')
                    ax.set_xlabel('P&L (%)')
                    ax.set_ylabel('Frequency')
        
        # 6. Performance metrics table
        ax = axes[2, 1]
        ax.axis('off')
        
        # Create metrics table
        metrics_text = f"""
        Total Return: {results.get('total_return', 0):.2%}
        Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}
        Max Drawdown: {results.get('max_drawdown', 0):.2%}
        Win Rate: {results.get('win_rate', 0):.2%}
        Total Trades: {results.get('total_trades', 0)}
        Profit Factor: {results.get('profit_factor', 0):.2f}
        """
        
        ax.text(0.1, 0.5, metrics_text, transform=ax.transAxes,
                fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax.set_title('Key Metrics')
        
        plt.tight_layout()
        plt.savefig(f'dl_models/{model_type}_backtest_report.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Backtest report saved to dl_models/{model_type}_backtest_report.png")


def create_signal_generator(model_type: str) -> Callable:
    """Create appropriate signal generator for model type"""
    
    def lstm_signal_generator(model, df, model_type):
        """Signal generator for LSTM models"""
        return model.generate_trading_signals(df)
    
    def cnn_signal_generator(model, df, model_type):
        """Signal generator for CNN models"""
        return model.detect_patterns(df)
    
    def transformer_signal_generator(model, df, model_type):
        """Signal generator for Transformer models"""
        return model.predict(df)
    
    def rl_signal_generator(model, df, model_type):
        """Signal generator for RL models"""
        return model.predict(df)
    
    generators = {
        'lstm': lstm_signal_generator,
        'cnn': cnn_signal_generator,
        'transformer': transformer_signal_generator,
        'rl': rl_signal_generator
    }
    
    return generators.get(model_type, lstm_signal_generator)


def main():
    """Test the backtesting framework"""
    logger.info("Testing DL Backtesting Framework...")
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'Date': dates,
        'Open': 100 * (1 + np.random.randn(len(dates)).cumsum() * 0.02),
        'High': 0,
        'Low': 0,
        'Close': 0,
        'Volume': np.random.randint(1000000, 5000000, len(dates))
    })
    
    # Add High/Low/Close based on Open
    df['High'] = df['Open'] * (1 + abs(np.random.randn(len(dates)) * 0.01))
    df['Low'] = df['Open'] * (1 - abs(np.random.randn(len(dates)) * 0.01))
    df['Close'] = df['Open'] + (df['High'] - df['Low']) * np.random.rand(len(dates))
    df.set_index('Date', inplace=True)
    
    # Create mock model and signal generator
    class MockModel:
        def generate_trading_signals(self, df):
            signals = pd.DataFrame(index=df.index)
            signals['signal'] = np.random.choice([-1, 0, 1], size=len(df), p=[0.2, 0.6, 0.2])
            signals['confidence'] = np.random.rand(len(df))
            return signals
    
    # Initialize backtester
    backtester = DeepLearningBacktester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        max_positions=5
    )
    
    # Run backtest
    model = MockModel()
    signal_generator = create_signal_generator('lstm')
    
    results = backtester.backtest_model(
        model=model,
        df=df,
        signal_generator=signal_generator,
        model_type='lstm',
        walk_forward=True,
        window_size=60,
        step_size=20
    )
    
    # Print results
    print("\nBacktest Results:")
    print("-" * 50)
    for key, value in results.items():
        if isinstance(value, (int, float)):
            if 'return' in key or 'ratio' in key or 'rate' in key:
                print(f"{key}: {value:.2%}")
            else:
                print(f"{key}: {value:.2f}")


if __name__ == "__main__":
    main()