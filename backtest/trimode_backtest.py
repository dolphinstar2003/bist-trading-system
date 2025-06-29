#!/usr/bin/env python3
"""
TriMode Orchestrator Backtest
Tests the dynamic mode switching strategy with historical data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import json
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.trimode_orchestrator import TriModeOrchestrator, TradingMode
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


class TriModeBacktest:
    """Backtest system for TriMode Orchestrator"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.orchestrator = TriModeOrchestrator(initial_capital)
        self.csv_manager = CSVDataManager()
        self.initial_capital = initial_capital
        
        # Tracking
        self.portfolio_history = []
        self.trade_history = []
        self.mode_history = []
        self.daily_returns = []
        
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, 
                     timeframe: str = '1d') -> Dict:
        """Run backtest on historical data"""
        
        logger.info(f"Starting TriMode Backtest")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Timeframe: {timeframe}")
        
        # Get date range from first symbol
        sample_data = self.csv_manager.load_raw_data(symbols[0], timeframe)
        if sample_data is None:
            logger.error("No data available")
            return {}
        
        # Filter date range
        sample_data = sample_data[start_date:end_date]
        dates = sample_data.index[200:]  # Skip first 200 for indicators
        
        # Initialize tracking
        current_capital = self.initial_capital
        positions = {}
        
        # Progress bar
        pbar = tqdm(total=len(dates), desc="Backtesting")
        
        # Main backtest loop
        for i, current_date in enumerate(dates):
            pbar.update(1)
            
            # Prepare market data for this date
            market_data = self._prepare_market_data(symbols, current_date, timeframe)
            
            # 1. Determine current mode
            mode = self.orchestrator.determine_mode(market_data)
            self.mode_history.append({
                'date': current_date,
                'mode': mode.value
            })
            
            # 2. Update position prices
            current_prices = {}
            for symbol in symbols:
                if symbol in market_data['symbol_data']:
                    current_prices[symbol] = market_data['symbol_data'][symbol]['close'].iloc[-1]
            
            # 3. Manage existing positions
            closed_trades = self.orchestrator.manage_positions(market_data['symbol_data'])
            for trade in closed_trades:
                self.trade_history.append({
                    **trade,
                    'date': current_date
                })
                # Update capital
                current_capital += trade['pnl']
            
            # 4. Generate new signals
            available_symbols = [s for s in symbols if s not in self.orchestrator.positions]
            signals = self.orchestrator.generate_signals(available_symbols, timeframe)
            
            # 5. Execute new trades
            executed_trades = self.orchestrator.execute_trades(signals)
            for trade in executed_trades:
                self.trade_history.append({
                    **trade,
                    'date': current_date
                })
            
            # 6. Calculate portfolio value
            portfolio_value = self.orchestrator.current_capital
            for symbol, position in self.orchestrator.positions.items():
                if symbol in current_prices:
                    position['current_price'] = current_prices[symbol]
                    portfolio_value += position['shares'] * current_prices[symbol]
            
            # 7. Record daily performance
            self.portfolio_history.append({
                'date': current_date,
                'total_value': portfolio_value,
                'cash': self.orchestrator.current_capital,
                'positions': len(self.orchestrator.positions),
                'mode': mode.value
            })
            
            # Calculate daily return
            if i > 0:
                prev_value = self.portfolio_history[-2]['total_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
                self.orchestrator.daily_returns.append(daily_return)
            
            # Update orchestrator's trades history for mode decisions
            if len(self.trade_history) > 0:
                recent_trades = [t for t in self.trade_history if t.get('action') == 'SELL']
                self.orchestrator.trades_history = recent_trades[-20:]  # Keep last 20
        
        pbar.close()
        
        # Calculate final metrics
        results = self._calculate_metrics()
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _prepare_market_data(self, symbols: List[str], current_date: pd.Timestamp, 
                           timeframe: str) -> Dict:
        """Prepare market data for a specific date"""
        
        market_data = {
            'symbol_data': {},
            'timeframe': timeframe
        }
        
        # Load data for each symbol up to current date
        for symbol in symbols:
            data = self.csv_manager.load_raw_data(symbol, timeframe)
            if data is not None:
                # Get data up to current date
                historical_data = data[data.index <= current_date]
                if len(historical_data) >= 200:
                    market_data['symbol_data'][symbol] = historical_data
        
        # Create synthetic index data (average of all symbols)
        if market_data['symbol_data']:
            closes = []
            for symbol_data in market_data['symbol_data'].values():
                # Normalize prices to percentage change
                pct_change = symbol_data['close'].pct_change().fillna(0)
                closes.append(pct_change)
            
            # Average percentage changes
            avg_returns = pd.concat(closes, axis=1).mean(axis=1)
            
            # Create index from returns
            index_prices = (1 + avg_returns).cumprod() * 100
            
            market_data['index_data'] = pd.DataFrame({
                'close': index_prices,
                'returns': avg_returns
            })
        
        return market_data
    
    def _calculate_metrics(self) -> Dict:
        """Calculate backtest performance metrics"""
        
        if not self.portfolio_history:
            return {}
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        
        # Basic metrics
        initial_value = self.initial_capital
        final_value = portfolio_df['total_value'].iloc[-1]
        total_return = (final_value - initial_value) / initial_value
        
        # Time metrics
        total_days = len(portfolio_df)
        total_months = total_days / 21  # Trading days per month
        
        # Returns analysis
        daily_returns = pd.Series(self.daily_returns)
        
        # Risk metrics
        if len(daily_returns) > 0:
            sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if daily_returns.std() > 0 else 0
            
            # Maximum drawdown
            cumulative = (1 + daily_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade analysis
        if not trades_df.empty:
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if not sell_trades.empty:
                total_trades = len(sell_trades)
                winning_trades = len(sell_trades[sell_trades['pnl'] > 0])
                losing_trades = len(sell_trades[sell_trades['pnl'] < 0])
                
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                avg_win = sell_trades[sell_trades['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = sell_trades[sell_trades['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
                
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
            else:
                total_trades = win_rate = avg_win = avg_loss = profit_factor = 0
        else:
            total_trades = win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Mode analysis
        mode_df = pd.DataFrame(self.mode_history)
        mode_distribution = mode_df['mode'].value_counts().to_dict()
        
        # Mode performance
        portfolio_df['returns'] = portfolio_df['total_value'].pct_change()
        mode_performance = {}
        
        for mode in ['AGGRESSIVE', 'BALANCED', 'DEFENSIVE']:
            mode_days = portfolio_df[portfolio_df['mode'] == mode]
            if not mode_days.empty:
                mode_returns = mode_days['returns'].dropna()
                mode_performance[mode] = {
                    'days': len(mode_days),
                    'avg_return': mode_returns.mean(),
                    'total_return': (1 + mode_returns).prod() - 1,
                    'volatility': mode_returns.std()
                }
        
        # Monthly returns
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        portfolio_df.set_index('date', inplace=True)
        monthly_returns = portfolio_df['total_value'].resample('M').last().pct_change().dropna()
        
        avg_monthly_return = monthly_returns.mean()
        
        # Compile results
        results = {
            'summary': {
                'initial_capital': initial_value,
                'final_capital': final_value,
                'total_return': total_return,
                'annualized_return': (1 + total_return) ** (252 / total_days) - 1 if total_days > 0 else 0,
                'monthly_return': avg_monthly_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'total_days': total_days
            },
            'trades': {
                'total_trades': total_trades,
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            },
            'mode_analysis': {
                'distribution': mode_distribution,
                'performance': mode_performance
            },
            'monthly_returns': monthly_returns.to_list(),
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history
        }
        
        return results
    
    def _save_results(self, results: Dict):
        """Save backtest results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary_file = f'backtest/trimode_results_{timestamp}.json'
        with open(summary_file, 'w') as f:
            # Remove large arrays for summary
            summary = {
                'summary': results['summary'],
                'trades': results['trades'],
                'mode_analysis': results['mode_analysis']
            }
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed history
        if results.get('portfolio_history'):
            portfolio_df = pd.DataFrame(results['portfolio_history'])
            portfolio_df.to_csv(f'backtest/trimode_portfolio_{timestamp}.csv', index=False)
        
        if results.get('trade_history'):
            trades_df = pd.DataFrame(results['trade_history'])
            trades_df.to_csv(f'backtest/trimode_trades_{timestamp}.csv', index=False)
        
        logger.info(f"Results saved to {summary_file}")
        
        # Print summary
        self._print_summary(results)
        
        # Generate plots
        self._generate_plots(results, timestamp)
    
    def _print_summary(self, results: Dict):
        """Print backtest summary"""
        
        print("\n" + "="*80)
        print("TRIMODE ORCHESTRATOR BACKTEST RESULTS")
        print("="*80)
        
        summary = results['summary']
        print(f"\nCAPITAL:")
        print(f"  Initial: ${summary['initial_capital']:,.0f}")
        print(f"  Final: ${summary['final_capital']:,.0f}")
        print(f"  Total Return: {summary['total_return']:.1%}")
        
        print(f"\nRETURNS:")
        print(f"  Monthly Average: {summary['monthly_return']:.1%}")
        print(f"  Annualized: {summary['annualized_return']:.1%}")
        print(f"  Sharpe Ratio: {summary['sharpe_ratio']:.2f}")
        print(f"  Max Drawdown: {summary['max_drawdown']:.1%}")
        
        trades = results['trades']
        print(f"\nTRADES:")
        print(f"  Total: {trades['total_trades']}")
        print(f"  Win Rate: {trades['win_rate']:.1%}")
        print(f"  Avg Win: ${trades['avg_win']:.2f}")
        print(f"  Avg Loss: ${trades['avg_loss']:.2f}")
        print(f"  Profit Factor: {trades['profit_factor']:.2f}")
        
        mode_dist = results['mode_analysis']['distribution']
        print(f"\nMODE DISTRIBUTION:")
        for mode, days in mode_dist.items():
            pct = days / sum(mode_dist.values()) * 100
            print(f"  {mode}: {days} days ({pct:.1f}%)")
        
        mode_perf = results['mode_analysis']['performance']
        print(f"\nMODE PERFORMANCE:")
        for mode, perf in mode_perf.items():
            print(f"  {mode}:")
            print(f"    Avg Daily Return: {perf['avg_return']:.2%}")
            print(f"    Total Return: {perf['total_return']:.1%}")
            print(f"    Volatility: {perf['volatility']:.2%}")
        
        # Check if we hit monthly target
        monthly_target = 0.10  # 10% monthly target
        if summary['monthly_return'] >= monthly_target:
            print(f"\n✅ MONTHLY TARGET ACHIEVED: {summary['monthly_return']:.1%} >= {monthly_target:.0%}")
        else:
            print(f"\n❌ MONTHLY TARGET MISSED: {summary['monthly_return']:.1%} < {monthly_target:.0%}")
            print(f"   Shortfall: {(monthly_target - summary['monthly_return']):.1%}")
        
        print("="*80)
    
    def _generate_plots(self, results: Dict, timestamp: str):
        """Generate performance plots"""
        
        if not results.get('portfolio_history'):
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('TriMode Orchestrator Backtest Results', fontsize=16)
        
        # 1. Portfolio value over time
        portfolio_df = pd.DataFrame(results['portfolio_history'])
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        
        ax = axes[0, 0]
        ax.plot(portfolio_df['date'], portfolio_df['total_value'], 'b-', linewidth=2)
        ax.set_title('Portfolio Value Over Time')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.grid(True, alpha=0.3)
        
        # 2. Mode distribution over time
        ax = axes[0, 1]
        mode_colors = {'AGGRESSIVE': 'red', 'BALANCED': 'blue', 'DEFENSIVE': 'green'}
        
        for mode, color in mode_colors.items():
            mode_data = portfolio_df[portfolio_df['mode'] == mode]
            if not mode_data.empty:
                ax.scatter(mode_data['date'], mode_data['total_value'], 
                         c=color, label=mode, alpha=0.6, s=10)
        
        ax.set_title('Portfolio Value by Mode')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Daily returns distribution
        ax = axes[1, 0]
        if results['summary']['total_days'] > 1:
            daily_returns = portfolio_df['total_value'].pct_change().dropna()
            ax.hist(daily_returns, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(daily_returns.mean(), color='red', linestyle='--', 
                      label=f'Mean: {daily_returns.mean():.3%}')
            ax.set_title('Daily Returns Distribution')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # 4. Cumulative returns by mode
        ax = axes[1, 1]
        for mode in ['AGGRESSIVE', 'BALANCED', 'DEFENSIVE']:
            mode_data = portfolio_df[portfolio_df['mode'] == mode].copy()
            if not mode_data.empty:
                mode_data['cum_return'] = (1 + mode_data['returns'].fillna(0)).cumprod()
                ax.plot(range(len(mode_data)), mode_data['cum_return'], 
                       label=mode, color=mode_colors[mode], linewidth=2)
        
        ax.set_title('Cumulative Returns by Mode')
        ax.set_xlabel('Days in Mode')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Monthly returns
        ax = axes[2, 0]
        if 'monthly_returns' in results and results['monthly_returns']:
            months = range(len(results['monthly_returns']))
            colors = ['green' if r > 0 else 'red' for r in results['monthly_returns']]
            ax.bar(months, results['monthly_returns'], color=colors, alpha=0.7)
            ax.axhline(0.10, color='blue', linestyle='--', label='10% Target')
            ax.set_title('Monthly Returns')
            ax.set_xlabel('Month')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 6. Win/Loss distribution
        ax = axes[2, 1]
        if results['trade_history']:
            trades_df = pd.DataFrame(results['trade_history'])
            sell_trades = trades_df[trades_df['action'] == 'SELL']
            
            if not sell_trades.empty:
                wins = sell_trades[sell_trades['pnl'] > 0]['pnl']
                losses = abs(sell_trades[sell_trades['pnl'] < 0]['pnl'])
                
                ax.boxplot([wins, losses], labels=['Wins', 'Losses'])
                ax.set_title('Win/Loss Distribution')
                ax.set_ylabel('P&L ($)')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'backtest/trimode_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Plots saved to backtest/trimode_plots_{timestamp}.png")


def main():
    """Run TriMode backtest"""
    
    print("="*80)
    print("TRIMODE ORCHESTRATOR BACKTEST")
    print("="*80)
    
    # Initialize backtester
    backtester = TriModeBacktest(initial_capital=100000)
    
    # Select symbols
    test_symbols = ASSETS[:20]  # Top 20 symbols
    
    # Run backtest
    results = backtester.run_backtest(
        symbols=test_symbols,
        start_date='2024-01-01',
        end_date='2025-06-26',
        timeframe='1d'
    )
    
    print("\nBacktest completed!")


if __name__ == "__main__":
    main()