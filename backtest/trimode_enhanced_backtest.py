#!/usr/bin/env python3
"""
Enhanced TriMode Orchestrator Backtest
Tests the enhanced dynamic mode switching strategy with Quick Win features
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

from strategies.trimode_orchestrator_enhanced import EnhancedTriModeOrchestrator, TradingMode
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS


class EnhancedTriModeBacktest:
    """Backtest system for Enhanced TriMode Orchestrator"""
    
    def __init__(self, initial_capital: float = 100000.0):
        self.orchestrator = EnhancedTriModeOrchestrator(initial_capital)
        self.csv_manager = CSVDataManager()
        self.initial_capital = initial_capital
        
        # Tracking
        self.portfolio_history = []
        self.trade_history = []
        self.mode_history = []
        self.daily_returns = []
        self.partial_exits = []  # Track partial profit taking
        
    def run_backtest(self, symbols: List[str], start_date: str, end_date: str, 
                     timeframe: str = '1d', optimize_ema: bool = True) -> Dict:
        """Run backtest on historical data"""
        
        logger.info(f"Starting Enhanced TriMode Backtest")
        logger.info(f"Symbols: {len(symbols)}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Timeframe: {timeframe}")
        logger.info(f"Features: ATR trailing stop, Partial profits, RVOL filter, Volatility sizing")
        
        # Optimize EMA parameters if requested
        if optimize_ema:
            logger.info("\nOptimizing EMA parameters for all symbols...")
            for symbol in tqdm(symbols, desc="EMA Optimization"):
                data = self.csv_manager.load_raw_data(symbol, timeframe)
                if data is not None and len(data) > 200:
                    self.orchestrator.optimize_ema_parameters(symbol, data)
        
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
            
            # 3. Manage existing positions (with enhanced features)
            closed_trades = self.orchestrator.manage_positions(market_data['symbol_data'])
            for trade in closed_trades:
                self.trade_history.append({
                    **trade,
                    'date': current_date
                })
                
                # Track partial exits separately
                if trade.get('action') == 'SELL_PARTIAL':
                    self.partial_exits.append(trade)
                
                # Update capital is already done in orchestrator
                
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
                'mode': mode.value,
                'partial_exits_active': sum(1 for p in self.orchestrator.positions.values() 
                                          if p.get('partial_exit_done', False))
            })
            
            # Calculate daily return
            if i > 0:
                prev_value = self.portfolio_history[-2]['total_value']
                daily_return = (portfolio_value - prev_value) / prev_value
                self.daily_returns.append(daily_return)
                self.orchestrator.daily_returns.append(daily_return)
            
            # Update orchestrator's trades history for mode decisions
            if len(self.trade_history) > 0:
                recent_trades = [t for t in self.trade_history if t.get('action') in ['SELL', 'SELL_PARTIAL']]
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
        
        # Create synthetic index data
        if market_data['symbol_data']:
            closes = []
            for symbol_data in market_data['symbol_data'].values():
                pct_change = symbol_data['close'].pct_change().fillna(0)
                closes.append(pct_change)
            
            avg_returns = pd.concat(closes, axis=1).mean(axis=1)
            index_prices = (1 + avg_returns).cumprod() * 100
            
            market_data['index_data'] = pd.DataFrame({
                'close': index_prices,
                'returns': avg_returns
            })
        
        return market_data
    
    def _calculate_metrics(self) -> Dict:
        """Calculate enhanced backtest performance metrics"""
        
        if not self.portfolio_history:
            return {}
        
        # Convert to DataFrame
        portfolio_df = pd.DataFrame(self.portfolio_history)
        trades_df = pd.DataFrame(self.trade_history) if self.trade_history else pd.DataFrame()
        partial_df = pd.DataFrame(self.partial_exits) if self.partial_exits else pd.DataFrame()
        
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
        
        # Enhanced trade analysis
        if not trades_df.empty:
            # Separate full exits and partial exits
            full_exits = trades_df[trades_df['action'] == 'SELL']
            partial_exits = trades_df[trades_df['action'] == 'SELL_PARTIAL']
            
            if not full_exits.empty:
                total_trades = len(full_exits)
                winning_trades = len(full_exits[full_exits['pnl'] > 0])
                losing_trades = len(full_exits[full_exits['pnl'] < 0])
                
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                avg_win = full_exits[full_exits['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                avg_loss = full_exits[full_exits['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
                
                profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
            else:
                total_trades = win_rate = avg_win = avg_loss = profit_factor = 0
            
            # Partial exit statistics
            if not partial_exits.empty:
                partial_stats = {
                    'count': len(partial_exits),
                    'total_pnl': partial_exits['pnl'].sum(),
                    'avg_pnl': partial_exits['pnl'].mean(),
                    'avg_exit_pct': partial_exits['pnl_pct'].mean()
                }
            else:
                partial_stats = {'count': 0, 'total_pnl': 0, 'avg_pnl': 0, 'avg_exit_pct': 0}
        else:
            total_trades = win_rate = avg_win = avg_loss = profit_factor = 0
            partial_stats = {'count': 0, 'total_pnl': 0, 'avg_pnl': 0, 'avg_exit_pct': 0}
        
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
        
        # RVOL effectiveness
        rvol_trades = [t for t in self.trade_history if 'rvol' in str(t.get('reasons', []))]
        rvol_effectiveness = len(rvol_trades) / len(self.trade_history) if self.trade_history else 0
        
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
            'partial_exits': partial_stats,
            'mode_analysis': {
                'distribution': mode_distribution,
                'performance': mode_performance
            },
            'enhancements': {
                'rvol_filter_usage': rvol_effectiveness,
                'optimal_ema_symbols': len(self.orchestrator.ema_params_cache),
                'trailing_stops_hit': len([t for t in self.trade_history if t.get('exit_reason') == 'stop_loss'])
            },
            'monthly_returns': monthly_returns.to_list(),
            'portfolio_history': self.portfolio_history,
            'trade_history': self.trade_history
        }
        
        return results
    
    def _save_results(self, results: Dict):
        """Save enhanced backtest results"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save summary
        summary_file = f'backtest/trimode_enhanced_results_{timestamp}.json'
        with open(summary_file, 'w') as f:
            # Remove large arrays for summary
            summary = {
                'summary': results['summary'],
                'trades': results['trades'],
                'partial_exits': results['partial_exits'],
                'mode_analysis': results['mode_analysis'],
                'enhancements': results['enhancements']
            }
            json.dump(summary, f, indent=2, default=str)
        
        # Save detailed history
        if results.get('portfolio_history'):
            portfolio_df = pd.DataFrame(results['portfolio_history'])
            portfolio_df.to_csv(f'backtest/trimode_enhanced_portfolio_{timestamp}.csv', index=False)
        
        if results.get('trade_history'):
            trades_df = pd.DataFrame(results['trade_history'])
            trades_df.to_csv(f'backtest/trimode_enhanced_trades_{timestamp}.csv', index=False)
        
        # Save optimal EMA parameters
        ema_params_file = f'backtest/optimal_ema_params_{timestamp}.json'
        with open(ema_params_file, 'w') as f:
            ema_params = {k: list(v) for k, v in self.orchestrator.ema_params_cache.items()}
            json.dump(ema_params, f, indent=2)
        
        logger.info(f"Results saved to {summary_file}")
        
        # Print summary
        self._print_enhanced_summary(results)
        
        # Generate enhanced plots
        self._generate_enhanced_plots(results, timestamp)
    
    def _print_enhanced_summary(self, results: Dict):
        """Print enhanced backtest summary"""
        
        print("\n" + "="*80)
        print("ENHANCED TRIMODE ORCHESTRATOR BACKTEST RESULTS")
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
        
        # Partial exits
        partial = results['partial_exits']
        print(f"\nPARTIAL PROFIT TAKING:")
        print(f"  Total Partial Exits: {partial['count']}")
        print(f"  Total PnL from Partials: ${partial['total_pnl']:.2f}")
        print(f"  Avg Partial Exit: {partial['avg_exit_pct']:.1%}")
        
        # Enhancements effectiveness
        enh = results['enhancements']
        print(f"\nENHANCEMENTS EFFECTIVENESS:")
        print(f"  RVOL Filter Usage: {enh['rvol_filter_usage']:.1%}")
        print(f"  Optimized EMA Symbols: {enh['optimal_ema_symbols']}")
        print(f"  Trailing Stops Triggered: {enh['trailing_stops_hit']}")
        
        mode_dist = results['mode_analysis']['distribution']
        print(f"\nMODE DISTRIBUTION:")
        for mode, days in mode_dist.items():
            pct = days / sum(mode_dist.values()) * 100
            print(f"  {mode}: {days} days ({pct:.1f}%)")
        
        # Check if we hit monthly target
        monthly_target = 0.10  # 10% monthly target
        if summary['monthly_return'] >= monthly_target:
            print(f"\n✅ MONTHLY TARGET ACHIEVED: {summary['monthly_return']:.1%} >= {monthly_target:.0%}")
        else:
            print(f"\n❌ MONTHLY TARGET MISSED: {summary['monthly_return']:.1%} < {monthly_target:.0%}")
            print(f"   Shortfall: {(monthly_target - summary['monthly_return']):.1%}")
        
        print("="*80)
    
    def _generate_enhanced_plots(self, results: Dict, timestamp: str):
        """Generate enhanced performance plots"""
        
        if not results.get('portfolio_history'):
            return
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Enhanced TriMode Orchestrator Backtest Results', fontsize=16)
        
        # 1. Portfolio value with partial exits marked
        portfolio_df = pd.DataFrame(results['portfolio_history'])
        portfolio_df['date'] = pd.to_datetime(portfolio_df['date'])
        
        ax = axes[0, 0]
        ax.plot(portfolio_df['date'], portfolio_df['total_value'], 'b-', linewidth=2, label='Portfolio Value')
        
        # Mark partial exits
        if self.partial_exits:
            partial_df = pd.DataFrame(self.partial_exits)
            if 'date' in partial_df.columns:
                partial_df['date'] = pd.to_datetime(partial_df['date'])
                ax.scatter(partial_df['date'], 
                          [portfolio_df[portfolio_df['date'] <= d]['total_value'].iloc[-1] 
                           for d in partial_df['date']], 
                          color='green', s=50, alpha=0.6, label='Partial Exits')
        
        ax.set_title('Portfolio Value with Partial Profit Taking')
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Trailing stop effectiveness
        ax = axes[0, 1]
        trades_df = pd.DataFrame(results['trade_history']) if results['trade_history'] else pd.DataFrame()
        
        if not trades_df.empty:
            exit_reasons = trades_df[trades_df['action'] == 'SELL']['exit_reason'].value_counts()
            ax.bar(exit_reasons.index, exit_reasons.values)
            ax.set_title('Exit Reasons Distribution')
            ax.set_xlabel('Exit Reason')
            ax.set_ylabel('Count')
            ax.tick_params(axis='x', rotation=45)
        
        # 3. RVOL distribution
        ax = axes[1, 0]
        if 'rvol' in trades_df.columns:
            rvol_values = trades_df['rvol'].dropna()
            ax.hist(rvol_values, bins=30, alpha=0.7)
            ax.axvline(1.5, color='red', linestyle='--', label='RVOL Filter (1.5)')
            ax.set_title('Relative Volume Distribution')
            ax.set_xlabel('RVOL')
            ax.set_ylabel('Frequency')
            ax.legend()
        
        # 4. Position sizing with volatility adjustment
        ax = axes[1, 1]
        if 'volatility_adjustment' in trades_df.columns:
            vol_adj = trades_df['volatility_adjustment'].dropna()
            ax.hist(vol_adj, bins=20, alpha=0.7)
            ax.set_title('Volatility-Based Position Sizing')
            ax.set_xlabel('Size Adjustment Factor')
            ax.set_ylabel('Frequency')
        
        # 5. Partial vs Full exit performance
        ax = axes[2, 0]
        if self.partial_exits:
            partial_pnls = [t['pnl_pct'] for t in self.partial_exits]
            full_exits = trades_df[trades_df['action'] == 'SELL']
            if not full_exits.empty:
                full_pnls = full_exits['pnl_pct'].tolist()
                
                ax.boxplot([partial_pnls, full_pnls], labels=['Partial Exits', 'Full Exits'])
                ax.set_title('Partial vs Full Exit Performance')
                ax.set_ylabel('Return (%)')
                ax.grid(True, alpha=0.3)
        
        # 6. Monthly returns with 10% target
        ax = axes[2, 1]
        if 'monthly_returns' in results and results['monthly_returns']:
            months = range(len(results['monthly_returns']))
            colors = ['green' if r >= 0.10 else 'red' for r in results['monthly_returns']]
            ax.bar(months, results['monthly_returns'], color=colors, alpha=0.7)
            ax.axhline(0.10, color='blue', linestyle='--', label='10% Target')
            ax.set_title('Monthly Returns vs Target')
            ax.set_xlabel('Month')
            ax.set_ylabel('Return')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'backtest/trimode_enhanced_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced plots saved to backtest/trimode_enhanced_plots_{timestamp}.png")


def main():
    """Run Enhanced TriMode backtest"""
    
    print("="*80)
    print("ENHANCED TRIMODE ORCHESTRATOR BACKTEST")
    print("With Quick Win Features:")
    print("- ATR-based trailing stops")
    print("- Partial profit taking (50% at targets)")
    print("- Volume spike filter (RVOL > 1.5)")
    print("- Optimal EMA parameter storage")
    print("- Volatility-based position sizing")
    print("="*80)
    
    # Initialize backtester
    backtester = EnhancedTriModeBacktest(initial_capital=100000)
    
    # Select symbols
    test_symbols = ASSETS[:30]  # Top 30 symbols for better diversification
    
    # Run backtest with EMA optimization
    results = backtester.run_backtest(
        symbols=test_symbols,
        start_date='2024-01-01',
        end_date='2025-06-26',
        timeframe='1d',
        optimize_ema=True  # Enable EMA optimization
    )
    
    print("\nEnhanced backtest completed!")


if __name__ == "__main__":
    main()