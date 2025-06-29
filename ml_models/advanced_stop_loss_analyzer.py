#!/usr/bin/env python3
"""
Advanced Stop Loss & Take Profit Analyzer
Analyzes optimal stop loss, take profit, trailing stop levels and multi-timeframe strategies
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import json
import logging
from typing import Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class AdvancedStopLossAnalyzer:
    """Advanced analyzer for stop loss, take profit and multi-timeframe strategies"""
    
    def __init__(self):
        self.results = {}
        self.timeframes = ['15m', '1h', '4h', '1d']
        self.load_stock_list()
        
    def load_stock_list(self):
        """Load BIST stock list from settings"""
        settings_path = 'settings.json'
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = json.load(f)
                self.symbols = settings.get('trading', {}).get('symbols', [])
                logger.info(f"Loaded {len(self.symbols)} symbols")
        else:
            self.symbols = ['THYAO', 'ASELS', 'SISE', 'TUPRS', 'EREGL']
            
    def load_stock_data(self, symbol: str, timeframe: str = '1d') -> pd.DataFrame:
        """Load historical data for a stock"""
        try:
            path = f"data/raw/{symbol}_{timeframe}_raw.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                # Filter data from 2020 onwards
                df = df[df.index >= '2020-01-01']
                return df
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {symbol} {timeframe}: {e}")
            return pd.DataFrame()
            
    def calculate_rallies_and_drawdowns(self, prices: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Calculate both rallies (upward moves) and drawdowns"""
        # Drawdowns (as before)
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        
        # Rallies - upward moves from local lows
        running_min = prices.expanding().min()
        rally = (prices - running_min) / running_min
        
        # Find rally periods
        rally_starts = []
        rally_ends = []
        rally_magnitudes = []
        rally_lengths = []
        
        # Find local minima and maxima
        window = 20  # Look for turns over 20 periods
        local_min = prices.rolling(window=window, center=True).min() == prices
        local_max = prices.rolling(window=window, center=True).max() == prices
        
        # Extract rally periods
        min_indices = prices[local_min].index.tolist()
        max_indices = prices[local_max].index.tolist()
        
        for i in range(len(min_indices) - 1):
            start = min_indices[i]
            # Find next maximum after this minimum
            next_maxes = [m for m in max_indices if m > start]
            if next_maxes:
                end = next_maxes[0]
                
                start_price = prices[start]
                end_price = prices[end]
                magnitude = (end_price - start_price) / start_price * 100
                
                if magnitude > 3:  # Only count rallies > 3%
                    rally_starts.append(start)
                    rally_ends.append(end)
                    rally_magnitudes.append(magnitude)
                    rally_lengths.append((end - start).days)
                    
        rallies_df = pd.DataFrame({
            'start_date': rally_starts,
            'end_date': rally_ends,
            'magnitude_pct': rally_magnitudes,
            'length_days': rally_lengths
        }) if rally_starts else pd.DataFrame()
        
        # Calculate drawdowns similarly
        drawdown_starts = []
        drawdown_ends = []
        drawdown_magnitudes = []
        drawdown_lengths = []
        
        for i in range(len(max_indices) - 1):
            start = max_indices[i]
            # Find next minimum after this maximum
            next_mins = [m for m in min_indices if m > start]
            if next_mins:
                end = next_mins[0]
                
                start_price = prices[start]
                end_price = prices[end]
                magnitude = abs((end_price - start_price) / start_price * 100)
                
                if magnitude > 2:  # Only count drawdowns > 2%
                    drawdown_starts.append(start)
                    drawdown_ends.append(end)
                    drawdown_magnitudes.append(magnitude)
                    drawdown_lengths.append((end - start).days)
                    
        drawdowns_df = pd.DataFrame({
            'start_date': drawdown_starts,
            'end_date': drawdown_ends,
            'magnitude_pct': drawdown_magnitudes,
            'length_days': drawdown_lengths
        }) if drawdown_starts else pd.DataFrame()
        
        return rallies_df, drawdowns_df
        
    def calculate_trailing_stop_effectiveness(self, df: pd.DataFrame, trailing_pct: float) -> Dict:
        """Test effectiveness of different trailing stop percentages"""
        trades = []
        in_position = False
        entry_price = 0
        highest_price = 0
        stop_price = 0
        
        for i in range(20, len(df)):
            if not in_position:
                # Simple entry: price crosses above 20-day SMA
                if df['close'].iloc[i] > df['close'].iloc[i-20:i].mean():
                    in_position = True
                    entry_price = df['close'].iloc[i]
                    highest_price = entry_price
                    stop_price = entry_price * (1 - trailing_pct / 100)
                    entry_date = df.index[i]
            else:
                current_price = df['close'].iloc[i]
                
                # Update trailing stop if price made new high
                if current_price > highest_price:
                    highest_price = current_price
                    stop_price = highest_price * (1 - trailing_pct / 100)
                    
                # Check if stopped out
                if current_price <= stop_price:
                    exit_price = stop_price
                    profit_pct = (exit_price - entry_price) / entry_price * 100
                    
                    trades.append({
                        'entry_date': entry_date,
                        'exit_date': df.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'highest_price': highest_price,
                        'profit_pct': profit_pct,
                        'max_profit_pct': (highest_price - entry_price) / entry_price * 100,
                        'efficiency': profit_pct / ((highest_price - entry_price) / entry_price * 100) if highest_price > entry_price else 0
                    })
                    
                    in_position = False
                    
        if trades:
            trades_df = pd.DataFrame(trades)
            winning_trades = trades_df[trades_df['profit_pct'] > 0]
            
            return {
                'trailing_pct': trailing_pct,
                'num_trades': len(trades_df),
                'win_rate': len(winning_trades) / len(trades_df) * 100,
                'avg_profit': trades_df['profit_pct'].mean(),
                'avg_efficiency': trades_df['efficiency'].mean() * 100,  # How much of max profit captured
                'total_return': trades_df['profit_pct'].sum()
            }
        else:
            return None
            
    def analyze_multi_timeframe(self, symbol: str) -> Dict:
        """Analyze correlations between different timeframes"""
        mtf_data = {}
        
        for tf in self.timeframes:
            df = self.load_stock_data(symbol, tf)
            if not df.empty and len(df) > 100:
                # Calculate trend indicators
                df['sma_20'] = df['close'].rolling(20).mean()
                df['sma_50'] = df['close'].rolling(50).mean()
                df['trend'] = np.where(df['close'] > df['sma_20'], 1, -1)
                
                # Calculate momentum
                df['rsi'] = self.calculate_rsi(df['close'])
                df['momentum'] = df['close'].pct_change(10)
                
                mtf_data[tf] = {
                    'trend': df['trend'].iloc[-1] if len(df) > 0 else 0,
                    'rsi': df['rsi'].iloc[-1] if len(df) > 0 else 50,
                    'momentum': df['momentum'].iloc[-1] if len(df) > 0 else 0,
                    'volatility': df['close'].pct_change().std() * np.sqrt(252) * 100
                }
                
        # Determine multi-timeframe alignment
        if mtf_data:
            trends = [mtf_data[tf]['trend'] for tf in self.timeframes if tf in mtf_data]
            alignment_score = sum(trends) / len(trends) if trends else 0
            
            # Trading recommendation based on MTF
            if len(mtf_data) >= 3:
                # Strong buy: All timeframes aligned bullish
                if alignment_score > 0.8:
                    mtf_signal = "STRONG_BUY"
                # Buy: Higher timeframes bullish, can buy on lower TF pullbacks
                elif alignment_score > 0.5:
                    mtf_signal = "BUY_ON_PULLBACK"
                # Neutral
                elif alignment_score > -0.5:
                    mtf_signal = "NEUTRAL"
                # Sell: Higher timeframes bearish
                elif alignment_score > -0.8:
                    mtf_signal = "SELL_ON_RALLY"
                else:
                    mtf_signal = "STRONG_SELL"
            else:
                mtf_signal = "INSUFFICIENT_DATA"
                
            return {
                'timeframe_data': mtf_data,
                'alignment_score': alignment_score,
                'signal': mtf_signal
            }
        else:
            return None
            
    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
        
    def analyze_stock_advanced(self, symbol: str) -> Dict:
        """Comprehensive analysis for a single stock"""
        logger.info(f"Analyzing {symbol}...")
        
        # Load daily data
        df = self.load_stock_data(symbol, '1d')
        if df.empty or len(df) < 200:
            return None
            
        # Basic statistics
        df['returns'] = df['close'].pct_change()
        daily_volatility = df['returns'].std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Calculate rallies and drawdowns
        rallies_df, drawdowns_df = self.calculate_rallies_and_drawdowns(df['close'])
        
        if rallies_df.empty or drawdowns_df.empty:
            return None
            
        # Analyze rallies for take profit levels
        rally_stats = {
            'avg_rally': rallies_df['magnitude_pct'].mean(),
            'median_rally': rallies_df['magnitude_pct'].median(),
            'percentile_50': rallies_df['magnitude_pct'].quantile(0.50),
            'percentile_70': rallies_df['magnitude_pct'].quantile(0.70),
            'percentile_80': rallies_df['magnitude_pct'].quantile(0.80),
            'percentile_90': rallies_df['magnitude_pct'].quantile(0.90),
            'max_rally': rallies_df['magnitude_pct'].max(),
            'avg_rally_days': rallies_df['length_days'].mean()
        }
        
        # Analyze drawdowns for stop loss levels
        drawdown_stats = {
            'avg_drawdown': drawdowns_df['magnitude_pct'].mean(),
            'median_drawdown': drawdowns_df['magnitude_pct'].median(),
            'percentile_70': drawdowns_df['magnitude_pct'].quantile(0.70),
            'percentile_80': drawdowns_df['magnitude_pct'].quantile(0.80),
            'percentile_90': drawdowns_df['magnitude_pct'].quantile(0.90),
            'max_drawdown': drawdowns_df['magnitude_pct'].max()
        }
        
        # Test different trailing stop levels
        trailing_results = []
        for trailing_pct in [3, 5, 7, 10, 15, 20]:
            result = self.calculate_trailing_stop_effectiveness(df, trailing_pct)
            if result:
                trailing_results.append(result)
                
        # Find optimal trailing stop
        if trailing_results:
            best_trailing = max(trailing_results, key=lambda x: x['total_return'])
            optimal_trailing_pct = best_trailing['trailing_pct']
        else:
            optimal_trailing_pct = 10  # Default
            
        # Multi-timeframe analysis
        mtf_analysis = self.analyze_multi_timeframe(symbol)
        
        # Compile results
        results = {
            'symbol': symbol,
            'annual_volatility': annual_volatility * 100,
            
            # Stop loss recommendations
            'optimal_stop_loss': drawdown_stats['percentile_80'],
            'tight_stop_loss': drawdown_stats['percentile_70'],
            'wide_stop_loss': drawdown_stats['percentile_90'],
            
            # Take profit recommendations
            'optimal_take_profit': rally_stats['percentile_70'],
            'conservative_take_profit': rally_stats['percentile_50'],
            'aggressive_take_profit': rally_stats['percentile_80'],
            
            # Trailing stop
            'optimal_trailing_stop': optimal_trailing_pct,
            'trailing_test_results': trailing_results,
            
            # Statistics
            'avg_rally': rally_stats['avg_rally'],
            'avg_drawdown': drawdown_stats['avg_drawdown'],
            'max_rally': rally_stats['max_rally'],
            'max_drawdown': drawdown_stats['max_drawdown'],
            'avg_rally_days': rally_stats['avg_rally_days'],
            
            # Multi-timeframe
            'mtf_analysis': mtf_analysis,
            
            # Risk reward ratio
            'risk_reward_ratio': rally_stats['percentile_70'] / drawdown_stats['percentile_80'] if drawdown_stats['percentile_80'] > 0 else 0
        }
        
        return results
        
    def analyze_all_stocks(self):
        """Analyze all stocks with parallel processing"""
        logger.info("Starting comprehensive analysis for all stocks...")
        
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_stock_advanced, symbol): symbol 
                for symbol in self.symbols
            }
            
            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    result = future.result()
                    if result:
                        self.results[symbol] = result
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")
                    
    def create_trading_strategies(self):
        """Create specific trading strategies for each stock"""
        strategies = {}
        
        for symbol, data in self.results.items():
            if not data:
                continue
                
            # Determine strategy type based on characteristics
            volatility = data['annual_volatility']
            rr_ratio = data['risk_reward_ratio']
            
            if volatility < 30:
                strategy_type = "SWING_TRADING"
                position_size = 0.20  # 20% of capital
            elif volatility < 50:
                strategy_type = "POSITION_TRADING"
                position_size = 0.15  # 15% of capital
            else:
                strategy_type = "DAY_TRADING"
                position_size = 0.10  # 10% of capital
                
            # Multi-timeframe strategy
            mtf = data.get('mtf_analysis', {})
            if mtf and mtf.get('signal'):
                if mtf['signal'] == "STRONG_BUY":
                    entry_strategy = "Immediate entry on any pullback"
                elif mtf['signal'] == "BUY_ON_PULLBACK":
                    entry_strategy = "Wait for pullback to support or moving average"
                elif mtf['signal'] == "NEUTRAL":
                    entry_strategy = "Wait for trend confirmation"
                elif mtf['signal'] == "SELL_ON_RALLY":
                    entry_strategy = "Only counter-trend trades on oversold"
                else:
                    entry_strategy = "Avoid trading"
            else:
                entry_strategy = "Standard entry on signal"
                
            strategies[symbol] = {
                'strategy_type': strategy_type,
                'position_size': position_size,
                'stop_loss': data['optimal_stop_loss'],
                'take_profit': data['optimal_take_profit'],
                'trailing_stop': data['optimal_trailing_stop'],
                'entry_strategy': entry_strategy,
                'risk_reward_ratio': rr_ratio,
                'expected_win_rate': 100 / (1 + rr_ratio) if rr_ratio > 0 else 50  # Breakeven win rate
            }
            
        return strategies
        
    def print_results(self):
        """Print comprehensive results"""
        if not self.results:
            logger.error("No results to display")
            return
            
        print("\n" + "="*120)
        print("ADVANCED STOP LOSS & TAKE PROFIT ANALYSIS")
        print("="*120)
        
        # Create summary dataframe
        summary_data = []
        for symbol, data in self.results.items():
            if data:
                summary_data.append({
                    'Symbol': symbol,
                    'Volatility': f"{data['annual_volatility']:.1f}%",
                    'Stop_Loss': f"{data['optimal_stop_loss']:.1f}%",
                    'Take_Profit': f"{data['optimal_take_profit']:.1f}%",
                    'Trailing': f"{data['optimal_trailing_stop']:.0f}%",
                    'RR_Ratio': f"{data['risk_reward_ratio']:.2f}",
                    'Avg_Rally': f"{data['avg_rally']:.1f}%",
                    'Avg_DD': f"{data['avg_drawdown']:.1f}%",
                    'MTF_Signal': data['mtf_analysis']['signal'] if data.get('mtf_analysis') else 'N/A'
                })
                
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            print("\nSummary Table:")
            print("-"*120)
            print(summary_df.to_string(index=False))
            
        # Print trading strategies
        print("\n" + "="*80)
        print("RECOMMENDED TRADING STRATEGIES")
        print("="*80)
        
        strategies = self.create_trading_strategies()
        for symbol, strategy in strategies.items():
            print(f"\n{symbol}:")
            print(f"  Strategy Type: {strategy['strategy_type']}")
            print(f"  Position Size: {strategy['position_size']*100:.0f}% of capital")
            print(f"  Stop Loss: {strategy['stop_loss']:.1f}%")
            print(f"  Take Profit: {strategy['take_profit']:.1f}%")
            print(f"  Trailing Stop: {strategy['trailing_stop']:.0f}%")
            print(f"  Risk/Reward: 1:{strategy['risk_reward_ratio']:.2f}")
            print(f"  Min Win Rate: {strategy['expected_win_rate']:.1f}%")
            print(f"  Entry: {strategy['entry_strategy']}")
            
    def save_results(self):
        """Save results to files"""
        output_dir = 'data/analysis'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        detailed_results = []
        for symbol, data in self.results.items():
            if data:
                # Flatten the data for CSV
                row = {
                    'symbol': symbol,
                    'annual_volatility': data['annual_volatility'],
                    'optimal_stop_loss': data['optimal_stop_loss'],
                    'tight_stop_loss': data['tight_stop_loss'],
                    'wide_stop_loss': data['wide_stop_loss'],
                    'optimal_take_profit': data['optimal_take_profit'],
                    'conservative_take_profit': data['conservative_take_profit'],
                    'aggressive_take_profit': data['aggressive_take_profit'],
                    'optimal_trailing_stop': data['optimal_trailing_stop'],
                    'avg_rally': data['avg_rally'],
                    'avg_drawdown': data['avg_drawdown'],
                    'max_rally': data['max_rally'],
                    'max_drawdown': data['max_drawdown'],
                    'risk_reward_ratio': data['risk_reward_ratio']
                }
                
                # Add MTF data
                if data.get('mtf_analysis'):
                    row['mtf_signal'] = data['mtf_analysis']['signal']
                    row['mtf_alignment'] = data['mtf_analysis']['alignment_score']
                    
                detailed_results.append(row)
                
        # Save to CSV
        if detailed_results:
            results_df = pd.DataFrame(detailed_results)
            results_df.to_csv(os.path.join(output_dir, 'advanced_trading_parameters.csv'), index=False)
            
            # Save strategies
            strategies = self.create_trading_strategies()
            strategies_df = pd.DataFrame.from_dict(strategies, orient='index')
            strategies_df.to_csv(os.path.join(output_dir, 'trading_strategies.csv'))
            
            logger.info(f"Results saved to {output_dir}")
            
    def plot_analysis(self, symbol: str = None):
        """Plot analysis for a specific symbol or top performers"""
        if symbol and symbol in self.results:
            self._plot_single_stock(symbol)
        else:
            self._plot_summary()
            
    def _plot_single_stock(self, symbol: str):
        """Detailed plot for a single stock"""
        data = self.results[symbol]
        df = self.load_stock_data(symbol, '1d')
        
        if df.empty:
            return
            
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        
        # 1. Price with stop loss and take profit levels
        ax1 = axes[0, 0]
        ax1.plot(df.index[-252:], df['close'][-252:], 'b-', label='Price')
        
        # Add levels
        current_price = df['close'].iloc[-1]
        sl_price = current_price * (1 - data['optimal_stop_loss']/100)
        tp_price = current_price * (1 + data['optimal_take_profit']/100)
        
        ax1.axhline(y=current_price, color='black', linestyle='-', alpha=0.5, label='Current')
        ax1.axhline(y=sl_price, color='red', linestyle='--', alpha=0.7, label=f"Stop Loss (-{data['optimal_stop_loss']:.1f}%)")
        ax1.axhline(y=tp_price, color='green', linestyle='--', alpha=0.7, label=f"Take Profit (+{data['optimal_take_profit']:.1f}%)")
        
        ax1.set_title(f'{symbol} - Price with SL/TP Levels')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Trailing stop effectiveness
        ax2 = axes[0, 1]
        if data.get('trailing_test_results'):
            trailing_df = pd.DataFrame(data['trailing_test_results'])
            ax2.plot(trailing_df['trailing_pct'], trailing_df['total_return'], 'go-', label='Total Return')
            ax2.plot(trailing_df['trailing_pct'], trailing_df['win_rate'], 'ro-', label='Win Rate')
            ax2.set_xlabel('Trailing Stop %')
            ax2.set_ylabel('Performance')
            ax2.set_title('Trailing Stop Optimization')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
        # 3. Multi-timeframe alignment
        ax3 = axes[1, 0]
        if data.get('mtf_analysis') and data['mtf_analysis'].get('timeframe_data'):
            mtf_data = data['mtf_analysis']['timeframe_data']
            timeframes = list(mtf_data.keys())
            trends = [mtf_data[tf]['trend'] for tf in timeframes]
            colors = ['green' if t > 0 else 'red' for t in trends]
            
            ax3.bar(timeframes, trends, color=colors, alpha=0.7)
            ax3.set_title('Multi-Timeframe Trend Alignment')
            ax3.set_ylabel('Trend Direction')
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax3.grid(True, alpha=0.3, axis='y')
            
        # 4. Risk/Reward visualization
        ax4 = axes[1, 1]
        categories = ['Conservative', 'Optimal', 'Aggressive']
        stop_losses = [data['tight_stop_loss'], data['optimal_stop_loss'], data['wide_stop_loss']]
        take_profits = [data['conservative_take_profit'], data['optimal_take_profit'], data['aggressive_take_profit']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax4.bar(x - width/2, stop_losses, width, label='Stop Loss', color='red', alpha=0.7)
        ax4.bar(x + width/2, take_profits, width, label='Take Profit', color='green', alpha=0.7)
        
        ax4.set_xlabel('Strategy Type')
        ax4.set_ylabel('Percentage')
        ax4.set_title('Risk/Reward Profiles')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # 5. Historical drawdowns and rallies
        ax5 = axes[2, 0]
        rallies, drawdowns = self.calculate_rallies_and_drawdowns(df['close'])
        
        if not rallies.empty:
            ax5.hist(rallies['magnitude_pct'], bins=20, alpha=0.5, color='green', label='Rallies')
        if not drawdowns.empty:
            ax5.hist(drawdowns['magnitude_pct'], bins=20, alpha=0.5, color='red', label='Drawdowns')
            
        ax5.set_xlabel('Magnitude (%)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Historical Rallies vs Drawdowns Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary text
        ax6 = axes[2, 1]
        ax6.axis('off')
        
        summary_text = f"""
        {symbol} Trading Parameters Summary
        =====================================
        
        Annual Volatility: {data['annual_volatility']:.1f}%
        
        Recommended Settings:
        - Stop Loss: {data['optimal_stop_loss']:.1f}%
        - Take Profit: {data['optimal_take_profit']:.1f}%
        - Trailing Stop: {data['optimal_trailing_stop']:.0f}%
        
        Risk/Reward Ratio: 1:{data['risk_reward_ratio']:.2f}
        
        Historical Performance:
        - Avg Rally: {data['avg_rally']:.1f}%
        - Avg Drawdown: {data['avg_drawdown']:.1f}%
        - Max Rally: {data['max_rally']:.1f}%
        - Max Drawdown: {data['max_drawdown']:.1f}%
        
        MTF Signal: {data['mtf_analysis']['signal'] if data.get('mtf_analysis') else 'N/A'}
        """
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'{symbol} - Advanced Trading Analysis', fontsize=16)
        plt.tight_layout()
        
        # Save
        output_dir = 'data/analysis/stock_reports'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'{symbol}_advanced_analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
    def _plot_summary(self):
        """Plot summary of all stocks"""
        if not self.results:
            return
            
        # Prepare data
        symbols = []
        stop_losses = []
        take_profits = []
        rr_ratios = []
        volatilities = []
        
        for symbol, data in self.results.items():
            if data:
                symbols.append(symbol)
                stop_losses.append(data['optimal_stop_loss'])
                take_profits.append(data['optimal_take_profit'])
                rr_ratios.append(data['risk_reward_ratio'])
                volatilities.append(data['annual_volatility'])
                
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Risk/Reward scatter
        ax1 = axes[0, 0]
        scatter = ax1.scatter(stop_losses, take_profits, c=volatilities, 
                            cmap='coolwarm', s=100, alpha=0.7)
        
        for i, symbol in enumerate(symbols[:20]):  # Label top 20
            ax1.annotate(symbol, (stop_losses[i], take_profits[i]), 
                        fontsize=8, alpha=0.7)
            
        ax1.set_xlabel('Optimal Stop Loss (%)')
        ax1.set_ylabel('Optimal Take Profit (%)')
        ax1.set_title('Risk vs Reward by Stock')
        
        # Add diagonal lines for RR ratios
        for rr in [1, 1.5, 2, 3]:
            x_line = np.array([0, max(stop_losses)])
            y_line = x_line * rr
            ax1.plot(x_line, y_line, '--', alpha=0.3, label=f'RR 1:{rr}')
            
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label('Annual Volatility (%)')
        
        # 2. Top Risk/Reward ratios
        ax2 = axes[0, 1]
        top_rr = sorted(zip(symbols, rr_ratios), key=lambda x: x[1], reverse=True)[:15]
        top_symbols, top_ratios = zip(*top_rr)
        
        ax2.barh(range(len(top_symbols)), top_ratios, color='purple', alpha=0.7)
        ax2.set_yticks(range(len(top_symbols)))
        ax2.set_yticklabels(top_symbols)
        ax2.set_xlabel('Risk/Reward Ratio')
        ax2.set_title('Top 15 Stocks by Risk/Reward Ratio')
        ax2.grid(True, alpha=0.3, axis='x')
        
        # 3. Volatility distribution
        ax3 = axes[1, 0]
        ax3.hist(volatilities, bins=20, color='blue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=30, color='green', linestyle='--', label='Low Risk')
        ax3.axvline(x=50, color='orange', linestyle='--', label='Medium Risk')
        ax3.axvline(x=70, color='red', linestyle='--', label='High Risk')
        ax3.set_xlabel('Annual Volatility (%)')
        ax3.set_ylabel('Number of Stocks')
        ax3.set_title('Volatility Distribution')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Strategy recommendations
        ax4 = axes[1, 1]
        strategies = self.create_trading_strategies()
        strategy_counts = {}
        for s in strategies.values():
            st = s['strategy_type']
            strategy_counts[st] = strategy_counts.get(st, 0) + 1
            
        if strategy_counts:
            ax4.pie(strategy_counts.values(), labels=strategy_counts.keys(), 
                   autopct='%1.1f%%', startangle=90)
            ax4.set_title('Recommended Strategy Distribution')
            
        plt.suptitle('BIST Stocks - Advanced Trading Analysis Summary', fontsize=16)
        plt.tight_layout()
        
        # Save
        plt.savefig('data/analysis/advanced_trading_summary.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """Main function"""
    analyzer = AdvancedStopLossAnalyzer()
    
    try:
        # Analyze all stocks
        analyzer.analyze_all_stocks()
        
        # Print results
        analyzer.print_results()
        
        # Save results
        analyzer.save_results()
        
        # Plot summary
        analyzer.plot_analysis()
        
        # Plot specific stock if requested
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('--symbol', type=str, help='Plot specific stock analysis')
        args = parser.parse_args()
        
        if args.symbol:
            analyzer.plot_analysis(args.symbol)
            
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise


if __name__ == "__main__":
    main()