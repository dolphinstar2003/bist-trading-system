#!/usr/bin/env python3
"""
Optimal Stop Loss Finder
Analyzes historical drawdowns to find stock-specific optimal stop loss levels
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


class OptimalStopLossFinder:
    """Find optimal stop loss levels based on historical drawdown analysis"""
    
    def __init__(self):
        self.results = {}
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
            
    def load_stock_data(self, symbol: str) -> pd.DataFrame:
        """Load historical data for a stock"""
        try:
            path = f"data/raw/{symbol}_1d_raw.csv"
            if os.path.exists(path):
                df = pd.read_csv(path)
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
                df.sort_index(inplace=True)
                
                # Filter data from 2020 onwards
                df = df[df.index >= '2020-01-01']
                return df
            else:
                logger.warning(f"Data not found for {symbol}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {symbol}: {e}")
            return pd.DataFrame()
            
    def calculate_drawdowns(self, prices: pd.Series) -> pd.DataFrame:
        """Calculate drawdown statistics"""
        # Calculate running maximum
        running_max = prices.expanding().max()
        
        # Calculate drawdown
        drawdown = (prices - running_max) / running_max
        
        # Find drawdown periods
        drawdown_start = []
        drawdown_end = []
        drawdown_length = []
        drawdown_magnitude = []
        recovery_time = []
        
        in_drawdown = False
        start_idx = None
        
        for i in range(len(drawdown)):
            if drawdown.iloc[i] < 0 and not in_drawdown:
                # Drawdown starts
                in_drawdown = True
                start_idx = i
            elif drawdown.iloc[i] == 0 and in_drawdown:
                # Drawdown ends
                in_drawdown = False
                if start_idx is not None:
                    # Record drawdown
                    drawdown_start.append(drawdown.index[start_idx])
                    drawdown_end.append(drawdown.index[i])
                    
                    # Find maximum drawdown in this period
                    dd_period = drawdown.iloc[start_idx:i+1]
                    max_dd = dd_period.min()
                    drawdown_magnitude.append(abs(max_dd))
                    
                    # Calculate lengths
                    drawdown_length.append(i - start_idx)
                    
                    # Recovery time (from max drawdown to recovery)
                    max_dd_idx = dd_period.idxmin()
                    recovery_days = (drawdown.index[i] - max_dd_idx).days
                    recovery_time.append(recovery_days)
        
        # Create DataFrame
        if drawdown_start:
            drawdowns_df = pd.DataFrame({
                'start_date': drawdown_start,
                'end_date': drawdown_end,
                'magnitude': drawdown_magnitude,
                'length_days': drawdown_length,
                'recovery_days': recovery_time
            })
            
            # Add percentage columns
            drawdowns_df['magnitude_pct'] = drawdowns_df['magnitude'] * 100
            
            return drawdowns_df
        else:
            return pd.DataFrame()
            
    def analyze_stock_drawdowns(self, symbol: str) -> Dict:
        """Analyze drawdowns for a single stock"""
        logger.info(f"Analyzing {symbol}...")
        
        # Load data
        df = self.load_stock_data(symbol)
        if df.empty:
            return None
            
        # Calculate returns and volatility
        df['returns'] = df['close'].pct_change()
        daily_volatility = df['returns'].std()
        annual_volatility = daily_volatility * np.sqrt(252)
        
        # Calculate drawdowns
        drawdowns_df = self.calculate_drawdowns(df['close'])
        
        if drawdowns_df.empty:
            return None
            
        # Analyze drawdown statistics
        stats = {
            'symbol': symbol,
            'total_days': len(df),
            'daily_volatility': daily_volatility,
            'annual_volatility': annual_volatility,
            'num_drawdowns': len(drawdowns_df),
            'avg_drawdown': drawdowns_df['magnitude_pct'].mean(),
            'median_drawdown': drawdowns_df['magnitude_pct'].median(),
            'max_drawdown': drawdowns_df['magnitude_pct'].max(),
            'percentile_95': drawdowns_df['magnitude_pct'].quantile(0.95),
            'percentile_90': drawdowns_df['magnitude_pct'].quantile(0.90),
            'percentile_80': drawdowns_df['magnitude_pct'].quantile(0.80),
            'percentile_70': drawdowns_df['magnitude_pct'].quantile(0.70),
            'percentile_60': drawdowns_df['magnitude_pct'].quantile(0.60),
            'percentile_50': drawdowns_df['magnitude_pct'].quantile(0.50),
            'avg_recovery_days': drawdowns_df['recovery_days'].mean(),
            'max_recovery_days': drawdowns_df['recovery_days'].max()
        }
        
        # Categorize drawdowns
        small_dd = drawdowns_df[drawdowns_df['magnitude_pct'] <= 5]
        medium_dd = drawdowns_df[(drawdowns_df['magnitude_pct'] > 5) & (drawdowns_df['magnitude_pct'] <= 10)]
        large_dd = drawdowns_df[drawdowns_df['magnitude_pct'] > 10]
        
        stats['small_dd_count'] = len(small_dd)
        stats['medium_dd_count'] = len(medium_dd)
        stats['large_dd_count'] = len(large_dd)
        
        # Calculate optimal stop loss based on drawdown patterns
        # Use percentile approach - avoid bottom 20% worst drawdowns
        optimal_stop_loss = drawdowns_df['magnitude_pct'].quantile(0.80)
        
        # Adjust based on volatility
        volatility_adjustment = daily_volatility * 100 * 2  # 2 standard deviations
        
        # Final optimal stop loss
        stats['optimal_stop_loss'] = min(optimal_stop_loss, volatility_adjustment * 2)
        stats['conservative_stop_loss'] = stats['percentile_70']  # Avoid bottom 30%
        stats['aggressive_stop_loss'] = stats['percentile_90']   # Avoid only worst 10%
        
        # Store detailed drawdown data
        stats['drawdowns'] = drawdowns_df
        
        return stats
        
    def find_optimal_stop_losses(self):
        """Find optimal stop losses for all stocks"""
        logger.info("Analyzing all stocks...")
        
        # Analyze each stock
        with ProcessPoolExecutor(max_workers=8) as executor:
            future_to_symbol = {
                executor.submit(self.analyze_stock_drawdowns, symbol): symbol 
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
                    
    def categorize_stocks(self):
        """Categorize stocks by volatility and risk profile"""
        if not self.results:
            return
            
        # Create DataFrame for easier analysis
        summary_data = []
        for symbol, stats in self.results.items():
            summary_data.append({
                'Symbol': symbol,
                'Annual_Volatility': stats['annual_volatility'] * 100,
                'Max_Drawdown': stats['max_drawdown'],
                'Avg_Drawdown': stats['avg_drawdown'],
                'Optimal_Stop_Loss': stats['optimal_stop_loss'],
                'Conservative_SL': stats['conservative_stop_loss'],
                'Aggressive_SL': stats['aggressive_stop_loss'],
                'Avg_Recovery_Days': stats['avg_recovery_days']
            })
            
        self.summary_df = pd.DataFrame(summary_data)
        
        # Categorize by volatility
        self.summary_df['Risk_Category'] = pd.cut(
            self.summary_df['Annual_Volatility'],
            bins=[0, 30, 50, 100],
            labels=['Low_Risk', 'Medium_Risk', 'High_Risk']
        )
        
    def print_results(self):
        """Print analysis results"""
        print("\n" + "="*100)
        print("OPTIMAL STOP LOSS ANALYSIS RESULTS")
        print("="*100)
        
        if hasattr(self, 'summary_df'):
            # Sort by optimal stop loss
            sorted_df = self.summary_df.sort_values('Optimal_Stop_Loss')
            
            print("\nStock-Specific Optimal Stop Loss Levels:")
            print("-"*100)
            print(f"{'Symbol':<10} {'Volatility':<12} {'Max DD':<10} {'Avg DD':<10} "
                  f"{'Optimal SL':<12} {'Conservative':<12} {'Aggressive':<12} {'Risk':<12}")
            print("-"*100)
            
            for _, row in sorted_df.iterrows():
                print(f"{row['Symbol']:<10} {row['Annual_Volatility']:>10.1f}% "
                      f"{row['Max_Drawdown']:>8.1f}% {row['Avg_Drawdown']:>8.1f}% "
                      f"{row['Optimal_Stop_Loss']:>10.1f}% {row['Conservative_SL']:>11.1f}% "
                      f"{row['Aggressive_SL']:>11.1f}% {row['Risk_Category']:<12}")
                      
            # Summary by risk category
            print("\n" + "-"*50)
            print("Summary by Risk Category:")
            print("-"*50)
            
            for category in ['Low_Risk', 'Medium_Risk', 'High_Risk']:
                cat_data = sorted_df[sorted_df['Risk_Category'] == category]
                if not cat_data.empty:
                    print(f"\n{category}:")
                    print(f"  Stocks: {len(cat_data)}")
                    print(f"  Avg Optimal Stop Loss: {cat_data['Optimal_Stop_Loss'].mean():.1f}%")
                    print(f"  Range: {cat_data['Optimal_Stop_Loss'].min():.1f}% - {cat_data['Optimal_Stop_Loss'].max():.1f}%")
                    
        print("\n" + "="*100)
        
    def save_results(self):
        """Save results to CSV"""
        if hasattr(self, 'summary_df'):
            output_dir = 'data/analysis'
            os.makedirs(output_dir, exist_ok=True)
            
            # Save summary
            summary_path = os.path.join(output_dir, 'optimal_stop_losses.csv')
            self.summary_df.to_csv(summary_path, index=False)
            logger.info(f"Summary saved to {summary_path}")
            
            # Save detailed drawdown data
            for symbol, stats in self.results.items():
                if 'drawdowns' in stats:
                    dd_path = os.path.join(output_dir, f'{symbol}_drawdowns.csv')
                    stats['drawdowns'].to_csv(dd_path, index=False)
                    
    def plot_analysis(self, top_n: int = 10):
        """Plot analysis results"""
        if not hasattr(self, 'summary_df'):
            return
            
        # Select top N stocks by trading volume or just first N
        plot_df = self.summary_df.head(top_n)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Stop Loss Comparison
        ax1 = axes[0, 0]
        x = np.arange(len(plot_df))
        width = 0.25
        
        ax1.bar(x - width, plot_df['Conservative_SL'], width, label='Conservative', alpha=0.8, color='green')
        ax1.bar(x, plot_df['Optimal_Stop_Loss'], width, label='Optimal', alpha=0.8, color='blue')
        ax1.bar(x + width, plot_df['Aggressive_SL'], width, label='Aggressive', alpha=0.8, color='red')
        
        ax1.set_xlabel('Stock')
        ax1.set_ylabel('Stop Loss (%)')
        ax1.set_title('Stop Loss Levels by Stock')
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_df['Symbol'], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Volatility vs Stop Loss
        ax2 = axes[0, 1]
        scatter = ax2.scatter(plot_df['Annual_Volatility'], plot_df['Optimal_Stop_Loss'], 
                            c=plot_df['Max_Drawdown'], cmap='coolwarm', s=100, alpha=0.7)
        
        for i, symbol in enumerate(plot_df['Symbol']):
            ax2.annotate(symbol, (plot_df['Annual_Volatility'].iloc[i], 
                                 plot_df['Optimal_Stop_Loss'].iloc[i]), 
                        fontsize=8, alpha=0.7)
            
        ax2.set_xlabel('Annual Volatility (%)')
        ax2.set_ylabel('Optimal Stop Loss (%)')
        ax2.set_title('Volatility vs Optimal Stop Loss')
        cbar = plt.colorbar(scatter, ax=ax2)
        cbar.set_label('Max Drawdown (%)')
        ax2.grid(True, alpha=0.3)
        
        # 3. Risk Category Distribution
        ax3 = axes[1, 0]
        risk_counts = self.summary_df['Risk_Category'].value_counts()
        ax3.pie(risk_counts.values, labels=risk_counts.index, autopct='%1.1f%%', startangle=90)
        ax3.set_title('Stock Distribution by Risk Category')
        
        # 4. Recovery Time Analysis
        ax4 = axes[1, 1]
        plot_df_sorted = plot_df.sort_values('Avg_Recovery_Days')
        ax4.barh(plot_df_sorted['Symbol'], plot_df_sorted['Avg_Recovery_Days'], 
                color='purple', alpha=0.7)
        ax4.set_xlabel('Average Recovery Days')
        ax4.set_ylabel('Stock')
        ax4.set_title('Average Drawdown Recovery Time')
        ax4.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('data/analysis/optimal_stop_loss_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def generate_trading_rules(self):
        """Generate stock-specific trading rules"""
        print("\n" + "="*60)
        print("RECOMMENDED TRADING RULES BY STOCK")
        print("="*60)
        
        if hasattr(self, 'summary_df'):
            for _, row in self.summary_df.iterrows():
                print(f"\n{row['Symbol']}:")
                print(f"  Risk Profile: {row['Risk_Category']}")
                print(f"  Recommended Stop Loss: {row['Optimal_Stop_Loss']:.1f}%")
                
                if row['Annual_Volatility'] < 30:
                    print(f"  Strategy: Low volatility stock, can use tighter stops")
                    print(f"  Position Size: Up to 20% of portfolio")
                elif row['Annual_Volatility'] < 50:
                    print(f"  Strategy: Medium volatility, use standard stops")
                    print(f"  Position Size: Up to 15% of portfolio")
                else:
                    print(f"  Strategy: High volatility, use wider stops or reduce position")
                    print(f"  Position Size: Max 10% of portfolio")
                    
                print(f"  Entry: Wait for drawdown of {row['Avg_Drawdown']/2:.1f}% for better entry")
                print(f"  Exit: Consider profit taking at {row['Optimal_Stop_Loss']*2:.1f}% gain")


def main():
    """Main function"""
    finder = OptimalStopLossFinder()
    
    try:
        # Find optimal stop losses
        finder.find_optimal_stop_losses()
        
        # Categorize stocks
        finder.categorize_stocks()
        
        # Print results
        finder.print_results()
        
        # Save results
        finder.save_results()
        
        # Plot analysis
        finder.plot_analysis()
        
        # Generate trading rules
        finder.generate_trading_rules()
        
    except Exception as e:
        logger.error(f"Error in analysis: {e}")
        raise


if __name__ == "__main__":
    main()