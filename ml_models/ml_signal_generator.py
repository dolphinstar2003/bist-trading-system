#!/usr/bin/env python3
"""
ML Signal Generator - Real-time signal generation using trained models
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger
import json
import joblib
from dataclasses import dataclass
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Proje imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS
from ml_models.ml_trading_system import MLTradingSystem


@dataclass
class TradingSignal:
    """Trading signal data class"""
    symbol: str
    timeframe: str
    timestamp: datetime
    signal: int  # -1: sell, 0: hold, 1: buy
    confidence: float
    ml_scores: Dict[str, float]
    indicator_states: Dict[str, Any]
    risk_metrics: Dict[str, float]
    
    def to_dict(self) -> Dict:
        return {
            'symbol': self.symbol,
            'timeframe': self.timeframe,
            'timestamp': self.timestamp.isoformat(),
            'signal': self.signal,
            'confidence': self.confidence,
            'ml_scores': self.ml_scores,
            'indicator_states': self.indicator_states,
            'risk_metrics': self.risk_metrics
        }


class MLSignalGenerator:
    """Generate trading signals using ML models and indicators"""
    
    def __init__(self, model_timestamp: str = None):
        self.csv_manager = CSVDataManager()
        self.ml_system = MLTradingSystem()
        
        # Risk parameters
        self.max_positions = 10
        self.position_size_pct = 0.1  # 10% per position
        self.max_correlation = 0.7
        self.min_confidence = 0.6
        
        # Signal cache
        self.signal_cache = {}
        self.last_update = {}
        
        # Load ML models if timestamp provided
        if model_timestamp:
            self.load_models(model_timestamp)
    
    def load_models(self, timestamp: str):
        """Load pre-trained ML models"""
        for timeframe in ['1h', '4h', '1d']:
            try:
                self.ml_system.load_models(timeframe, timestamp)
                logger.info(f"Loaded models for {timeframe}")
            except Exception as e:
                logger.error(f"Error loading models for {timeframe}: {e}")
    
    def calculate_risk_metrics(self, symbol: str, timeframe: str, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate risk metrics for a symbol"""
        try:
            # Recent volatility
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1]
            
            # Maximum drawdown (last 100 bars)
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.tail(100).min()
            
            # Sharpe ratio (last 100 bars)
            sharpe = returns.tail(100).mean() / returns.tail(100).std() * np.sqrt(252)
            
            # ATR percentage
            atr = self.calculate_atr(df, 14)
            atr_pct = (atr / df['close']).iloc[-1]
            
            # Support/Resistance distance
            support = df['low'].rolling(20).min().iloc[-1]
            resistance = df['high'].rolling(20).max().iloc[-1]
            support_distance = (df['close'].iloc[-1] - support) / df['close'].iloc[-1]
            resistance_distance = (resistance - df['close'].iloc[-1]) / df['close'].iloc[-1]
            
            return {
                'volatility': volatility,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe,
                'atr_pct': atr_pct,
                'support_distance': support_distance,
                'resistance_distance': resistance_distance
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return {
                'volatility': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'atr_pct': 0,
                'support_distance': 0,
                'resistance_distance': 0
            }
    
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def get_indicator_states(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Get current states of all indicators"""
        states = {}
        
        try:
            # Supertrend
            st_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'supertrend')
            if st_data is not None and len(st_data) > 0:
                latest = st_data.iloc[-1]
                states['supertrend'] = {
                    'signal': int(latest.get('buy_signal', 0)) - int(latest.get('sell_signal', 0)),
                    'trend': latest.get('trend_direction', 'unknown')
                }
            
            # Squeeze Momentum
            sqz_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'squeeze_momentum')
            if sqz_data is not None and len(sqz_data) > 0:
                latest = sqz_data.iloc[-1]
                states['squeeze_momentum'] = {
                    'signal': int(latest.get('sqz_buy_signal', 0)) - int(latest.get('sqz_sell_signal', 0)),
                    'momentum': latest.get('momentum', 0),
                    'squeeze_on': bool(latest.get('squeeze_on', False))
                }
            
            # MACD
            macd_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'macd')
            if macd_data is not None and len(macd_data) > 0:
                latest = macd_data.iloc[-1]
                states['macd'] = {
                    'signal': int(latest.get('macd_buy_signal', 0)) - int(latest.get('macd_sell_signal', 0)),
                    'histogram': latest.get('macd_histogram', 0)
                }
            
            # Lorentzian
            lor_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'lorentzian')
            if lor_data is not None and len(lor_data) > 0:
                latest = lor_data.iloc[-1]
                states['lorentzian'] = {
                    'signal': latest.get('signal', 0),
                    'confidence': latest.get('confidence', 0.5)
                }
            
            # Trend Vanguard
            tv_data = self.csv_manager.load_indicator_data(symbol, timeframe, 'trend_vanguard')
            if tv_data is not None and len(tv_data) > 0:
                latest = tv_data.iloc[-1]
                states['trend_vanguard'] = {
                    'signal': latest.get('signal', 0),
                    'strength': latest.get('strength', 0),
                    'confidence': latest.get('confidence', 0.5),
                    'regime': latest.get('market_regime', 'neutral')
                }
            
        except Exception as e:
            logger.error(f"Error getting indicator states: {e}")
        
        return states
    
    def generate_signal(self, symbol: str, timeframe: str) -> Optional[TradingSignal]:
        """Generate trading signal for a symbol"""
        try:
            # Get ML prediction
            ml_prediction = self.ml_system.predict_ensemble(symbol, timeframe)
            if not ml_prediction:
                return None
            
            # Get indicator states
            indicator_states = self.get_indicator_states(symbol, timeframe)
            
            # Load price data for risk metrics
            price_data = self.csv_manager.load_raw_data(symbol, timeframe)
            if price_data is None or len(price_data) < 100:
                return None
            
            # Calculate risk metrics
            risk_metrics = self.calculate_risk_metrics(symbol, timeframe, price_data)
            
            # Combine ML and indicator signals
            ml_signal = ml_prediction['ensemble_prediction']
            ml_confidence = ml_prediction['confidence']
            
            # Count confirming indicators
            indicator_confirmations = 0
            for ind_name, ind_state in indicator_states.items():
                if isinstance(ind_state, dict) and 'signal' in ind_state:
                    if ind_state['signal'] * ml_signal > 0:  # Same direction
                        indicator_confirmations += 1
            
            # Adjust confidence based on confirmations
            confirmation_ratio = indicator_confirmations / max(len(indicator_states), 1)
            adjusted_confidence = ml_confidence * (0.7 + 0.3 * confirmation_ratio)
            
            # Risk adjustment
            if risk_metrics['volatility'] > 0.03:  # High volatility
                adjusted_confidence *= 0.9
            if risk_metrics['max_drawdown'] < -0.1:  # Recent drawdown
                adjusted_confidence *= 0.95
            
            # Final signal decision
            if adjusted_confidence < self.min_confidence:
                final_signal = 0  # Hold
            else:
                final_signal = ml_signal
            
            # Create signal object
            signal = TradingSignal(
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                signal=final_signal,
                confidence=adjusted_confidence,
                ml_scores=ml_prediction['individual_predictions'],
                indicator_states=indicator_states,
                risk_metrics=risk_metrics
            )
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {e}")
            return None
    
    async def generate_signals_async(self, symbols: List[str], timeframe: str) -> List[TradingSignal]:
        """Generate signals for multiple symbols asynchronously"""
        signals = []
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all tasks
            futures = {
                executor.submit(self.generate_signal, symbol, timeframe): symbol 
                for symbol in symbols
            }
            
            # Collect results
            for future in futures:
                try:
                    signal = future.result(timeout=30)
                    if signal:
                        signals.append(signal)
                except Exception as e:
                    logger.error(f"Error processing {futures[future]}: {e}")
        
        return signals
    
    def rank_signals(self, signals: List[TradingSignal]) -> List[TradingSignal]:
        """Rank signals by quality"""
        # Score each signal
        for signal in signals:
            score = 0
            
            # Confidence weight (40%)
            score += signal.confidence * 0.4
            
            # Risk-adjusted score (30%)
            risk_score = 1 - signal.risk_metrics.get('volatility', 0) * 10
            risk_score *= 1 - abs(signal.risk_metrics.get('max_drawdown', 0))
            score += risk_score * 0.3
            
            # Indicator agreement (30%)
            confirmations = sum(
                1 for ind in signal.indicator_states.values()
                if isinstance(ind, dict) and ind.get('signal', 0) == signal.signal
            )
            agreement_score = confirmations / max(len(signal.indicator_states), 1)
            score += agreement_score * 0.3
            
            # Special bonuses
            if signal.indicator_states.get('trend_vanguard', {}).get('regime') == 'bullish' and signal.signal == 1:
                score += 0.05
            if signal.indicator_states.get('squeeze_momentum', {}).get('squeeze_on'):
                score += 0.03
            
            signal.quality_score = score
        
        # Sort by quality score
        return sorted(signals, key=lambda s: s.quality_score, reverse=True)
    
    def filter_by_correlation(self, signals: List[TradingSignal], max_corr: float = 0.7) -> List[TradingSignal]:
        """Filter signals to avoid correlated positions"""
        # This is a simplified version - in production you'd calculate actual correlation
        selected = []
        sectors = {
            'THYAO': 'transport',
            'GARAN': 'banking',
            'SAHOL': 'holding',
            'EREGL': 'steel',
            'AKBNK': 'banking',
            'SISE': 'glass',
            'TUPRS': 'energy',
            'ARCLK': 'durable'
        }
        
        selected_sectors = set()
        
        for signal in signals:
            symbol_sector = sectors.get(signal.symbol, 'other')
            
            # Skip if we already have this sector
            if symbol_sector in selected_sectors and symbol_sector != 'other':
                continue
            
            selected.append(signal)
            selected_sectors.add(symbol_sector)
            
            if len(selected) >= self.max_positions:
                break
        
        return selected
    
    def generate_portfolio_signals(self, symbols: List[str], timeframe: str) -> Dict[str, Any]:
        """Generate complete portfolio signals"""
        logger.info(f"Generating signals for {len(symbols)} symbols on {timeframe}")
        
        # Generate all signals
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        all_signals = loop.run_until_complete(self.generate_signals_async(symbols, timeframe))
        
        # Filter for buy/sell signals only
        active_signals = [s for s in all_signals if s.signal != 0]
        
        # Rank by quality
        ranked_signals = self.rank_signals(active_signals)
        
        # Filter by correlation
        final_signals = self.filter_by_correlation(ranked_signals)
        
        # Prepare portfolio summary
        portfolio = {
            'timestamp': datetime.now().isoformat(),
            'timeframe': timeframe,
            'total_symbols_analyzed': len(symbols),
            'active_signals': len(active_signals),
            'selected_signals': len(final_signals),
            'signals': [s.to_dict() for s in final_signals],
            'summary': {
                'buy_signals': sum(1 for s in final_signals if s.signal == 1),
                'sell_signals': sum(1 for s in final_signals if s.signal == -1),
                'avg_confidence': np.mean([s.confidence for s in final_signals]) if final_signals else 0,
                'top_opportunity': final_signals[0].symbol if final_signals else None
            }
        }
        
        return portfolio
    
    def save_signals(self, portfolio: Dict[str, Any], filename: str = None):
        """Save signals to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"ml_models/signals/portfolio_signals_{timestamp}.json"
        
        # Create directory if needed
        Path(filename).parent.mkdir(exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(i) for i in obj]
            return obj
        
        serializable_portfolio = convert_to_serializable(portfolio)
        
        with open(filename, 'w') as f:
            json.dump(serializable_portfolio, f, indent=2)
        
        logger.info(f"Signals saved to {filename}")
    
    def create_trading_report(self, portfolio: Dict[str, Any]) -> str:
        """Create human-readable trading report"""
        report = []
        report.append("=" * 60)
        report.append("ML TRADING SIGNALS REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {portfolio['timestamp']}")
        report.append(f"Timeframe: {portfolio['timeframe']}")
        report.append(f"Symbols Analyzed: {portfolio['total_symbols_analyzed']}")
        report.append(f"Active Signals: {portfolio['active_signals']}")
        report.append(f"Selected for Portfolio: {portfolio['selected_signals']}")
        report.append("")
        
        if portfolio['signals']:
            report.append("TOP TRADING OPPORTUNITIES:")
            report.append("-" * 60)
            
            for i, signal in enumerate(portfolio['signals'][:5], 1):
                action = "BUY" if signal['signal'] == 1 else "SELL"
                report.append(f"\n{i}. {signal['symbol']} - {action}")
                report.append(f"   Confidence: {signal['confidence']:.1%}")
                report.append(f"   ML Models: {signal['ml_scores']}")
                
                # Indicator summary
                ind_summary = []
                for ind, state in signal['indicator_states'].items():
                    if isinstance(state, dict) and 'signal' in state:
                        if state['signal'] != 0:
                            ind_summary.append(f"{ind}({'BUY' if state['signal'] > 0 else 'SELL'})")
                
                if ind_summary:
                    report.append(f"   Indicators: {', '.join(ind_summary)}")
                
                # Risk metrics
                risk = signal['risk_metrics']
                report.append(f"   Risk: Vol={risk['volatility']:.1%}, DD={risk['max_drawdown']:.1%}")
        else:
            report.append("No trading opportunities found meeting criteria.")
        
        report.append("\n" + "=" * 60)
        return "\n".join(report)


def main():
    """Generate trading signals"""
    # Initialize generator
    generator = MLSignalGenerator()
    
    # Load pre-trained models (use your actual timestamp)
    # generator.load_models('20240627_120000')
    
    # Or train new models
    ml_system = MLTradingSystem()
    training_symbols = ['THYAO', 'GARAN', 'SAHOL', 'EREGL', 'AKBNK', 'SISE', 'TUPRS', 'ARCLK']
    ml_system.train_ensemble_models(training_symbols, '1h')
    generator.ml_system = ml_system
    
    # Generate signals for all symbols
    all_symbols = ASSETS[:20]  # Top 20 symbols
    portfolio = generator.generate_portfolio_signals(all_symbols, '1h')
    
    # Save signals
    generator.save_signals(portfolio)
    
    # Print report
    report = generator.create_trading_report(portfolio)
    print(report)
    
    # Live monitoring example
    logger.info("\nStarting live signal monitoring...")
    while False:  # Set to True for live monitoring
        try:
            portfolio = generator.generate_portfolio_signals(all_symbols, '1h')
            
            if portfolio['signals']:
                logger.info(f"Found {len(portfolio['signals'])} opportunities")
                for signal in portfolio['signals'][:3]:
                    logger.info(f"  {signal['symbol']}: {signal['signal']} (conf: {signal['confidence']:.1%})")
            
            # Wait before next update
            asyncio.run(asyncio.sleep(300))  # 5 minutes
            
        except KeyboardInterrupt:
            logger.info("Monitoring stopped")
            break
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")


if __name__ == "__main__":
    main()