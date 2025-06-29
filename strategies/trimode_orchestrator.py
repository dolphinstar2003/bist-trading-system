#!/usr/bin/env python3
"""
TriMode Orchestrator - Dynamic Strategy Selection System
Automatically selects between Aggressive, Balanced, and Defensive modes
Target: 10% monthly return with adaptive risk management
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
import json
from typing import Dict, List, Tuple, Optional
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.market_regime_detector import MarketRegimeDetector
from strategies.adaptive_ensemble_system import AdaptiveEnsembleSystem
from ml_models.ml_trading_system_fixed import MLTradingSystem
from ml_models.ml_signal_generator import MLSignalGenerator
from risk.dynamic_risk_manager import DynamicRiskManager
from utils.csv_data_manager import CSVDataManager
from config.assets import ASSETS
from backtest.ema_cross_optimizer_fixed import EMACrossOptimizerFixed


class TradingMode(Enum):
    AGGRESSIVE = "AGGRESSIVE"
    BALANCED = "BALANCED"
    DEFENSIVE = "DEFENSIVE"


class TriModeOrchestrator:
    """
    Orchestrates between three trading modes based on market conditions
    Aggressive: Simple strategies, larger positions, trend following
    Balanced: Enhanced strategies, normal positions, mixed signals
    Defensive: ML-heavy strategies, small positions, capital preservation
    """
    
    def __init__(self, initial_capital: float = 100000.0):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        
        # Initialize subsystems
        self.csv_manager = CSVDataManager()
        self.regime_detector = MarketRegimeDetector()
        self.risk_manager = DynamicRiskManager(initial_capital)
        self.ml_system = MLTradingSystem()
        self.ml_signal_generator = MLSignalGenerator()
        self.ema_optimizer = EMACrossOptimizerFixed()
        
        # Mode settings
        self.current_mode = TradingMode.BALANCED
        self.mode_history = []
        
        # Performance tracking
        self.performance_window = 30  # days
        self.trades_history = []
        self.daily_returns = []
        self.positions = {}
        
        # Mode-specific parameters
        self.mode_params = {
            TradingMode.AGGRESSIVE: {
                'position_size': 0.20,  # 20% of capital per position
                'max_positions': 7,
                'stop_loss': 0.10,  # 10% stop loss
                'min_confirmations': 0,  # No confirmations needed
                'target_monthly_return': 0.15,  # 15% target
                'filters': ['volume'],  # Minimal filters
                'ml_weight': 0.1  # Low ML influence
            },
            TradingMode.BALANCED: {
                'position_size': 0.10,  # 10% of capital
                'max_positions': 5,
                'stop_loss': 0.05,  # 5% stop loss
                'min_confirmations': 2,  # 2 confirmations
                'target_monthly_return': 0.10,  # 10% target
                'filters': ['volume', 'adx', 'ml'],
                'ml_weight': 0.3  # Moderate ML influence
            },
            TradingMode.DEFENSIVE: {
                'position_size': 0.05,  # 5% of capital
                'max_positions': 3,
                'stop_loss': 0.03,  # 3% stop loss
                'min_confirmations': 3,  # 3 confirmations
                'target_monthly_return': 0.05,  # 5% target
                'filters': ['volume', 'adx', 'ml', 'mtf', 'volatility'],
                'ml_weight': 0.6  # High ML influence
            }
        }
        
        # Optimal EMA parameters cache (symbol -> (fast, slow))
        self.ema_params_cache = {}
        
        # Load ML models
        self._load_ml_models()
        
    def _load_ml_models(self):
        """Load ML models for the system"""
        try:
            success = self.ml_system.load_models('1d', timestamp=None)
            if success:
                self.ml_signal_generator.ml_system = self.ml_system
                logger.info("ML models loaded successfully")
            else:
                logger.warning("Could not load ML models")
        except Exception as e:
            logger.warning(f"ML loading failed: {e}")
    
    def determine_mode(self, market_data: Dict) -> TradingMode:
        """
        Determine the appropriate trading mode based on:
        1. Market regime
        2. Portfolio performance
        3. Market volatility
        4. Drawdown status
        """
        
        # 1. Get market regime
        # Use a representative symbol for regime detection (e.g., index or largest stock)
        if 'symbol_data' in market_data and market_data['symbol_data']:
            # Use first available symbol for regime detection
            sample_symbol = list(market_data['symbol_data'].keys())[0]
            regime_str, regime_analysis = self.regime_detector.detect_regime(
                sample_symbol, 
                market_data.get('timeframe', '1d')
            )
            regime = {'regime': regime_str, 'confidence': regime_analysis.get('confidence', 0)}
        else:
            # Fallback if no data
            regime = {'regime': 'unknown', 'confidence': 0}
        
        # 2. Calculate recent performance
        monthly_return = self._calculate_period_return(30)
        weekly_return = self._calculate_period_return(7)
        current_drawdown = self._calculate_current_drawdown()
        
        # 3. Calculate market volatility
        market_volatility = self._calculate_market_volatility(market_data)
        
        # 4. Count losing streak
        losing_streak = self._get_losing_streak()
        
        # Decision Logic
        logger.info(f"Mode Decision Factors:")
        logger.info(f"  Regime: {regime['regime']}")
        logger.info(f"  Monthly Return: {monthly_return:.1%}")
        logger.info(f"  Drawdown: {current_drawdown:.1%}")
        logger.info(f"  Volatility: {market_volatility:.1%}")
        logger.info(f"  Losing Streak: {losing_streak}")
        
        # DEFENSIVE conditions (highest priority)
        if (current_drawdown < -0.10 or  # 10% drawdown
            losing_streak >= 5 or  # 5 consecutive losses
            market_volatility > 0.04 or  # 4% daily volatility
            regime['regime'] == 'high_volatility' or
            monthly_return < -0.10):  # -10% monthly loss
            new_mode = TradingMode.DEFENSIVE
            
        # AGGRESSIVE conditions
        elif (regime['regime'] == 'strong_trend_up' and
              market_volatility < 0.02 and  # Low volatility
              monthly_return > 0.05 and  # Positive momentum
              current_drawdown > -0.05 and  # Small drawdown
              losing_streak < 2):  # Not many losses
            new_mode = TradingMode.AGGRESSIVE
            
        # BALANCED (default)
        else:
            new_mode = TradingMode.BALANCED
        
        # Mode transition smoothing (avoid rapid switching)
        if len(self.mode_history) >= 3:
            # If mode changed too frequently, stay in BALANCED
            recent_modes = [m['to_mode'] for m in self.mode_history[-3:]]
            if len(set(recent_modes)) == 3:  # All different
                logger.info("Mode switching too frequent, forcing BALANCED")
                new_mode = TradingMode.BALANCED
        
        # Record mode change
        if new_mode != self.current_mode:
            logger.info(f"Mode Change: {self.current_mode.value} -> {new_mode.value}")
            self.mode_history.append({
                'timestamp': datetime.now(),
                'from_mode': self.current_mode,
                'to_mode': new_mode,
                'reason': {
                    'regime': regime['regime'],
                    'monthly_return': monthly_return,
                    'drawdown': current_drawdown,
                    'volatility': market_volatility
                }
            })
        
        self.current_mode = new_mode
        return new_mode
    
    def generate_signals(self, symbols: List[str], timeframe: str = '1d') -> List[Dict]:
        """Generate trading signals based on current mode"""
        
        mode_params = self.mode_params[self.current_mode]
        signals = []
        
        logger.info(f"Generating signals in {self.current_mode.value} mode")
        
        for symbol in symbols:
            try:
                # Get market data
                data = self.csv_manager.load_raw_data(symbol, timeframe)
                if data is None or len(data) < 200:
                    continue
                
                # Generate signal based on mode
                if self.current_mode == TradingMode.AGGRESSIVE:
                    signal = self._generate_aggressive_signal(symbol, data, timeframe)
                elif self.current_mode == TradingMode.DEFENSIVE:
                    signal = self._generate_defensive_signal(symbol, data, timeframe)
                else:  # BALANCED
                    signal = self._generate_balanced_signal(symbol, data, timeframe)
                
                if signal and signal['strength'] > 0:
                    signals.append(signal)
                    
            except Exception as e:
                logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        # Sort by strength and apply position limits
        signals.sort(key=lambda x: x['strength'], reverse=True)
        max_positions = mode_params['max_positions']
        
        # Consider existing positions
        available_slots = max_positions - len(self.positions)
        
        return signals[:available_slots]
    
    def _generate_aggressive_signal(self, symbol: str, data: pd.DataFrame, 
                                   timeframe: str) -> Optional[Dict]:
        """
        Aggressive mode: Simple EMA cross with minimal filters
        Focus on trend following, larger positions
        """
        
        # Get optimal EMA parameters (or use defaults)
        if symbol not in self.ema_params_cache:
            # Use common aggressive parameters
            fast_ema = 10
            slow_ema = 30
        else:
            fast_ema, slow_ema = self.ema_params_cache[symbol]
        
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=fast_ema, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_ema, adjust=False).mean()
        
        # Simple crossover
        current_signal = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
        prev_signal = 1 if ema_fast.iloc[-2] > ema_slow.iloc[-2] else -1
        
        # Check for crossover
        if current_signal != prev_signal and current_signal == 1:
            # Buy signal
            
            # Minimal confirmations - just volume
            volume_confirmed = data['volume'].iloc[-1] > data['volume'].rolling(20).mean().iloc[-1]
            
            if volume_confirmed:
                # Calculate signal strength (simplified)
                ema_diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
                strength = min(abs(ema_diff) * 10, 1.0)  # Normalize to 0-1
                
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'strength': strength,
                    'mode': 'AGGRESSIVE',
                    'entry_price': data['close'].iloc[-1],
                    'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.AGGRESSIVE]['stop_loss']),
                    'position_size': self.mode_params[TradingMode.AGGRESSIVE]['position_size'],
                    'reasons': ['ema_cross', 'volume_confirmed'],
                    'timestamp': data.index[-1]
                }
        
        return None
    
    def _generate_balanced_signal(self, symbol: str, data: pd.DataFrame, 
                                 timeframe: str) -> Optional[Dict]:
        """
        Balanced mode: Enhanced strategy with multiple confirmations
        Mix of technical and ML signals
        """
        
        confirmations = []
        
        # 1. EMA Signal
        ema_20 = data['close'].ewm(span=20, adjust=False).mean()
        ema_50 = data['close'].ewm(span=50, adjust=False).mean()
        
        if ema_20.iloc[-1] > ema_50.iloc[-1] and ema_20.iloc[-2] <= ema_50.iloc[-2]:
            confirmations.append('ema_cross')
        
        # 2. Volume confirmation
        volume_ratio = data['volume'].iloc[-1] / data['volume'].rolling(20).mean().iloc[-1]
        if volume_ratio > 1.3:
            confirmations.append('high_volume')
        
        # 3. ADX confirmation
        adx = self._calculate_adx(data)
        if adx > 25:
            confirmations.append('strong_trend')
        
        # 4. ML Signal (if available)
        ml_signal = self._get_ml_signal(symbol, timeframe)
        if ml_signal and ml_signal['confidence'] > 0.4:
            confirmations.append(f'ml_{ml_signal["signal"]}')
        
        # Need at least 2 confirmations
        if len(confirmations) >= self.mode_params[TradingMode.BALANCED]['min_confirmations']:
            # Calculate weighted signal strength
            strength = len(confirmations) / 4.0  # Max 4 confirmations
            
            # Adjust for ML confidence
            if ml_signal:
                strength = strength * 0.7 + ml_signal['confidence'] * 0.3
            
            return {
                'symbol': symbol,
                'action': 'BUY',
                'strength': min(strength, 1.0),
                'mode': 'BALANCED',
                'entry_price': data['close'].iloc[-1],
                'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.BALANCED]['stop_loss']),
                'position_size': self.mode_params[TradingMode.BALANCED]['position_size'],
                'reasons': confirmations,
                'timestamp': data.index[-1]
            }
        
        return None
    
    def _generate_defensive_signal(self, symbol: str, data: pd.DataFrame, 
                                  timeframe: str) -> Optional[Dict]:
        """
        Defensive mode: ML-heavy with strict filters
        Focus on capital preservation
        """
        
        # Start with ML signal as primary
        ml_signal = self._get_ml_signal(symbol, timeframe)
        if not ml_signal or ml_signal['confidence'] < 0.5:
            return None  # No trade without strong ML signal
        
        confirmations = ['ml_signal']
        
        # Additional strict confirmations
        
        # 1. Volatility filter - avoid high volatility
        returns = data['close'].pct_change()
        recent_volatility = returns.tail(20).std()
        if recent_volatility < 0.03:  # Less than 3% daily volatility
            confirmations.append('low_volatility')
        
        # 2. Trend alignment
        sma_50 = data['close'].rolling(50).mean()
        sma_200 = data['close'].rolling(200).mean()
        if data['close'].iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]:
            confirmations.append('trend_aligned')
        
        # 3. RSI not overbought
        rsi = self._calculate_rsi(data['close'])
        if 30 < rsi < 70:
            confirmations.append('rsi_neutral')
        
        # 4. Support level nearby
        recent_low = data['low'].tail(20).min()
        if (data['close'].iloc[-1] - recent_low) / recent_low < 0.05:
            confirmations.append('near_support')
        
        # Need at least 3 confirmations for defensive mode
        if len(confirmations) >= self.mode_params[TradingMode.DEFENSIVE]['min_confirmations']:
            # Conservative strength calculation
            strength = ml_signal['confidence'] * 0.6 + (len(confirmations) / 5.0) * 0.4
            
            return {
                'symbol': symbol,
                'action': 'BUY',
                'strength': min(strength * 0.8, 1.0),  # Further reduce strength
                'mode': 'DEFENSIVE',
                'entry_price': data['close'].iloc[-1],
                'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.DEFENSIVE]['stop_loss']),
                'position_size': self.mode_params[TradingMode.DEFENSIVE]['position_size'],
                'reasons': confirmations,
                'timestamp': data.index[-1]
            }
        
        return None
    
    def execute_trades(self, signals: List[Dict]) -> List[Dict]:
        """Execute trades based on signals and current mode"""
        
        executed_trades = []
        mode_params = self.mode_params[self.current_mode]
        
        for signal in signals:
            # Check if we already have a position
            if signal['symbol'] in self.positions:
                continue
            
            # Calculate actual position size based on available capital
            available_capital = self.current_capital - sum(p['cost'] for p in self.positions.values())
            position_value = available_capital * signal['position_size']
            
            # Risk checks
            if position_value < 1000:  # Minimum position size
                continue
            
            # Additional mode-specific checks
            if self.current_mode == TradingMode.DEFENSIVE:
                # Extra cautious in defensive mode
                if len(self.positions) >= 2:  # Max 2 positions in defensive
                    break
            
            # Execute trade
            shares = int(position_value / signal['entry_price'])
            cost = shares * signal['entry_price'] * 1.002  # Including commission
            
            if cost <= available_capital:
                trade = {
                    'symbol': signal['symbol'],
                    'action': 'BUY',
                    'shares': shares,
                    'entry_price': signal['entry_price'],
                    'cost': cost,
                    'stop_loss': signal['stop_loss'],
                    'mode': signal['mode'],
                    'timestamp': signal['timestamp'],
                    'reasons': signal['reasons']
                }
                
                # Update positions
                self.positions[signal['symbol']] = trade
                self.current_capital -= cost
                executed_trades.append(trade)
                
                logger.info(f"Executed {signal['mode']} trade: {signal['symbol']} "
                          f"@ {signal['entry_price']:.2f} ({shares} shares)")
        
        return executed_trades
    
    def manage_positions(self, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """Manage existing positions based on mode and market conditions"""
        
        closed_trades = []
        mode_params = self.mode_params[self.current_mode]
        
        for symbol, position in list(self.positions.items()):
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]['close'].iloc[-1]
            
            # Calculate P&L
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            
            # Exit conditions based on mode
            should_exit = False
            exit_reason = ""
            
            # 1. Stop loss hit
            if current_price <= position['stop_loss']:
                should_exit = True
                exit_reason = "stop_loss"
            
            # 2. Mode-specific exits
            elif self.current_mode == TradingMode.AGGRESSIVE:
                # Aggressive: Let profits run, use trailing stop
                if pnl_pct > 0.10:  # 10% profit
                    # Tighten stop loss
                    new_stop = current_price * 0.95
                    position['stop_loss'] = max(position['stop_loss'], new_stop)
                
            elif self.current_mode == TradingMode.DEFENSIVE:
                # Defensive: Take profits early
                if pnl_pct > 0.03:  # 3% profit
                    should_exit = True
                    exit_reason = "profit_target"
                # Or exit if mode was different when entered
                elif position.get('mode') != 'DEFENSIVE':
                    should_exit = True
                    exit_reason = "mode_change"
            
            else:  # BALANCED
                # Balanced: Standard exit rules
                if pnl_pct > 0.08:  # 8% profit
                    # Trailing stop
                    new_stop = current_price * 0.96
                    position['stop_loss'] = max(position['stop_loss'], new_stop)
                elif pnl_pct < -0.05:  # 5% loss
                    should_exit = True
                    exit_reason = "stop_loss"
            
            # Execute exit
            if should_exit:
                exit_value = position['shares'] * current_price * 0.998  # Including commission
                self.current_capital += exit_value
                
                closed_trade = {
                    'symbol': symbol,
                    'action': 'SELL',
                    'shares': position['shares'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl': exit_value - position['cost'],
                    'pnl_pct': pnl_pct,
                    'exit_reason': exit_reason,
                    'mode': position.get('mode', 'UNKNOWN'),
                    'timestamp': current_data[symbol].index[-1]
                }
                
                closed_trades.append(closed_trade)
                del self.positions[symbol]
                
                logger.info(f"Closed position: {symbol} @ {current_price:.2f} "
                          f"(PnL: {pnl_pct:.1%}, Reason: {exit_reason})")
        
        return closed_trades
    
    # Helper methods
    
    def _calculate_period_return(self, days: int) -> float:
        """Calculate return over specified period"""
        if len(self.daily_returns) < days:
            return 0.0
        return sum(self.daily_returns[-days:])
    
    def _calculate_current_drawdown(self) -> float:
        """Calculate current drawdown from peak"""
        if not self.daily_returns:
            return 0.0
        
        cumulative = np.cumprod(1 + np.array(self.daily_returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown[-1] if len(drawdown) > 0 else 0.0
    
    def _calculate_market_volatility(self, market_data: Dict) -> float:
        """Calculate overall market volatility"""
        if 'index_data' in market_data:
            returns = market_data['index_data']['close'].pct_change().dropna()
            return returns.tail(20).std()
        return 0.02  # Default 2% if no data
    
    def _get_losing_streak(self) -> int:
        """Count consecutive losing trades"""
        if not self.trades_history:
            return 0
        
        streak = 0
        for trade in reversed(self.trades_history):
            if trade.get('pnl', 0) < 0:
                streak += 1
            else:
                break
        return streak
    
    def _get_ml_signal(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get ML signal if available"""
        try:
            if not self.ml_system:
                return None
                
            prediction = self.ml_system.predict_ensemble(symbol, timeframe)
            if prediction:
                return {
                    'signal': prediction['ensemble_prediction'],
                    'confidence': prediction['confidence']
                }
        except:
            pass
        return None
    
    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> float:
        """Calculate ADX indicator"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        plus_dm = high.diff()
        minus_dm = low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        
        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
        
        atr = tr.rolling(period).mean()
        
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.abs().rolling(period).mean() / atr)
        
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.iloc[-1] if not adx.empty else 0
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def get_status_report(self) -> Dict:
        """Get current system status"""
        return {
            'current_mode': self.current_mode.value,
            'mode_params': self.mode_params[self.current_mode],
            'positions': len(self.positions),
            'current_capital': self.current_capital,
            'total_value': self.current_capital + sum(
                p['shares'] * p.get('current_price', p['entry_price']) 
                for p in self.positions.values()
            ),
            'monthly_return': self._calculate_period_return(30),
            'drawdown': self._calculate_current_drawdown(),
            'active_positions': list(self.positions.keys()),
            'last_mode_change': self.mode_history[-1] if self.mode_history else None
        }


def main():
    """Test the TriMode Orchestrator"""
    print("="*60)
    print("TRIMODE ORCHESTRATOR TEST")
    print("="*60)
    
    orchestrator = TriModeOrchestrator(initial_capital=100000)
    
    # Test mode determination with different scenarios
    test_scenarios = [
        {
            'name': 'Bull Market',
            'index_data': pd.DataFrame({
                'close': np.linspace(100, 120, 100),
                'volume': np.random.rand(100) * 1000000
            }),
            'timeframe': '1d'
        },
        {
            'name': 'High Volatility',
            'index_data': pd.DataFrame({
                'close': 100 + np.random.randn(100) * 5,
                'volume': np.random.rand(100) * 1000000
            }),
            'timeframe': '1d'
        }
    ]
    
    for scenario in test_scenarios:
        print(f"\nTesting scenario: {scenario['name']}")
        mode = orchestrator.determine_mode(scenario)
        print(f"Selected mode: {mode.value}")
    
    # Test signal generation
    print("\nTesting signal generation...")
    test_symbols = ['THYAO', 'GARAN', 'AKBNK']
    signals = orchestrator.generate_signals(test_symbols)
    
    print(f"\nGenerated {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal['symbol']}: {signal['action']} "
              f"(strength: {signal['strength']:.2f}, mode: {signal['mode']})")
    
    # Get status report
    print("\nSystem Status:")
    status = orchestrator.get_status_report()
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()