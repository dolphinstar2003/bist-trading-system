#!/usr/bin/env python3
"""
Enhanced TriMode Orchestrator - Dynamic Strategy Selection System with Quick Wins
Includes: ATR trailing stop, partial profits, volume spike filter, optimal EMA storage, volatility sizing
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


class EnhancedTriModeOrchestrator:
    """
    Enhanced Orchestrator with Quick Win features:
    - ATR-based trailing stops for each mode
    - Partial profit taking (50% at targets)
    - Volume spike filter (RVOL > 1.5)
    - Optimal EMA parameter storage and usage
    - Volatility-based position sizing
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
        
        # Enhanced position tracking for partial profits
        self.partial_exits = {}  # Track partial exit status
        
        # Mode-specific parameters with enhanced settings
        self.mode_params = {
            TradingMode.AGGRESSIVE: {
                'position_size': 0.20,  # Base position size
                'max_positions': 7,
                'stop_loss': 0.10,
                'trailing_stop_atr_multiplier': 3.0,  # 3x ATR for aggressive
                'profit_target_atr_multiplier': 4.0,  # 4x ATR first target
                'partial_exit_percent': 0.5,  # Exit 50% at first target
                'min_confirmations': 0,
                'target_monthly_return': 0.15,
                'filters': ['volume', 'rvol'],
                'ml_weight': 0.1,
                'volatility_adjustment': True  # Enable volatility-based sizing
            },
            TradingMode.BALANCED: {
                'position_size': 0.10,
                'max_positions': 5,
                'stop_loss': 0.05,
                'trailing_stop_atr_multiplier': 2.5,  # 2.5x ATR for balanced
                'profit_target_atr_multiplier': 3.0,  # 3x ATR first target
                'partial_exit_percent': 0.5,
                'min_confirmations': 2,
                'target_monthly_return': 0.10,
                'filters': ['volume', 'adx', 'ml', 'rvol'],
                'ml_weight': 0.3,
                'volatility_adjustment': True
            },
            TradingMode.DEFENSIVE: {
                'position_size': 0.05,
                'max_positions': 3,
                'stop_loss': 0.03,
                'trailing_stop_atr_multiplier': 2.0,  # 2x ATR for defensive
                'profit_target_atr_multiplier': 2.0,  # 2x ATR first target
                'partial_exit_percent': 0.6,  # Exit 60% at first target (more conservative)
                'min_confirmations': 3,
                'target_monthly_return': 0.05,
                'filters': ['volume', 'adx', 'ml', 'mtf', 'volatility', 'rvol'],
                'ml_weight': 0.6,
                'volatility_adjustment': True
            }
        }
        
        # Optimal EMA parameters cache (symbol -> (fast, slow))
        self.ema_params_cache = self._load_optimal_ema_params()
        
        # ATR cache for efficiency
        self.atr_cache = {}
        
        # Load ML models
        self._load_ml_models()
    
    def _load_optimal_ema_params(self) -> Dict[str, Tuple[int, int]]:
        """Load pre-optimized EMA parameters from file or calculate"""
        params_file = Path('backtest/optimal_ema_params.json')
        
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
                # Convert to tuples
                return {k: tuple(v) for k, v in params.items()}
        else:
            # Default parameters by market cap/volatility
            logger.info("No optimal EMA parameters found, using defaults")
            return {
                # High volatility stocks: shorter periods
                'THYAO': (8, 21),
                'GARAN': (10, 30),
                'AKBNK': (10, 30),
                # Default for others
                'DEFAULT': (12, 26)
            }
    
    def _save_optimal_ema_params(self):
        """Save optimal EMA parameters to file"""
        params_file = Path('backtest/optimal_ema_params.json')
        with open(params_file, 'w') as f:
            # Convert tuples to lists for JSON
            params_list = {k: list(v) for k, v in self.ema_params_cache.items()}
            json.dump(params_list, f, indent=2)
    
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
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(period).mean()
        
        return atr
    
    def _calculate_relative_volume(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Relative Volume (RVOL)"""
        avg_volume = df['volume'].rolling(period).mean()
        rvol = df['volume'] / avg_volume
        return rvol.fillna(1.0)
    
    def _get_volatility_adjustment(self, symbol: str, data: pd.DataFrame) -> float:
        """Calculate position size adjustment based on volatility"""
        # Calculate recent volatility
        returns = data['close'].pct_change()
        recent_vol = returns.tail(20).std()
        
        # Normal volatility benchmark (2% daily)
        normal_vol = 0.02
        
        # Inverse volatility adjustment (lower size for higher vol)
        adjustment = min(normal_vol / (recent_vol + 0.001), 1.5)
        adjustment = max(adjustment, 0.5)  # Between 0.5x and 1.5x
        
        return adjustment
    
    def determine_mode(self, market_data: Dict) -> TradingMode:
        """
        Determine the appropriate trading mode based on:
        1. Market regime
        2. Portfolio performance
        3. Market volatility
        4. Drawdown status
        """
        
        # 1. Get market regime
        if 'symbol_data' in market_data and market_data['symbol_data']:
            sample_symbol = list(market_data['symbol_data'].keys())[0]
            regime_str, regime_analysis = self.regime_detector.detect_regime(
                sample_symbol, 
                market_data.get('timeframe', '1d')
            )
            regime = {'regime': regime_str, 'confidence': regime_analysis.get('confidence', 0)}
        else:
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
        
        # Mode transition smoothing
        if len(self.mode_history) >= 3:
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
                
                # Calculate ATR for this symbol
                atr = self._calculate_atr(data)
                self.atr_cache[symbol] = atr.iloc[-1]
                
                # Calculate RVOL
                rvol = self._calculate_relative_volume(data)
                
                # Check volume spike filter
                if 'rvol' in mode_params['filters'] and rvol.iloc[-1] < 1.5:
                    continue  # Skip if no volume spike
                
                # Generate signal based on mode
                if self.current_mode == TradingMode.AGGRESSIVE:
                    signal = self._generate_aggressive_signal(symbol, data, timeframe, rvol)
                elif self.current_mode == TradingMode.DEFENSIVE:
                    signal = self._generate_defensive_signal(symbol, data, timeframe, rvol)
                else:  # BALANCED
                    signal = self._generate_balanced_signal(symbol, data, timeframe, rvol)
                
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
                                   timeframe: str, rvol: pd.Series) -> Optional[Dict]:
        """
        Aggressive mode: Optimal EMA cross with volume spike
        """
        
        # Get optimal EMA parameters
        if symbol in self.ema_params_cache:
            fast_ema, slow_ema = self.ema_params_cache[symbol]
        else:
            fast_ema, slow_ema = self.ema_params_cache.get('DEFAULT', (10, 30))
        
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=fast_ema, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_ema, adjust=False).mean()
        
        # Simple crossover
        current_signal = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
        prev_signal = 1 if ema_fast.iloc[-2] > ema_slow.iloc[-2] else -1
        
        # Check for crossover
        if current_signal != prev_signal and current_signal == 1:
            # Buy signal
            
            # Volume confirmation with RVOL
            volume_confirmed = rvol.iloc[-1] > 1.5
            
            if volume_confirmed:
                # Calculate signal strength
                ema_diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
                strength = min(abs(ema_diff) * 10, 1.0)
                
                # Get volatility adjustment
                vol_adjustment = self._get_volatility_adjustment(symbol, data) if self.mode_params[TradingMode.AGGRESSIVE]['volatility_adjustment'] else 1.0
                
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'strength': strength,
                    'mode': 'AGGRESSIVE',
                    'entry_price': data['close'].iloc[-1],
                    'atr': self.atr_cache.get(symbol, data['close'].iloc[-1] * 0.02),
                    'rvol': rvol.iloc[-1],
                    'volatility_adjustment': vol_adjustment,
                    'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.AGGRESSIVE]['stop_loss']),
                    'position_size': self.mode_params[TradingMode.AGGRESSIVE]['position_size'] * vol_adjustment,
                    'reasons': ['optimal_ema_cross', f'rvol_{rvol.iloc[-1]:.1f}'],
                    'timestamp': data.index[-1],
                    'ema_params': (fast_ema, slow_ema)
                }
        
        return None
    
    def _generate_balanced_signal(self, symbol: str, data: pd.DataFrame, 
                                 timeframe: str, rvol: pd.Series) -> Optional[Dict]:
        """
        Balanced mode: Enhanced strategy with multiple confirmations
        """
        
        confirmations = []
        
        # 1. EMA Signal with optimal parameters
        if symbol in self.ema_params_cache:
            fast_ema, slow_ema = self.ema_params_cache[symbol]
        else:
            fast_ema, slow_ema = 20, 50  # Default balanced
        
        ema_fast = data['close'].ewm(span=fast_ema, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_ema, adjust=False).mean()
        
        if ema_fast.iloc[-1] > ema_slow.iloc[-1] and ema_fast.iloc[-2] <= ema_slow.iloc[-2]:
            confirmations.append('ema_cross')
        
        # 2. Volume spike confirmation
        if rvol.iloc[-1] > 1.5:
            confirmations.append(f'rvol_{rvol.iloc[-1]:.1f}')
        
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
            strength = len(confirmations) / 4.0
            
            # Adjust for ML confidence
            if ml_signal:
                strength = strength * 0.7 + ml_signal['confidence'] * 0.3
            
            # Get volatility adjustment
            vol_adjustment = self._get_volatility_adjustment(symbol, data) if self.mode_params[TradingMode.BALANCED]['volatility_adjustment'] else 1.0
            
            return {
                'symbol': symbol,
                'action': 'BUY',
                'strength': min(strength, 1.0),
                'mode': 'BALANCED',
                'entry_price': data['close'].iloc[-1],
                'atr': self.atr_cache.get(symbol, data['close'].iloc[-1] * 0.02),
                'rvol': rvol.iloc[-1],
                'volatility_adjustment': vol_adjustment,
                'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.BALANCED]['stop_loss']),
                'position_size': self.mode_params[TradingMode.BALANCED]['position_size'] * vol_adjustment,
                'reasons': confirmations,
                'timestamp': data.index[-1],
                'ema_params': (fast_ema, slow_ema)
            }
        
        return None
    
    def _generate_defensive_signal(self, symbol: str, data: pd.DataFrame, 
                                  timeframe: str, rvol: pd.Series) -> Optional[Dict]:
        """
        Defensive mode: ML-heavy with strict filters
        """
        
        # Start with ML signal as primary
        ml_signal = self._get_ml_signal(symbol, timeframe)
        if not ml_signal or ml_signal['confidence'] < 0.5:
            return None
        
        confirmations = ['ml_signal']
        
        # Additional strict confirmations
        
        # 1. Volatility filter
        returns = data['close'].pct_change()
        recent_volatility = returns.tail(20).std()
        if recent_volatility < 0.03:
            confirmations.append('low_volatility')
        
        # 2. Volume spike required even in defensive
        if rvol.iloc[-1] > 1.3:  # Lower threshold for defensive
            confirmations.append(f'rvol_{rvol.iloc[-1]:.1f}')
        
        # 3. Trend alignment
        sma_50 = data['close'].rolling(50).mean()
        sma_200 = data['close'].rolling(200).mean()
        if data['close'].iloc[-1] > sma_50.iloc[-1] > sma_200.iloc[-1]:
            confirmations.append('trend_aligned')
        
        # 4. RSI not overbought
        rsi = self._calculate_rsi(data['close'])
        if 30 < rsi < 70:
            confirmations.append('rsi_neutral')
        
        # Need at least 3 confirmations
        if len(confirmations) >= self.mode_params[TradingMode.DEFENSIVE]['min_confirmations']:
            # Conservative strength calculation
            strength = ml_signal['confidence'] * 0.6 + (len(confirmations) / 5.0) * 0.4
            
            # Get volatility adjustment (more conservative in defensive)
            vol_adjustment = self._get_volatility_adjustment(symbol, data) * 0.8 if self.mode_params[TradingMode.DEFENSIVE]['volatility_adjustment'] else 1.0
            
            return {
                'symbol': symbol,
                'action': 'BUY',
                'strength': min(strength * 0.8, 1.0),
                'mode': 'DEFENSIVE',
                'entry_price': data['close'].iloc[-1],
                'atr': self.atr_cache.get(symbol, data['close'].iloc[-1] * 0.02),
                'rvol': rvol.iloc[-1],
                'volatility_adjustment': vol_adjustment,
                'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.DEFENSIVE]['stop_loss']),
                'position_size': self.mode_params[TradingMode.DEFENSIVE]['position_size'] * vol_adjustment,
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
                if len(self.positions) >= 2:  # Max 2 positions in defensive
                    break
            
            # Execute trade
            shares = int(position_value / signal['entry_price'])
            cost = shares * signal['entry_price'] * 1.002  # Including commission
            
            if cost <= available_capital:
                # Calculate profit targets based on ATR
                atr = signal['atr']
                mode_params = self.mode_params[self.current_mode]
                
                profit_target_1 = signal['entry_price'] + (atr * mode_params['profit_target_atr_multiplier'])
                trailing_stop_distance = atr * mode_params['trailing_stop_atr_multiplier']
                
                trade = {
                    'symbol': signal['symbol'],
                    'action': 'BUY',
                    'shares': shares,
                    'entry_price': signal['entry_price'],
                    'cost': cost,
                    'stop_loss': signal['stop_loss'],
                    'profit_target_1': profit_target_1,
                    'trailing_stop_distance': trailing_stop_distance,
                    'current_stop': signal['stop_loss'],  # Initial stop
                    'mode': signal['mode'],
                    'timestamp': signal['timestamp'],
                    'reasons': signal['reasons'],
                    'atr': atr,
                    'rvol': signal.get('rvol', 1.0),
                    'partial_exit_done': False  # Track partial exit
                }
                
                # Update positions
                self.positions[signal['symbol']] = trade
                self.current_capital -= cost
                executed_trades.append(trade)
                
                logger.info(f"Executed {signal['mode']} trade: {signal['symbol']} "
                          f"@ {signal['entry_price']:.2f} ({shares} shares) "
                          f"Target: {profit_target_1:.2f}, Trail: {trailing_stop_distance:.2f}")
        
        return executed_trades
    
    def manage_positions(self, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Enhanced position management with:
        - ATR-based trailing stops
        - Partial profit taking
        - Mode-specific exit rules
        """
        
        closed_trades = []
        mode_params = self.mode_params[self.current_mode]
        
        for symbol, position in list(self.positions.items()):
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]['close'].iloc[-1]
            
            # Calculate P&L
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            shares_to_exit = position['shares']
            
            # 1. Check stop loss (including trailing)
            if current_price <= position['current_stop']:
                should_exit = True
                exit_reason = "stop_loss"
            
            # 2. Partial profit taking at first target
            elif current_price >= position['profit_target_1'] and not position['partial_exit_done']:
                # Take partial profits
                partial_percent = mode_params['partial_exit_percent']
                shares_to_exit = int(position['shares'] * partial_percent)
                
                if shares_to_exit > 0:
                    exit_value = shares_to_exit * current_price * 0.998
                    self.current_capital += exit_value
                    
                    partial_pnl = exit_value - (position['cost'] * partial_percent)
                    
                    closed_trade = {
                        'symbol': symbol,
                        'action': 'SELL_PARTIAL',
                        'shares': shares_to_exit,
                        'entry_price': position['entry_price'],
                        'exit_price': current_price,
                        'pnl': partial_pnl,
                        'pnl_pct': pnl_pct,
                        'exit_reason': 'partial_profit_target',
                        'mode': position.get('mode', 'UNKNOWN'),
                        'timestamp': current_data[symbol].index[-1]
                    }
                    
                    closed_trades.append(closed_trade)
                    
                    # Update position
                    position['shares'] -= shares_to_exit
                    position['cost'] *= (1 - partial_percent)
                    position['partial_exit_done'] = True
                    
                    # Move stop to breakeven
                    position['current_stop'] = position['entry_price']
                    
                    logger.info(f"Partial exit: {symbol} @ {current_price:.2f} "
                              f"({shares_to_exit} shares, PnL: {pnl_pct:.1%})")
                    
                    continue  # Don't fully exit
            
            # 3. Update trailing stop if in profit
            elif pnl_pct > 0:
                # Calculate new trailing stop
                new_stop = current_price - position['trailing_stop_distance']
                
                # Only move stop up, never down
                if new_stop > position['current_stop']:
                    position['current_stop'] = new_stop
                    logger.debug(f"{symbol} trailing stop updated to {new_stop:.2f}")
            
            # 4. Mode-specific exits
            if self.current_mode == TradingMode.AGGRESSIVE:
                # Let profits run, but tighten stop after big gains
                if pnl_pct > 0.15:  # 15% profit
                    position['trailing_stop_distance'] *= 0.7  # Tighten by 30%
                
            elif self.current_mode == TradingMode.DEFENSIVE:
                # Take remaining profits early
                if pnl_pct > 0.05 and position['partial_exit_done']:  # 5% after partial
                    should_exit = True
                    exit_reason = "defensive_profit_target"
                # Exit if mode changed from entry
                elif position.get('mode') != 'DEFENSIVE':
                    should_exit = True
                    exit_reason = "mode_change"
            
            else:  # BALANCED
                # Standard management
                if pnl_pct > 0.12 and position['partial_exit_done']:  # 12% after partial
                    # Very tight stop
                    position['trailing_stop_distance'] *= 0.5
            
            # Execute full exit if needed
            if should_exit:
                exit_value = position['shares'] * current_price * 0.998
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
                    'timestamp': current_data[symbol].index[-1],
                    'partial_exit_done': position['partial_exit_done']
                }
                
                closed_trades.append(closed_trade)
                del self.positions[symbol]
                
                logger.info(f"Closed position: {symbol} @ {current_price:.2f} "
                          f"(PnL: {pnl_pct:.1%}, Reason: {exit_reason})")
        
        return closed_trades
    
    def optimize_ema_parameters(self, symbol: str, data: pd.DataFrame) -> Tuple[int, int]:
        """
        Optimize EMA parameters for a specific symbol if not cached
        """
        if symbol not in self.ema_params_cache or symbol == 'DEFAULT':
            logger.info(f"Optimizing EMA parameters for {symbol}")
            
            # Use the optimizer - grid_search_optimization expects start_date and end_date
            result = self.ema_optimizer.grid_search_optimization(
                symbol,
                start_date=None,  # Will use all available data
                end_date=None
            )
            
            if result and result['sharpe_ratio'] > 0:
                optimal_fast = result['fast_ema']
                optimal_slow = result['slow_ema']
                self.ema_params_cache[symbol] = (optimal_fast, optimal_slow)
                self._save_optimal_ema_params()
                logger.info(f"Optimal EMA for {symbol}: ({optimal_fast}, {optimal_slow})")
                return optimal_fast, optimal_slow
        
        return self.ema_params_cache.get(symbol, self.ema_params_cache.get('DEFAULT', (12, 26)))
    
    # Helper methods (same as original)
    
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
        return 0.02  # Default 2%
    
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
        """Get current system status with enhanced metrics"""
        
        # Calculate total position value including partial exits
        total_position_value = 0
        for symbol, position in self.positions.items():
            # Assume current price is available
            current_price = position.get('current_price', position['entry_price'])
            total_position_value += position['shares'] * current_price
        
        # Performance metrics
        total_value = self.current_capital + total_position_value
        total_return = (total_value - self.initial_capital) / self.initial_capital
        
        return {
            'current_mode': self.current_mode.value,
            'mode_params': self.mode_params[self.current_mode],
            'positions': len(self.positions),
            'current_capital': self.current_capital,
            'total_value': total_value,
            'total_return': total_return,
            'monthly_return': self._calculate_period_return(30),
            'drawdown': self._calculate_current_drawdown(),
            'active_positions': list(self.positions.keys()),
            'last_mode_change': self.mode_history[-1] if self.mode_history else None,
            'optimal_ema_params': dict(list(self.ema_params_cache.items())[:10]),  # Show first 10
            'partial_exits_done': sum(1 for p in self.positions.values() if p.get('partial_exit_done', False))
        }


def main():
    """Test the Enhanced TriMode Orchestrator"""
    print("="*60)
    print("ENHANCED TRIMODE ORCHESTRATOR TEST")
    print("="*60)
    
    orchestrator = EnhancedTriModeOrchestrator(initial_capital=100000)
    
    # Test with a few symbols
    test_symbols = ['THYAO', 'GARAN', 'AKBNK']
    
    print("\nOptimizing EMA parameters for test symbols...")
    for symbol in test_symbols:
        data = orchestrator.csv_manager.load_raw_data(symbol, '1d')
        if data is not None and len(data) > 100:
            fast, slow = orchestrator.optimize_ema_parameters(symbol, data)
            print(f"{symbol}: Fast={fast}, Slow={slow}")
    
    print("\nTesting signal generation...")
    signals = orchestrator.generate_signals(test_symbols)
    
    print(f"\nGenerated {len(signals)} signals:")
    for signal in signals:
        print(f"  {signal['symbol']}: {signal['action']} "
              f"(strength: {signal['strength']:.2f}, mode: {signal['mode']}, "
              f"RVOL: {signal.get('rvol', 0):.1f})")
    
    # Test status report
    print("\nSystem Status:")
    status = orchestrator.get_status_report()
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()