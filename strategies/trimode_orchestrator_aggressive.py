#!/usr/bin/env python3
"""
Aggressive TriMode Orchestrator - More aggressive settings for 10% monthly target
Key changes:
- Lower thresholds for mode switching
- Wider stops and smaller position sizes for better risk management
- Lower RVOL requirements
- More aggressive mode conditions
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


class AggressiveTriModeOrchestrator:
    """
    More aggressive orchestrator targeting 10% monthly returns
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
        self.partial_exits = {}
        
        # ADJUSTED MODE PARAMETERS FOR MORE AGGRESSIVE TRADING
        self.mode_params = {
            TradingMode.AGGRESSIVE: {
                'position_size': 0.15,  # Reduced from 0.20 for better risk
                'max_positions': 10,    # Increased from 7
                'stop_loss': 0.12,      # Wider stop from 0.10
                'trailing_stop_atr_multiplier': 4.0,  # Wider from 3.0
                'profit_target_atr_multiplier': 3.0,  # Lower from 4.0 for quicker profits
                'partial_exit_percent': 0.4,  # Take less (40%) to let more run
                'min_confirmations': 0,
                'target_monthly_return': 0.15,
                'filters': ['volume'],  # Removed RVOL requirement
                'ml_weight': 0.1,
                'volatility_adjustment': False,  # Disabled for more consistent sizing
                'min_rvol': 1.0  # No RVOL filter
            },
            TradingMode.BALANCED: {
                'position_size': 0.10,
                'max_positions': 8,     # Increased from 5
                'stop_loss': 0.08,      # Wider from 0.05
                'trailing_stop_atr_multiplier': 3.5,  # Wider from 2.5
                'profit_target_atr_multiplier': 2.5,  # Lower from 3.0
                'partial_exit_percent': 0.5,
                'min_confirmations': 1,  # Reduced from 2
                'target_monthly_return': 0.10,
                'filters': ['volume', 'adx'],  # Removed ML and RVOL
                'ml_weight': 0.2,
                'volatility_adjustment': False,
                'min_rvol': 1.2  # Lower RVOL requirement
            },
            TradingMode.DEFENSIVE: {
                'position_size': 0.05,
                'max_positions': 5,     # Increased from 3
                'stop_loss': 0.05,      # Wider from 0.03
                'trailing_stop_atr_multiplier': 3.0,  # Wider from 2.0
                'profit_target_atr_multiplier': 2.0,
                'partial_exit_percent': 0.6,
                'min_confirmations': 2,  # Reduced from 3
                'target_monthly_return': 0.05,
                'filters': ['volume', 'adx', 'ml'],  # Removed strict filters
                'ml_weight': 0.4,  # Reduced from 0.6
                'volatility_adjustment': False,
                'min_rvol': 1.0  # No RVOL requirement in defensive
            }
        }
        
        # Optimal EMA parameters cache
        self.ema_params_cache = self._load_optimal_ema_params()
        
        # ATR cache for efficiency
        self.atr_cache = {}
        
        # Load ML models
        self._load_ml_models()
    
    def _load_optimal_ema_params(self) -> Dict[str, Tuple[int, int]]:
        """Load pre-optimized EMA parameters from file or calculate"""
        # Use reasonable parameters instead of over-optimized ones
        params_file = Path('backtest/reasonable_ema_params.json')
        
        if params_file.exists():
            with open(params_file, 'r') as f:
                params = json.load(f)
                return {k: tuple(v) for k, v in params.items()}
        else:
            # Fallback to original optimal params
            params_file = Path('backtest/optimal_ema_params.json')
            if params_file.exists():
                with open(params_file, 'r') as f:
                    params = json.load(f)
                    # Filter out unreasonable slow EMAs (> 50)
                    filtered = {}
                    for k, v in params.items():
                        if v[1] <= 50:  # Keep only reasonable slow EMAs
                            filtered[k] = tuple(v)
                        else:
                            # Use more reasonable values
                            filtered[k] = (min(v[0], 15), min(v[1], 35))
                    return filtered
            else:
                logger.info("No optimal EMA parameters found, using defaults")
                return {
                    'THYAO': (9, 21),
                    'GARAN': (10, 25),
                    'AKBNK': (12, 26),
                    'DEFAULT': (10, 25)
                }
    
    def _save_optimal_ema_params(self):
        """Save optimal EMA parameters to file"""
        params_file = Path('backtest/optimal_ema_params.json')
        with open(params_file, 'w') as f:
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
        # Disabled by default in mode params
        return 1.0
    
    def determine_mode(self, market_data: Dict) -> TradingMode:
        """
        MORE AGGRESSIVE mode determination
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
        
        # Start with AGGRESSIVE as default for testing
        new_mode = TradingMode.AGGRESSIVE
        
        # Only go DEFENSIVE in extreme conditions
        if (current_drawdown < -0.30 or  # Very deep drawdown
            losing_streak >= 10 or  # Many consecutive losses
            market_volatility > 0.08):  # Extreme volatility
            new_mode = TradingMode.DEFENSIVE
            
        # BALANCED for moderate conditions
        elif (current_drawdown < -0.15 or  # Moderate drawdown
              losing_streak >= 5 or  # Some losses
              market_volatility > 0.04):  # Higher volatility
            new_mode = TradingMode.BALANCED
        
        # Remove mode transition smoothing to allow quicker adaptation
        
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
        """Generate trading signals with more lenient filters"""
        
        mode_params = self.mode_params[self.current_mode]
        signals = []
        
        logger.info(f"Generating signals in {self.current_mode.value} mode")
        
        for symbol in symbols:
            try:
                # Get market data
                data = self.csv_manager.load_raw_data(symbol, timeframe)
                if data is None or len(data) < 100:  # Reduced from 200
                    continue
                
                # Calculate ATR for this symbol
                atr = self._calculate_atr(data)
                self.atr_cache[symbol] = atr.iloc[-1]
                
                # Calculate RVOL
                rvol = self._calculate_relative_volume(data)
                
                # Skip RVOL filter in aggressive mode for more signals
                if self.current_mode != TradingMode.AGGRESSIVE:
                    min_rvol = mode_params.get('min_rvol', 1.0)
                    if rvol.iloc[-1] < min_rvol:
                        continue
                
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
        Aggressive mode: Simple EMA cross, no strict filters
        """
        
        # Get optimal EMA parameters
        if symbol in self.ema_params_cache:
            fast_ema, slow_ema = self.ema_params_cache[symbol]
        else:
            fast_ema, slow_ema = self.ema_params_cache.get('DEFAULT', (10, 25))
        
        # Calculate EMAs
        ema_fast = data['close'].ewm(span=fast_ema, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_ema, adjust=False).mean()
        
        # Simple crossover
        current_signal = 1 if ema_fast.iloc[-1] > ema_slow.iloc[-1] else -1
        prev_signal = 1 if ema_fast.iloc[-2] > ema_slow.iloc[-2] else -1
        
        # More flexible conditions for aggressive mode
        # 1. Fresh crossover
        # 2. Strong trend (fast EMA significantly above slow)
        # 3. Momentum (price above both EMAs)
        
        fresh_crossover = current_signal == 1 and current_signal != prev_signal
        strong_trend = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1] > 0.005  # 0.5% diff
        price_momentum = data['close'].iloc[-1] > ema_fast.iloc[-1]
        
        if current_signal == 1 and (fresh_crossover or (strong_trend and price_momentum)):
            # Buy signal
            
            # Very minimal volume check - just ensure there's some volume
            volume_confirmed = data['volume'].iloc[-1] > 0
            
            if volume_confirmed:
                # Calculate signal strength
                ema_diff = (ema_fast.iloc[-1] - ema_slow.iloc[-1]) / ema_slow.iloc[-1]
                strength = min(abs(ema_diff) * 20, 1.0)  # More sensitive
                
                return {
                    'symbol': symbol,
                    'action': 'BUY',
                    'strength': strength,
                    'mode': 'AGGRESSIVE',
                    'entry_price': data['close'].iloc[-1],
                    'atr': self.atr_cache.get(symbol, data['close'].iloc[-1] * 0.02),
                    'rvol': rvol.iloc[-1],
                    'volatility_adjustment': 1.0,
                    'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.AGGRESSIVE]['stop_loss']),
                    'position_size': self.mode_params[TradingMode.AGGRESSIVE]['position_size'],
                    'reasons': ['ema_cross', f'rvol_{rvol.iloc[-1]:.1f}'],
                    'timestamp': data.index[-1],
                    'ema_params': (fast_ema, slow_ema)
                }
        
        return None
    
    def _generate_balanced_signal(self, symbol: str, data: pd.DataFrame, 
                                 timeframe: str, rvol: pd.Series) -> Optional[Dict]:
        """
        Balanced mode: EMA + minimal confirmations
        """
        
        confirmations = []
        
        # 1. EMA Signal with optimal parameters
        if symbol in self.ema_params_cache:
            fast_ema, slow_ema = self.ema_params_cache[symbol]
        else:
            fast_ema, slow_ema = 15, 35  # Balanced defaults
        
        ema_fast = data['close'].ewm(span=fast_ema, adjust=False).mean()
        ema_slow = data['close'].ewm(span=slow_ema, adjust=False).mean()
        
        # More lenient EMA conditions
        if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
            confirmations.append('ema_bullish')
        
        # 2. Volume check
        if data['volume'].iloc[-1] > data['volume'].rolling(20).mean().iloc[-1]:
            confirmations.append('volume_ok')
        
        # 3. ADX confirmation (optional)
        if 'adx' in self.mode_params[TradingMode.BALANCED]['filters']:
            adx = self._calculate_adx(data)
            if adx > 20:  # Lowered from 25
                confirmations.append('trend_ok')
        
        # Need only 1 confirmation
        if len(confirmations) >= self.mode_params[TradingMode.BALANCED]['min_confirmations']:
            # Calculate signal strength
            strength = len(confirmations) / 3.0
            
            return {
                'symbol': symbol,
                'action': 'BUY',
                'strength': min(strength, 1.0),
                'mode': 'BALANCED',
                'entry_price': data['close'].iloc[-1],
                'atr': self.atr_cache.get(symbol, data['close'].iloc[-1] * 0.02),
                'rvol': rvol.iloc[-1],
                'volatility_adjustment': 1.0,
                'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.BALANCED]['stop_loss']),
                'position_size': self.mode_params[TradingMode.BALANCED]['position_size'],
                'reasons': confirmations,
                'timestamp': data.index[-1],
                'ema_params': (fast_ema, slow_ema)
            }
        
        return None
    
    def _generate_defensive_signal(self, symbol: str, data: pd.DataFrame, 
                                  timeframe: str, rvol: pd.Series) -> Optional[Dict]:
        """
        Defensive mode: More lenient than before
        """
        
        # Start with ML signal as preference (not requirement)
        ml_signal = self._get_ml_signal(symbol, timeframe)
        
        confirmations = []
        
        # 1. ML signal (optional)
        if ml_signal and ml_signal['confidence'] > 0.3:  # Lowered from 0.5
            confirmations.append('ml_signal')
        
        # 2. Basic trend check
        sma_20 = data['close'].rolling(20).mean()
        if data['close'].iloc[-1] > sma_20.iloc[-1]:
            confirmations.append('above_sma20')
        
        # 3. Not oversold
        rsi = self._calculate_rsi(data['close'])
        if rsi > 30:  # Just not oversold
            confirmations.append('rsi_ok')
        
        # 4. Volume present
        if data['volume'].iloc[-1] > 0:
            confirmations.append('volume_present')
        
        # Need only 2 confirmations
        if len(confirmations) >= self.mode_params[TradingMode.DEFENSIVE]['min_confirmations']:
            # Calculate strength
            strength = len(confirmations) / 4.0
            if ml_signal:
                strength = strength * 0.7 + ml_signal['confidence'] * 0.3
            
            return {
                'symbol': symbol,
                'action': 'BUY',
                'strength': min(strength * 0.9, 1.0),
                'mode': 'DEFENSIVE',
                'entry_price': data['close'].iloc[-1],
                'atr': self.atr_cache.get(symbol, data['close'].iloc[-1] * 0.02),
                'rvol': rvol.iloc[-1],
                'volatility_adjustment': 1.0,
                'stop_loss': data['close'].iloc[-1] * (1 - self.mode_params[TradingMode.DEFENSIVE]['stop_loss']),
                'position_size': self.mode_params[TradingMode.DEFENSIVE]['position_size'],
                'reasons': confirmations,
                'timestamp': data.index[-1]
            }
        
        return None
    
    def execute_trades(self, signals: List[Dict]) -> List[Dict]:
        """Execute trades based on signals"""
        
        executed_trades = []
        mode_params = self.mode_params[self.current_mode]
        
        for signal in signals:
            if signal['symbol'] in self.positions:
                continue
            
            # Calculate actual position size
            available_capital = self.current_capital - sum(p['cost'] for p in self.positions.values())
            position_value = available_capital * signal['position_size']
            
            if position_value < 1000:
                continue
            
            # Execute trade
            shares = int(position_value / signal['entry_price'])
            cost = shares * signal['entry_price'] * 1.002
            
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
                    'current_stop': signal['stop_loss'],
                    'mode': signal['mode'],
                    'timestamp': signal['timestamp'],
                    'reasons': signal['reasons'],
                    'atr': atr,
                    'rvol': signal.get('rvol', 1.0),
                    'partial_exit_done': False
                }
                
                self.positions[signal['symbol']] = trade
                self.current_capital -= cost
                executed_trades.append(trade)
                
                logger.info(f"Executed {signal['mode']} trade: {signal['symbol']} "
                          f"@ {signal['entry_price']:.2f} ({shares} shares)")
        
        return executed_trades
    
    def manage_positions(self, current_data: Dict[str, pd.DataFrame]) -> List[Dict]:
        """
        Position management with wider stops
        """
        
        closed_trades = []
        mode_params = self.mode_params[self.current_mode]
        
        for symbol, position in list(self.positions.items()):
            if symbol not in current_data:
                continue
                
            current_price = current_data[symbol]['close'].iloc[-1]
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            
            should_exit = False
            exit_reason = ""
            shares_to_exit = position['shares']
            
            # 1. Check stop loss (wider stops)
            if current_price <= position['current_stop']:
                should_exit = True
                exit_reason = "stop_loss"
            
            # 2. Partial profit taking at first target
            elif current_price >= position['profit_target_1'] and not position['partial_exit_done']:
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
                    
                    # Move stop to breakeven minus small buffer
                    position['current_stop'] = position['entry_price'] * 0.98
                    
                    logger.info(f"Partial exit: {symbol} @ {current_price:.2f} "
                              f"({shares_to_exit} shares, PnL: {pnl_pct:.1%})")
                    
                    continue
            
            # 3. Update trailing stop if in profit
            elif pnl_pct > 0.02:  # Only trail after 2% profit
                # Calculate new trailing stop
                new_stop = current_price - position['trailing_stop_distance']
                
                # Only move stop up
                if new_stop > position['current_stop']:
                    position['current_stop'] = new_stop
            
            # 4. Mode-specific exits
            if self.current_mode == TradingMode.AGGRESSIVE:
                # Let profits run
                if pnl_pct > 0.20:  # 20% profit
                    position['trailing_stop_distance'] *= 0.8  # Tighten slightly
                
            elif self.current_mode == TradingMode.DEFENSIVE:
                # Take remaining profits at reasonable levels
                if pnl_pct > 0.08 and position['partial_exit_done']:
                    should_exit = True
                    exit_reason = "defensive_profit_target"
            
            else:  # BALANCED
                # Standard management
                if pnl_pct > 0.15 and position['partial_exit_done']:
                    position['trailing_stop_distance'] *= 0.7
            
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
        return 0.02
    
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
        """Simple ADX calculation"""
        high = data['high']
        low = data['low']
        close = data['close']
        
        high_low = high - low
        high_close = abs(high - close.shift())
        low_close = abs(low - close.shift())
        
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        
        # Simplified ADX
        price_change = close.pct_change(period)
        adx = abs(price_change) / (atr / close) * 100
        adx = adx.rolling(period).mean()
        
        return adx.iloc[-1] if not adx.empty else 25
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def optimize_ema_parameters(self, symbol: str, data: pd.DataFrame) -> Tuple[int, int]:
        """Optimize EMA parameters for a specific symbol"""
        if symbol not in self.ema_params_cache or symbol == 'DEFAULT':
            logger.info(f"Optimizing EMA parameters for {symbol}")
            
            result = self.ema_optimizer.grid_search_optimization(
                symbol,
                start_date=None,
                end_date=None
            )
            
            if result and result['sharpe_ratio'] > 0:
                optimal_fast = result['fast_ema']
                optimal_slow = result['slow_ema']
                self.ema_params_cache[symbol] = (optimal_fast, optimal_slow)
                self._save_optimal_ema_params()
                logger.info(f"Optimal EMA for {symbol}: ({optimal_fast}, {optimal_slow})")
                return optimal_fast, optimal_slow
        
        return self.ema_params_cache.get(symbol, self.ema_params_cache.get('DEFAULT', (10, 25)))
    
    def get_status_report(self) -> Dict:
        """Get current system status"""
        
        total_position_value = 0
        for symbol, position in self.positions.items():
            current_price = position.get('current_price', position['entry_price'])
            total_position_value += position['shares'] * current_price
        
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
            'last_mode_change': self.mode_history[-1] if self.mode_history else None
        }


def main():
    """Test the Aggressive TriMode Orchestrator"""
    print("="*60)
    print("AGGRESSIVE TRIMODE ORCHESTRATOR TEST")
    print("="*60)
    
    orchestrator = AggressiveTriModeOrchestrator(initial_capital=100000)
    
    # Get status
    status = orchestrator.get_status_report()
    print("\nSystem Status:")
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    main()