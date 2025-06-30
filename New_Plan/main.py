"""
Hybrid Trading System - Main Entry Point
Paper/Live trading orchestrator
"""

import asyncio
from datetime import datetime, time
import json
from pathlib import Path
import sys
import argparse
from loguru import logger
import signal as sig
import pandas as pd
from typing import Dict, List

# Add parent path
sys.path.append(str(Path(__file__).parent))

from core.csv_data_manager import CSVDataManager
from core.data_collector import UnifiedDataCollector
from core.signal_generator import SignalGenerator
from core.portfolio_manager import PortfolioManager
from execution.algolab_connector import AlgolabConnector


class TradingSystem:
    """Main trading system orchestrator"""
    
    def __init__(self, config_path: str = 'config.json', mode: str = 'paper'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.mode = mode
        self.running = False
        
        # Initialize components
        self.csv_manager = CSVDataManager()
        self.data_collector = UnifiedDataCollector(self.config)
        self.signal_generator = SignalGenerator(self.config)
        self.portfolio_manager = PortfolioManager(self.config)
        
        # Algolab connector only for live mode
        self.algolab = None
        if mode == 'live':
            self.algolab = AlgolabConnector(self.config)
        
        # Paper trading state
        self.paper_prices = {}
        self.paper_orders = []
        
        logger.info(f"Trading system initialized in {mode.upper()} mode")
    
    def is_market_open(self) -> bool:
        """Check if BIST market is open"""
        now = datetime.now()
        
        # Market closed on weekends
        if now.weekday() >= 5:
            return False
        
        # BIST market hours: 09:10 - 18:00
        market_open = time(9, 10)
        market_close = time(18, 0)
        current_time = now.time()
        
        return market_open <= current_time <= market_close
    
    async def update_market_data(self, symbols: List[str]) -> Dict:
        """Update market data for all symbols"""
        market_data = {}
        
        for symbol in symbols:
            try:
                # Get multi-timeframe data
                data = await self.data_collector.collect_multi_timeframe_data(symbol)
                if data:
                    market_data[symbol] = data
                    
                    # Update paper prices
                    if self.mode == 'paper' and '1h' in data:
                        self.paper_prices[symbol] = data['1h']['close'].iloc[-1]
                        
            except Exception as e:
                logger.error(f"Error updating data for {symbol}: {e}")
        
        return market_data
    
    async def execute_signals(self, signals: List[Dict]):
        """Execute trading signals"""
        for signal in signals:
            try:
                # Process signal through portfolio manager
                order = await self.portfolio_manager.process_signal(signal)
                
                if order:
                    if self.mode == 'live' and self.algolab:
                        # Live trading
                        result = await self.algolab.place_order(order)
                        logger.info(f"Live order placed: {order['symbol']} {order['side']} - Result: {result}")
                    else:
                        # Paper trading
                        self._execute_paper_order(order)
                        
            except Exception as e:
                logger.error(f"Error executing signal for {signal['symbol']}: {e}")
    
    def _execute_paper_order(self, order: Dict):
        """Execute order in paper trading mode"""
        # Simulate order execution
        execution_price = self.paper_prices.get(order['symbol'], order['price'])
        
        # Add some slippage
        if order['side'] == 'buy':
            execution_price *= 1.001  # 0.1% slippage
        else:
            execution_price *= 0.999
        
        # Execute in portfolio manager
        self.portfolio_manager.execute_order(order, execution_price)
        
        # Record paper trade
        self.paper_orders.append({
            'timestamp': datetime.now(),
            'symbol': order['symbol'],
            'side': order['side'],
            'quantity': order['quantity'],
            'order_price': order['price'],
            'execution_price': execution_price,
            'status': 'filled'
        })
        
        logger.info(f"Paper order executed: {order['symbol']} {order['side']} "
                   f"{order['quantity']} @ {execution_price:.2f}")
    
    async def update_positions(self):
        """Update all open positions"""
        positions = self.portfolio_manager.positions.copy()
        
        for symbol, position in positions.items():
            try:
                if self.mode == 'live' and self.algolab:
                    # Get live price
                    current_price = await self.algolab.get_current_price(symbol)
                else:
                    # Use paper price
                    current_price = self.paper_prices.get(symbol, position['current_price'])
                
                # Update position
                result = self.portfolio_manager.update_position(symbol, current_price)
                
                # Check if position was closed
                if result and result.get('status') == 'closed':
                    logger.info(f"Position closed: {symbol} - P&L: {result['pnl']:.2f} "
                              f"({result['pnl_percent']:.2f}%) - Reason: {result['close_reason']}")
                    
            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")
    
    async def trading_loop(self):
        """Main trading loop"""
        self.running = True
        cycle_count = 0
        
        logger.info("Starting trading loop...")
        
        while self.running:
            try:
                cycle_count += 1
                
                # Check market hours
                if not self.is_market_open() and self.mode == 'live':
                    logger.info("Market closed, waiting...")
                    await asyncio.sleep(300)  # Wait 5 minutes
                    continue
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Trading cycle #{cycle_count} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                logger.info(f"{'='*60}")
                
                # Get current portfolio status
                status = self.portfolio_manager.get_portfolio_status()
                logger.info(f"Portfolio: {status['total_equity']:,.0f} TRY "
                           f"({status['total_return']:+.1f}%) - "
                           f"Positions: {status['open_positions']}")
                
                # Update market data
                symbols = self.config['symbols']
                market_data = await self.update_market_data(symbols)
                logger.info(f"Updated data for {len(market_data)} symbols")
                
                # Generate signals
                if market_data:
                    signals = await self.signal_generator.generate_signals(list(market_data.keys()))
                    logger.info(f"Generated {len(signals)} signals")
                    
                    # Display signals
                    for signal in signals:
                        logger.info(f"Signal: {signal['symbol']} {signal['direction'].upper()} "
                                   f"@ {signal['entry_price']:.2f} - "
                                   f"Confidence: {signal['confidence']:.1%}")
                    
                    # Execute signals
                    if signals:
                        await self.execute_signals(signals)
                
                # Update existing positions
                await self.update_positions()
                
                # Risk check
                risk_status = self.portfolio_manager.risk_check()
                if not risk_status['within_limit']:
                    logger.warning(f"Risk limit exceeded: {risk_status['portfolio_risk']:.1f}%")
                
                # Display current positions
                if self.portfolio_manager.positions:
                    logger.info("\nOpen Positions:")
                    for symbol, pos in self.portfolio_manager.positions.items():
                        logger.info(f"  {symbol}: {pos['quantity']} @ {pos['entry_price']:.2f} - "
                                   f"P&L: {pos['pnl']:+.2f} ({pos['pnl_percent']:+.1f}%)")
                
                # Wait for next cycle
                wait_time = 60 if self.mode == 'paper' else self.config.get('execution', {}).get('update_interval', 60)
                logger.info(f"\nWaiting {wait_time} seconds for next cycle...")
                await asyncio.sleep(wait_time)
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                break
            except Exception as e:
                logger.error(f"Trading loop error: {e}")
                await asyncio.sleep(60)
    
    def stop(self):
        """Stop trading system"""
        logger.info("Stopping trading system...")
        self.running = False
        
        # Save final state
        if self.mode == 'paper' and self.paper_orders:
            # Save paper trading results
            df = pd.DataFrame(self.paper_orders)
            filename = f"paper_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)
            logger.info(f"Paper trades saved to {filename}")
        
        # Display final results
        status = self.portfolio_manager.get_portfolio_status()
        logger.info("\n" + "="*60)
        logger.info("FINAL RESULTS")
        logger.info("="*60)
        logger.info(f"Final Equity: {status['total_equity']:,.0f} TRY")
        logger.info(f"Total Return: {status['total_return']:+.2f}%")
        logger.info(f"Total Trades: {self.portfolio_manager.performance.get('total_trades', 0)}")
        logger.info(f"Win Rate: {status['win_rate']:.1f}%")
        logger.info(f"Max Drawdown: {status['max_drawdown']:.1f}%")


def signal_handler(signum, frame):
    """Handle interrupt signals"""
    global trading_system
    if trading_system:
        trading_system.stop()
    sys.exit(0)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Hybrid Trading System')
    parser.add_argument('--mode', choices=['paper', 'live', 'backtest'], 
                       default='paper', help='Trading mode')
    parser.add_argument('--config', default='config.json', help='Config file path')
    parser.add_argument('--symbols', nargs='+', help='Override symbols from config')
    parser.add_argument('--confirm', action='store_true', help='Confirm live trading')
    
    args = parser.parse_args()
    
    # Safety check for live trading
    if args.mode == 'live' and not args.confirm:
        response = input("⚠️  WARNING: You are about to start LIVE TRADING. Type 'YES' to confirm: ")
        if response != 'YES':
            logger.info("Live trading cancelled")
            return
    
    # Create trading system
    global trading_system
    trading_system = TradingSystem(args.config, args.mode)
    
    # Override symbols if provided
    if args.symbols:
        trading_system.config['symbols'] = args.symbols
    
    # Setup signal handlers
    sig.signal(sig.SIGINT, signal_handler)
    sig.signal(sig.SIGTERM, signal_handler)
    
    # Display startup info
    logger.info("\n" + "="*60)
    logger.info("HYBRID TRADING SYSTEM")
    logger.info("="*60)
    logger.info(f"Mode: {args.mode.upper()}")
    logger.info(f"Symbols: {', '.join(trading_system.config['symbols'][:5])}... "
               f"({len(trading_system.config['symbols'])} total)")
    logger.info(f"Initial Capital: {trading_system.config['portfolio']['initial_capital']:,.0f} TRY")
    logger.info(f"Max Risk per Trade: {trading_system.config['risk']['max_risk_per_trade']*100:.0f}%")
    logger.info("="*60 + "\n")
    
    # Run trading loop
    try:
        asyncio.run(trading_system.trading_loop())
    except KeyboardInterrupt:
        pass
    finally:
        trading_system.stop()


if __name__ == "__main__":
    main()