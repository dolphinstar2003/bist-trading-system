"""
Reinforcement Learning Trading Agent using DQN and PPO

This module implements deep reinforcement learning agents for portfolio
optimization and automated trading decisions.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import logging
from typing import Dict, List, Tuple, Optional, Union
import random
from collections import deque
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingEnvironment(gym.Env):
    """Custom trading environment for RL agents"""
    
    def __init__(self, df: pd.DataFrame, initial_balance: float = 100000,
                 commission: float = 0.001, max_positions: int = 10,
                 reward_scaling: float = 1e-4, window_size: int = 30):
        """
        Initialize trading environment
        
        Args:
            df: DataFrame with OHLCV and indicator data
            initial_balance: Starting capital
            commission: Trading commission rate
            max_positions: Maximum number of positions
            reward_scaling: Scaling factor for rewards
            window_size: Observation window size
        """
        super(TradingEnvironment, self).__init__()
        
        self.df = df.copy()
        self.initial_balance = initial_balance
        self.commission = commission
        self.max_positions = max_positions
        self.reward_scaling = reward_scaling
        self.window_size = window_size
        
        # Prepare features
        self._prepare_features()
        
        # Action space: 0=hold, 1=buy, 2=sell
        self.action_space = spaces.Discrete(3)
        
        # Observation space
        n_features = self.features.shape[1]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(window_size * n_features + 3,),  # +3 for portfolio state
            dtype=np.float32
        )
        
        # Episode variables
        self.current_step = 0
        self.done = False
        
        # Portfolio state
        self.balance = initial_balance
        self.positions = {}  # {symbol: {'quantity': int, 'avg_price': float}}
        self.portfolio_value = initial_balance
        self.trades = []
        
        # Performance tracking
        self.episode_returns = []
        self.episode_trades = 0
    
    def _prepare_features(self):
        """Prepare and normalize features"""
        feature_columns = []
        
        # Price features
        self.df['returns'] = self.df['Close'].pct_change()
        self.df['log_returns'] = np.log(self.df['Close'] / self.df['Close'].shift(1))
        self.df['high_low_ratio'] = self.df['High'] / self.df['Low']
        self.df['close_open_ratio'] = self.df['Close'] / self.df['Open']
        
        feature_columns.extend(['returns', 'log_returns', 'high_low_ratio', 'close_open_ratio'])
        
        # Volume features
        self.df['volume_ratio'] = self.df['Volume'] / self.df['Volume'].rolling(20).mean()
        feature_columns.append('volume_ratio')
        
        # Technical indicators
        indicator_columns = [
            'rsi', 'macd', 'macd_signal', 'macd_hist',
            'atr', 'adx', 'plus_di', 'minus_di',
            'bb_upper', 'bb_middle', 'bb_lower'
        ]
        
        for col in indicator_columns:
            if col in self.df.columns:
                feature_columns.append(col)
                # Normalize indicators
                if col == 'rsi':
                    self.df[col] = (self.df[col] - 50) / 50
                else:
                    self.df[col] = (self.df[col] - self.df[col].rolling(50).mean()) / \
                                  (self.df[col].rolling(50).std() + 1e-8)
        
        # Fill NaN values
        self.df[feature_columns] = self.df[feature_columns].fillna(0)
        
        # Extract features
        self.features = self.df[feature_columns].values
        self.prices = self.df['Close'].values
    
    def reset(self, seed=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = self.window_size
        self.done = False
        
        # Reset portfolio
        self.balance = self.initial_balance
        self.positions = {}
        self.portfolio_value = self.initial_balance
        self.trades = []
        
        # Reset tracking
        self.episode_returns = [0]
        self.episode_trades = 0
        
        return self._get_observation(), {}
    
    def _get_observation(self):
        """Get current observation"""
        # Get window of features
        window_features = self.features[self.current_step - self.window_size:self.current_step]
        window_features = window_features.flatten()
        
        # Portfolio state
        position_value = sum(pos['quantity'] * self.prices[self.current_step] 
                           for pos in self.positions.values())
        
        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized cash
            position_value / self.initial_balance,  # Normalized position value
            len(self.positions) / self.max_positions  # Position utilization
        ])
        
        # Combine features
        observation = np.concatenate([window_features, portfolio_state])
        
        return observation.astype(np.float32)
    
    def step(self, action):
        """Execute action and return results"""
        prev_portfolio_value = self._calculate_portfolio_value()
        
        # Execute action
        if action == 1:  # Buy
            self._execute_buy()
        elif action == 2:  # Sell
            self._execute_sell()
        # action == 0 is hold, do nothing
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        new_portfolio_value = self._calculate_portfolio_value()
        step_return = (new_portfolio_value - prev_portfolio_value) / prev_portfolio_value
        
        # Reward shaping
        reward = step_return * self.reward_scaling
        
        # Add penalty for excessive trading
        if action != 0:
            reward -= self.commission * self.reward_scaling
        
        # Check if done
        self.done = (self.current_step >= len(self.prices) - 1) or \
                   (new_portfolio_value < self.initial_balance * 0.5)  # 50% drawdown
        
        # Track performance
        self.portfolio_value = new_portfolio_value
        self.episode_returns.append(step_return)
        
        # Get info
        info = {
            'portfolio_value': new_portfolio_value,
            'balance': self.balance,
            'positions': len(self.positions),
            'trades': self.episode_trades,
            'return': (new_portfolio_value - self.initial_balance) / self.initial_balance
        }
        
        return self._get_observation(), reward, self.done, False, info
    
    def _calculate_portfolio_value(self):
        """Calculate total portfolio value"""
        position_value = sum(pos['quantity'] * self.prices[self.current_step] 
                           for pos in self.positions.values())
        return self.balance + position_value
    
    def _execute_buy(self):
        """Execute buy order"""
        if len(self.positions) >= self.max_positions:
            return
        
        # Calculate position size
        position_size = self.balance / (self.max_positions - len(self.positions))
        position_size = min(position_size, self.balance * 0.95)  # Keep some cash
        
        if position_size < 1000:  # Minimum position size
            return
        
        # Execute trade
        price = self.prices[self.current_step]
        quantity = position_size / price
        cost = quantity * price * (1 + self.commission)
        
        if cost <= self.balance:
            self.balance -= cost
            
            # Update positions
            position_id = f"pos_{self.current_step}"
            self.positions[position_id] = {
                'quantity': quantity,
                'avg_price': price,
                'entry_step': self.current_step
            }
            
            # Record trade
            self.trades.append({
                'step': self.current_step,
                'action': 'buy',
                'price': price,
                'quantity': quantity,
                'cost': cost
            })
            
            self.episode_trades += 1
    
    def _execute_sell(self):
        """Execute sell order"""
        if not self.positions:
            return
        
        # Sell oldest position
        position_id = min(self.positions.keys(), 
                         key=lambda x: self.positions[x]['entry_step'])
        position = self.positions[position_id]
        
        # Execute trade
        price = self.prices[self.current_step]
        revenue = position['quantity'] * price * (1 - self.commission)
        
        self.balance += revenue
        
        # Calculate profit
        cost = position['quantity'] * position['avg_price']
        profit = revenue - cost
        
        # Record trade
        self.trades.append({
            'step': self.current_step,
            'action': 'sell',
            'price': price,
            'quantity': position['quantity'],
            'revenue': revenue,
            'profit': profit
        })
        
        # Remove position
        del self.positions[position_id]
        self.episode_trades += 1
    
    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"Step: {self.current_step}")
            print(f"Portfolio Value: ${self.portfolio_value:,.2f}")
            print(f"Balance: ${self.balance:,.2f}")
            print(f"Positions: {len(self.positions)}")
            print(f"Return: {(self.portfolio_value - self.initial_balance) / self.initial_balance:.2%}")
            print("-" * 50)


class DQNNetwork(nn.Module):
    """Deep Q-Network for trading"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_actions: int = 3):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc4 = nn.Linear(hidden_dim // 2, n_actions)
        
        # Dueling DQN architecture
        self.value_stream = nn.Linear(hidden_dim // 2, 1)
        self.advantage_stream = nn.Linear(hidden_dim // 2, n_actions)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        
        # Dueling streams
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        
        # Combine streams
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class PPONetwork(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, n_actions: int = 3):
        super(PPONetwork, self).__init__()
        
        # Shared layers
        self.shared_fc1 = nn.Linear(input_dim, hidden_dim)
        self.shared_fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head
        self.actor_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.actor_out = nn.Linear(hidden_dim // 2, n_actions)
        
        # Critic head
        self.critic_fc = nn.Linear(hidden_dim, hidden_dim // 2)
        self.critic_out = nn.Linear(hidden_dim // 2, 1)
        
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # Shared layers
        x = F.relu(self.shared_fc1(x))
        x = self.dropout(x)
        x = F.relu(self.shared_fc2(x))
        x = self.dropout(x)
        
        # Actor
        actor = F.relu(self.actor_fc(x))
        action_logits = self.actor_out(actor)
        
        # Critic
        critic = F.relu(self.critic_fc(x))
        value = self.critic_out(critic)
        
        return action_logits, value


class ReinforcementLearningTrader:
    """Main RL trading system"""
    
    def __init__(self, algorithm: str = 'ppo', learning_rate: float = 3e-4,
                 buffer_size: int = 100000, batch_size: int = 64,
                 gamma: float = 0.99, device: str = 'auto'):
        """
        Initialize RL Trader
        
        Args:
            algorithm: 'dqn', 'ppo', or 'a2c'
            learning_rate: Learning rate
            buffer_size: Replay buffer size (for DQN)
            batch_size: Batch size for training
            gamma: Discount factor
            device: Device to use ('auto', 'cpu', or 'cuda')
        """
        self.algorithm = algorithm
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        
        # Model and environment
        self.model = None
        self.env = None
        
        # Training history
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'portfolio_values': []
        }
    
    def create_environment(self, df: pd.DataFrame, **env_kwargs) -> TradingEnvironment:
        """Create trading environment"""
        return TradingEnvironment(df, **env_kwargs)
    
    def train(self, df: pd.DataFrame, total_timesteps: int = 100000,
              eval_freq: int = 5000, n_eval_episodes: int = 5,
              save_freq: int = 10000):
        """Train the RL agent"""
        logger.info(f"Starting {self.algorithm.upper()} training...")
        
        # Create environment
        self.env = self.create_environment(df)
        eval_env = self.create_environment(df)
        
        # Create model based on algorithm
        if self.algorithm == 'dqn':
            self.model = DQN(
                'MlpPolicy',
                self.env,
                learning_rate=self.learning_rate,
                buffer_size=self.buffer_size,
                batch_size=self.batch_size,
                gamma=self.gamma,
                exploration_fraction=0.1,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.01,
                train_freq=4,
                gradient_steps=1,
                target_update_interval=1000,
                verbose=1,
                device=self.device
            )
        
        elif self.algorithm == 'ppo':
            self.model = PPO(
                'MlpPolicy',
                self.env,
                learning_rate=self.learning_rate,
                n_steps=2048,
                batch_size=self.batch_size,
                n_epochs=10,
                gamma=self.gamma,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                verbose=1,
                device=self.device
            )
        
        elif self.algorithm == 'a2c':
            self.model = A2C(
                'MlpPolicy',
                self.env,
                learning_rate=self.learning_rate,
                n_steps=5,
                gamma=self.gamma,
                gae_lambda=0.95,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                verbose=1,
                device=self.device
            )
        
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
        
        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=f'./dl_models/{self.algorithm}_best/',
            log_path=f'./dl_models/{self.algorithm}_logs/',
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
            render=False
        )
        
        # Train model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback,
            progress_bar=True
        )
        
        # Save final model
        self.model.save(f"dl_models/{self.algorithm}_trader_final")
        
        logger.info("Training completed!")
    
    def predict(self, df: pd.DataFrame, deterministic: bool = True) -> pd.DataFrame:
        """Generate trading signals using trained agent"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Create environment
        env = self.create_environment(df)
        
        # Run predictions
        obs, _ = env.reset()
        
        signals = []
        portfolio_values = []
        
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            # Record signal
            signal = 0
            if action == 1:
                signal = 1  # Buy
            elif action == 2:
                signal = -1  # Sell
            
            signals.append(signal)
            portfolio_values.append(info['portfolio_value'])
        
        # Create results dataframe
        start_idx = env.window_size
        end_idx = start_idx + len(signals)
        
        results = pd.DataFrame(index=df.index[start_idx:end_idx])
        results['signal'] = signals
        results['portfolio_value'] = portfolio_values
        results['returns'] = pd.Series(portfolio_values).pct_change().fillna(0).values
        
        # Calculate performance metrics
        total_return = (portfolio_values[-1] - env.initial_balance) / env.initial_balance
        sharpe_ratio = results['returns'].mean() / (results['returns'].std() + 1e-8) * np.sqrt(252)
        
        results.attrs['total_return'] = total_return
        results.attrs['sharpe_ratio'] = sharpe_ratio
        results.attrs['num_trades'] = env.episode_trades
        
        return results
    
    def backtest(self, df: pd.DataFrame, test_split: float = 0.2) -> Dict:
        """Backtest the trained agent"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        # Split data
        split_idx = int(len(df) * (1 - test_split))
        test_df = df.iloc[split_idx:].copy()
        
        # Run backtest
        results = self.predict(test_df)
        
        # Calculate metrics
        metrics = {
            'total_return': results.attrs['total_return'],
            'sharpe_ratio': results.attrs['sharpe_ratio'],
            'num_trades': results.attrs['num_trades'],
            'win_rate': (results[results['signal'] != 0]['returns'] > 0).mean(),
            'max_drawdown': self._calculate_max_drawdown(results['portfolio_value'].values),
            'avg_trade_return': results[results['signal'] != 0]['returns'].mean()
        }
        
        # Plot results
        self._plot_backtest_results(test_df, results)
        
        return metrics
    
    def _calculate_max_drawdown(self, portfolio_values: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_dd = 0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _plot_backtest_results(self, df: pd.DataFrame, results: pd.DataFrame):
        """Plot backtest results"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Price and signals
        ax1.plot(df.index[:len(results)], df['Close'].iloc[:len(results)], 
                label='Price', color='black', linewidth=1)
        
        # Buy signals
        buy_signals = results[results['signal'] == 1]
        ax1.scatter(buy_signals.index, df.loc[buy_signals.index, 'Close'],
                   color='green', marker='^', s=100, label='Buy')
        
        # Sell signals
        sell_signals = results[results['signal'] == -1]
        ax1.scatter(sell_signals.index, df.loc[sell_signals.index, 'Close'],
                   color='red', marker='v', s=100, label='Sell')
        
        ax1.set_ylabel('Price')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Portfolio value
        ax2.plot(results.index, results['portfolio_value'], 
                label='Portfolio Value', color='blue', linewidth=2)
        ax2.axhline(y=100000, color='gray', linestyle='--', label='Initial Balance')
        ax2.set_ylabel('Portfolio Value ($)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Returns
        ax3.plot(results.index, results['returns'].cumsum() * 100,
                label='Cumulative Returns', color='green', linewidth=2)
        ax3.set_ylabel('Cumulative Return (%)')
        ax3.set_xlabel('Date')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.suptitle(f'{self.algorithm.upper()} Trading Agent Backtest Results')
        plt.tight_layout()
        plt.savefig(f'dl_models/{self.algorithm}_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save(self, path: str):
        """Save model"""
        if self.model is not None:
            self.model.save(path)
            logger.info(f"Model saved to {path}")
    
    def load(self, path: str, env: Optional[TradingEnvironment] = None):
        """Load model"""
        if self.algorithm == 'dqn':
            self.model = DQN.load(path, env=env, device=self.device)
        elif self.algorithm == 'ppo':
            self.model = PPO.load(path, env=env, device=self.device)
        elif self.algorithm == 'a2c':
            self.model = A2C.load(path, env=env, device=self.device)
        
        logger.info(f"Model loaded from {path}")


def main():
    """Test the RL trading system"""
    import sys
    sys.path.append('..')
    from utils.csv_data_manager import CSVDataManager
    
    # Initialize data manager
    data_manager = CSVDataManager()
    
    # Load data
    symbol = "SISE"
    df = data_manager.load_data(symbol, "1d")
    
    if df is not None and len(df) > 500:
        # Test with PPO
        logger.info("Testing PPO agent...")
        
        ppo_trader = ReinforcementLearningTrader(
            algorithm='ppo',
            learning_rate=3e-4,
            batch_size=64,
            gamma=0.99
        )
        
        # Train agent (reduced timesteps for testing)
        ppo_trader.train(df, total_timesteps=50000, eval_freq=5000)
        
        # Backtest
        metrics = ppo_trader.backtest(df, test_split=0.2)
        
        print("\nBacktest Results:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Test with DQN
        logger.info("\nTesting DQN agent...")
        
        dqn_trader = ReinforcementLearningTrader(
            algorithm='dqn',
            learning_rate=1e-3,
            buffer_size=50000,
            batch_size=32,
            gamma=0.99
        )
        
        # Train agent
        dqn_trader.train(df, total_timesteps=50000, eval_freq=5000)
        
        # Backtest
        dqn_metrics = dqn_trader.backtest(df, test_split=0.2)
        
        print("\nDQN Backtest Results:")
        print("-" * 50)
        for metric, value in dqn_metrics.items():
            print(f"{metric}: {value:.4f}")
    else:
        print(f"Insufficient data for {symbol}")


if __name__ == "__main__":
    main()