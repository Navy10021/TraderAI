# TraderAI: 🚀 Deep Reinforcement Learning for Smarter Investing 💹

Welcome to the **Deep Reinforcement Learning-based Stock Trading Framework**! 
This project leverages financial data, key technical indicators, and deep reinforcement learning techniques to simulate a robust trading agent that learns to maximize portfolio returns.

## 🌟 Features

### 📊 Financial Data Integration

  - Download real-time stock market data using the yfinance library.
  - Compute popular indicators: SMA, RSI, MACD.

### 🤖 Reinforcement Learning Environment
  - Custom trading environment with realistic constraints like transaction costs and balance tracking.
  - Reward structure encouraging portfolio optimization and volatility management.

### 🧠 Advanced Policy Networks
  - HighPerformancePolicyNetwork: Residual connections, BatchNorm, and Dropout for stability.
  - SimplePolicyNetwork: LayerNorm, GELU activation, and improved regularization.

### 📉 Visualization Tools
  - Intuitive graphs for portfolio value, actions, and price trends.
  - Easy-to-interpret results for training and testing phases.


## 📂 File Overview
### 🛠️ Modules
#### 1. Data Preprocessing
  - Download and process stock market data.
  - Compute SMA, RSI, and MACD indicators.

#### 2. Environment Setup
  - TradingEnv: Custom RL environment for portfolio management.

#### 3. Policy Networks

  - Two versions of neural networks for action prediction:
    1) **HighPerformancePolicyNetwork**: Residuals & BatchNorm for high performance.
    2) **SimplePolicyNetwork**: Lightweight with GELU and Dropout.

#### 4. Training Loop
  - Implements policy gradient optimization with entropy regularization.
  - Early stopping mechanism to prevent overfitting.

#### 5. Inference and Visualization
  - Predict portfolio behavior and plot results with easy-to-read graphs.
