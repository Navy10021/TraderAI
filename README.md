# 💹 TraderAI: Deep Reinforcement Learning for Smarter Investing

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


## 🚀 How It Works
### 1️⃣ Data Preprocessing
  - Download stock price data using yfinance (eg.NVIDIA(NVDA)).
 ```python
   data = get_stock_data('NVDA', start_date='2021-01-01', end_date='2025-01-01')
   ```
  - Calculate key indicators:
    1) Simple Moving Average (SMA): Tracks long-term trends.
    2) Relative Strength Index (RSI): Measures momentum.
    3) Moving Average Convergence Divergence (MACD): Tracks trend changes.
      
### 2️⃣ Custom Reinforcement Learning Environment
  - Simulates realistic trading conditions:
    1) Balance Tracking: Start with $1000.
    2) Transaction Costs: 0.5% cost per trade.
    3) State Representation: Combines past 5 days' data into one state.
  ```python
   env = TradingEnv(data, window_size=5)
   ```

### 3️⃣ Reinforcement Learning with Policy Networks
  - Two policy network options:
    1) HighPerformancePolicyNetwork 🏋️: Designed for large-scale training.
    2) SimplePolicyNetwork 🎯: Lightweight and efficient.
  - Training Process
    1) Policy gradient method with discounted rewards.
    2) Gradient clipping and entropy regularization for stability.
  ```python
   portfolio_values, initial_balance = train(env, policy_net, optimizer)
   ```

### 4️⃣ Inference and Visualization
  - Evaluate the model by visualizing:
    1) Portfolio value during the prediction phase.
    2) Buy and sell actions overlaid on price trends.
  ```python
   visualize_training(portfolio_values, initial_balance)
   predict(env, policy_net)
   ```


## 🏋️‍♂️ Training Results
### Portfolio Value Over Episodes
💡 The following graph shows the change in portfolio value during training.

![1](https://github.com/user-attachments/assets/dbcff93b-c20c-4617-977a-8fc21b4427f3)

## 🕵️‍♂️ Prediction Results

### Actions and Prices During Prediction
  - 🟢 Buy Action (🔼)
  - 🔴 Sell Action (🔽)

![2](https://github.com/user-attachments/assets/886744b4-3c5e-4ea0-b910-b52ef8e0ea12)


## 🛠️ Future Improvements
  1) Add additional technical indicators (e.g., Bollinger Bands, ATR).
  2) Integrate more diverse datasets and multiple stocks.
  3) Experiment with other RL algorithms (e.g., PPO, A3C).
   
## 👨‍💻 Contributors
Seoul National University Graduate School of Data Science (SNU GSDS)
Under the guidance of Navy Lee

## 📬 Contact
For any questions or feedback, contact us at:
📧 iyunseob4@gmail.com or navy10021@snu.ac.kr
