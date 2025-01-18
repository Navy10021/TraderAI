import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 0. Build the dataset
def get_stock_data(ticker, start_date, end_date):
    df = pd.DataFrame()
    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    df['Close'] = stock_data['Close']
    df['Volume'] = stock_data['Volume']
    return df

# Download stock data
data = get_stock_data('NVDA', start_date='2022-01-01', end_date='2025-01-01')

# Simple Moving Average (SMA)
data['MA'] = data['Close'].rolling(window=50).mean()

# Calculate RSI
window_length = 14
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=window_length).mean()
avg_loss = loss.rolling(window=window_length).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Calculate MACD
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

ema_fast = ema(data['Close'], span=12)
ema_slow = ema(data['Close'], span=26)
data['MACD'] = ema_fast - ema_slow
data['MACD_signal'] = ema(data['MACD'], span=9)

# Remove missing values
data = data.dropna(subset=['MA', 'RSI', 'MACD', 'MACD_signal'])

# Scaling
scaler = MinMaxScaler()
data[['Close', 'Volume', 'MA', 'RSI', 'MACD', 'MACD_signal']] = scaler.fit_transform(
    data[['Close', 'Volume', 'MA', 'RSI', 'MACD', 'MACD_signal']]
)

print(data.tail())


# 1. Financial data preprocessing class
class TradingEnv:
    def __init__(self, data, window_size=5, initial_balance=1000, transaction_cost=0.005):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # Transaction cost rate
        self.reset()

    def reset(self):
        self.current_step = self.window_size
        self.balance = self.initial_balance
        self.portfolio_value = self.balance
        self.holdings = 0
        self.done = False
        return self._get_state()

    def _get_state(self):
        state = self.data.iloc[self.current_step - self.window_size: self.current_step].values.flatten()
        return state

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']

        if current_price <= 0:
            self.current_step += 1
            if self.current_step >= len(self.data):
                self.done = True
            return self._get_state(), 0, self.done

        reward = 0
        previous_portfolio_value = self.portfolio_value

        # **Action Handling**
        if action == 0:  # Sell
            if self.holdings > 0:
                sell_value = self.holdings * current_price
                transaction_fee = sell_value * self.transaction_cost
                reward = sell_value - transaction_fee
                self.balance += reward
                self.holdings = 0

        elif action == 2:  # Buy
            buy_amount = self.balance // current_price
            if buy_amount > 0:
                buy_value = buy_amount * current_price
                transaction_fee = buy_value * self.transaction_cost
                self.balance -= (buy_value + transaction_fee)
                self.holdings += buy_amount

        # Calculate portfolio value
        self.portfolio_value = self.balance + self.holdings * current_price

        # **Reward Design**
        portfolio_change = self.portfolio_value - previous_portfolio_value
        reward = portfolio_change / previous_portfolio_value  # Portfolio change rate

        # Volatility suppression (penalty for large changes)
        volatility_penalty = -0.01 * abs(portfolio_change)

        # Adjust reward with volatility penalty
        reward += volatility_penalty

        # Move to the next step
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True

        return self._get_state(), reward, self.done


# 2. Build reinforcement learning models
# 2-1. Policy Network (ver1)
class HighPerformancePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(HighPerformancePolicyNetwork, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.bn1 = nn.BatchNorm1d(256)

        self.fc2 = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        self.bn2 = nn.BatchNorm1d(256)

        self.residual = nn.Linear(256, 256)

        self.fc3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.output_layer = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        if x.size(0) > 1:  # Apply BatchNorm only if batch size > 1
            x = self.fc1(x)
            x = self.bn1(x)
            residual = x
            x = self.fc2(x)
            x = self.bn2(x) + residual  # Residual Connection
        else:  # Bypass BatchNorm for batch size = 1
            x = self.fc1(x)
            residual = x
            x = self.fc2(x) + residual

        x = self.fc3(x)
        return self.output_layer(x)

# 2-2. Policy Network (ver2)
class SimplePolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SimplePolicyNetwork, self).__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),  # Layer Normalization for input stabilization
            nn.ReLU()
        )

        self.hidden_layers = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(0.3),  # Increased Dropout to reduce overfitting

            nn.Linear(256, 128),
            nn.GELU(),  # GELU activation for smoother gradient flow
            nn.Dropout(0.2)
        )

        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)  # Softmax for action probability distribution
        )

    def forward(self, x):
        # Forward pass with normalization and improved layers
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)

# 3. Training Loop
def train(env, policy_net, optimizer, gamma=0.99, episodes=500, patience=30):
    portfolio_values = []
    initial_balance = env.initial_balance
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # Variables for early stopping
    best_value = -float('inf')  # Best portfolio value
    no_improvement = 0         # Number of episodes with no improvement
    best_model_state = None    # To store weights of the best model

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        log_probs = []
        rewards = []
        states = []

        epsilon = max(0.01, 1.0 - episode / episodes)  # Exploration rate (decreases over time)
        while not env.done:
            action_probs = policy_net(state)
            if np.random.rand() < epsilon:
                action = np.random.choice(3)  # Random exploration
            else:
                action = torch.argmax(action_probs).item()  # Action based on policy

            log_prob = Categorical(action_probs).log_prob(torch.tensor(action))
            log_probs.append(log_prob)

            next_state, reward, done = env.step(action)
            rewards.append(reward)
            states.append(state)

            state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        portfolio_values.append(env.portfolio_value)

        # Calculate discounted rewards
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # Policy gradient loss + entropy regularization
        entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-9))
        loss = torch.cat([-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]).sum()
        loss = loss + 0.01 * entropy_loss  # Entropy regularization

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()

        # Update the best model and check early stopping condition
        if env.portfolio_value > best_value:
            best_value = env.portfolio_value
            no_improvement = 0
            best_model_state = policy_net.state_dict()  # Save model weights
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at episode {episode + 1}. Best portfolio value: {best_value:.2f}")
            break

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Portfolio Value: {env.portfolio_value:.2f}")

    # Load the best model's weights
    if best_model_state:
        policy_net.load_state_dict(best_model_state)

    return portfolio_values, initial_balance


# 4. Visualization Function
def visualize_training(portfolio_values, initial_balance):
    plt.figure(figsize=(10, 6))
    episodes = range(1, len(portfolio_values) + 1)
    plt.plot(episodes, portfolio_values, label='Portfolio Value', color='blue', linewidth=2)
    plt.axhline(y=initial_balance, color='red', linestyle='--', label='Initial Balance')
    plt.title("Training Results: Portfolio Value Over Episodes", fontsize=16)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Portfolio Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()


# 5. Inference Function
def predict(env, policy_net):
    # Initialize the environment
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    portfolio_history = []  # Record portfolio value
    actions = []  # Record actions at each step
    prices = []  # Record closing prices at each step

    while not env.done:
        with torch.no_grad():
            # Compute action probabilities using the policy network
            action_probs = policy_net(state)
            action = torch.argmax(action_probs).item()  # Choose the action with the highest probability

        # Step into the next state
        next_state, reward, done = env.step(action)

        # Record data
        portfolio_history.append(env.portfolio_value)
        actions.append(action)  # Record action
        prices.append(env.data.iloc[env.current_step - 1]['Close'])  # Record closing price

        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

    # Display the final portfolio value
    print(f"Final Portfolio Value: {env.portfolio_value:.2f}")

    # Visualization: Portfolio value and actions
    plt.figure(figsize=(12, 8))

    # 1. Portfolio value trend
    plt.subplot(2, 1, 1)
    plt.plot(range(len(portfolio_history)), portfolio_history, color='blue', label='Portfolio Value')
    plt.axhline(y=env.initial_balance, color='red', linestyle='--', label='Initial Balance')
    plt.title("Portfolio Value During Prediction", fontsize=16)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Portfolio Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # 2. Closing prices and actions
    plt.subplot(2, 1, 2)
    plt.plot(range(len(prices)), prices, color='black', label='Price (Close)', alpha=0.8)

    # Mark buy (▲) and sell (▼) actions
    for i, action in enumerate(actions):
        if action == 2:  # Buy
            plt.scatter(i, prices[i], color='green', marker='^', s=100)
        elif action == 0:  # Sell
            plt.scatter(i, prices[i], color='red', marker='v', s=100)

    # Add forced labels for buy/sell actions
    plt.scatter([], [], color='green', marker='^', s=100, label='Buy')  # Buy label
    plt.scatter([], [], color='red', marker='v', s=100, label='Sell')   # Sell label

    plt.title("Actions and Prices During Prediction", fontsize=16)
    plt.xlabel("Step (Date)", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# 6. Training / Inference / Visualization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = TradingEnv(data, window_size=5)
state_dim = 5 * 6
action_dim = 3
# policy_net = SimplePolicyNetwork(state_dim, action_dim).to(device)
policy_net = HighPerformancePolicyNetwork(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# Train the model
portfolio_values, initial_balance = train(env, policy_net, optimizer)

# Visualize training
visualize_training(portfolio_values, initial_balance)

# Run inference
predict(env, policy_net)
