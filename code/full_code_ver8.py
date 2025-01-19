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
data = get_stock_data('TSLA', start_date='2022-01-01', end_date='2025-01-01')

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
        if action == 0:     # Sell
            if self.holdings > 0:
                sell_value = self.holdings * current_price
                transaction_fee = sell_value * self.transaction_cost
                reward = sell_value - transaction_fee
                self.balance += reward
                self.holdings = 0

        elif action == 1:   # Hold
            reward = 0      # No reward or penalty for holding

        elif action == 2:   # Buy
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

            if np.random.rand() < epsilon:  # Exploration: Random action
                action = np.random.choice(3)  # Randomly choose between 0, 1, 2 (sell, hold, buy)
            else:  # Exploitation: Choose based on policy
                action = torch.argmax(action_probs).item()

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

        # Normalize the discounted rewards
        discounted_rewards = torch.tensor(discounted_rewards).float().to(device)
        rewards_tensor = discounted_rewards - discounted_rewards.mean()

        # Compute the loss (Policy Gradient)
        loss = -torch.sum(torch.stack(log_probs) * rewards_tensor)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step()

        # Check for early stopping based on portfolio value
        if env.portfolio_value > best_value:
            best_value = env.portfolio_value
            best_model_state = policy_net.state_dict()  # Save best model weights
            no_improvement = 0
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at episode {episode + 1}")
            break

        # Print progress
        if episode % 10 == 0:
            print(f"Episode {episode + 1}/{episodes}, Portfolio Value: {env.portfolio_value:.2f}")

    # Load the best model
    if best_model_state:
        policy_net.load_state_dict(best_model_state)

    return portfolio_values, env.initial_balance


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


# 4. Inference: Simulate the Trading Agent
def test(env, policy_net):
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    portfolio_values = []

    while not env.done:
        action_probs = policy_net(state)
        action = torch.argmax(action_probs).item()

        next_state, reward, done = env.step(action)
        portfolio_values.append(env.portfolio_value)

        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

    return portfolio_values


# 5. Run the training and testing
def predict(env, policy_net):
    # Initialize the environment
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    portfolio_history = []  # Record portfolio values
    actions = []  # Record actions
    prices = []  # Record closing prices

    while not env.done:
        with torch.no_grad():
            # Compute action probabilities using the policy network
            action_probs = policy_net(state)
            action = torch.argmax(action_probs).item()  # Choose action with the highest probability

        # Take a step in the environment
        next_state, reward, done = env.step(action)

        # Record data
        portfolio_history.append(env.portfolio_value)
        actions.append(action)
        prices.append(env.data.iloc[env.current_step - 1]['Close'])

        # Update state
        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

    # Display the final portfolio value
    print(f"Final Portfolio Value: {env.portfolio_value:.2f}")

    # Visualization
    plt.figure(figsize=(12, 8))

    # 1. Portfolio Value Trend
    plt.subplot(2, 1, 1)
    plt.plot(range(len(portfolio_history)), portfolio_history, color='blue', label='Portfolio Value', linewidth=2)
    plt.axhline(y=env.initial_balance, color='red', linestyle='--', label='Initial Balance', linewidth=1.5)
    plt.title("Portfolio Value During Prediction", fontsize=16)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Portfolio Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # 2. Closing Prices and Actions
    plt.subplot(2, 1, 2)
    plt.plot(range(len(prices)), prices, color='black', label='Price (Close)', alpha=0.8, linewidth=1.5)

    # Mark buy (▲), sell (▼), and hold (-) actions
    for i, action in enumerate(actions):
        if action == 2:  # Buy
            plt.scatter(i, prices[i], color='green', marker='^', s=100, label='Buy' if i == 0 else "")
        elif action == 0:  # Sell
            plt.scatter(i, prices[i], color='red', marker='v', s=100, label='Sell' if i == 0 else "")
        elif action == 1:  # Hold
            plt.scatter(i, prices[i], color='blue', marker='_', s=100, label='Hold' if i == 0 else "")

    # Ensure the legend only shows "Buy", "Sell", and "Hold" once
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=12)

    plt.title("Actions and Prices During Prediction", fontsize=16)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# ===============================================================================
# 6. Training / Inference / Visualization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Split data into training and testing
train_data = data.iloc[:int(len(data) * 0.8)]
test_data = data.iloc[int(len(data) * 0.8):]

# Initialize environments
train_env = TradingEnv(train_data, window_size=5)
test_env = TradingEnv(test_data, window_size=5)

state_dim = 5 * 6
action_dim = 3

# Initialize policy network and optimizer
policy_net = HighPerformancePolicyNetwork(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# Train the model
portfolio_values, initial_balance = train(train_env, policy_net, optimizer)

# Visualize training
visualize_training(portfolio_values, initial_balance)

# Save the trained model
torch.save(policy_net.state_dict(), 'policy_net.pth')

# Load the trained model (optional)
policy_net.load_state_dict(torch.load('policy_net.pth'))
policy_net.eval()  # Set the model to evaluation mode

# Run inference on test data
predict(test_env, policy_net)
