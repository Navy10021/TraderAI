import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# 0. 데이터셋 구축
def get_stock_data(ticker, start_date, end_date):
    df = pd.DataFrame()
    # 데이터 다운로드
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    df['Close'] = stock_data['Close']
    df['Volume'] = stock_data['Volume']
    return df

# 주가 데이터 다운로드
data = get_stock_data('NVDA', start_date='2021-01-01', end_date='2025-01-01')

# 이동평균(SMA)
data['MA'] = data['Close'].rolling(window=50).mean()

# RSI 계산
window_length = 14
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=window_length).mean()
avg_loss = loss.rolling(window=window_length).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# MACD 계산
def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()

ema_fast = ema(data['Close'], span=12)
ema_slow = ema(data['Close'], span=26)
data['MACD'] = ema_fast - ema_slow
data['MACD_signal'] = ema(data['MACD'], span=9)

# 결측값 제거
data = data.dropna(subset=['MA', 'RSI', 'MACD', 'MACD_signal'])

# 스캐일링
scaler = MinMaxScaler()
data[['Close', 'Volume', 'MA', 'RSI', 'MACD', 'MACD_signal']] = scaler.fit_transform(
    data[['Close', 'Volume', 'MA', 'RSI', 'MACD', 'MACD_signal']]
)

print(data.tail())


# 1. 금융 데이터 전처리 클래스
class TradingEnv:
    def __init__(self, data, window_size=5, initial_balance=1000, transaction_cost=0.005):
        self.data = data
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost  # 거래 비용 비율
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
        if action == 0:  # 매도
            if self.holdings > 0:
                sell_value = self.holdings * current_price
                transaction_fee = sell_value * self.transaction_cost
                reward = sell_value - transaction_fee
                self.balance += reward
                self.holdings = 0

        elif action == 2:  # 매수
            buy_amount = self.balance // current_price
            if buy_amount > 0:
                buy_value = buy_amount * current_price
                transaction_fee = buy_value * self.transaction_cost
                self.balance -= (buy_value + transaction_fee)
                self.holdings += buy_amount

        # 포트폴리오 가치 계산
        self.portfolio_value = self.balance + self.holdings * current_price

        # **Reward Design**
        portfolio_change = self.portfolio_value - previous_portfolio_value
        reward = portfolio_change / previous_portfolio_value  # 포트폴리오 변화율

        # 변동성 억제 (큰 변화를 억제하기 위한 페널티)
        volatility_penalty = -0.01 * abs(portfolio_change)

        # 리워드에 변동성 억제 반영
        reward += volatility_penalty

        # 다음 스텝으로 이동
        self.current_step += 1
        if self.current_step >= len(self.data):
            self.done = True

        return self._get_state(), reward, self.done


# 2. 강화학습 모델 구축
# 2-1. 정책 신경망(ver1)
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
        if x.size(0) > 1:  # 배치 크기가 1보다 큰 경우에만 BatchNorm 적용
            x = self.fc1(x)
            x = self.bn1(x)
            residual = x
            x = self.fc2(x)
            x = self.bn2(x) + residual  # Residual Connection
        else:  # 배치 크기가 1인 경우 BatchNorm 우회
            x = self.fc1(x)
            residual = x
            x = self.fc2(x) + residual

        x = self.fc3(x)
        return self.output_layer(x)

# 2-2. 정책 신경망(ver2)
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


# 3. 훈련 루프
def train(env, policy_net, optimizer, gamma=0.99, episodes=500, patience=30):
    portfolio_values = []
    initial_balance = env.initial_balance
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

    # 조기 종료 관련 변수
    best_value = -float('inf')  # 최고 포트폴리오 가치
    no_improvement = 0         # 개선되지 않은 에피소드 수
    best_model_state = None    # 베스트 모델의 가중치 저장

    for episode in range(episodes):
        state = env.reset()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        log_probs = []
        rewards = []
        states = []

        epsilon = max(0.01, 1.0 - episode / episodes)  # 탐색 비율 (점진적으로 감소)
        while not env.done:
            action_probs = policy_net(state)
            if np.random.rand() < epsilon:
                action = np.random.choice(3)  # 무작위 탐색
            else:
                action = torch.argmax(action_probs).item()  # 정책에 따른 선택

            log_prob = Categorical(action_probs).log_prob(torch.tensor(action))
            log_probs.append(log_prob)

            next_state, reward, done = env.step(action)
            rewards.append(reward)
            states.append(state)

            state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

        portfolio_values.append(env.portfolio_value)

        # Discounted rewards 계산
        discounted_rewards = []
        cumulative_reward = 0
        for reward in reversed(rewards):
            cumulative_reward = reward + gamma * cumulative_reward
            discounted_rewards.insert(0, cumulative_reward)
        discounted_rewards = torch.FloatTensor(discounted_rewards).to(device)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        # 정책 경사 손실 + 엔트로피 정규화
        entropy_loss = -torch.sum(action_probs * torch.log(action_probs + 1e-9))
        loss = torch.cat([-log_prob * reward for log_prob, reward in zip(log_probs, discounted_rewards)]).sum()
        loss = loss + 0.01 * entropy_loss  # 엔트로피 정규화

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)  # Gradient Clipping
        optimizer.step()
        scheduler.step()

        # 베스트 모델 업데이트 및 조기 종료 조건 체크
        if env.portfolio_value > best_value:
            best_value = env.portfolio_value
            no_improvement = 0
            best_model_state = policy_net.state_dict()  # 모델 가중치 저장
        else:
            no_improvement += 1

        if no_improvement >= patience:
            print(f"Early stopping at episode {episode + 1}. Best portfolio value: {best_value:.2f}")
            break

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}, Portfolio Value: {env.portfolio_value:.2f}")

    # 최종적으로 베스트 모델의 가중치 로드
    if best_model_state:
        policy_net.load_state_dict(best_model_state)

    return portfolio_values, initial_balance



# 4. 시각화 함수
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


# 5. 추론 함수
def predict(env, policy_net):
    # 환경 초기화
    state = env.reset()
    state = torch.FloatTensor(state).unsqueeze(0).to(device)

    portfolio_history = []  # 포트폴리오 값 기록
    actions = []  # 각 스텝에서의 행동 기록
    prices = []  # 각 스텝의 종가 기록

    while not env.done:
        with torch.no_grad():
            # 정책 신경망에서 행동 확률 계산
            action_probs = policy_net(state)
            action = torch.argmax(action_probs).item()  # 가장 높은 확률의 행동 선택

        # 환경에서 다음 상태로 이동
        next_state, reward, done = env.step(action)

        # 데이터 기록
        portfolio_history.append(env.portfolio_value)
        actions.append(action)  # 행동 기록
        prices.append(env.data.iloc[env.current_step - 1]['Close'])  # 종가 기록

        state = torch.FloatTensor(next_state).unsqueeze(0).to(device)

    # 최종 포트폴리오 가치 출력
    print(f"Final Portfolio Value: {env.portfolio_value:.2f}")

    # 시각화: 포트폴리오 가치 및 행동
    plt.figure(figsize=(12, 8))

    # 1. 포트폴리오 가치 변화
    plt.subplot(2, 1, 1)
    plt.plot(range(len(portfolio_history)), portfolio_history, color='blue', label='Portfolio Value')
    plt.axhline(y=env.initial_balance, color='red', linestyle='--', label='Initial Balance')
    plt.title("Portfolio Value During Prediction", fontsize=16)
    plt.xlabel("Step", fontsize=14)
    plt.ylabel("Portfolio Value", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    # 2. 종가와 행동 시각화
    plt.subplot(2, 1, 2)
    plt.plot(range(len(prices)), prices, color='black', label='Price (Close)', alpha=0.8)

    # 매수(🔼)와 매도(🔽) 표시
    for i, action in enumerate(actions):
        if action == 2:  # 매수
            plt.scatter(i, prices[i], color='green', marker='^', s=100)
        elif action == 0:  # 매도
            plt.scatter(i, prices[i], color='red', marker='v', s=100)

    # 레이블 강제로 추가
    plt.scatter([], [], color='green', marker='^', s=100, label='Buy')  # 매수 레이블
    plt.scatter([], [], color='red', marker='v', s=100, label='Sell')   # 매도 레이블

    plt.title("Actions and Prices During Prediction", fontsize=16)
    plt.xlabel("Step (Date)", fontsize=14)
    plt.ylabel("Price", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)
    # 날짜 레이블로 x축 수정 (Step 대신 실제 날짜 표시)
    #plt.xticks(ticks=range(0, len(prices), step=50), labels=env.data.index[env.window_size:len(prices):50], rotation=45)

    plt.tight_layout()
    plt.show()



# 6. 훈련 / 추론 / 시각화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

env = TradingEnv(data, window_size=5)
state_dim = 5 * 6
action_dim = 3
#policy_net = SimplePolicyNetwork(state_dim, action_dim).to(device)
policy_net = HighPerformancePolicyNetwork(state_dim, action_dim).to(device)
optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

# 훈련 시행
portfolio_values, initial_balance = train(env, policy_net, optimizer)

# 훈련 시각화
visualize_training(portfolio_values, initial_balance)

# 추론 실행
predict(env, policy_net)
