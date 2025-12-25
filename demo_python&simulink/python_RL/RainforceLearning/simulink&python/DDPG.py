import socket
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import math
import time
import os
import signal
import sys

# ================= 配置参数 =================
# 通信配置
HOST = '127.0.0.1'
PORT_RECV = 50000
PORT_SEND = 50001

# 物理限制
FORCE_MAX = 30.0            # 最大输出力 (连续空间范围 [-30, 30])
ANGLE_LIMIT_DEG = 25
ANGLE_LIMIT_RAD = ANGLE_LIMIT_DEG * (math.pi / 180)
POS_LIMIT = 1.0

# 训练配置
MAX_EPISODES = 3000
TARGET_AVG_REWARD = 600.0   # DDPG上限更高，目标分可以设高点
AUTOSAVE_INTERVAL = 10

# DDPG 超参数
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005                 # 软更新系数 (Soft Update)
LR_ACTOR = 0.0001           # 策略网络学习率 (通常比Critic小)
LR_CRITIC = 0.001           # 价值网络学习率
MEMORY_SIZE = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= 1. OU 噪声 (用于连续动作探索) =================
class OUNoise:
    """Ornstein-Uhlenbeck Process: 模拟物理惯性的随机噪声"""
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

# ================= 2. 经验回放池 =================
class ReplayMemory:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)

# ================= 3. DDPG 网络结构 =================

# Actor: 输入状态 -> 输出确定的动作值 (Continuous Action)
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.max_action = max_action
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh() # 输出 [-1, 1]
        )

    def forward(self, state):
        return self.net(state) * self.max_action

# Critic: 输入状态 + 动作 -> 输出 Q 值
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # 先处理状态
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, state, action):
        # 将状态和动作拼接
        q = torch.cat([state, action], 1)
        q = nn.functional.relu(self.l1(q))
        q = nn.functional.relu(self.l2(q))
        return self.l3(q)

# ================= 4. DDPG Agent =================
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.memory = ReplayMemory(MEMORY_SIZE)
        self.noise = OUNoise(action_dim)
        self.max_action = max_action

    def select_action(self, state, noise=True):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state).cpu().data.numpy().flatten()

        if noise:
            action = action + self.noise.sample()

        return np.clip(action, -self.max_action, self.max_action)

    def update(self):
        if len(self.memory) < BATCH_SIZE: return

        state, action, reward, next_state, done = self.memory.sample(BATCH_SIZE)

        state = torch.FloatTensor(state).to(device)
        action = torch.FloatTensor(action).to(device)
        reward = torch.FloatTensor(reward).reshape((BATCH_SIZE, 1)).to(device)
        next_state = torch.FloatTensor(next_state).to(device)
        done = torch.FloatTensor(done).reshape((BATCH_SIZE, 1)).to(device)

        # ----------------------
        # 1. 更新 Critic
        # ----------------------
        with torch.no_grad():
            # 计算目标 Q 值: Q_target = r + gamma * Q_target_net(s', Actor_target_net(s'))
            next_action = self.actor_target(next_state)
            target_Q = self.critic_target(next_state, next_action)
            target_Q = reward + (1 - done) * GAMMA * target_Q

        current_Q = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_Q, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------
        # 2. 更新 Actor
        # ----------------------
        # 目标: 最大化 Q 值 -> 最小化 -Q
        actor_loss = -self.critic(state, self.actor(state)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------
        # 3. 软更新目标网络
        # ----------------------
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

    def save(self, filename_prefix):
        torch.save(self.actor.state_dict(), f'{filename_prefix}_actor.pth')
        torch.save(self.critic.state_dict(), f'{filename_prefix}_critic.pth')

# ================= 5. 通信服务器 =================
class SimulinkServer:
    def __init__(self):
        print("Initializing Server Sockets...")
        self.sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_recv.bind((HOST, PORT_RECV))
        self.sock_recv.listen(1)

        self.sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock_send.bind((HOST, PORT_SEND))
        self.sock_send.listen(1)

        self.conn_recv = None
        self.conn_send = None
        print(f"Server Ready. Ports: {PORT_RECV}, {PORT_SEND}")

    def wait_for_connection(self):
        # print(">>> Waiting for Simulink simulation...")
        self.conn_recv, _ = self.sock_recv.accept()
        self.conn_send, _ = self.sock_send.accept()
        self.send_action(0.0, 0.0)

    def get_state(self):
        if self.conn_recv is None: return None
        try:
            data = self.conn_recv.recv(32)
            if not data: return None
            state = struct.unpack('<dddd', data)
            return np.array(state, dtype=np.float32)
        except: return None

    def send_action(self, force, stop_sig):
        if self.conn_send is None: return
        try:
            # DDPG 输出连续值，直接发送 float 即可
            packet = struct.pack('<dd', float(force), float(stop_sig))
            self.conn_send.send(packet)
        except: pass

    def close_current_connection(self):
        if self.conn_recv:
            try:
                self.conn_recv.close()
            except:
                pass

        if self.conn_send:
            try:
                self.conn_send.close()
            except:
                pass

        self.conn_recv = None
        self.conn_send = None

    def shutdown(self):
        self.close_current_connection()
        self.sock_recv.close()
        self.sock_send.close()

# 全局信号变量
stop_requested = False
def signal_handler(signum, frame):
    global stop_requested
    print("\n>>> 安全停止请求已接收... <<<")
    stop_requested = True

# ================= 6. 主循环 =================
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server = SimulinkServer()

    state_dim = 4
    action_dim = 1 # 连续动作只有1维：力
    agent = DDPGAgent(state_dim, action_dim, FORCE_MAX)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    angle_line, = ax1.plot([], [], 'b-', linewidth=1.5)
    ax1.set_ylim(-1.0, 1.0); ax1.grid(True)
    reward_line, = ax2.plot([], [], 'r.-', linewidth=1.5)
    ax2.set_title("DDPG Continuous Control Training"); ax2.grid(True)
    ax2.axhline(y=TARGET_AVG_REWARD, color='g', linestyle='--')

    episode_count = 0
    episode_rewards = []

    print(">>> DDPG Training Started (Continuous Force Mode)...")

    try:
        while episode_count < MAX_EPISODES and not stop_requested:
            print(">>> Waiting for Simulink...")
            server.wait_for_connection()

            state_np = server.get_state()
            if state_np is None:
                server.close_current_connection()
                continue

            # 坏局检测
            if abs(state_np[1]) > 0.3:
                print(f"\033[91m [WARN] Bad Init (Angle={state_np[1]:.2f}). Resetting.\033[0m")
                server.send_action(0.0, 1.0)
                server.close_current_connection()
                continue

            # 重置噪声
            agent.noise.reset()

            current_ep_reward = 0
            angle_history = []

            while True:
                if stop_requested:
                    server.send_action(0.0, 1.0)
                    break

                # 1. 选择动作 (连续值)
                action = agent.select_action(state_np)
                force = action[0] # 取出标量

                # 2. 发送给 Simulink
                # DDPG 输出的 force 已经是 -30 到 30 之间的实数，无需映射

                # 3. 失败判断
                curr_x, curr_theta = state_np[0], state_np[1]
                done = False
                stop_signal = 0.0

                if abs(curr_theta) > ANGLE_LIMIT_RAD or abs(curr_x) > POS_LIMIT:
                    stop_signal = 1.0
                    reward = -10.0
                    done = True
                else:
                    # 连续控制的奖励函数
                    r_angle = 1.0 - (curr_theta / ANGLE_LIMIT_RAD)**2
                    r_pos = 0.1 * (1.0 - (curr_x / POS_LIMIT)**2)

                    # 惩罚动作幅度 (鼓励节能)
                    r_action = -0.01 * (abs(force) / FORCE_MAX)**2

                    reward = r_angle + r_pos + r_action

                current_ep_reward += reward
                server.send_action(force, stop_signal)

                next_state_np = server.get_state()
                if next_state_np is None: break

                # 4. 存储与更新
                # 注意：DDPG 需要存 (s, a, r, s', done)
                agent.memory.push(state_np, action, reward, next_state_np, done)
                agent.update()

                state_np = next_state_np
                angle_history.append(curr_theta)

            server.close_current_connection()
            episode_count += 1
            episode_rewards.append(current_ep_reward)

            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 0 else current_ep_reward

            print(f"Ep {episode_count}: Reward={current_ep_reward:.1f}, Avg={avg_reward:.1f}")

            # 绘图
            if episode_count % 1 == 0:
                angle_line.set_ydata(angle_history)
                angle_line.set_xdata(range(len(angle_history)))
                ax1.set_xlim(0, max(len(angle_history), 10))
                reward_line.set_ydata(episode_rewards)
                reward_line.set_xdata(range(len(episode_rewards)))
                ax2.set_xlim(0, max(len(episode_rewards), 10))
                ax2.set_ylim(min(min(episode_rewards), -20), max(max(episode_rewards), 100))
                plt.draw(); plt.pause(0.01)

            # 自动保存
            if episode_count % AUTOSAVE_INTERVAL == 0 or stop_requested:
                try:
                    agent.save('cartpole_ddpg_autosave')
                    fig.savefig('training_result_ddpg.png')
                    print(f"   >>> Auto-Saved at Ep {episode_count}")
                except: pass

            if len(episode_rewards) >= 50 and avg_reward >= TARGET_AVG_REWARD:
                print(f"\n>>> SUCCESS: Converged! Avg Reward {avg_reward:.1f}")
                break

    except Exception as e:
        print(f"\nError: {e}")
    finally:
        print("\n=== Final Cleanup ===")
        try:
            agent.save('cartpole_ddpg_final')
            print("Model saved.")
        except:
            pass

        try:
            plt.ioff()
            fig.savefig('training_result_ddpg_final.png')
            print("Plot saved.")
        except:
            pass

        try:
            server.shutdown()
            print("Server closed.")
        except:
            pass
