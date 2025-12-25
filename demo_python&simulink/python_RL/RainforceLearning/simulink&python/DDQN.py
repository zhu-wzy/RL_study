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
import signal  # [新增] 用于捕获退出信号
import sys     # [新增]

# ================= 配置参数 =================
# 通信配置
HOST = '127.0.0.1'
PORT_RECV = 50000
PORT_SEND = 50001

# 物理限制
FORCE_MAG = 15.0
ANGLE_LIMIT_DEG = 25
ANGLE_LIMIT_RAD = ANGLE_LIMIT_DEG * (math.pi / 180)
POS_LIMIT = 1.0

# 训练配置
MAX_EPISODES = 2000
TARGET_AVG_REWARD = 450.0
AUTOSAVE_INTERVAL = 10      # [新增] 每隔多少回合自动保存一次

# DQN 超参数
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 3000
TARGET_UPDATE = 200
LR = 0.0005
MEMORY_SIZE = 50000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= 1. Dueling DQN 网络结构 =================
class DuelingDQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DuelingDQN, self).__init__()
        # 公共特征层
        self.feature_layer = nn.Sequential(
            nn.Linear(n_observations, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        # 支路1: Value Stream (状态价值)
        self.value_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        # 支路2: Advantage Stream (动作优势)
        self.advantage_stream = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # 聚合公式
        return values + (advantages - advantages.mean(dim=1, keepdim=True))

# ================= 2. 经验回放池 =================
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(args)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# ================= 3. 通信服务器 =================
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
        # print(">>> Waiting for Simulink simulation...") # 减少刷屏
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
            packet = struct.pack('<dd', float(force), float(stop_sig))
            self.conn_send.send(packet)
        except: pass

    def close_current_connection(self):
        if self.conn_recv:
            try: self.conn_recv.close()
            except: pass
        if self.conn_send:
            try: self.conn_send.close()
            except: pass
        self.conn_recv = None
        self.conn_send = None

    def shutdown(self):
        self.close_current_connection()
        self.sock_recv.close()
        self.sock_send.close()

# ================= 4. 辅助函数 =================
def select_action(state, steps_done, policy_net):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        np.exp(-1. * steps_done / EPS_DECAY)
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(3)]], device=device, dtype=torch.long)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE: return
    transitions = memory.sample(BATCH_SIZE)
    batch = list(zip(*transitions))

    state_batch = torch.cat(batch[0])
    action_batch = torch.cat(batch[1])
    reward_batch = torch.cat(batch[2])
    next_state_batch = torch.cat(batch[3])

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_state_values = target_net(next_state_batch).gather(1, next_actions).squeeze(1)

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch.squeeze()

    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# ================= 5. 主循环 =================
# 全局变量用于信号处理
stop_requested = False

def signal_handler(signum, frame):
    global stop_requested
    print("\n\n>>> 信号捕获: 正在请求安全停止... (请稍候，将在本回合结束后保存) <<<")
    stop_requested = True

if __name__ == "__main__":
    # 注册信号处理 (Ctrl+C 或 停止信号)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    server = SimulinkServer()

    n_obs = 4
    n_actions = 3
    policy_net = DuelingDQN(n_obs, n_actions).to(device)
    target_net = DuelingDQN(n_obs, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    angle_line, = ax1.plot([], [], 'b-', linewidth=1.5)
    ax1.set_ylim(-1.0, 1.0); ax1.grid(True)
    reward_line, = ax2.plot([], [], 'r.-', linewidth=1.5)
    ax2.set_title("Dueling DDQN Training Progress"); ax2.grid(True)
    ax2.axhline(y=TARGET_AVG_REWARD, color='g', linestyle='--')

    steps_done = 0
    episode_count = 0
    episode_rewards = []

    print(">>> Dueling DDQN Training Started...")
    print(f">>> Auto-Save Enabled: Every {AUTOSAVE_INTERVAL} episodes.")

    try:
        while episode_count < MAX_EPISODES and not stop_requested:
            print(">>> Waiting for Simulink simulation...")
            server.wait_for_connection()

            state_np = server.get_state()
            if state_np is None:
                server.close_current_connection()
                continue

            if abs(state_np[1]) > 0.3:
                print(f"\033[91m [WARN] Bad Init (Angle={state_np[1]:.2f}). Skipping.\033[0m")
                server.send_action(0.0, 1.0)
                server.close_current_connection()
                continue

            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            current_ep_reward = 0
            angle_history = []

            while True:
                # 如果在回合中收到停止信号，尽量让物理仿真跑完这一帧或直接退出
                if stop_requested:
                    server.send_action(0.0, 1.0) # 发送重置信号尝试让 Simulink 停下
                    break

                curr_x, curr_theta, curr_x_dot, curr_theta_dot = state_np

                action = select_action(state, steps_done, policy_net)
                action_item = action.item()

                if action_item == 0: force = -FORCE_MAG
                elif action_item == 1: force = 0.0
                else: force = FORCE_MAG

                done = False
                stop_signal = 0.0
                if abs(curr_theta) > ANGLE_LIMIT_RAD or abs(curr_x) > POS_LIMIT:
                    stop_signal = 1.0
                    reward = -10.0
                    done = True
                else:
                    r_angle = 1.0 - (curr_theta / ANGLE_LIMIT_RAD)**2
                    r_pos = 0.1 * (1.0 - (curr_x / POS_LIMIT)**2)
                    r_stable = -0.1 * abs(curr_theta_dot)
                    reward = r_angle + r_pos + r_stable

                current_ep_reward += reward
                server.send_action(force, stop_signal)
                next_state_np = server.get_state()

                if next_state_np is None: break

                next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
                reward_t = torch.tensor([reward], device=device)

                memory.push(state, action, reward_t, next_state)
                optimize_model(memory, policy_net, target_net, optimizer)

                state = next_state
                state_np = next_state_np
                steps_done += 1
                angle_history.append(curr_theta)

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

            server.close_current_connection()
            episode_count += 1
            episode_rewards.append(current_ep_reward)

            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 0 else current_ep_reward
            epsilon = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)

            print(f"Ep {episode_count}: Reward={current_ep_reward:.1f}, Avg={avg_reward:.1f}, Eps={epsilon:.3f}")

            if episode_count % 1 == 0:
                angle_line.set_ydata(angle_history)
                angle_line.set_xdata(range(len(angle_history)))
                ax1.set_xlim(0, max(len(angle_history), 10))
                reward_line.set_ydata(episode_rewards)
                reward_line.set_xdata(range(len(episode_rewards)))
                ax2.set_xlim(0, max(len(episode_rewards), 10))
                ax2.set_ylim(min(min(episode_rewards), -20), max(max(episode_rewards), 50))
                plt.draw(); plt.pause(0.01)

            # [新增] 自动保存逻辑
            if episode_count % AUTOSAVE_INTERVAL == 0 or stop_requested:
                try:
                    torch.save(policy_net.state_dict(), 'cartpole_dueling_ddqn_autosave.pth')
                    # 同时更新主图
                    fig.savefig('training_result_ddqn.png')
                    print(f"   >>> [Auto-Save] Model and plot saved at Ep {episode_count}")
                except Exception as e:
                    print(f"   >>> [Auto-Save Error] {e}")

            if len(episode_rewards) >= 50 and avg_reward >= TARGET_AVG_REWARD:
                print(f"\n>>> SUCCESS: Converged! Avg Reward {avg_reward:.1f}")
                break

    except Exception as e:
        print(f"\nError occurred: {e}")
    finally:
        print("\n=== Final Saving & Cleanup ===")
        try:
            torch.save(policy_net.state_dict(), 'cartpole_dueling_ddqn_final.pth')
            print("-> Final model saved successfully.")
        except Exception as e:
            print(f"-> Failed to save model: {e}")

        try:
            plt.ioff()
            fig.savefig('training_result_ddqn_final.png')
            print("-> Final plot saved successfully.")
        except Exception as e:
            print(f"-> Failed to save plot: {e}")

        try:
            server.shutdown()
            print("-> Server shutdown.")
        except Exception as e:
            print(f"-> Failed to shutdown server: {e}")
