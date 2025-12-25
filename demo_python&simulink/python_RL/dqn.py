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

# ================= 配置参数 =================
# 通信配置
HOST = '127.0.0.1'
PORT_RECV = 50000   # 接收 Simulink 状态
PORT_SEND = 50001   # 发送控制指令
DT = 0.01           # 仿真步长

# 物理与奖励配置
FORCE_MAG = 15.0            # 推力大小
ANGLE_LIMIT_DEG = 25        # 失败角度限制 (度)
ANGLE_LIMIT_RAD = ANGLE_LIMIT_DEG * (math.pi / 180) # ≈ 0.4363 rad
POS_LIMIT = 1.0             # 失败位置限制 (米)

# DQN 超参数
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000    # 全局步数，跨 Episode 累积
TARGET_UPDATE = 100
LR = 0.001
MEMORY_SIZE = 10000

# 设备检测
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= 1. DQN 网络定义 =================
class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        return self.layer3(x)

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

# ================= 3. 支持重连的通信服务器 =================
class SimulinkServer:
    def __init__(self):
        print("Initializing Server Sockets...")

        # 创建 Socket (只 Bind 一次)
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
        print(f"Server Ready. Listening on ports {PORT_RECV} & {PORT_SEND}")

    def wait_for_connection(self):
        """阻塞等待 Simulink 连接"""
        print(">>> Waiting for Simulink to start simulation...")
        self.conn_recv, addr_r = self.sock_recv.accept()
        # print(f"Connected Recv from {addr_r}")
        self.conn_send, addr_s = self.sock_send.accept()
        # print(f"Connected Send from {addr_s}")
        print(">>> Simulink Connected! Episode Starting.")

        # 握手：发送初始零指令
        self.send_action(0.0, 0.0)

    def get_state(self):
        if self.conn_recv is None: return None
        try:
            data = self.conn_recv.recv(32) # 4 * double
            if not data: return None
            state = struct.unpack('<dddd', data)
            return np.array(state, dtype=np.float32)
        except Exception:
            return None

    def send_action(self, force_value, stop_signal):
        if self.conn_send is None: return
        try:
            packet = struct.pack('<dd', float(force_value), float(stop_signal))
            self.conn_send.send(packet)
        except Exception:
            pass

    def close_current_connection(self):
        """只关闭当前的连接，不关闭 Server Socket"""
        if self.conn_recv:
            try: self.conn_recv.close()
            except: pass
        if self.conn_send:
            try: self.conn_send.close()
            except: pass
        self.conn_recv = None
        self.conn_send = None
        print(">>> Connection closed. Waiting for next episode...\n")

    def shutdown(self):
        """关闭整个服务器"""
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
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# ================= Main Loop =================
if __name__ == "__main__":
    # 初始化服务器
    server = SimulinkServer()

    # 初始化 DQN
    n_actions = 3
    n_observations = 4
    policy_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    # 绘图设置
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    # 图1: 当前 Episode 的角度变化
    angle_line, = ax1.plot([], [], 'b-')
    ax1.set_title("Current Episode Angle")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_ylabel("Angle (rad)")
    ax1.grid()

    # 图2: 历史 Reward 曲线
    episode_rewards = []
    reward_line, = ax2.plot([], [], 'r-')
    ax2.set_title("Episode Total Reward History")
    ax2.set_ylabel("Total Reward")
    ax2.grid()

    steps_done = 0 # 全局步数
    episode_count = 0

    try:
        # ==========================================
        # 外层循环：Episode Loop (对应 MATLAB 的多次仿真)
        # ==========================================
        while True:
            # 1. 等待 Simulink 连接 (阻塞直到 sim() 启动)
            server.wait_for_connection()

            # 获取初始状态
            state_np = server.get_state()
            if state_np is None:
                print("Failed to get initial state. Retrying...")
                server.close_current_connection()
                continue

            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)

            current_episode_reward = 0
            angle_history = [] # 仅用于当前回合绘图

            # ==========================================
            # 内层循环：Step Loop (单次仿真内)
            # ==========================================
            while True:
                # A. 状态判断
                current_x = state_np[0]
                current_theta = state_np[1]

                stop_signal = 0.0
                force = 0.0
                done = False

                # 失败判定
                if abs(current_theta) > ANGLE_LIMIT_RAD or abs(current_x) > POS_LIMIT:
                    stop_signal = 1.0 # 触发重置
                    reward = -10.0
                    done = True
                    # 失败时的虚拟动作
                    action = torch.tensor([[1]], device=device, dtype=torch.long)
                    # print(f"  -> Fail: Angle {current_theta:.3f}")
                else:
                    stop_signal = 0.0
                    action = select_action(state, steps_done, policy_net)
                    action_item = action.item()

                    if action_item == 0: force = -FORCE_MAG
                    elif action_item == 1: force = 0.0
                    else: force = FORCE_MAG

                    # 存活奖励
                    r1 = (ANGLE_LIMIT_RAD - abs(current_theta)) / ANGLE_LIMIT_RAD
                    reward = r1 + 0.1

                current_episode_reward += reward

                # B. 发送指令
                server.send_action(force, stop_signal)

                # C. 获取下一步状态
                next_state_np = server.get_state()

                # --- 关键：检测连接是否断开 (Episode 结束) ---
                if next_state_np is None:
                    # 如果还没触发 done=True 也可以在这里给个结束奖励
                    break # 跳出内层循环，进入下一次 Episode

                next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
                reward_tensor = torch.tensor([reward], device=device)

                # D. 存储与训练
                memory.push(state, action, reward_tensor, next_state)
                optimize_model(memory, policy_net, target_net, optimizer)

                # 更新状态
                state = next_state
                state_np = next_state_np
                steps_done += 1

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                # 简单绘图数据收集
                angle_history.append(current_theta)

                # 如果我们主动发了 Stop 信号，Simulink 应该会在几步内停止
                # 我们继续循环直到 receive 返回 None

            # ==========================================
            # Episode 结束处理
            # ==========================================
            server.close_current_connection()
            episode_count += 1
            episode_rewards.append(current_episode_reward)

            print(f"Episode {episode_count} Finished. Total Reward: {current_episode_reward:.2f}, Epsilon: {EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY):.3f}")

            # 更新绘图 (每回合画一次)
            # 画角度 (最后一次回合的)
            angle_line.set_ydata(angle_history)
            angle_line.set_xdata(range(len(angle_history)))
            ax1.set_xlim(0, max(len(angle_history), 10))

            # 画总奖励历史
            reward_line.set_ydata(episode_rewards)
            reward_line.set_xdata(range(len(episode_rewards)))
            ax2.set_xlim(0, max(len(episode_rewards), 10))
            ax2.set_ylim(min(episode_rewards)-10, max(episode_rewards)+10)

            plt.draw()
            plt.pause(0.01)

            # 定期保存
            if episode_count % 50 == 0:
                torch.save(policy_net.state_dict(), 'cartpole_dqn_autosave.pth')

    except KeyboardInterrupt:
        print("\nTraining stopped manually.")
    finally:
        server.shutdown()
        plt.ioff()
        plt.show()
        torch.save(policy_net.state_dict(), 'cartpole_dqn_final.pth')
        print("Final Model saved.")
