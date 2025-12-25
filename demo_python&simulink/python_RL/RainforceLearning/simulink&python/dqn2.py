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

# 训练终止与收敛配置 (新功能)
MAX_EPISODES = 2000         # 最大训练回合数 (防止无限运行)
TARGET_AVG_REWARD = 400.0   # [目标分数] 当最近50回合平均分超过此值时，认为收敛并停止
                            # 估算逻辑: 1步约1.1分。400分约等于坚持3.6秒(360步)不倒。
                            # 您可以根据实际情况调整这个值。

# DQN 超参数
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 5000
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

# ================= 3. 通信服务器 (支持重连) =================
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
        print(f"Server Ready. Ports: {PORT_RECV} (Recv), {PORT_SEND} (Send)")

    def wait_for_connection(self):
        """阻塞等待 Simulink 连接"""
        print(">>> Waiting for Simulink simulation...")
        self.conn_recv, _ = self.sock_recv.accept()
        self.conn_send, _ = self.sock_send.accept()
        # 握手
        self.send_action(0.0, 0.0)

    def get_state(self):
        if self.conn_recv is None: return None
        try:
            data = self.conn_recv.recv(32) # 4 * double
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
    next_state_values = target_net(next_state_batch).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# ================= 5. 主循环 =================
if __name__ == "__main__":
    server = SimulinkServer()

    # 初始化网络
    policy_net = DQN(4, 3).to(device)
    target_net = DQN(4, 3).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(MEMORY_SIZE)

    # 绘图配置
    plt.ion()
    # 创建一个宽一点的图，方便保存
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    angle_line, = ax1.plot([], [], 'b-', linewidth=1.5)
    ax1.set_title("Real-time Angle (Current Episode)")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_ylabel("Angle (rad)")
    ax1.grid(True)

    reward_line, = ax2.plot([], [], 'r.-', linewidth=1.5, markersize=3)
    ax2.set_title(f"Training Progress (Moving Avg Window: 50)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.grid(True)

    # 增加一条“目标分数线”
    ax2.axhline(y=TARGET_AVG_REWARD, color='g', linestyle='--', label='Target Reward')
    ax2.legend()

    steps_done = 0
    episode_count = 0
    episode_rewards = [] # 记录每个回合的总分

    print(f"Training Config: Max Episodes={MAX_EPISODES}, Target Avg Reward={TARGET_AVG_REWARD}")

    try:
        # --- 外层循环：控制 Episode ---
        while episode_count < MAX_EPISODES:
            server.wait_for_connection()

            state_np = server.get_state()
            if state_np is None:
                server.close_current_connection()
                continue

            state = torch.tensor(state_np, dtype=torch.float32, device=device).unsqueeze(0)
            current_ep_reward = 0
            angle_history = []

            # --- 内层循环：Step ---
            while True:
                # 状态获取
                curr_x, curr_theta = state_np[0], state_np[1]

                # 失败判断
                done = False
                stop_signal = 0.0
                if abs(curr_theta) > ANGLE_LIMIT_RAD or abs(curr_x) > POS_LIMIT:
                    stop_signal = 1.0 # 触发重置
                    reward = -10.0
                    done = True
                    # 失败时动作为空，给一个占位符
                    action = torch.tensor([[1]], device=device, dtype=torch.long)
                    force = 0.0
                else:
                    action = select_action(state, steps_done, policy_net)
                    # 动作映射
                    act_item = action.item()
                    force = -FORCE_MAG if act_item == 0 else (FORCE_MAG if act_item == 2 else 0.0)

                    # 奖励计算 (存活越久分越高，角度越小分越高)
                    r_angle = (ANGLE_LIMIT_RAD - abs(curr_theta)) / ANGLE_LIMIT_RAD
                    reward = r_angle + 0.1 # 基础分0.1保证只要活着就是正反馈

                current_ep_reward += reward

                # 发送与接收
                server.send_action(force, stop_signal)
                next_state_np = server.get_state()

                if next_state_np is None: break # Episode 结束

                # 训练
                next_state = torch.tensor(next_state_np, dtype=torch.float32, device=device).unsqueeze(0)
                reward_t = torch.tensor([reward], device=device)
                memory.push(state, action, reward_t, next_state)
                optimize_model(memory, policy_net, target_net, optimizer)

                state = next_state
                state_np = next_state_np
                steps_done += 1

                if steps_done % TARGET_UPDATE == 0:
                    target_net.load_state_dict(policy_net.state_dict())

                angle_history.append(curr_theta)

            # --- Episode 结束处理 ---
            server.close_current_connection()
            episode_count += 1
            episode_rewards.append(current_ep_reward)

            # 计算平均分 (最近50局)
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 0 else current_ep_reward

            print(f"Episode {episode_count}: Reward={current_ep_reward:.1f}, Avg(50)={avg_reward:.1f}, Epsilon={EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY):.3f}")

            # 实时绘图更新
            if episode_count % 1 == 0: # 每局都更一次图
                # 图1: 角度
                angle_line.set_ydata(angle_history)
                angle_line.set_xdata(range(len(angle_history)))
                ax1.set_xlim(0, max(len(angle_history), 10))

                # 图2: 奖励历史
                reward_line.set_ydata(episode_rewards)
                reward_line.set_xdata(range(len(episode_rewards)))
                ax2.set_xlim(0, max(len(episode_rewards), 10))
                ax2.set_ylim(min(min(episode_rewards), -20), max(max(episode_rewards), 20))
                ax2.set_title(f"Reward History (Ep {episode_count}, Avg: {avg_reward:.1f})")

                plt.draw()
                plt.pause(0.01)

            # ================= [新功能] 终止条件判断 =================
            # 条件1: 达到收敛目标
            if len(episode_rewards) >= 50 and avg_reward >= TARGET_AVG_REWARD:
                print(f"\n>>>>>>>> SUCCESS: Model Converged! Avg Reward {avg_reward:.1f} >= {TARGET_AVG_REWARD}")
                print(">>>>>>>> Stopping training automatically.")
                break # 跳出主循环

    except KeyboardInterrupt:
        print("\nTraining stopped manually.")

    finally:
        # ================= [新功能] 保存结果 =================
        print("\nSaving results...")

        # 1. 保存模型参数
        torch.save(policy_net.state_dict(), 'cartpole_dqn_final.pth')
        print("-> Model saved as 'cartpole_dqn_final.pth'")

        # 2. 保存奖励曲线图
        # 重新绘制一张干净的静态图用于保存
        plt.ioff() # 关闭交互模式
        save_fig, save_ax = plt.subplots(figsize=(12, 6))
        save_ax.plot(episode_rewards, label='Episode Reward', alpha=0.6)

        # 计算并绘制移动平均线
        if len(episode_rewards) >= 10:
            moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
            save_ax.plot(range(9, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (10)')

        save_ax.set_title(f"Training Result: Total Episodes {episode_count}")
        save_ax.set_xlabel("Episode")
        save_ax.set_ylabel("Reward")
        save_ax.legend()
        save_ax.grid(True)

        plot_filename = 'training_result.png'
        save_fig.savefig(plot_filename, dpi=300)
        print(f"-> Reward plot saved as '{plot_filename}'")

        server.shutdown()
        print("Done.")
