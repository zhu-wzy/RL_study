import socket
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import math
import time
import os

# ================= 配置参数 =================
# 通信配置
HOST = '127.0.0.1'
PORT_RECV = 50000
PORT_SEND = 50001

# 物理与奖励配置
FORCE_MAG = 15.0
ANGLE_LIMIT_DEG = 25
ANGLE_LIMIT_RAD = ANGLE_LIMIT_DEG * (math.pi / 180)
POS_LIMIT = 1.0

# 训练终止配置
MAX_EPISODES = 3000         # PPO可能需要多一点回合数
TARGET_AVG_REWARD = 450.0   # 目标收敛分数

# PPO 超参数
K_EPOCHS = 10               # 每次更新循环多少次
EPS_CLIP = 0.2              # PPO截断范围
GAMMA = 0.99                # 折扣因子
LR_ACTOR = 0.0003           # Actor 学习率
LR_CRITIC = 0.001           # Critic 学习率
UPDATE_TIMESTEP = 2048      # 每收集多少步经验更新一次网络

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ================= 1. PPO 记忆库 =================
class PPOMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

# ================= 2. Actor-Critic 网络 =================
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor: 输出动作概率
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

        # Critic: 输出状态价值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        """选择动作"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):
        """评估动作（用于更新阶段）"""
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

# ================= 3. PPO 算法核心 =================
class PPO:
    def __init__(self, state_dim, action_dim):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': LR_ACTOR},
            {'params': self.policy.critic.parameters(), 'lr': LR_CRITIC}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        return action.item(), action_logprob.item()

    def update(self, memory):
        # 蒙特卡洛估计回报 (Monte Carlo Estimate)
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)

        # 归一化奖励
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # 转换旧数据为 Tensor
        old_states = torch.squeeze(torch.stack(memory.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(memory.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs, dim=0)).detach().to(device)

        # K 轮更新
        for _ in range(K_EPOCHS):
            # 评估旧状态和动作
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)

            # 计算比率 (ratios)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 计算优势 (advantages)
            advantages = rewards - state_values.detach()

            # PPO Loss 公式
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - EPS_CLIP, 1 + EPS_CLIP) * advantages

            # 总损失 = Actor损失 - 0.5*Critic损失 - 0.01*熵(鼓励探索)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # 梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())

# ================= 4. 通信服务器 (保持不变) =================
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
        print(">>> Waiting for Simulink simulation...")
        self.conn_recv, _ = self.sock_recv.accept()
        self.conn_send, _ = self.sock_send.accept()
        self.send_action(0.0, 0.0) # 握手

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

# ================= 5. 主循环 =================
if __name__ == "__main__":
    server = SimulinkServer()

    # 初始化 PPO
    state_dim = 4
    action_dim = 3
    ppo_agent = PPO(state_dim, action_dim)
    memory = PPOMemory()

    # 绘图配置
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    angle_line, = ax1.plot([], [], 'b-', linewidth=1.5)
    ax1.set_title("Real-time Angle (Current Episode)")
    ax1.set_ylim(-1.0, 1.0)
    ax1.set_ylabel("Angle (rad)")
    ax1.grid(True)

    reward_line, = ax2.plot([], [], 'r.-', linewidth=1.5, markersize=3)
    ax2.set_title(f"PPO Training Progress")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Total Reward")
    ax2.grid(True)
    ax2.axhline(y=TARGET_AVG_REWARD, color='g', linestyle='--', label='Target Reward')
    ax2.legend()

    time_step = 0
    episode_count = 0
    episode_rewards = []

    print(f"PPO Training Config: Update every {UPDATE_TIMESTEP} steps. Target Avg Reward={TARGET_AVG_REWARD}")

    try:
        while episode_count < MAX_EPISODES:
            server.wait_for_connection()

            state_np = server.get_state()
            if state_np is None:
                server.close_current_connection()
                continue

            current_ep_reward = 0
            angle_history = []

            # --- 内层循环：Step ---
            while True:
                # 1. PPO Action Selection
                action_idx, log_prob = ppo_agent.select_action(state_np)

                # 2. 物理映射
                if action_idx == 0: force = -FORCE_MAG
                elif action_idx == 1: force = 0.0
                else: force = FORCE_MAG

                # 3. 失败判断
                curr_x, curr_theta = state_np[0], state_np[1]
                done = False
                stop_signal = 0.0

                if abs(curr_theta) > ANGLE_LIMIT_RAD or abs(curr_x) > POS_LIMIT:
                    stop_signal = 1.0 # 重置
                    reward = -10.0
                    done = True
                else:
                    # 奖励函数
                    r_angle = (ANGLE_LIMIT_RAD - abs(curr_theta)) / ANGLE_LIMIT_RAD
                    reward = r_angle + 0.1

                # 4. 存储 PPO 数据 (状态转为Tensor存入)
                memory.states.append(torch.FloatTensor(state_np))
                memory.actions.append(torch.tensor(action_idx))
                memory.logprobs.append(torch.tensor(log_prob))
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

                # 5. 与环境交互
                server.send_action(force, stop_signal)
                next_state_np = server.get_state()

                current_ep_reward += reward
                time_step += 1
                angle_history.append(curr_theta)

                # 6. PPO 更新 (每 UPDATE_TIMESTEP 步更新一次)
                if time_step % UPDATE_TIMESTEP == 0:
                    print(f"  [PPO Update] Updating policy at step {time_step}...")
                    ppo_agent.update(memory)
                    memory.clear()
                    print("  [PPO Update] Done.")

                if next_state_np is None: break
                state_np = next_state_np

            # --- Episode 结束处理 ---
            server.close_current_connection()
            episode_count += 1
            episode_rewards.append(current_ep_reward)

            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) > 0 else current_ep_reward

            print(f"Episode {episode_count}: Reward={current_ep_reward:.1f}, Avg(50)={avg_reward:.1f}")

            # 绘图
            if episode_count % 1 == 0:
                angle_line.set_ydata(angle_history)
                angle_line.set_xdata(range(len(angle_history)))
                ax1.set_xlim(0, max(len(angle_history), 10))

                reward_line.set_ydata(episode_rewards)
                reward_line.set_xdata(range(len(episode_rewards)))
                ax2.set_xlim(0, max(len(episode_rewards), 10))
                ax2.set_ylim(min(min(episode_rewards), -20), max(max(episode_rewards), 20))

                plt.draw()
                plt.pause(0.01)

            # 终止条件
            if len(episode_rewards) >= 50 and avg_reward >= TARGET_AVG_REWARD:
                print(f"\n>>> SUCCESS: PPO Converged! Avg Reward {avg_reward:.1f}")
                break

    except KeyboardInterrupt:
        print("\nTraining stopped manually.")

    finally:
        print("\nSaving results...")
        torch.save(ppo_agent.policy.state_dict(), 'cartpole_ppo_final.pth')

        # 保存图片
        plt.ioff()
        save_fig, save_ax = plt.subplots(figsize=(12, 6))
        save_ax.plot(episode_rewards, label='Episode Reward', alpha=0.6)
        if len(episode_rewards) >= 10:
            moving_avg = np.convolve(episode_rewards, np.ones(10)/10, mode='valid')
            save_ax.plot(range(9, len(episode_rewards)), moving_avg, 'r-', linewidth=2, label='Moving Avg (10)')
        save_ax.set_title(f"PPO Training Result")
        save_ax.grid(True)
        save_fig.savefig('ppo_training_result.png', dpi=300)

        server.shutdown()
        print("Done.")
