import socket
import struct
import matplotlib.pyplot as plt

# ================= 配置区 =================
HOST = '127.0.0.1'
PORT_RECV = 50000   # [Simulink -> Python]
PORT_SEND = 50001   # [Python -> Simulink]
DT = 0.01           # 仿真步长 (秒)

# ================= 1. 建立双通道连接 =================
print(f"1. [接收通道] 监听端口 {PORT_RECV}...")
sock_recv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_recv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_recv.bind((HOST, PORT_RECV))
sock_recv.listen(1)

print(f"2. [发送通道] 监听端口 {PORT_SEND}...")
sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock_send.bind((HOST, PORT_SEND))
sock_send.listen(1)

print(">>> 请现在运行 Simulink (点击 Run) <<<")

# 按顺序等待连接
conn_recv, addr_recv = sock_recv.accept()
print(f"   [接收通道] 已连接: {addr_recv}")

conn_send, addr_send = sock_send.accept()
print(f"   [发送通道] 已连接: {addr_send}")

# --- 发送握手包 (0.0) ---
conn_send.send(struct.pack('<d', 0.0))

# ================= 2. 绘图初始化 =================
plt.ion()
fig, ax = plt.subplots(figsize=(10, 5))

# 初始化一条蓝线
line, = ax.plot([], [], 'b-', linewidth=1.5, label='Signal History')

ax.set_title(f"Full History Signal (dt={DT}s)")
ax.set_xlabel("Time (seconds)")
ax.set_ylabel("Value")

# --- Y轴范围设置 ---
# 如果你知道大概范围，建议固定，比如 -10 到 10，这样看起来不晕
ax.set_ylim(-10, 10)
# 如果你想让Y轴也自动缩放，请注释掉上面这行，并在循环里使用 ax.relim() 和 ax.autoscale_view()

ax.grid(True)
ax.legend()

# 数据缓存 (无限增长)
time_buffer = []
data_buffer = []

print("--- 开始记录完整波形 ---")

try:
    step_index = 0
    while True:
        # A. 接收
        raw_data = conn_recv.recv(8)
        if not raw_data:
            print("Simulink 断开连接")
            break

        val_in = struct.unpack('<d', raw_data)[0]

        # B. 计算时间并存储
        current_time = step_index * DT

        # --- 核心修改：只添加，不删除 (No pop) ---
        time_buffer.append(current_time)
        data_buffer.append(val_in)

        step_index += 1

        # C. 绘图 (每 10 步刷新一次，数据多了可以设大一点防止卡顿)
        if step_index % 10 == 0:
            line.set_xdata(time_buffer)
            line.set_ydata(data_buffer)

            # --- 核心修改：X 轴始终从 0 到 当前时间 ---
            if len(time_buffer) > 0:
                ax.set_xlim(0, current_time + 0.1)

            plt.draw()
            plt.pause(0.001)

        # D. 发送控制量 (反转信号测试)
        val_out = -1.0
        conn_send.send(struct.pack('<d', val_out))

except KeyboardInterrupt:
    print("手动停止")
finally:
    conn_recv.close()
    conn_send.close()
    sock_recv.close()
    sock_send.close()
    plt.ioff()
    plt.show()
