% =========================================================================
% 强化学习训练主控脚本 (强制重置版)
% 功能：循环运行 Simulink 模型，并在每回合前强制刷新初始角度
% =========================================================================

% 1. 模型名称配置 (请修改为您实际的文件名，不带 .slx)
model_name = 'cartpole_sim'; 

% 2. 加载模型
if ~bdIsLoaded(model_name)
    load_system(model_name);
    disp(['>>> Model ''', model_name, ''' loaded.']);
end

% ================= [关键修复 1] 关闭快速重启 =================
% FastRestart 会缓存编译结果和初始状态，导致 workspace 里的变量修改无效
% 必须将其关闭，强制 Simulink 每次都重新读取 init_theta
set_param(model_name, 'FastRestart', 'off');

% ================= [关键修复 2] 强制关闭状态加载 =================
% 防止 Simulink 自动加载上一次仿真结束时的状态快照
set_param(model_name, 'LoadInitialState', 'off');
set_param(model_name, 'SaveFinalState', 'off');

% 设置最大训练回合
max_episodes = 5000;

disp('-----------------------------------------------------------');
disp('>>> RL Training Started.');
disp('>>> Check Python console for training progress.');
disp('>>> Press Ctrl+C in THIS window to stop manually.');
disp('-----------------------------------------------------------');

for episode = 1:max_episodes
    
    % ---------------------------------------------------------------------
    % A. 生成并写入新的初始条件
    % ---------------------------------------------------------------------
    % 随机生成一个很小的初始角度 (例如 -0.05 到 0.05 弧度)
    % 注意：不要设得太大，否则开局就倒了
    init_angle = (rand() - 0.5) * 0.1; 
    
    % [核心步骤] 将变量写入基础工作区 (Base Workspace)
    % Simulink 模型运行时会从这里读取 'init_theta'
    assignin('base', 'init_theta', init_angle);
    
    % 打印信息以便调试 (看看角度变了没有)
    fprintf('[Ep %d] Setting Init Theta = %.4f rad... ', episode, init_angle);
    
    % ---------------------------------------------------------------------
    % B. 运行单次仿真
    % ---------------------------------------------------------------------
    try
        % 启动仿真 (阻塞模式，直到 Python 发送停止信号或超时)
        sim(model_name);
        fprintf('Done.\n');
        
    catch ME
        % -----------------------------------------------------------------
        % C. 异常处理 (防止脚本因为 Simulink 的 Stop 命令而崩溃)
        % -----------------------------------------------------------------
        % 如果错误是因为我们主动调用的 stop_simulation_func，则是正常的
        if contains(ME.message, 'Stop') || contains(ME.message, 'assertion')
             fprintf('Stopped by Python.\n');
        else
             % 如果是其他错误 (如 TCP 连接失败)，则打印红色警告
             fprintf(2, '\n[ERROR] Simulation Error: %s\n', ME.message);
             
             % 尝试强制停止，防止卡死
             try
                set_param(model_name, 'SimulationCommand', 'stop');
             catch
             end
             
             disp('>>> Waiting 2 seconds before retry...');
             pause(2);
        end
    end
    
    % ---------------------------------------------------------------------
    % D. 资源清理与延时
    % ---------------------------------------------------------------------
    % 短暂暂停，确保 TCP 端口被操作系统释放，防止"端口被占用"错误
    pause(0.1); 
    
end

disp('>>> Training Finished.');