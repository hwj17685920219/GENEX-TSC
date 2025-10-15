import argparse


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--sumo_gui', type=int, default=0, help="是否显示SUMO图形界面")
    parser.add_argument('--N_VEHICLE', type=int, default=20, help="车辆节点数")
    parser.add_argument('--N_LANE', type=int, default=2, help="车道节点数")
    parser.add_argument('--N_PHASE', type=int, default=4, help="相位节点数")

    parser.add_argument('--F_LANE_AGG', type=int, default=10, help="车道节点聚合特征")
    parser.add_argument('--F_PHASE_AGG', type=int, default=22, help="相位节点聚合特征")
    parser.add_argument('--F_INTERSECTION_AGG', type=int, default=44, help="交叉口节点聚合特征")

    parser.add_argument('--F_VEHICLE', type=int, default=3, help="车辆节点特征")
    parser.add_argument('--F_LANE', type=int, default=12, help="车道节点特征")
    parser.add_argument('--F_PHASE', type=int, default=24, help="相位节点特征")
    parser.add_argument('--F_INTERSECTION', type=int, default=48, help="交叉口节点特征")

    parser.add_argument('--F_NEIGHBOR_IN', type=int, default=6, help="邻居交叉口特征输入维度")
    parser.add_argument('--F_NEIGHBOR_OUT', type=int, default=8, help="邻居交叉口特征输出维度")
    parser.add_argument('--N_DIRECTION', type=int, default=4, help="邻居交叉口数目")

    parser.add_argument('--alpha', type=float, default=0.2, help="GAT层leakyrelu参数")
    parser.add_argument('--dropout', type=float, default=0.2, help="GAT层dropout参数")
    parser.add_argument('--num_heads', type=int, default=4, help="特征聚合层中GAT的注意力头数")

    parser.add_argument('--h1_dim', type=int, default=512, help="第一层隐藏层维度")
    parser.add_argument('--h2_dim', type=int, default=256, help="第二层隐藏层维度")
    parser.add_argument('--action_dim', type=int, default=4, help="输出动作维度")

    parser.add_argument('--max_action', type=int, default=1, help="动作边界")
    parser.add_argument('--memory_capacity', type=int, default=5000, help="记忆库大小")
    parser.add_argument('--batch_size', type=int, default=128, help="批处理大小")                            # 128

    parser.add_argument('--lr_actor', type=float, default=0.001, help="actor网络学习率")
    parser.add_argument('--lr_critic', type=float, default=0.001, help="critic网络学习率")

    parser.add_argument('--step_size_actor', type=int, default=200, help="actor网络学习率衰减步数")   # 100
    parser.add_argument('--step_size_critic', type=int, default=200, help="critic网络学习率衰减步数") # 100
    parser.add_argument('--gamma_actor', type=float, default=0.5, help="actor网络学习率衰减大小")
    parser.add_argument('--gamma_critic', type=float, default=0.5, help="critic网络学习率衰减大小")

    parser.add_argument('--policy_freq', type=int, default=2, help="actor网络延迟更新参数")
    parser.add_argument('--reward_gamma', type=float, default=0.99, help="Q函数的折扣因子")
    parser.add_argument('--tau', type=float, default=0.005, help="目标网络的软交换系数")

    parser.add_argument('--noise_sigma', type=float, default=0.2, help='目标网络噪声探索因子')
    parser.add_argument('--noise_c', type=float, default=0.5, help='目标网络噪声探索边界')
    parser.add_argument('--var', type=float, default=0, help='智能体的动作探索因子')      # 0.5

    parser.add_argument('--kappa', type=float, default=0.15, help='奖励函数系数1')
    parser.add_argument('--rho', type=int, default=2, help='奖励函数系数2')
    parser.add_argument('--varsigma', type=int, default=60, help='奖励参考等待时长')

    parser.add_argument('--max_episodes', type=int, default=1, help='仿真回合数')
    parser.add_argument('--episode_time', type=int, default=3600, help='每回合持续时间')
    parser.add_argument('--start_learn', type=int, default=256, help='智能体开始训练时的指示符')               # 256

    parser.add_argument('--max_speed', type=float, default=13.89, help='车辆速度归一化系数')
    parser.add_argument('--max_waiting_time', type=int, default=100, help='车辆等待时间归一化系数')
    parser.add_argument('--max_distance2intersection', type=int, default=200, help='车辆距停止线距离归一化系数')

    parser.add_argument('--min_num', type=float, default=1e-5, help='一个很小的数')
    parser.add_argument('--MIN_GREEN', type=int, default=5, help='最短绿灯时间')
    parser.add_argument('--GREEN_BASE', type=int, default=30, help='最长绿灯时间')
    parser.add_argument('--YELLOW_TIME', type=int, default=3, help='黄灯持续时间')
    parser.add_argument('--RED_TIME', type=int, default=2, help='红灯持续时间')

    parser.add_argument('--inference_timesteps', type=int, default=5, help='推断网络的时间步长')
    parser.add_argument('--inference_output_dim', type=int, default=4, help='推断网络的场景编码维度')
    parser.add_argument('--N_positive', type=int, default=8, help='正样本数')
    parser.add_argument('--N_mid_1', type=int, default=16, help='中间样本1数')
    parser.add_argument('--N_mid_2', type=int, default=32, help='中间样本2数')
    parser.add_argument('--N_negative', type=int, default=64, help='负样本数')
    parser.add_argument('--TAU', type=float, default=0.3, help='推断网络温度系数')
    parser.add_argument('--lambda_positive', type=float, default=2, help='正样本损失系数')
    parser.add_argument('--lambda_mid_1', type=float, default=1, help='中间样本1损失系数')
    parser.add_argument('--lambda_mid_2', type=float, default=0.5, help='中间样本2损失系数')
    parser.add_argument('--lambda_negative', type=float, default=1, help='负样本损失系数')

    parser.add_argument('--task_time', type=int, default=900, help='推断网络中一个场景任务的时长')
    parser.add_argument('--timesteps_inference', type=int, default=5000, help='推断网络一个任务的时长')
    parser.add_argument('--non_primary_flow_low', type=int, default=10, help='非主干道车流生成左区间')
    parser.add_argument('--non_primary_flow_high', type=int, default=50, help='非主干道车流生成右区间')
    parser.add_argument('--primary_flow_low_4', type=int, default=50, help='十字路口主干道车流生成左区间')
    parser.add_argument('--primary_flow_high_4', type=int, default=200, help='十字路口主干道车流生成右区间')
    parser.add_argument('--primary_flow_low_3', type=int, default=150, help='三岔路口主干道车流生成左区间')
    parser.add_argument('--primary_flow_high_3', type=int, default=250, help='三岔路口主干道车流生成右区间')
    parser.add_argument('--n_positive_sample', type=int, default=5, help='当前样本的前后5个样本都视为其正样本')
    parser.add_argument('--inference_epoch', type=int, default=30, help='推断网络训练回合数')
    parser.add_argument('--inference_batch_size', type=int, default=64, help='推断网络的批处理大小')

    parser.add_argument('--task_num', type=int, default=3, help='元训练过程中每次采样的子任务数')
    parser.add_argument('--episode_task_num', type=int, default=5, help='元训练时每个子任务的训练回合数')
    parser.add_argument('--trajectory_len', type=int, default=64, help='元训练时每个任务的轨迹长度')
    parser.add_argument('--episode_meta_learn', type=int, default=1000, help='元训练过程回合数')



    return parser.parse_args()