import torch
import numpy as np
from operator import itemgetter
import torch.nn.functional as F
from config import parse_arguments
from layers import ActorNet, CriticNet

args = parse_arguments()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

class agent(object):
    """智能体的类"""
    def __init__(self, state_dim, h1_dim, h2_dim, action_dim, max_action, memory_capacity, batch_size, lr_actor,
                 lr_critic, step_size_actor, step_size_critic, gamma_actor, gamma_critic, policy_freq):
        self.s_dim = state_dim                      # 状态维度 - 输入actor网络全连接层的状态维度
        self.h1_dim = h1_dim                        # 隐藏层1维度
        self.h2_dim = h2_dim                        # 隐藏层2维度
        self.a_dim = action_dim                     # 动作维度
        self.max_action = max_action                # 动作边界
        self.memory_capacity = memory_capacity      # 记忆库大小
        self.memory = dict()                        # 记忆库
        self.pointer = 0                            # 记忆库存储数据指针
        self.var = args.var                         # 智能体的动作探索因子
        self.batch_size = batch_size                # 批处理大小
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic

        # 定义actor网络和critic网络
        self.actor = ActorNet(state_dim=state_dim, h1_dim=h1_dim, h2_dim=h2_dim, action_dim=action_dim, max_action=max_action, task_net='3')
        self.actor_target = ActorNet(state_dim=state_dim, h1_dim=h1_dim, h2_dim=h2_dim, action_dim=action_dim, max_action=max_action, task_net='3')
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.scheduler_actor = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=step_size_actor,
                                                               gamma=gamma_actor)

        self.critic = CriticNet(state_dim=state_dim, h1_dim=h1_dim, h2_dim=h2_dim, action_dim=action_dim)
        self.critic_target = CriticNet(state_dim=state_dim, h1_dim=h1_dim, h2_dim=h2_dim, action_dim=action_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.scheduler_critic = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=step_size_critic,
                                                                gamma=gamma_critic)

        self.max_action = max_action
        self.policy_freq = policy_freq              # 策略网络延迟更新参数
        self.total_it = 0                           # 总迭代次数


    def store_transition(self, s, a, r, s_):
        """存储训练数据"""
        index = self.pointer % self.memory_capacity           # 样本数据编号
        self.memory[index] = [s, a, r, s_]                    # 存储数据
        self.pointer += 1                                     # 记忆库指示符+1

    def choose_action(self, state, task_net):
        """选择动作"""
        state_veh = np.array(state[0]).reshape(-1, args.N_PHASE, args.N_LANE, args.N_VEHICLE, args.F_VEHICLE)  # 当前时刻目标交叉口的车辆状态
        state_nei = np.array(state[1]).reshape(-1, args.N_DIRECTION, args.F_NEIGHBOR_IN)                       # 当前时刻邻居交叉口状态
        # 转换为tensor格式，作为模型输入
        x_v = torch.tensor(state_veh, dtype=torch.float)
        x_n = torch.tensor(state_nei, dtype=torch.float)

        if task_net.split('_')[0] == '3':
            net = '3'
            phase_state = torch.eye(4)
            phase_state[-1, :] = 0                          # 三岔口的相位编码方式为 [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,0]]
            intersection_state = torch.tensor([1, 0])       # 三岔口的交叉口编码方式为 [1,0]
        else:
            net = '4'
            phase_state = torch.eye(4)                      # 十字路口的相位编码方式为 [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
            intersection_state = torch.tensor([0, 1])       # 十字路口的交叉口编码方式为 [0,1]

        batch_size = x_v.shape[0]

        # 根据状态信息制定动作
        action = self.actor(x_v, torch.ones(batch_size * args.N_PHASE * args.N_LANE, 2),
                            phase_state.repeat(batch_size, 1), intersection_state.repeat(batch_size, 1), x_n, net)
        return np.array(action.unsqueeze(0).flatten().detach())

    def learn(self, task_net):
        """智能体训练"""
        self.total_it += 1
        indices = np.random.choice(len(self.memory), size=self.batch_size)      # 随机选择训练样本编号
        batch_trans = itemgetter(*indices)(self.memory)                         # 从记忆库中抽取训练样本
        state_veh = torch.from_numpy((np.array([x[0][0] for x in batch_trans]))).float().squeeze(1)
        state_nei = torch.from_numpy((np.array([x[0][1] for x in batch_trans]))).float().squeeze(1)

        action = torch.from_numpy((np.array([x[1] for x in batch_trans]))).float()
        reward = torch.FloatTensor([x[2] for x in batch_trans])
        next_state_veh = torch.from_numpy((np.array([x[3][0] for x in batch_trans]))).float().squeeze(1)
        next_state_nei = torch.from_numpy((np.array([x[3][1] for x in batch_trans]))).float().squeeze(1)

        # 判断为何种交叉口
        if task_net.split('_')[0] == '3':
            net = '3'
            phase_state = torch.eye(4)
            phase_state[-1, :] = 0
            intersection_state = torch.tensor([1, 0])
        else:
            net = '4'
            phase_state = torch.eye(4)
            intersection_state = torch.tensor([0, 1])

        b_size = len(state_veh)
        x_lane = torch.ones(args.N_PHASE * args.N_LANE * b_size, 2)
        x_phase = phase_state.repeat(args.N_PHASE // 4 * b_size, 1)
        x_intersection = intersection_state.repeat(b_size, 1)


        with torch.no_grad():
            noise = (torch.randn_like(action) * args.noise_sigma).clamp(-args.noise_c, args.noise_c)
            next_action = (self.actor_target(next_state_veh, x_lane, x_phase, x_intersection, next_state_nei, net)
                           + noise).clamp(-self.max_action, self.max_action)

            # 计算 target Q
            target_Q1, target_Q2 = self.critic_target(next_state_veh, x_lane, x_phase, x_intersection,
                                                      next_state_nei, next_action)
            target_Q = torch.min(target_Q1, target_Q2)    # TD3中选择较小值
            target_Q = reward.reshape(-1, 1) + (args.reward_gamma * target_Q).detach()

        current_Q1, current_Q2 = self.critic(state_veh, x_lane, x_phase, x_intersection, state_nei, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward(retain_graph=True)
        self.critic_optimizer.step()
        self.scheduler_critic.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state_veh, x_lane, x_phase, x_intersection, state_nei,
                                         self.actor(state_veh, x_lane, x_phase, x_intersection, state_nei, net)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optimizer.step()

            # 采用软更新方式更新目标网络参数
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            self.scheduler_actor.step()

    # def save_model(self, episode):
    #     """保存模型"""
    #     torch.save(self.actor.state_dict(), "model/actor_" + str(episode) + ".pth")
    #     torch.save(self.critic.state_dict(), "model/critic_" + str(episode) + ".pth")
    #     # torch.save(self.actor_optimizer.state_dict(), "actor_optimizer_" + str(episode))
    #
    # def load_model(self, episode):
    #     """加载模型"""
    #     self.actor.load_state_dict(torch.load("model/actor_" + str(episode) + ".pth", weights_only=True))
    #     self.critic.load_state_dict(torch.load("model/critic_" + str(episode) + ".pth", weights_only=True))
    # def save_model(self, episode):
    #     """保存模型"""
    #     torch.save(self.actor, "model/actor_" + str(episode) + ".pth")
    #     torch.save(self.critic, "model/critic_" + str(episode) + ".pth")
    #
    # def load_model(self, episode):
    #     """加载模型"""
    #     self.actor = torch.load("model/actor_" + str(episode) + ".pth")
    #     self.critic = torch.load("model/critic_" + str(episode) + ".pth")
    def save_model(self, episode, test_net, is_best=False):
        """保存模型"""
        import os
        import glob

        if is_best:
            # 删除该路网之前保存的所有最佳模型文件
            actor_pattern = f"model/actor_{test_net}_*_best_rou.pth"
            critic_pattern = f"model/critic_{test_net}_*_best_rou.pth"

            for f in glob.glob(actor_pattern):
                os.remove(f)
            for f in glob.glob(critic_pattern):
                os.remove(f)

            # 保存新的最佳模型，文件名包含回合数
            torch.save(self.actor.state_dict(), f"model/actor_{test_net}_{episode}_best_rou.pth")
            torch.save(self.critic.state_dict(), f"model/critic_{test_net}_{episode}_best_rou.pth")
        else:
            # 常规保存带回合数
            torch.save(self.actor.state_dict(), f"model/actor_{test_net}_{episode}.pth")
            torch.save(self.critic.state_dict(), f"model/critic_{test_net}_{episode}.pth")
        # torch.save(self.actor_optimizer.state_dict(), "actor_optimizer_" + str(episode))

    def load_model(self, test_net,flow_id=0):
        """加载指定路网的最新最佳模型"""
        import glob

        # 构建通配符匹配路径
        actor_pattern = f"models/actor_{test_net}_*_best_rou{flow_id}.pth"
        critic_pattern = f"models/critic_{test_net}_*_best_rou{flow_id}.pth"

        # 查找所有匹配的模型文件
        actor_files = glob.glob(actor_pattern)
        critic_files = glob.glob(critic_pattern)

        if not actor_files or not critic_files:
            raise FileNotFoundError(f"未找到 {test_net} 路网的模型文件")

        # 按回合数排序，选择最新的文件
        latest_actor = max(actor_files, key=lambda x: int(x.split("_")[-3]))  # 解析文件名中的回合数
        latest_critic = max(critic_files, key=lambda x: int(x.split("_")[-3]))

        # 加载模型参数
        self.actor.load_state_dict(torch.load(latest_actor, weights_only=True))
        self.critic.load_state_dict(torch.load(latest_critic, weights_only=True))

if __name__ == '__main__':
    batch_size = 2
    agent = agent(args.F_INTERSECTION+args.F_NEIGHBOR_OUT, args.h1_dim, args.h2_dim, args.action_dim, args.max_action,
                  args.memory_capacity, args.batch_size, args.lr_actor, args.lr_critic, args.step_size_actor, args.step_size_critic,
                  args.gamma_actor, args.gamma_critic, args.policy_freq)
    # state = torch.randn(size=(1, args.N_PHASE, args.N_LANE, args.N_VEHICLE, args.F_VEHICLE))
    state_veh = np.random.rand(1, args.N_PHASE, args.N_LANE, args.N_VEHICLE, args.F_VEHICLE)
    state_nei = np.random.rand(1, args.N_DIRECTION, args.F_NEIGHBOR_IN)
    action = agent.choose_action(state_veh, state_nei)
    reward = 1
    state_veh_next = np.random.rand(1, args.N_PHASE, args.N_LANE, args.N_VEHICLE, args.F_VEHICLE)
    state_nei_next = np.random.rand(1, args.N_DIRECTION, args.F_NEIGHBOR_IN)
    state = [state_veh, state_nei]
    state_next = [state_veh, state_nei]

    for _ in range(10):
        agent.store_transition(state, action, reward, state_next)
    agent.learn()
    print(action)