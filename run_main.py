import json
import os
import re
import utils

import torch
from loguru import logger
import numpy as np
from agent import agent
from env import Environment
from changerou import Changerou
from config import parse_arguments
import PN_data_chat
from prompt import prompt
args = parse_arguments()

target_intersection = 'intersection_0'          # 目标交叉口名称
np.random.seed(args.seed)

env = Environment(gui='sumo-gui' if args.sumo_gui == 0 else 'sumo', target_intersection=target_intersection)   # 初始化环境
change_flow = Changerou()       # 初始化车流变更的类


net_list =  ['4_3_3_3_3', '4_3_3_3_2', '4_3_2_3_2', '4_3_2_2_3', '4_3_2_2_2', '4_2_2_2_2', '3_3_3_3', '3_3_3_2', '3_3_2_2', '3_2_2_2']


def parse_timing_values(response,net):
    """从响应文本中提取四个时间值"""
    if net.split('_')[0] == '3':
        pattern = r"""
        south_left:\s*([\d.]+)\s*seconds,.*?
        east_west_straight:\s*([\d.]+)\s*seconds,.*?
        east_left_straight:\s*([\d.]+)\s*seconds
        """
        matches = re.search(pattern, response, re.DOTALL | re.VERBOSE)
        if matches:
            return [int(round(float(x))) for x in matches.groups()]
        else:
            return [15, 15, 15]

    else:
        pattern = r"""
        north_south_straight:\s*([\d.]+)\s*seconds,.*?
        north_south_left:\s*([\d.]+)\s*seconds,.*?
        east_west_straight:\s*([\d.]+)\s*seconds,.*?
        east_west_left:\s*([\d.]+)\s*seconds
        """
        matches = re.search(pattern, response, re.DOTALL | re.VERBOSE)
        if matches:
            return [int(round(float(x))) for x in matches.groups()]
        else:
            return [15, 15, 15, 15]

def save_to_json(new_data, filename="time.json"):
    """将数据追加到JSON文件"""
    try:
        # 确保新数据是列表形式
        if not isinstance(new_data, list):
            new_data = [new_data]

        # 检查文件是否存在
        if os.path.exists(filename):
            # 读取现有数据
            with open(filename, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)

            # 确保现有数据是列表
            if not isinstance(existing_data, list):
                existing_data = [existing_data]
        else:
            # 文件不存在时创建空列表
            existing_data = []

        # 添加新数据
        existing_data.extend(new_data)

        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, ensure_ascii=False, indent=2)

        logger.info(f"成功追加 {len(new_data)} 条记录到 {filename}")
    except Exception as e:
        logger.error(f"保存数据到JSON文件失败: {str(e)}")
        # 尝试创建新文件
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(new_data, f, ensure_ascii=False, indent=2)
            logger.info(f"创建新文件 {filename} 并保存数据")
        except Exception as e2:
            logger.error(f"创建新文件失败: {str(e2)}")

# 修改后的保存函数
def save_results(net, traffic_states, responses, state):

    # 为每个response创建记录

    for i, response in enumerate(responses):
        # 创建基础记录
        record = {
            "output": response,
            "parsed_values": []
        }

        # 解析生成的数值
        values = parse_timing_values(response,net=net)
        record["parsed_values"] = values


    result_sequence = []

    for value in values:
        result_sequence.append(value)
        result_sequence.append(3)
        result_sequence.append(2)
    # 添加元数据
    result_data = {
        "responses": record,
    }

    # 保存结果
    # save_to_json(result_data)
    # print(result_sequence)

    return result_sequence


for task_net in net_list:
    ep_reward_list = []  # 存储每一回合的奖励
    ep_halting_list = []
    ep_avg_travel_time_list = []
    ep_travel_time_list = []
    ep_waiting_time_list = []
    ep_total_CO2_list = []
    ep_total_fuel_list = []
    ep_avg_speed_list = []
    ep_vehicle_count = []

    task_agent = agent(args.F_INTERSECTION + args.F_NEIGHBOR_OUT, args.h1_dim, args.h2_dim,
                        args.action_dim, args.max_action, args.memory_capacity, args.batch_size, args.lr_actor,
                        args.lr_critic, args.step_size_actor, args.step_size_critic, args.gamma_actor, args.gamma_critic,
                        args.policy_freq)  # 初始化智能体（涵盖了场景推断网络）
    for flow_id in range(5):

        task_agent.load_model(task_net,flow_id)
        task_agent.actor.eval()
        task_agent.critic.eval()
        for episode in range(args.max_episodes):  # 遍历每一个回合
            state_list, action_list, reward_list = [], [], []  # 用于存储当前回合的 transition
            halting_num, waiting_time = 0, 0
            total_CO2 = 0.0
            total_fuel = 0.0
            total_speed = 0.0
            vehicle_count = 0
            total_time = 3600
            # 统计当前回合的指标信息-每秒钟的停车数目和排队长度
            env.reset_light(net=task_net,flow_id=flow_id)  # 开启当前环境，可传入路网和车流信息

            cycle_index = 0  # 周期指示符，当其为0时，需要对目标交叉口进行重新配时
            for time in range(args.episode_time):  # 开启当前回合
                if cycle_index == 0:  # 当周期指示符为 0，说明需要重新配时
                    state = env.get_state(task_net)  # 获取当前时刻交叉口的车辆信息（论文中的[s^{c, veh}, s^{c, nei}]）
                    # action = task_agent.choose_action(state, task_net)
                    # if task_net.split('_')[0] == '3':
                    #     noise = np.random.normal(0, args.max_action * task_agent.var, size=args.action_dim - 1)
                    #     noise = np.append(noise, 0, axis=None)
                    # else:
                    #     noise = np.random.normal(0, args.max_action * task_agent.var, size=args.action_dim)
                    # action = (action + noise).clip(-args.max_action, args.max_action)
                    # timing_strategy2= utils.process_action(action, task_net)                    # 将输出的动作映射到配时空间，转换为配时方案
                    traffic_states = env.analyze_traffic_state(state, task_net)
                    # print("\nRL配时方案：",timing_strategy2)

                    timing_strategy1 = PN_data_chat.main(traffic_states,net=task_net)
                    timing_strategy = save_results(task_net, traffic_states, timing_strategy1, state)
                    print(timing_strategy)
                    reward = env.get_reward()  # 获取当前时刻将奖励值
                    reward_list.append(reward)  # 添加奖励信息


                    cycle_index = sum(timing_strategy)
                    total_time = total_time -cycle_index

                # 下述代码即是根据 timing_strategy 调整下一周期的目标交叉口配时方案
                for i in range(len(timing_strategy) + 1):
                    if sum(timing_strategy) - cycle_index < sum(timing_strategy[: i]):
                        if task_net.split('_')[0] == '4':  # 当为十字路口时，相位数为12
                            time_index = 12
                        else:
                            time_index = 9  # 三岔路口时，相位数为9
                        phase_index = i % time_index - 1 if i % time_index != 0 else (time_index - 1)  # 计算当前时刻应该处于哪个相位
                        break

                env.change_light(intersection=target_intersection, phase_index=phase_index)  # 设置当前时刻相位编号

                env.step()  # 当前环境向前运行1秒
                CO2, fuel, speed, count = env.get_emission_indicators()
                total_CO2 += CO2
                total_fuel += fuel  # fuel是平均值，需要乘以数量
                total_speed += speed * count  # speed是平均值，需要乘以数量
                vehicle_count += count

                travel_stats = env.get_travel_time_stats()
                ep_avg_travel_time_list.append(travel_stats['average_travel_time'])
                ep_travel_time_list.append(travel_stats['total_travel_time'])

                cycle_index -= 1  # 周期指示符-1
                # if episode % 10 == 0 and episode >= 10:           # 满足条件时进行一次数据统计
                halting_num += env.get_indicator_halting_num()  # 统计每一秒的车辆停车数目
                waiting_time += env.get_indicator_waiting_time()                            # 统计每一秒的车辆等待时间

            env.end()  # 结束当前回合
            avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
            # 打印当前回合数据
            ep_reward_list.append(sum(reward_list))
            ep_halting_list.append(halting_num)
            ep_waiting_time_list.append(waiting_time)
            ep_total_CO2_list.append(total_CO2)
            ep_total_fuel_list.append(total_fuel)
            ep_avg_speed_list.append(avg_speed)
            ep_vehicle_count.append(vehicle_count)
            # 打印当前回合数据
            print(f"{task_net} flow {flow_id} episode {episode}: "
                  f"reward: {sum(reward_list)}, "
                  f"halting: {halting_num}, "
                  f"waiting: {waiting_time}, "
                  f"CO2: {total_CO2:.2f}mg, "
                  f"fuel: {total_fuel:.2f}ml, "
                  f"speed: {avg_speed:.2f}m/s")

        # 保存当前flow_id的所有指标到TXT文件
    utils.text_save(f'indicator_dataP/1reward_{task_net}_ours.txt', ep_reward_list)
    utils.text_save(f'indicator_dataP/1halting_num_{task_net}_ours.txt', ep_halting_list)
    utils.text_save(f'indicator_dataP/1waiting_time_{task_net}_ours.txt', ep_waiting_time_list)
    utils.text_save(f'indicator_dataP/1total_CO2_{task_net}_ours.txt', ep_total_CO2_list)
    utils.text_save(f'indicator_dataP/1total_fuel_{task_net}_ours.txt', ep_total_fuel_list)
    utils.text_save(f'indicator_dataP/pavg_speed_{task_net}_ours.txt', ep_avg_speed_list)
    utils.text_save(f'indicator_dataP/pvehicle_count_{task_net}_ours.txt', ep_vehicle_count)
