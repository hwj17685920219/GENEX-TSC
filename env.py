import os
import sys
import utils
import optparse
import numpy as np
from config import parse_arguments

args = parse_arguments()

np.set_printoptions(threshold=np.inf)
np.random.seed(1)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME")

import traci
from sumolib import checkBinary

def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true", default=False, help="run the commandline version of sumo")

    options, args = optParser.parse_args()
    return options


class Environment(object):
    def __init__(self, gui, target_intersection):
        """初始化env类 param gui: 是否显示图形界面"""
        self.control = traci                    # 定义traci接口
        self.time = 0                           # 从时间0开始
        self.sumoBinary = checkBinary(gui)      # 是否显示界面
        self.target_intersection = target_intersection
        # self.intersec, self.net = None, None
        self.vehicle_entry_times = {}           # 车辆ID -> 进入时间
        self.vehicle_travel_times = {}          # 车辆ID -> 行程时间
        self.total_travel_time = 0.0            # 总行程时间
        self.vehicle_count = 0

    def reset_light(self, net,flow_id=0):
        """根据网络信息重置当前环境"""
        self.net = net
        self.vehicle_entry_times = {}
        self.vehicle_travel_times = {}
        self.total_travel_time = 0.0
        self.vehicle_count = 0

        f = open("env.sumocfg", 'r')
        new = []
        for line in f:
            new.append(line)
        new[5] = '        <net-file value="' + 'network/intersection_' + str(net) + '.net.xml"/>\n'
        new[6] = f'        <route-files value="test/rou/rou.rou{flow_id}_{str(net)}.xml"/>\n'
        f.close()
        f = open("env.sumocfg", 'w')
        for n in new:
            f.write(n)
        f.close()
        self.control.start([self.sumoBinary, "-c", "env.sumocfg"])

    def get_vehicle_id(self):
        return traci.vehicle.getIDList()

    def get_phase_lanes(self):
        """返回目标交叉口各相位对应的车道"""
        if self.net == '4_3_3_3_3':
            pl_dict = {0: ['road_2_0_1', 'road_4_0_1'], 3: ['road_2_0_2', 'road_4_0_2'],
                       6: ['road_1_0_1', 'road_3_0_1'], 9: ['road_1_0_2', 'road_3_0_2']}
        elif self.net == '4_3_3_3_2':
            pl_dict = {0: ['road_2_0_1', 'road_4_0_1'], 3: ['road_2_0_2', 'road_4_0_2'],
                       6: ['road_1_0_1', 'road_3_0_0'], 9: ['road_1_0_2', 'road_3_0_1']}
        elif self.net == '4_3_2_2_3':
            pl_dict = {0: ['road_2_0_1', 'road_4_0_0'], 3: ['road_2_0_2', 'road_4_0_1'],
                       6: ['road_1_0_1', 'road_3_0_0'], 9: ['road_1_0_2', 'road_3_0_1']}
        elif self.net == '4_3_2_3_2':
            pl_dict = {0: ['road_2_0_1', 'road_4_0_1'], 3: ['road_2_0_2', 'road_4_0_2'],
                       6: ['road_1_0_0', 'road_3_0_0'], 9: ['road_1_0_1', 'road_3_0_1']}
        elif self.net == '4_3_2_2_2':
            pl_dict = {0: ['road_2_0_0', 'road_4_0_0'], 3: ['road_2_0_1', 'road_4_0_1'],
                       6: ['road_1_0_1', 'road_3_0_0'], 9: ['road_1_0_2', 'road_3_0_1']}
        elif self.net == '4_2_2_2_2':
            pl_dict = {0: ['road_2_0_0', 'road_4_0_0'], 3: ['road_2_0_1', 'road_4_0_1'],
                       6: ['road_1_0_0', 'road_3_0_0'], 9: ['road_1_0_1', 'road_3_0_1']}

        elif self.net == '3_3_3_3':
            pl_dict = {0: ['road_4_0_1', 'road_4_0_2'], 3: [['road_3_0_0', 'road_3_0_1'], ['road_1_0_1', 'road_1_0_2']],
                       6: [['road_3_0_0', 'road_3_0_1'], ['road_3_0_2']]}
        elif self.net == '3_3_3_2':
            pl_dict = {0: ['road_4_0_1'], 3: [['road_3_0_0', 'road_3_0_1'], ['road_1_0_1', 'road_1_0_2']],
                       6: [['road_3_0_0', 'road_3_0_1'], ['road_3_0_2']]}
        elif self.net == '3_3_2_2':
            pl_dict = {0: ['road_4_0_1'], 3: [['road_3_0_0'], ['road_1_0_1', 'road_1_0_2']],
                       6: ['road_3_0_0', 'road_3_0_1']}
        else:               #if self.net == '3_2_2_2':
            pl_dict = {0: ['road_4_0_1'], 3: ['road_3_0_0', 'road_1_0_1'],
                       6: ['road_3_0_0', 'road_3_0_1']}

        return pl_dict

    def get_intersection_lanes(self):
        connections = traci.trafficlight.getControlledLinks(self.target_intersection)

        incoming_lanes = set()  # 存储流入车道
        outgoing_lanes = set()  # 存储流出车道

        for connection in connections:
            # connection 是 (incoming_lane, outgoing_lane, via_lane) 组成的元组
            incoming_lane = connection[0][0]  # 获取流入车道ID
            outgoing_lane = connection[0][1]  # 获取流出车道ID

            incoming_lanes.add(incoming_lane)
            outgoing_lanes.add(outgoing_lane)

        return incoming_lanes, outgoing_lanes


    def get_neighbor_lanes(self, task_net):
        if task_net.split('_')[0] == '3':    # 三岔路口下没有北方相连的交叉口
            nl_dict = {'n': [],
                       'e': ['road_6_3_0', 'road_14_3_1', 'road_7_3_2'],
                       's': ['road_7_4_0', 'road_17_4_1', 'road_8_4_2'],
                       'w': ['road_8_1_0', 'road_20_1_1', 'road_5_1_2']}
        else:
            nl_dict = {'n': ['road_5_2_0', 'road_11_2_1', 'road_6_2_2'],
                       'e': ['road_6_3_0', 'road_14_3_1', 'road_7_3_2'],
                       's': ['road_7_4_0', 'road_17_4_1', 'road_8_4_2'],
                       'w': ['road_8_1_0', 'road_20_1_1', 'road_5_1_2']}
        return nl_dict

    def get_vehicle_speed(self, car):
        """获取指定车辆速度"""
        return traci.vehicle.getSpeed(car)

    def get_vehicle_waitingtime(self, car):
        """获取指定车辆等待时间"""
        return traci.vehicle.getWaitingTime(car)

    def get_lane_total_vehicles(self, lane):
        return traci.lane.getLastStepVehicleNumber(lane)

    def get_lane_halting_vehicles(self, lane):
        return traci.lane.getLastStepHaltingNumber(lane)

    def get_vehicle_CO2_emission(self, car):
        """获取指定车辆CO2排放量"""
        return traci.vehicle.getCO2Emission(car)

    def get_vehicle_fuel_consumption(self, car):
        """获取指定车辆燃油消耗"""
        return traci.vehicle.getFuelConsumption(car)

    def get_emission_indicators(self):
        """获取排放指标：CO2总排放量、平均燃油消耗、平均速度"""
        lanes = []  # 存储车道编号
        for (key, value) in self.get_phase_lanes().items():
            lanes.append(value)
        lanes = sum(lanes, [])  # 获取该路口对应的所有道路编号
        lanes = utils.flatten_list(lanes)
        lanes = list(set(lanes))

        total_CO2 = 0.0
        total_fuel = 0.0
        total_speed = 0.0
        vehicle_count = 0

        for lane in lanes:  # 遍历每条车道
            vehicle_ids = traci.lane.getLastStepVehicleIDs(lane)
            for car in vehicle_ids:
                total_CO2 += self.get_vehicle_CO2_emission(car)
                total_fuel += self.get_vehicle_fuel_consumption(car)
                total_speed += self.get_vehicle_speed(car)
                vehicle_count += 1

        avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0

        return total_CO2, total_fuel, avg_speed, vehicle_count
    
    def get_vehicle_distance_to_stopline(self, car):
        """获取指定车辆距离停止线距离"""
        lane = traci.vehicle.getLaneID(car)          # 获取当前车辆所在车道
        lane_length = traci.lane.getLength(lane)     # 获取当前道路长度
        vehicle_pos = traci.vehicle.getLanePosition(car)        # 获取当前车辆在道路上的位置
        distance_to_stopline = lane_length - vehicle_pos        # 指定车辆距离停止线距离
        return distance_to_stopline

    def get_lane_information(self):
        """获取车道上的车辆信息"""
        lanes = []     # 存储车道编号
        for (key, value) in self.get_phase_lanes().items():
            lanes.append(value)
        lanes = sum(lanes, [])                            # 获取该路口对应的所有道路编号
        lanes = utils.flatten_list(lanes)
        lanes = list(set(lanes))
        lane_info = dict([(lane, []) for lane in lanes])  # 存储每条道路上的车辆信息
        for lane in lanes:  # 遍历每条车道
            vehicle_id = traci.lane.getLastStepVehicleIDs(lane)  # 获取该车道上的车辆编号
            if len(vehicle_id) != 0:
                for car in vehicle_id:  # 遍历该车道上的每辆车
                    v_speed = self.get_vehicle_speed(car)                 # 获取该车的速度信息
                    v_waitingtime = self.get_vehicle_waitingtime(car)     # 获取该车的等待时间
                    v_dis2intersec = self.get_vehicle_distance_to_stopline(car)  # 获取该车距离路口位置
                    car_info = [v_speed / args.max_speed,
                                v_waitingtime / args.max_waiting_time,
                                v_dis2intersec / args.max_distance2intersection]   # 将该车的速度等信息作为该车的特征表示，并做归一化处理
                    lane_info[lane].append(car_info)                      # 将该车信息与所在路段相关联
            lane_info[lane] = sorted(lane_info[lane], key=lambda x: x[-1])[: args.N_VEHICLE] \
                if len(lane_info[lane]) >= args.N_VEHICLE else sorted(lane_info[lane], key=lambda x: x[-1])
            while len(lane_info[lane]) < args.N_VEHICLE:            # 补全至相同维度
                lane_info[lane].append([0., 0., 0.])
            # print(lane, lane_info[lane])
        return lane_info  # 返回全部路段信息-路段名称及其所包含的所有车辆信息

    def get_state_vehicle(self):
        """获取当前时刻目标交叉口的车辆状态信息 return: state_veh"""
        lane_info = self.get_lane_information()  # 获取道路与车辆信息的对应关系
        phase_lane = self.get_phase_lanes()      # 获取相位与道路信息的对应关系
        state_p = []          # 存储每一个相位的信息
        for p in phase_lane.keys():              # 遍历每一个相位
            lane = phase_lane[p]                 # 获取当前相位所对应的车道
            state_l = []                         # 存储当前相位下车道信息
            # 将相同流向的车道合并为同一个
            if utils.is_nested_list(lane) is False:
                for l in lane:                       # 遍历每一条车道
                    state_l.append(lane_info[l])     # 将车道对应的车辆信息加入至车道信息
            else:
                for i in range(len(lane)):
                    state_l_d = []
                    for j in range(len(lane[i])):
                        state_l_d.append(lane_info[lane[i][j]])
                    state_l_d = sum(state_l_d, [])
                    state_l_d = [row for row in state_l_d if row != [0.0, 0.0, 0.0]]
                    state_l_d = sorted(state_l_d, key=lambda x: x[-1])[: args.N_VEHICLE]
                    while len(state_l_d) < args.N_VEHICLE:  # 补全至相同维度
                        state_l_d.append([0., 0., 0.])
                    state_l.append(state_l_d)

            while len(state_l) < args.N_LANE:       # 当当前相位控制车道数少于规定数量时，补零
                state_l.append([[0., 0., 0.][:] for _ in range(args.N_VEHICLE)])
            state_p.append(state_l)              # 添加至相位特征集合

        while len(state_p) < args.N_PHASE:              # 当当前相位数少于规定数量时，补零
            state_p.append([[[0., 0., 0.][:] for _ in range(args.N_VEHICLE)][:] for _ in range(args.N_LANE)])
        return state_p

    def get_state_neighbor(self, task_net):
        """获取当前时刻目标交叉口的邻居交叉口信息 return: state_nei"""
        neighbor_lane = self.get_neighbor_lanes(task_net)         # 获取周围交叉口车道信息
        state_neighbor = []                               # 用于存储交叉口状态
        for dir in neighbor_lane.keys():                  # 遍历四个方向
            lane = neighbor_lane[dir]                     # 获取该方向车道信息
            state_dir = []                                # 存储该方向状态
            if len(lane) == 0:                            # 当没有该方向时
                state_dir.append([0, 0, 0, 0, 0, 0])      # 补零
            else:
                for l in lane:                            # 反之遍历每条车道，获取其对应车辆信息
                    state_dir.append([self.get_lane_halting_vehicles(l), self.get_lane_total_vehicles(l)])
            state_neighbor.append([item for sublist in state_dir for item in sublist])    # 添加至邻居状态集合
        return state_neighbor                             # 返回邻居状态

    def get_state(self, task_net):
        """获取当前交叉口状态信息，包括state_vehicle和state_neighbor"""
        state = [self.get_state_vehicle(), self.get_state_neighbor(task_net)]
        return state

    def get_state_inference(self):
        return self.get_state_vehicle()

    def change_light(self, intersection, phase_index):
        """改变信号灯配时方案"""
        traci.trafficlight.setPhase(intersection, phase_index)

    def get_reward(self):
        reward = 0
        incoming_lanes, outgoing_lanes = self.get_intersection_lanes()
        for i_l in incoming_lanes:
            pressure_neg = traci.lane.getLastStepHaltingNumber(i_l)
            reward -= pressure_neg * 0.2
        for o_l in outgoing_lanes:
            pressure_pos = traci.lane.getLastStepHaltingNumber(o_l)
            reward += pressure_pos * 0.1
        return reward
        # reward = 0
        # incoming_lanes, outgoing_lanes = self.get_intersection_lanes()
        # for i_l in incoming_lanes:
        #     reward += traci.lane.getLastStepHaltingNumber(i_l)
        # return -reward + 20
    # def get_reward(self):
    #     """获取当前时刻的奖励信息"""
    #     lanes = []  # 存储车道编号
    #     for (key, value) in self.get_phase_lanes().items():     # 仅统计控制车道上的车辆信息
    #         lanes.append(value)
    #     lanes = list(itertools.chain.from_iterable(lanes))      # 转成控制车道列表
    #     waitingTime_fairness, vehicle_num = 0, 0
    #     for l in lanes:       # 遍历每一条车道
    #         l_vehicles = traci.lane.getLastStepVehicleIDs(l)    # 统计该车道上的车辆信息
    #         for v in l_vehicles:                                # 遍历每一辆车
    #             waitingtime_fairness_v = args.kappa * (1 - (self.get_vehicle_waitingtime(v) / args.varsigma) ** args.rho) # 获取该车等待时间信息
    #             waitingTime_fairness += waitingtime_fairness_v  # 累加等待时间
    #             vehicle_num += 1
    #     reward = waitingTime_fairness / (vehicle_num + min_num) # 获取奖励
    #     return reward

    def get_indicator_halting_num(self):
        """返回当前时刻交叉口的排队车辆数"""
        lanes = []  # 存储车道编号
        for (key, value) in self.get_phase_lanes().items():
            lanes.append(value)
        lanes = sum(lanes, [])  # 获取该路口对应的所有道路编号
        lanes = utils.flatten_list(lanes)
        lanes = list(set(lanes))
        halting_num = 0
        for l in lanes:
            halting_num += traci.lane.getLastStepHaltingNumber(l)
        return halting_num

    def get_indicator_waiting_time(self):
        """返回当前时刻交叉口的等待时间"""
        lanes = []  # 存储车道编号
        for (key, value) in self.get_phase_lanes().items():
            lanes.append(value)
        lanes = sum(lanes, [])  # 获取该路口对应的所有道路编号
        lanes = utils.flatten_list(lanes)
        lanes = list(set(lanes))
        waiting_time = 0
        for l in lanes:  # 遍历每条车道
            vehicle_id = traci.lane.getLastStepVehicleIDs(l)  # 获取该车道上的车辆编号
            for car in vehicle_id:  # 遍历该车道上的每辆车
                waiting_time += self.get_vehicle_waitingtime(car)  # 获取该车的等待时间
        return waiting_time

    def get_state_lane_haltingNum(self):
        lanes = []  # 存储车道编号
        for (key, value) in self.get_phase_lanes().items():
            lanes.append(value)
        # lanes = sum(lanes, [])  # 获取该路口对应的所有道路编号
        lanes = utils.remove_third_layer(lanes)
        halting_num = []
        for p in range(len(lanes)):
            halting_num_p = 0
            for i in range(len(lanes[p])):
                halting_num_p += traci.lane.getLastStepHaltingNumber(lanes[p][i])
            halting_num.append(halting_num_p)
        return halting_num

        # return [traci.lane.getLastStepHaltingNumber(lanes[0]) + traci.lane.getLastStepHaltingNumber(lanes[1]),
        #         traci.lane.getLastStepHaltingNumber(lanes[2]) + traci.lane.getLastStepHaltingNumber(lanes[3]),
        #         traci.lane.getLastStepHaltingNumber(lanes[4]) + traci.lane.getLastStepHaltingNumber(lanes[5]),
        #         traci.lane.getLastStepHaltingNumber(lanes[6]) + traci.lane.getLastStepHaltingNumber(lanes[7])]
    def analyze_traffic_state(self, state, task_net):
        """根据task_net动态分析交通状态"""
        state_vehicle, state_neighbor = state[0], state[1]

        # ================== 动态获取车道映射 ==================
        phase_lanes = self.get_phase_lanes()  # 根据task_net自动匹配
        neighbor_lanes = self.get_neighbor_lanes(task_net)

        # ================== 当前交叉口统计 ==================
        lane_stats = {}

        # 遍历每个相位及对应车道
        for phase_idx, lanes in phase_lanes.items():
            # 扁平化处理车道结构（移除嵌套）
            flat_lanes = []
            if isinstance(lanes, list):
                for item in lanes:
                    if isinstance(item, list):
                        flat_lanes.extend(item)
                    else:
                        flat_lanes.append(item)
            else:
                flat_lanes = [lanes]

            # 统计该相位下所有车道的总体指标
            total_waiting = 0
            total_vehicles = 0
            total_time = 0.0

            for lane_id in flat_lanes:
                try:
                    # 从SUMO直接获取实时数据
                    total_waiting += traci.lane.getLastStepHaltingNumber(lane_id)
                    total_vehicles += traci.lane.getLastStepVehicleNumber(lane_id)

                    # 计算该车道所有车辆的等待时间总和
                    for veh_id in traci.lane.getLastStepVehicleIDs(lane_id):
                        total_time += traci.vehicle.getWaitingTime(veh_id)
                except traci.TraCIException:
                    print(f"车道 {lane_id} 不存在，已跳过")
                    continue

            # 保存该相位的总体统计信息
            lane_stats[f"phase{phase_idx}"] = {
                'waiting': total_waiting,
                'total': total_vehicles,
                'time': round(total_time, 2)
            }

        # ================== 邻居交叉口统计 ==================
        neighbor_stats = {}
        for direction, lanes in neighbor_lanes.items():
            direction_total = 0
            for lane_id in lanes:
                direction_total += traci.lane.getLastStepVehicleNumber(lane_id)
            neighbor_stats[direction] = int(direction_total)

        return {
            'current_intersection': lane_stats,
            'neighbor_intersections': {'by_direction': neighbor_stats}
        }


    def step(self):
        """路网执行1秒"""
        # 记录新进入路网的车辆
        departed_vehicles = traci.simulation.getDepartedIDList()
        current_time = traci.simulation.getTime()

        for veh_id in departed_vehicles:
            self.vehicle_entry_times[veh_id] = current_time

        # 记录离开路网的车辆并计算行程时间
        arrived_vehicles = traci.simulation.getArrivedIDList()

        for veh_id in arrived_vehicles:
            if veh_id in self.vehicle_entry_times:
                entry_time = self.vehicle_entry_times[veh_id]
                travel_time = current_time - entry_time

                # 存储行程时间
                self.vehicle_travel_times[veh_id] = travel_time
                self.total_travel_time += travel_time
                self.vehicle_count += 1

                # 移除已离开的车辆
                del self.vehicle_entry_times[veh_id]

        # 执行仿真步进
        traci.simulationStep()

    def get_travel_time_stats(self):
        """获取行程时间统计信息"""
        if self.vehicle_count == 0:
            return {
                'average_travel_time': 0,
                'min_travel_time': 0,
                'max_travel_time': 0,
                'total_vehicles': 0,
                'total_travel_time': 0
            }

        # 计算统计指标
        all_times = list(self.vehicle_travel_times.values())
        avg_time = self.total_travel_time / self.vehicle_count
        min_time = min(all_times)
        max_time = max(all_times)

        return {
            'average_travel_time': round(avg_time, 2),
            'min_travel_time': round(min_time, 2),
            'max_travel_time': round(max_time, 2),
            'total_vehicles': self.vehicle_count,
            'total_travel_time': round(self.total_travel_time, 2)
        }

    def get_current_travel_times(self):
        """获取当前仍在路网中的车辆的已行驶时间"""
        current_time = traci.simulation.getTime()
        travel_times = {}

        for veh_id, entry_time in self.vehicle_entry_times.items():
            travel_time = current_time - entry_time
            travel_times[veh_id] = round(travel_time, 2)

        return travel_times

    def end(self):
        """结束当前环境"""
        # 在结束前记录所有尚未离开的车辆的行程时间
        current_time = traci.simulation.getTime()

        for veh_id, entry_time in self.vehicle_entry_times.items():
            travel_time = current_time - entry_time
            self.vehicle_travel_times[veh_id] = travel_time
            self.total_travel_time += travel_time
            self.vehicle_count += 1

        self.vehicle_entry_times = {}
        self.control.close()