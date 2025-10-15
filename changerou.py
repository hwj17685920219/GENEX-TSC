import random
from config import parse_arguments

args = parse_arguments()

non_primary_flow_low, non_primary_flow_high = args.non_primary_flow_low, args.non_primary_flow_high

road_id_in = ['9_5', '10_5', '11_2', '12_6', '13_6', '14_3', '15_7', '16_7', '17_4', '18_8', '19_8', '20_1']
road_id_out = ['5_9', '5_10', '2_11', '6_12', '6_13', '3_14', '7_15', '7_16', '4_17', '8_18', '8_19', '1_20']


TIME_PERIOD = args.task_time

class Changerou(object):
    def rewrite_flow(self, net):
        with open('rou.rou_' + str(net) + '.xml', 'w', encoding='utf-8') as f:
            f.write('<routes>\n')
            f.write('  <vType accel="3" decel="8" id="CarA" length="5" maxSpeed="13.89" '
                    'reroute="false" sigma="0.5" color="0,1,0"/>\n\n')

            flow_num = 0

            start_time = 1
            end_time = args.episode_time

            for l_in in road_id_in:
                for l_out in road_id_out:
                    if set(l_in.split('_')) == set(l_out.split('_')):
                        continue
                    else: # 西东直行， 东西直行， 北南直行， 南北直行，西东左转， 东西左转， 北南左转， 南北左转
                        if net.split('_')[0] == '4':
                            if (l_in == '20_1' and l_out == '3_14') or (l_in == '14_3' and l_out == '1_20') \
                                    or (l_in == '11_2' and l_out == '4_17') or (l_in == '17_4' and l_out == '2_11') \
                                    or (l_in == '20_1' and l_out == '2_11') or (l_in == '14_3' and l_out == '4_17') \
                                    or (l_in == '11_2' and l_out == '3_14') or (l_in == '17_4' and l_out == '1_20'):
                                if random.random() < 0.5:
                                    num = random.randint(args.primary_flow_low_4, args.primary_flow_high_4)
                                else:
                                    num = random.randint(non_primary_flow_low, non_primary_flow_high)

                                if l_in == '20_1' and l_out == '3_14':
                                    route = "road_20_1  road_1_0  road_0_3  road_3_14"
                                elif l_in == '14_3' and l_out == '1_20':
                                    route = "road_14_3  road_3_0  road_0_1  road_1_20"
                                elif l_in == '11_2' and l_out == '4_17':
                                    route = "road_11_2  road_2_0  road_0_4  road_4_17"
                                elif l_in == '17_4' and l_out == '2_11':
                                    route = "road_17_4  road_4_0  road_0_2  road_2_11"
                                elif l_in == '20_1' and l_out == '2_11':
                                    route = "road_20_1  road_1_0  road_0_2  road_2_11"
                                elif l_in == '14_3' and l_out == '4_17':
                                    route = "road_14_3  road_3_0  road_0_4  road_4_17"
                                elif l_in == '11_2' and l_out == '3_14':
                                    route = "road_11_2  road_2_0  road_0_3  road_3_14"
                                else:
                                    route = "road_17_4  road_4_0  road_0_1  road_1_20"

                                f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
                                        'type="CarA"> <route edges="' + str(route) + '"/> </flow>')
                                f.write('\n')

                            else:
                                num = random.randint(non_primary_flow_low, non_primary_flow_high)
                                f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
                                        'type="CarA" from="road_' + str(l_in) + '" to="road_' + str(l_out) + '"/>')
                                f.write('\n')
                        else:
                            # 三叉路口下随机选择直行、左转车道为主干道
                            if (l_in == '20_1' and l_out == '3_14') or (l_in == '14_3' and l_out == '1_20') \
                                    or (l_in == '14_3' and l_out == '4_17') \
                                    or (l_in == '17_4' and l_out == '1_20'):
                                if random.random() < 0.5:
                                    num = random.randint(args.primary_flow_low_3, args.primary_flow_high_3)
                                else:
                                    num = random.randint(non_primary_flow_low, non_primary_flow_high)

                                if l_in == '20_1' and l_out == '3_14':
                                    route = "road_20_1  road_1_0  road_0_3  road_3_14"
                                elif l_in == '14_3' and l_out == '1_20':
                                    route = "road_14_3  road_3_0  road_0_1  road_1_20"
                                elif l_in == '14_3' and l_out == '4_17':
                                    route = "road_14_3  road_3_0  road_0_4  road_4_17"
                                else:
                                    route = "road_17_4  road_4_0  road_0_1  road_1_20"

                                f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
                                        'type="CarA"> <route edges="' + str(route) + '"/> </flow>')
                                f.write('\n')
                            else:
                                num = random.randint(non_primary_flow_low, non_primary_flow_high)
                                f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
                                        'type="CarA" from="road_' + str(l_in) + '" to="road_' + str(l_out) + '"/>')
                                f.write('\n')

                    flow_num += 1
            f.write('</routes>\n')


    def rewrite_flow_for_inference(self, net):
        with open('rou.rou_' + str(net) + '.xml', 'w', encoding='utf-8') as f:
            f.write('<routes>\n')
            f.write('  <vType accel="3" decel="8" id="CarA" length="5" maxSpeed="13.89" '
                    'reroute="false" sigma="0.5" color="0,1,0"/>\n\n')

            flow_num = 0
            flow_period = args.timesteps_inference

            for period in range(flow_period):
                start_time = period * TIME_PERIOD + 1
                end_time = period * TIME_PERIOD + TIME_PERIOD

                for l_in in road_id_in:
                    for l_out in road_id_out:
                        if set(l_in.split('_')) == set(l_out.split('_')):
                            continue
                        else: # 西东直行， 东西直行， 北南直行， 南北直行，西东左转， 东西左转， 北南左转， 南北左转
                            if net.split('_')[0] == '4':
                                if (l_in == '20_1' and l_out == '3_14') or (l_in == '14_3' and l_out == '1_20') \
                                        or (l_in == '11_2' and l_out == '4_17') or (l_in == '17_4' and l_out == '2_11') \
                                        or (l_in == '20_1' and l_out == '2_11') or (l_in == '14_3' and l_out == '4_17') \
                                        or (l_in == '11_2' and l_out == '3_14') or (l_in == '17_4' and l_out == '1_20'):
                                    if random.random() < 0.5:
                                        num = random.randint(args.primary_flow_low_4, args.primary_flow_high_4)
                                    else:
                                        num = random.randint(non_primary_flow_low, non_primary_flow_high)

                                    if l_in == '20_1' and l_out == '3_14':
                                        route = "road_20_1  road_1_0  road_0_3  road_3_14"
                                    elif l_in == '14_3' and l_out == '1_20':
                                        route = "road_14_3  road_3_0  road_0_1  road_1_20"
                                    elif l_in == '11_2' and l_out == '4_17':
                                        route = "road_11_2  road_2_0  road_0_4  road_4_17"
                                    elif l_in == '17_4' and l_out == '2_11':
                                        route = "road_17_4  road_4_0  road_0_2  road_2_11"
                                    elif l_in == '20_1' and l_out == '2_11':
                                        route = "road_20_1  road_1_0  road_0_2  road_2_11"
                                    elif l_in == '14_3' and l_out == '4_17':
                                        route = "road_14_3  road_3_0  road_0_4  road_4_17"
                                    elif l_in == '11_2' and l_out == '3_14':
                                        route = "road_11_2  road_2_0  road_0_3  road_3_14"
                                    else:
                                        route = "road_17_4  road_4_0  road_0_1  road_1_20"

                                    f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
                                            'type="CarA"> <route edges="' + str(route) + '"/> </flow>')
                                    f.write('\n')

                                else:
                                    num = random.randint(non_primary_flow_low, non_primary_flow_high)
                                    f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
                                            'type="CarA" from="road_' + str(l_in) + '" to="road_' + str(l_out) + '"/>')
                                    f.write('\n')
                            else:
                                # 三叉路口下随机选择直行、左转车道为主干道
                                if (l_in == '20_1' and l_out == '3_14') or (l_in == '14_3' and l_out == '1_20') \
                                        or (l_in == '14_3' and l_out == '4_17') \
                                        or (l_in == '17_4' and l_out == '1_20'):
                                    if random.random() < 0.5:
                                        num = random.randint(args.primary_flow_low_3, args.primary_flow_high_3)
                                    else:
                                        num = random.randint(non_primary_flow_low, non_primary_flow_high)

                                    if l_in == '20_1' and l_out == '3_14':
                                        route = "road_20_1  road_1_0  road_0_3  road_3_14"
                                    elif l_in == '14_3' and l_out == '1_20':
                                        route = "road_14_3  road_3_0  road_0_1  road_1_20"
                                    elif l_in == '14_3' and l_out == '4_17':
                                        route = "road_14_3  road_3_0  road_0_4  road_4_17"
                                    else:
                                        route = "road_17_4  road_4_0  road_0_1  road_1_20"

                                    f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
                                            'type="CarA"> <route edges="' + str(route) + '"/> </flow>')
                                    f.write('\n')
                                else:
                                    num = random.randint(non_primary_flow_low, non_primary_flow_high)
                                    f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
                                            'type="CarA" from="road_' + str(l_in) + '" to="road_' + str(l_out) + '"/>')
                                    f.write('\n')

                        flow_num += 1
            f.write('</routes>\n')



        # if net.split('_') == 4:
        #     with open('rou.rou_' + str(net) + '.xml', 'w', encoding='utf-8') as f:
        #         f.write('<routes>\n')
        #         f.write('  <vType accel="3" decel="8" id="CarA" length="5" maxSpeed="13.89" '
        #                 'reroute="false" sigma="0.5" color="0,1,0"/>\n\n')
        #
        #         flow_num = 0
        #         flow_period = args.timesteps_inference
        #
        #         for period in range(flow_period):
        #             start_time = period * TIME_PERIOD + 1
        #             end_time = period * TIME_PERIOD + TIME_PERIOD
        #
        #             for l_in in road_id_in:
        #                 for l_out in road_id_out:
        #                     if set(l_in.split('_')) == set(l_out.split('_')):
        #                         continue
        #                     else: # 西东直行， 东西直行， 北南直行， 南北直行，西东左转， 东西左转， 北南左转， 南北左转
        #                         if (l_in == '20_1' and l_out == '3_14') or (l_in == '14_3' and l_out == '1_20') \
        #                                 or (l_in == '11_2' and l_out == '4_17') or (l_in == '17_4' and l_out == '2_11') \
        #                                 or (l_in == '20_1' and l_out == '2_11') or (l_in == '14_3' and l_out == '4_17') \
        #                                 or (l_in == '11_2' and l_out == '3_14') or (l_in == '17_4' and l_out == '1_20'):
        #                             if random.random() < 0.5:
        #                                 num = random.randint(primary_flow_low, primary_flow_high)
        #                             else:
        #                                 num = random.randint(non_primary_flow_low, non_primary_flow_high)
        #
        #                             if l_in == '20_1' and l_out == '3_14':
        #                                 route = "road_20_1  road_1_0  road_0_3  road_3_14"
        #                             elif l_in == '14_3' and l_out == '1_20':
        #                                 route = "road_14_3  road_3_0  road_0_1  road_1_20"
        #                             elif l_in == '11_2' and l_out == '4_17':
        #                                 route = "road_11_2  road_2_0  road_0_4  road_4_17"
        #                             elif l_in == '17_4' and l_out == '2_11':
        #                                 route = "road_17_4  road_4_0  road_0_2  road_2_11"
        #                             elif l_in == '20_1' and l_out == '2_11':
        #                                 route = "road_20_1  road_1_0  road_0_2  road_2_11"
        #                             elif l_in == '14_3' and l_out == '4_17':
        #                                 route = "road_14_3  road_3_0  road_0_4  road_4_17"
        #                             elif l_in == '11_2' and l_out == '3_14':
        #                                 route = "road_11_2  road_2_0  road_0_3  road_3_14"
        #                             else:
        #                                 route = "road_17_4  road_4_0  road_0_1  road_1_20"
        #
        #                             f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
        #                                     'type="CarA"> <route edges="' + str(route) + '"/> </flow>')
        #                             f.write('\n')
        #
        #                         else:
        #                             num = random.randint(non_primary_flow_low, non_primary_flow_high)
        #                             f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
        #                                     'type="CarA" from="road_' + str(l_in) + '" to="road_' + str(l_out) + '"/>')
        #                             f.write('\n')
        #                     flow_num += 1
        #         f.write('</routes>\n')
        #
        # else:
        #     with open('rou.rou_' + str(net) + '.xml', 'w', encoding='utf-8') as f:
        #         f.write('<routes>\n')
        #         f.write('  <vType accel="3" decel="8" id="CarA" length="5" maxSpeed="13.89" '
        #                 'reroute="false" sigma="0.5" color="0,1,0"/>\n\n')
        #
        #         flow_num = 0
        #         flow_period = args.timesteps_inference
        #
        #         for period in range(flow_period):
        #             start_time = period * TIME_PERIOD + 1
        #             end_time = period * TIME_PERIOD + TIME_PERIOD
        #
        #             for l_in in road_id_in:
        #                 for l_out in road_id_out:
        #                     if set(l_in.split('_')) == set(l_out.split('_')):
        #                         continue
        #                     else:  # 西东直行， 东西直行， 北南直行， 南北直行，西东左转， 东西左转， 北南左转， 南北左转
        #                         if (l_in == '20_1' and l_out == '3_14') or (l_in == '14_3' and l_out == '1_20') \
        #                                 or (l_in == '14_3' and l_out == '4_17') \
        #                                 or (l_in == '17_4' and l_out == '1_20'):
        #                             if random.random() < 0.5:
        #                                 num = random.randint(primary_flow_low, primary_flow_high)
        #                             else:
        #                                 num = random.randint(non_primary_flow_low, non_primary_flow_high)
        #
        #                             if l_in == '20_1' and l_out == '3_14':
        #                                 route = "road_20_1  road_1_0  road_0_3  road_3_14"
        #                             elif l_in == '14_3' and l_out == '1_20':
        #                                 route = "road_14_3  road_3_0  road_0_1  road_1_20"
        #                             elif l_in == '14_3' and l_out == '4_17':
        #                                 route = "road_14_3  road_3_0  road_0_4  road_4_17"
        #                             else:
        #                                 route = "road_17_4  road_4_0  road_0_1  road_1_20"
        #
        #                             f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(
        #                                 start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
        #                                                                                                          'type="CarA"> <route edges="' + str(
        #                                 route) + '"/> </flow>')
        #                             f.write('\n')
        #
        #                         else:
        #                             num = random.randint(non_primary_flow_low, non_primary_flow_high)
        #                             f.write('  <flow id="flow_' + str(flow_num) + '" color="0,1,1"  begin="' + str(
        #                                 start_time) + '" end="' + str(end_time) + '" vehsPerHour="' + str(num) + '" '
        #                                                                                                          'type="CarA" from="road_' + str(
        #                                 l_in) + '" to="road_' + str(l_out) + '"/>')
        #                             f.write('\n')
        #                     flow_num += 1
        #         f.write('</routes>\n')



if __name__ == '__main__':
    change = Changerou()
    change.rewrite_flow_for_inference('4_3_3_3_3')
