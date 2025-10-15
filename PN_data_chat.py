
import json


from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import logging
import os
from datetime import datetime

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
#and
# the timing provided by the Agent action .

# 加载本地模型和分词器
PROMPT_TWO= [
    {"role": "system",
     "content": "You are an expert in traffic management. You can use your knowledge of traffic common sense to solve this traffic signal timing task."},
    {"role": "user",
     "content": "A traffic light controls a three-way intersection with east, south, and west directions. "
                "Each direction has an adjacent intersection, and each direction contains multiple lanes. "
                "On each lane, there may be waiting vehicles, vehicles currently approaching the intersection, and vehicles that will soon arrive at this intersection from adjacent intersections. "
                "Vehicles that queued early have already reached the intersection and are waiting for permission to proceed. "
                "Approaching vehicles will arrive at the intersection in the future.\n\n"
                "The traffic light has 3 signal phases. Each phase allows vehicles in specific lanes to proceed. "
                "The state of the intersection is described as follows:\n"
                "- The number of waiting vehicles in each lane.\n"
                "- The number of approaching vehicles currently en route to the intersection in each lane.\n"
                "- The waiting time of the waiting vehicles in each lane.\n"
                "- The number of vehicles approaching this direction from adjacent intersections.\n"
                "Intersection State: {env_conditions}\n"
                "Please Answer:\n"
                "Which traffic signal timing plan will most effectively improve traffic conditions? "
                "(Minimize the number of waiting vehicles and their total waiting time)\n\n"
                "Traffic Signal Timing Requirements:\n"
                "- The timing plan must include three phases:\n"
                "south_left: [between 5 to 30]seconds,\n"
                "east_west_straight: [between 5 to 30] seconds,\n"
                "east_left_straight: [between 5 to 30] seconds,\n"
                "- The sum of all green light times must be between 15 seconds and 90 seconds."
                "- The green light time for each phase should be an integer between 5 and 30 seconds.\n"
                "- IMPORTANT: Avoid multiples of 5, use diverse timing values, including all integers from 5 to 30.\n"

                "Requirements:\n"
                "- Think step by step.\n"
                "- You must provide your analysis according to the following steps:"
                "Step 1: Analyze traffic conditions and prioritize lanes with higher congestion.\n"
                "Step 2: Provide a timing plan with diverse timing values.\n"
                "Step 3: Explain your timing plan.\n"
                "- Your output must adhere to the specified format.\n"
                "- Please strictly follow the format below:\n"
                "south_left: [value] seconds,\n"
                "east_west_straight: [value] seconds,\n"
                "east_left_straight: [value] seconds,\n\n"
                "Explanation: Explanation of the timing plan.\n"
     }
]

PROMPT_ONE = [
    {"role": "system",
     "content": "You are an expert in traffic management. You can use your knowledge of traffic commonsense to solve this Traffic signal cycle timing task."},
    {"role": "user",
     "content": "A traffic light controls a four-way intersection with north, south, east, and west directions. "
                "Each direction has a neighboring intersection, and each direction contains a through lane and a left-turn lane. "
                "On each lane, there may be waiting vehicles, approaching vehicles currently in transit, "
                "and vehicles from neighboring intersections that will soon arrive at this intersection. "
                "Early queued vehicles have already reached the intersection and are waiting for permission to proceed. "
                "Approaching vehicles will arrive at the intersection in the future.\n\n"
                "The traffic light has 4 signal phases. Each phase allows vehicles in two specific lanes to proceed. "
                "The state of the intersection is described as follows:\n"
                "- The number of vehicles waiting on each lane.\n"
                "- The number of approaching vehicles currently en route to the intersection on each lane.\n"
                "- The waiting time of the vehicles waiting on each lane.\n"
                "- The number of vehicles from neighboring intersections that are approaching this direction.\n"
                "Intersection State: {env_conditions}\n"
                "Please Answer:\n"
                "Which traffic signal timing plan will most effectively improve the traffic conditions? "
                "(Minimize the number of waiting vehicles and their total waiting time)\n\n"
                "Traffic Signal Timing Requirements:\n"
                "- The timing plan must include four phases:\n"
                "north_south_straight: [between 5 to 30] seconds,\n"
                "north_south_left: [between 5 to 30] seconds,\n"
                "east_west_straight: [between 5 to 30] seconds,\n"
                "east_west_left: [between 5 to 30] seconds,\n"
                "- The sum of all green light times must be between 20 seconds to 120 seconds."
                "- Each phase's green light time should be an integer between 5 and 30 seconds.\n"
                "- IMPORTANT: Avoid multiples of 5, Use diverse timing values , including all integers from 5 to 30.\n"

                "Requirements:\n"
                "- Let's think step by step.\n"
                "- You must follow the following steps to provide your analysis:"
                "Step 1: Analyze the traffic conditions and prioritize lanes with higher congestion.\n"
                "Step 2: Provide your timing plan with diverse timing values.\n"
                "Step 3: Explain your timing plan.\n"
                "- Your output must adhere to the specified output format.\n"
                "- Please follow the format below strictly:\n"
                "north_south_straight: [value] seconds,\n"
                "north_south_left: [value] seconds,\n"
                "east_west_straight: [value] seconds,\n"
                "east_west_left: [value] seconds,\n\n"
                "Explanation:  explanation of your timing plan.\n"
     }
]

model_path = "./finetuned_model_pos_neg/final_merge_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
model.eval()

def create_prompt_template(net_type):
    """根据交叉口类型创建提示词模板"""
    if net_type.split('_')[0] == '3':
        prompt_list = PROMPT_TWO
        template = (
            prompt_list[0]['content'] +
            "\n\n### Instruction:\n" +
            prompt_list[1]['content'] +
            "\n\n### Response:\n"
        )
    else:  # 四路交叉口
        prompt_list = PROMPT_ONE
        template = (
            prompt_list[0]['content'] +
            "\n\n### Instruction:\n" +
            prompt_list[1]['content'] +
            "\n\n### Response:\n"
        )
    return template

def setup_chain_one(prompt_template):
    """设置LangChain工作流"""
    prompt = PromptTemplate(
        input_variables=["env_conditions"],
        template=prompt_template
    )

    lambda_step = RunnableLambda(lambda x: call_local_model(prompt.format(**x)))
    return lambda_step


NUM_GENERATIONS = 1

def call_local_model(prompt):
    """调用本地大语言模型"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        if 'token_type_ids' in inputs:
            del inputs['token_type_ids']
        output = model.generate(
            **inputs,
            max_new_tokens=1024,          # 缩短生成长度
            temperature=0.1,             # 适度降低温度
            # top_k = 50,#最高的50个中选
            top_p = 0.9,               # 缩小核采样范围                 # 添加top-k限制
            do_sample=True,
            num_return_sequences=1,      # 批量生成5个结果
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
            use_cache=True,               # 启用缓存
        )
        response = tokenizer.decode(output[0], skip_special_tokens=True)
        generated_text = response[len(prompt):].strip()
        return generated_text
    except Exception as e:
        logger.error(f"本地模型推理失败: {str(e)}")
        return f"模型错误: {str(e)}"


def main(traffic_states,net):
    try:
        def safe_get(data, keys, default=0):
            value = data
            for key in keys:
                if key in value:
                    value = value[key]
                else:
                    return default
            return value if isinstance(value, (int, float)) else default
        lane_data = traffic_states['current_intersection']
        neighbor_stats = traffic_states['neighbor_intersections']['by_direction']
    # 构建增强版环境参数
        if net.split('_')[0] == '3':
            ENV_CONDITIONS = f"""

            There are {lane_data['phase0']['waiting']} waiting vehicles in the sorth-left phase, with a total of {lane_data['phase0']['total']} vehicles and a total waiting time of {lane_data['phase0']['time']} seconds;
            There are {lane_data['phase3']['waiting']} waiting vehicles in the east-west through phase, with a total of  {lane_data['phase3']['total']} vehicles and a total waiting time of {lane_data['phase3']['time']} seconds;
            There are {lane_data['phase6']['waiting']} waiting vehicles in the east left turn and straight through phase, with a total of  {lane_data['phase6']['total']} vehicles and a total waiting time of  {lane_data['phase6']['time']} seconds;
            The total number of vehicles about to enter from the neighboring intersection in the south direction is {safe_get(neighbor_stats, ['s'], 0)} vehicles;
            The total number of vehicles about to enter from the neighboring intersection in the east direction is  {safe_get(neighbor_stats, ['e'], 0)} vehicles;
            The total number of vehicles about to enter from the neighboring intersection in the west direction is {safe_get(neighbor_stats, ['w'], 0)} vehicles.

            """
            # RL based timing:
            # south_left_turn: {timing_strategy[0]}
            # east_west_straight: {timing_strategy[3]}
            # east_left_straight: {timing_strategy[6]}
        else:
            ENV_CONDITIONS = f"""

            There are {lane_data['phase0']['waiting']} waiting vehicles in the north-south through lane, with a total of {lane_data['phase0']['total']} vehicles and a total waiting time of {lane_data['phase0']['time']} seconds;
            There are {lane_data['phase3']['waiting']} waiting vehicles in the north-south left-turn lane, with a total of  {lane_data['phase3']['total']} vehicles and a total waiting time of {lane_data['phase3']['time']} seconds;
            There are {lane_data['phase6']['waiting']} waiting vehicles in the east-west through lane, with a total of  {lane_data['phase6']['total']} vehicles and a total waiting time of  {lane_data['phase6']['time']} seconds;
            There are {lane_data['phase9']['waiting']}waiting vehicles in the east-west left-turn lane, with a total of {lane_data['phase9']['total']} vehicles and a total waiting time of {lane_data['phase9']['time']} seconds;
            The total number of vehicles about to enter from the neighboring intersection in the north direction is {safe_get(neighbor_stats, ['n'], 0)} vehicles;
            The total number of vehicles about to enter from the neighboring intersection in the south direction is {safe_get(neighbor_stats, ['s'], 0)} vehicles;
            The total number of vehicles about to enter from the neighboring intersection in the east direction is  {safe_get(neighbor_stats, ['e'], 0)} vehicles;
            The total number of vehicles about to enter from the neighboring intersection in the west direction is {safe_get(neighbor_stats, ['w'], 0)} vehicles.


            """
        prompt_template = create_prompt_template(net)
        # 运行推理链
        chain = setup_chain_one(prompt_template)
        all_responses = []

        # 修改为调用三次
        for _ in range(1) :
            response = chain.invoke({"env_conditions": ENV_CONDITIONS})
            # print(ENV_CONDITIONS)
            all_responses.append(response)
            print(f"第{_ + 1}次生成结果：\n{response}\n{'=' * 50}")

        return all_responses

    except Exception as e:
        logger.error(f"主流程执行失败: {str(e)}")
        return None