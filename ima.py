import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    GenerationConfig,
    TrainerCallback

)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
import re
import transformers

# 加载基础模型和分词器
model_name = "deepseek-ai/deepseek-llm-7b-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True)
model.generation_config = GenerationConfig.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id


cutoff_len = 2048
train_on_inputs = False  # 控制是否在输入部分计算损失
val_set_size = 0.05



def clean_input_text(text):
    """
    清洗输入文本，执行以下操作：
    1. 删除 "This is a four-way intersection" 或 "This is a three-way intersection" 句子
    2. 删除从 "RL based timing：" 开始到 "Please Answer:" 之前的内容
    """
    if not isinstance(text, str):
        # 确保处理的是字符串
        try:
            text = str(text)
        except:
            return ""

    # 第一步: 删除特定句子
    text = re.sub(r'\bThis is a (?:four|three)-way intersection\b\.?\s*', '', text, flags=re.IGNORECASE)

    return text


def tokenize_and_mask(data):
    """
    处理输入和输出，构建训练样本
    - prompt: 清理后的输入文本
    - output: 对应的输出文本
    """
    # 构建完整文本: 输入 + 输出 + EOS
    full_text = data["input"] + data["output"] + tokenizer.eos_token

    # 分词完整文本
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None
    )

    # 如果不训练输入部分，则创建掩码
    if not train_on_inputs:
        # 仅对输入部分分词
        user_prompt = data["input"]
        tokenized_user = tokenizer(
            user_prompt,
            truncation=True,
            max_length=cutoff_len,
            padding=False,
            return_tensors=None
        )
        user_prompt_len = len(tokenized_user["input_ids"])

        # 创建标签掩码: 输入部分设为-100(忽略损失)，输出部分保留
        labels = [-100] * user_prompt_len + tokenized["input_ids"][user_prompt_len:]
    else:
        labels = tokenized["input_ids"].copy()

    return {
        "input_ids": tokenized["input_ids"],
        "attention_mask": tokenized["attention_mask"],
        "labels": labels
    }


class TrafficSignalDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # 加载funting_data.json文件
        data_path = os.path.join(data_dir, "merged_output.json")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"数据文件 {data_path} 不存在")

        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                data_list = json.load(f)
                print(f"成功加载 {len(data_list)} 条数据样本")

                # 处理每条数据样本
                for idx, outer_dict in enumerate(data_list):
                    try:
                        data_point = outer_dict["responses"]
                        clean_input = clean_input_text(data_point["input"])

                        # 清理输入文本 - 删除从"RL based timing："开始往后到Please Answer:前的内容
                        cleaned_data_point = {
                            "input": clean_input,
                            "output": data_point["output"]
                        }

                        # 生成并分词提示
                        tokenized = tokenize_and_mask(cleaned_data_point)

                        # 转换为Tensor
                        input_ids = torch.tensor(tokenized["input_ids"])
                        attention_mask = torch.tensor(tokenized["attention_mask"])
                        labels = torch.tensor(tokenized["labels"])

                        self.samples.append({
                            "input_ids": input_ids,
                            "attention_mask": attention_mask,
                            "labels": labels,
                        })

                    except Exception as e:
                        print(f"处理数据点 #{idx} 时出错: {str(e)}")

        except Exception as e:
            print(f"加载数据文件时出错: {str(e)}")
            raise

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class CustomDataCollator(DataCollatorForLanguageModeling):
    def __call__(self, batch):
        return super().__call__(batch)


# 参数高效微调配置
lora_config = LoraConfig(
    r=4,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)

# 数据加载
train_dataset = TrafficSignalDataset(".", tokenizer)  # 当前目录查找funting_data.json
data_collator = CustomDataCollator(tokenizer=tokenizer, mlm=False)


class LossLoggingCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.training_losses = []
        self.log_file = "./imitation_finetuned_model/training_losses.json"

    def on_log(self, args, state, control, logs=None, **kwargs):
        # 在日志事件中捕获损失
        if logs is not None and "loss" in logs:
            self.training_losses.append({
                "step": state.global_step,
                "epoch": state.epoch,
                "loss": logs["loss"]
            })

    def on_train_end(self, args, state, control, **kwargs):
        # 训练结束时保存所有损失记录
        with open(self.log_file, "w") as f:
            json.dump(self.training_losses, f, indent=2)
        print(f"训练损失已保存至 {self.log_file}")
# 修改后的训练参数配置
training_args = TrainingArguments(
    output_dir="./imitation_finetuned_model1",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    num_train_epochs=20,
    logging_dir="./logs",
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=True,  # 启用混合精度
    gradient_checkpointing=False,
    report_to="none",
    optim="adafactor",  # 使用更适合低秩适配的优化器
    remove_unused_columns=False,
    warmup_steps=50,
    logging_steps=50,
    # 保留自定义字段
)

# 初始化训练器
model = get_peft_model(model, lora_config)
model.train()
loss_callback = LossLoggingCallback()
trainer = transformers.Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    callbacks=[loss_callback]
)

# 开始训练
print("Starting training...")
train_result = trainer.train()
trainer.save_metrics("train", train_result.metrics)
trainer.save_state()
# 保存最终模型
model.save_pretrained("./imitation_finetuned_model/imitation_model2")
tokenizer.save_pretrained("./imitation_finetuned_model/imitation_model2")

final_loss = train_result.metrics.get("train_loss", "未知")
print(f"训练完成! 最终损失: {final_loss}")

# 分析损失曲线（可选）
if len(loss_callback.training_losses) > 0:
    import matplotlib.pyplot as plt

    losses = [x["loss"] for x in loss_callback.training_losses]
    steps = [x["step"] for x in loss_callback.training_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses)
    plt.title("训练损失曲线")
    plt.xlabel("训练步数")
    plt.ylabel("损失")
    plt.grid(True)
    plt.savefig("./imitation_finetuned_model/training_loss_curve.png")
    print("训练损失曲线已保存")