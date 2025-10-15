import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,

)
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss  # 导入 CrossEntropyLoss
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json
import os
from aft_rank_loss import make_supervised_data_module
import warnings
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling, TrainerCallback
)

warnings.filterwarnings("ignore", message="Can't initialize NVML")
model_path = "./imitation_finetuned_model/merge_model4"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    load_in_8bit=True,
    # 自动分配模型层到可用设备
    # max_memory={0: "20GiB", 1: "20GiB"},  # 为每个GPU分配内存# 使用半精度减少内存占用
)
lora_config = LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)
accelerator = Accelerator()
model.train()
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# 定义提示模板（与原始代码一致）

class TrafficSignalDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        # 加载数据集文件
        file_path = os.path.join(data_dir, "PN_data_time.json")
        with open(file_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        # 处理每个数据点
        for data_point in self.data:
            self.process_sample(data_point)

    def process_sample(self, data_point):
        data = data_point["responses"]
        data=data[0]
        query = data["input"]
        responses = [gen["output"] for gen in data_point["responses"]]
        scores = [gen["q_score"] for gen in data_point["responses"]]

        # 构建符合排序学习框架的数据结构
        sample = {
            "query": query,
            "responses": responses,
            "scores": scores
        }
        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def train():
    # 加载数据集
    dataset = TrafficSignalDataset("", tokenizer)

    # 转换为排序学习所需的数据格式
    train_data = []
    for sample in dataset:
        train_data.append({
                "query": sample["query"],
                "responses": sample["responses"],
                "scores": sample["scores"]
        })

    # 创建数据模块
    data_module = make_supervised_data_module(tokenizer, train_data)
    IGNORE_INDEX = -100
    class ContrastiveTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                labels=inputs["labels"]
            )
            scores = inputs["scores"]  # 形状: (batch_size, num_responses_per_query)
            weights = F.softmax(scores, dim=1)
            # 动态损失权重计算
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            loss_fct = CrossEntropyLoss(reduction='none')
            nll_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            ).view_as(shift_labels)

            seq_mask = shift_labels != IGNORE_INDEX  # 忽略填充部分的mask
            per_seq_loss = (nll_loss * seq_mask).sum(dim=1) / seq_mask.sum(dim=1).clamp(min=1e-8)

            # 应用softmax权重计算加权损失
            batch_size, num_responses = scores.shape
            weighted_losses = []
            idx = 0
            for i in range(batch_size):
                # 取出当前查询对应的所有回复损失
                rsp_losses = per_seq_loss[idx:idx + num_responses]
                # 按权重加权求和
                weighted_loss = (rsp_losses * weights[i]).sum()
                weighted_losses.append(weighted_loss)
                idx += num_responses

            # 计算批次总损失
            total_loss = torch.stack(weighted_losses).mean()

            return (total_loss, outputs) if return_outputs else total_loss


    class LossLoggingCallback(TrainerCallback):
        def __init__(self):
            super().__init__()
            self.training_losses = []
            self.log_file = "./finetuned_model_pos_negstep/training_loss.json"

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None and "loss" in logs:
                self.training_losses.append({
                    "step": state.global_step,
                    "epoch": state.epoch,
                    "loss": logs["loss"]
                })

        def on_train_end(self, args, state, control, **kwargs):
            with open(self.log_file, "w") as f:
                json.dump(self.training_losses, f, indent=2)
            print(f"训练损失已保存至 {self.log_file}")


    training_args = TrainingArguments(
        output_dir="./finetuned_model_pos_negstep",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=500,
        evaluation_strategy="no",
        fp16=True,  # 启用混合精度
        gradient_checkpointing=False,
        report_to="none",
        optim="adafactor",  # 使用更适合低秩适配的优化器
        remove_unused_columns=False,
        # warmup_steps=50,
        logging_steps=50
        # 保留自定义字段
    )


    loss_callback = LossLoggingCallback()
    trainer = ContrastiveTrainer(
        model=model,
        args=training_args,
        **data_module,
        callbacks=[loss_callback]
    )
    trainer = accelerator.prepare(trainer)

    print("Starting training...")
    train_result = trainer.train()
    trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    # 保存最终模型
    # 保存模型（使用Accelerator处理）
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained("./finetuned_model_pos_negstep/final_model", save_function=accelerator.save)
    tokenizer.save_pretrained("./finetuned_model_pos_negstep/final_model")

    final_loss = train_result.metrics.get("train_loss", "未知")
    print(f"训练完成! 最终损失: {final_loss}")



if __name__ == "__main__":
    train()
