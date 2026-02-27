import json
import logging
import re
import time
import argparse
import requests
from pathlib import Path

# 配置
API_KEY = "sk-f4a50882f4f44dc8a1126ffc185a72b7"
API_BASE = "https://api.deepseek.com"
MODEL_NAME = "deepseek-chat"
BATCH_SIZE = 10
MAX_RETRIES = 3
TIMEOUT = 30
DEFAULT_SCORE = 0.0

# 日志配置
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# 提示词模板
SYSTEM_PROMPT = """你是一个SFT语料质量评估专家，任务是从原始数据集中筛选出高质量样本，特别关注思维链合成数据的难度和质量。

[质量检查维度]
1. 基础质量（0-3分）：
   - 准确性：需正确解决问题，无事实性错误或逻辑矛盾
   - 信息量：避免冗余回复或过于简略的答案，内容充实有价值
   - 语法规范：无拼写错误、标点误用或语义不通顺问题
   - 格式规范：符合对话格式要求，角色分明

2. 思维链难度评估（0-5分）：
   - 推理复杂度：需要多步推理、逻辑跳跃或复杂分析的难度越高
   - 知识深度：涉及专业知识、跨领域知识或深度理解的难度
   - 创造性：需要创造性思维、新颖视角或独特解决方案
   - 上下文理解：对复杂上下文、隐含信息或微妙语义的理解要求
   - 问题复杂度：原始问题的复杂程度和挑战性

3. 合成质量评估（0-2分）：
   - 自然流畅度：思维链是否自然连贯，无明显人工拼接痕迹
   - 逻辑一致性：推理过程逻辑严密，前后一致无矛盾
   - 步骤完整性：关键推理步骤完整，无跳跃或缺失
   - 可解释性：推理过程清晰可理解，便于人类follow

[评估步骤]
1. 初步筛选：检查敏感词、广告、无效占位符、明显低质量内容（直接0分）
2. 基础质量评分：按准确性、信息量、语法规范打分（0-3分）
3. 难度评估：评估思维链合成难度，难度越高得分越低（0-5分，越高越难）
4. 合成质量：评估思维链的自然性、逻辑性和完整性（0-2分）

[最终得分计算]
最终得分 = 基础质量分 × (1 - 难度分/10) + 合成质量分
说明：难度越高（难度分越大），对基础质量的折扣越大，体现合成难度对质量的负面影响

[格式要求]
---
语料得分：[最终得分（保留2位小数）]
---"""

def user_prompt(instruction, input_text, output):
    return f"""请评估以下语料质量：

Instruction: {instruction}

Input: {input_text}

Output: {output}

请按照要求进行评分并返回格式化的结果。"""

def extract_score(content):
    """从响应中提取分数"""
    if not content:
        return None
    
    # 主模式
    match = re.search(r"语料得分：[^\d]*(\d+(?:\.\d+)?)", content)
    if match:
        return float(match.group(1))
    
    # 回退：取最后一个数字
    matches = re.findall(r"(\d+(?:\.\d+)?)", content)
    return float(matches[-1]) if matches else None

def api_request(messages, api_key, api_base, model_name):
    """发送API请求"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 100,
    }
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.post(
                f"{api_base}/chat/completions",
                headers=headers,
                json=payload,
                timeout=TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logging.error(f"API错误 (第{attempt}次尝试): {resp.status_code}")
        except Exception as e:
            logging.error(f"请求异常 (第{attempt}次尝试): {e}")
        
        if attempt < MAX_RETRIES:
            time.sleep(2 ** (attempt - 1))
    return None

def read_jsonl(file_path):
    """读取JSONL文件"""
    path = Path(file_path)
    if not path.exists():
        logging.error(f"文件不存在: {file_path}")
        return []
    
    data = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                logging.warning(f"第{lineno}行JSON解析失败，已跳过")
    return data

def write_jsonl(file_path, data):
    """写入JSONL文件"""
    Path(file_path).write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in data) + "\n",
        encoding="utf-8",
    )

def score_single(instruction, input_text, output, api_key, api_base, model_name, system_prompt=None):
    """对单条样本评分"""
    messages = [
        {"role": "system", "content": system_prompt or SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(instruction, input_text, output)},
    ]
    content = api_request(messages, api_key, api_base, model_name)
    score = extract_score(content) if content else None
    return score, content or ""

def score_dataset(data, api_key, api_base, model_name, batch_size=BATCH_SIZE, system_prompt=None):
    """批量评分"""
    scored = []
    total = len(data)
    
    for idx, item in enumerate(data):
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if not (instruction and output):
            logging.warning(f"第{idx + 1}条样本缺少必要字段，使用默认分数")
            score = DEFAULT_SCORE
        else:
            score, _ = score_single(instruction, input_text, output, api_key, api_base, model_name, system_prompt=system_prompt)
            if score is None:
                score = DEFAULT_SCORE
        
        new_item = item.copy()
        new_item["idx"] = idx
        new_item["score"] = score
        scored.append(new_item)
        
        # 批处理限速
        if (idx + 1) % batch_size == 0:
            logging.info(f"已完成 {idx + 1}/{total}，暂停2秒...")
            time.sleep(2)
    return scored

def main():
    parser = argparse.ArgumentParser(description="JSONL数据质量评分工具")
    parser.add_argument("input_file", help="输入JSONL文件路径")
    parser.add_argument("-o", "--output", help="输出文件路径（默认覆盖原文件）")
    parser.add_argument("--api-key", default=API_KEY, help="API密钥")
    parser.add_argument("--api-base", default=API_BASE, help="API基础URL")
    parser.add_argument("--model", default=MODEL_NAME, help="模型名称")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="批处理大小")
    
    args = parser.parse_args()
    
    logging.info(f"开始读取文件: {args.input_file}")
    data = read_jsonl(args.input_file)
    if not data:
        logging.error("未读取到有效数据")
        return
    
    logging.info(f"共{len(data)}条样本，开始评分...")
    scored = score_dataset(data, args.api_key, args.api_base, args.model, args.batch_size)
    
    out_path = args.output or args.input_file
    logging.info(f"写入结果到: {out_path}")
    write_jsonl(out_path, scored)
    
    success = sum(1 for item in scored if item["score"] > 0)
    logging.info(f"处理完成！成功评分: {success}/{len(data)}，结果已保存至: {out_path}")

if __name__ == "__main__":
    main()
