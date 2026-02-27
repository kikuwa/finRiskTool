# -*- coding: utf-8 -*-
"""
@Time    : 2025/11/25
@File    : optimized_cot_infer.py
@Desc    : DeepSeek R1 API 批量推理优化版
           1. 多线程并发请求 (大幅提升速度)
           2. 实时增量写入结果 (防止数据丢失)
           3. 支持断点续传 (跳过已处理数据)
           4. 适配 DeepSeek R1 的 reasoning_content 字段
"""

import json
import os
import logging
import requests
import time
import hashlib
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import Dict, Any, Optional

try:
    from dotenv import load_dotenv
    # 显式加载当前脚本同级目录下的 .env 文件
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
    load_dotenv(env_path)
except ImportError:
    pass

# ================= 配置区 =================
class Config:
    API_KEY = os.getenv("DEEPSEEK_API_KEY")
    BASE_URL = "https://api.deepseek.com/chat/completions"
    MODEL_NAME = "deepseek-reasoner"
    
    # 动态获取当前脚本所在目录，确保路径正确
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(BASE_DIR, "data", "alpaca_dataset.jsonl")
    OUTPUT_FILE = os.path.join(BASE_DIR, "data", "alpaca_dataset_with_output.jsonl")
    
    MAX_WORKERS = 5
    TEMPERATURE = 0.6
    MAX_RETRIES = 5
    RETRY_DELAY = 2
    TIMEOUT = 180

# ================= 日志配置 =================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 文件写入锁，防止多线程并发写入导致数据错乱
write_lock = threading.Lock()

def get_data_hash(item: Dict) -> str:
    """
    生成数据的唯一哈希指纹，用于断点续传的去重识别
    通过instruction和input字段组合生成MD5哈希值
    """
    content = f"{item.get('instruction', '')}{item.get('input', '')}"
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def load_processed_hashes(output_file: str) -> set:
    """
    从输出文件中加载已处理数据的哈希值集合
    用于断点续传，跳过已完成的记录
    """
    processed = set()
    if not os.path.exists(output_file):
        return processed
    
    with open(output_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                if not line.strip(): continue
                item = json.loads(line)
                # 重新计算哈希值以识别已处理的记录
                h = get_data_hash(item)
                processed.add(h)
            except Exception:
                continue
    logger.info(f"发现已处理数据: {len(processed)} 条")
    return processed

def call_deepseek_stream(instruction: str, user_input: str, ground_truth: str) -> Dict[str, str]:
    """
    调用 DeepSeek API，专门适配 deepseek-reasoner 的返回结构
    支持流式响应，分别获取思维链和最终答案
    返回字典: {'reasoning': str, 'content': str}
    """
    
    # 构造对话消息，R1推荐直接通过system/user交互
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"{instruction}\n\n{user_input}".strip()},
    ]



    headers = {
        "Authorization": f"Bearer {Config.API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": Config.MODEL_NAME,
        "messages": messages,
        "temperature": Config.TEMPERATURE,
        "stream": True
    }

    # 初始化变量存储思维链和内容
    reasoning_text = ""
    content_text = ""

    for attempt in range(Config.MAX_RETRIES):
        try:
            response = requests.post(
                Config.BASE_URL,
                headers=headers,
                json=payload,
                stream=True,
                timeout=Config.TIMEOUT
            )
            
            if response.status_code != 200:
                error_msg = response.text
                logger.warning(f"HTTP {response.status_code} - 尝试 {attempt+1}/{Config.MAX_RETRIES}: {error_msg}")
                # 429状态码表示限速，使用指数退避策略
                if response.status_code == 429:
                    time.sleep(Config.RETRY_DELAY * (2 ** attempt))
                    continue
                raise Exception(f"API Error: {response.status_code}")

            # 流式解析API响应数据
            for line in response.iter_lines():
                if not line: continue
                decoded_line = line.decode('utf-8').strip()
                
                if decoded_line.startswith("data:"):
                    data_str = decoded_line[5:].strip()
                    if data_str == "[DONE]": break
                    
                    try:
                        data_json = json.loads(data_str)
                        delta = data_json['choices'][0]['delta']
                        
                        # R1模型特有的思维链字段，存储推理过程
                        if 'reasoning_content' in delta and delta['reasoning_content']:
                            reasoning_text += delta['reasoning_content']
                        
                        # 最终答案内容字段
                        if 'content' in delta and delta['content']:
                            content_text += delta['content']
                            
                    except Exception:
                        continue
            
            # 成功获取到内容（思维链或答案），返回结果
            if content_text or reasoning_text:
                return {
                    "reasoning": reasoning_text,
                    "content": content_text
                }
            else:
                raise Exception("API 返回内容为空")

        except Exception as e:
            logger.error(f"Request failed: {e}")
            if attempt < Config.MAX_RETRIES - 1:
                time.sleep(Config.RETRY_DELAY)
            else:
                return {"content": "", "reasoning": ""}

    return {"content": "", "reasoning": ""}

def process_item(item: Dict) -> Optional[Dict]:
    """
    单个数据项处理函数，供线程池调用
    将API返回结果整合到原始数据项中
    """
    try:
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")
        ground_truth=item.get("gt", "")
        # 即使input为空也可能有有效问题，检查instruction和input是否都为空
        if not instruction and not input_text:
            return None

        result = call_deepseek_stream(instruction, input_text,ground_truth)
        
        # 将思维链和答案整合为指定格式
        item["output"]='<think>'+result["reasoning"]+'</think> <answer>'+result["content"]+'</answer>'
        
        return item
    except Exception as e:
        logger.error(f"处理数据出错: {e}")
        return None

def main():
    """主函数：执行批量推理流程"""
    parser = argparse.ArgumentParser(description="DeepSeek R1 API 批量推理优化版")
    parser.add_argument("--api_key", type=str, help="API Key (或者设置 DEEPSEEK_API_KEY 环境变量)")
    parser.add_argument("--input", type=str, help="输入文件路径")
    parser.add_argument("--output", type=str, help="输出文件路径")
    parser.add_argument("--workers", type=int, help="并发线程数")
    parser.add_argument("--model", type=str, help="模型名称")
    parser.add_argument("--log_file", type=str, help="日志文件路径")
    
    args = parser.parse_args()
    
    # 更新配置
    if args.api_key:
        Config.API_KEY = args.api_key
    if args.input:
        Config.INPUT_FILE = args.input
    if args.output:
        Config.OUTPUT_FILE = args.output
    if args.workers:
        Config.MAX_WORKERS = args.workers
    if args.model:
        Config.MODEL_NAME = args.model
        
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

    # API Key 校验
    if not Config.API_KEY:
        logger.error("未设置 API KEY。请通过 --api_key 参数或 DEEPSEEK_API_KEY 环境变量设置。")
        return

    try:
        Config.API_KEY.encode('ascii')
    except UnicodeEncodeError:
        logger.error(f"API KEY 格式错误: 包含非 ASCII 字符 (如中文)。当前 Key: {Config.API_KEY[:5]}...{Config.API_KEY[-5:] if len(Config.API_KEY)>5 else ''}")
        logger.error("请检查环境变量 DEEPSEEK_API_KEY 或传入的参数是否包含了说明性文字 (如 'sk-您的Key')。")
        return

    # 1. 加载输入数据
    if not os.path.exists(Config.INPUT_FILE):
        logger.error(f"输入文件不存在: {Config.INPUT_FILE}")
        return

    logger.info("正在加载数据...")
    all_data = []
    with open(Config.INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    all_data.append(json.loads(line))
                except:
                    pass
    
    # 2. 断点续传：检查已处理的数据
    processed_hashes = load_processed_hashes(Config.OUTPUT_FILE)
    
    tasks = []
    skipped_count = 0
    
    for item in all_data:
        h = get_data_hash(item)
        if h in processed_hashes:
            skipped_count += 1
        else:
            tasks.append(item)
            
    logger.info(f"总数据: {len(all_data)}, 已完成: {skipped_count}, 待处理: {len(tasks)}")
    
    if not tasks:
        logger.info("所有数据已处理完毕。")
        return

    # 3. 多线程批量处理
    # 确保输出目录存在
    os.makedirs(os.path.dirname(Config.OUTPUT_FILE), exist_ok=True)
      
    logger.info(f"开始推理，线程数: {Config.MAX_WORKERS}...")
    
    pbar = tqdm(total=len(tasks), desc="Processing")
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        # 提交所有处理任务到线程池
        future_to_item = {executor.submit(process_item, item): item for item in tasks}
        
        # 等待任务完成并处理结果
        for future in as_completed(future_to_item):
            result_item = future.result()
            
            if result_item:
                # 使用线程锁确保并发写入安全
                with write_lock:
                    with open(Config.OUTPUT_FILE, 'a', encoding='utf-8') as f_out:
                        f_out.write(json.dumps(result_item, ensure_ascii=False) + '\n')
            
            pbar.update(1)
            
    pbar.close()
    logger.info(f"推理完成！结果已保存至 {Config.OUTPUT_FILE}")

if __name__ == "__main__":
    main()
