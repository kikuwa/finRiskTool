
import json
import os
import logging
import requests
import time
import hashlib
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional, List, Any

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    推理引擎服务：封装 DeepSeek R1 的批量推理逻辑
    """
    
    def __init__(self):
        self._stop_event = threading.Event()
        self._status = {
            "status": "idle", # idle, running, stopped, completed, error
            "total": 0,
            "processed": 0,
            "current_file": None,
            "error": None
        }
        self._lock = threading.Lock()

    def get_status(self) -> Dict:
        """获取当前运行状态"""
        return self._status

    def stop(self):
        """停止任务"""
        self._stop_event.set()
        with self._lock:
            if self._status["status"] == "running":
                self._status["status"] = "stopped"

    def run(self, config: Dict):
        """
        运行推理任务
        config: {
            'api_key': str,
            'input_file': str,
            'output_file': str,
            'model': str,
            'workers': int
        }
        """
        self._stop_event.clear()
        
        with self._lock:
            self._status = {
                "status": "running",
                "total": 0,
                "processed": 0,
                "current_file": config.get('input_file'),
                "error": None
            }

        try:
            self._execute_inference(config)
            with self._lock:
                if self._status["status"] == "running":
                    self._status["status"] = "completed"
        except Exception as e:
            logger.error(f"推理任务异常: {e}")
            with self._lock:
                self._status["status"] = "error"
                self._status["error"] = str(e)

    def _execute_inference(self, config: Dict):
        input_file = config['input_file']
        output_file = config['output_file']
        api_key = config['api_key']
        workers = config.get('workers', 5)
        model = config.get('model', 'deepseek-reasoner')
        base_url = config.get('base_url', "https://api.deepseek.com/chat/completions")

        # 1. 加载数据
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"输入文件不存在: {input_file}")

        all_data = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        all_data.append(json.loads(line))
                    except:
                        pass
        
        # 2. 断点续传
        processed_hashes = self._load_processed_hashes(output_file)
        tasks = []
        for item in all_data:
            if self._get_data_hash(item) not in processed_hashes:
                tasks.append(item)
        
        with self._lock:
            self._status["total"] = len(all_data)
            self._status["processed"] = len(all_data) - len(tasks)

        if not tasks:
            return

        # 3. 确保输出目录
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # 4. 多线程处理
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_item = {
                executor.submit(self._process_item, item, api_key, model, base_url): item 
                for item in tasks
            }
            
            for future in as_completed(future_to_item):
                if self._stop_event.is_set():
                    break
                
                result_item = future.result()
                if result_item:
                    # 写入结果
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result_item, ensure_ascii=False) + '\n')
                
                with self._lock:
                    self._status["processed"] += 1

    def _process_item(self, item: Dict, api_key: str, model: str, base_url: str) -> Optional[Dict]:
        try:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            if not instruction and not input_text:
                return None

            result = self._call_llm(instruction, input_text, api_key, model, base_url)
            
            # 只有成功获取结果才返回
            if result['content'] or result['reasoning']:
                item["output"] = f"<think>{result['reasoning']}</think> <answer>{result['content']}</answer>"
                return item
            return None
        except Exception as e:
            logger.error(f"处理单条数据失败: {e}")
            return None

    def _call_llm(self, instruction: str, user_input: str, api_key: str, model: str, base_url: str) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"{instruction}\n\n{user_input}".strip()},
        ]
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "temperature": 0.6
        }
        
        reasoning_text = ""
        content_text = ""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(base_url, headers=headers, json=payload, stream=True, timeout=120)
                if response.status_code != 200:
                    if response.status_code == 429:
                        time.sleep(2 * (attempt + 1))
                        continue
                    raise Exception(f"API Error: {response.text}")
                
                for line in response.iter_lines():
                    if not line: continue
                    decoded = line.decode('utf-8').strip()
                    if decoded.startswith("data:"):
                        data_str = decoded[5:].strip()
                        if data_str == "[DONE]": break
                        try:
                            data = json.loads(data_str)
                            delta = data['choices'][0]['delta']
                            if 'reasoning_content' in delta and delta['reasoning_content']:
                                reasoning_text += delta['reasoning_content']
                            if 'content' in delta and delta['content']:
                                content_text += delta['content']
                        except:
                            pass
                
                return {"reasoning": reasoning_text, "content": content_text}
                
            except Exception as e:
                logger.warning(f"API调用尝试 {attempt+1} 失败: {e}")
                time.sleep(2)
        
        return {"reasoning": "", "content": ""}

    def _get_data_hash(self, item: Dict) -> str:
        content = f"{item.get('instruction', '')}{item.get('input', '')}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()

    def _load_processed_hashes(self, output_file: str) -> set:
        processed = set()
        if not os.path.exists(output_file):
            return processed
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    if not line.strip(): continue
                    item = json.loads(line)
                    processed.add(self._get_data_hash(item))
                except:
                    continue
        return processed
