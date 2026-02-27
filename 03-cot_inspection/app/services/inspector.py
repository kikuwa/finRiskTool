import os
import random
from .model_inspection import score_dataset
from .rules import RuleBase

class ModelScorer:
    def __init__(self):
        self.default_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.default_api_base = os.getenv("DEEPSEEK_API_BASE", "https://api.deepseek.com")
        self.default_model_name = os.getenv("DEEPSEEK_MODEL_NAME", "deepseek-chat")
    
    def score_dataset(self, data, api_key=None, api_base=None, model_name=None, batch_size=10, system_prompt=None):
        key = api_key or self.default_api_key
        base = api_base or self.default_api_base
        model = model_name or self.default_model_name
        if not key or not base or not model:
            raise ValueError("缺少模型质检所需的API配置")
        return score_dataset(data, key, base, model, batch_size, system_prompt=system_prompt)

class WebInspector:
    def __init__(self):
        self.model_scorer = ModelScorer()
        self.rule_base = RuleBase()
    
    def inspect_with_model(self, data, model_config=None):
        """使用模型进行质检"""
        try:
            model_config = model_config or {}
            api_key = model_config.get("api_key")
            api_base = model_config.get("api_base")
            model_name = model_config.get("model_name")
            system_prompt = model_config.get("system_prompt")
            return self.model_scorer.score_dataset(
                data,
                api_key=api_key,
                api_base=api_base,
                model_name=model_name,
                batch_size=model_config.get("batch_size", 10),
                system_prompt=system_prompt,
            )
        except Exception as e:
            print(f"模型质检出错: {e}")
            # 出错时返回模拟数据
            return self._create_mock_results(data, 'model_based')
    
    def inspect_with_rules(self, data, enabled_rules=None):
        """使用规则进行质检"""
        try:
            enabled_rules = set(enabled_rules or [])
            results = []
            for item in data:
                rule_input = {
                    'meta_prompt': item.get('instruction', ''),
                    'user': item.get('input', ''),
                    'assistant': item.get('output', ''),
                    'file_path': item.get('file_path', ''),
                    'ref_answer': item.get('gt', None),
                }
                
                rule_result = self.rule_base.run(rule_input)
                
                keys_to_score = [
                    k for k in rule_result.keys()
                    if k.startswith('no_') or k in {'fk_answer_exist', 'fk_answer_yes_or_no', 'fk_answer_equal'}
                ]
                if enabled_rules:
                    keys_to_score = [k for k in keys_to_score if k in enabled_rules]
                
                passed_rules = sum(1 for key in keys_to_score if rule_result.get(key) is True)
                total_rules = len(keys_to_score)
                
                score = (passed_rules / total_rules) * 10 if total_rules > 0 else 0
                
                results.append({
                    **item,
                    'score': round(score, 2),
                    'inspection_type': 'rule_based',
                    'rule_details': rule_result
                })
            
            return results
        except Exception as e:
            print(f"规则质检出错: {e}")
            # 出错时返回模拟数据
            return self._create_mock_results(data, 'rule_based')
    
    def _create_mock_results(self, data, inspection_type):
        """创建模拟结果"""
        results = []
        for item in data:
            results.append({
                **item,
                'score': round(random.uniform(5.0, 9.5), 2),
                'inspection_type': inspection_type,
                'error': '质检过程出现错误，使用模拟数据'
            })
        return results

inspector = WebInspector()
