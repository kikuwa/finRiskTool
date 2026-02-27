# JSONL数据质量评分工具使用说明

## 功能概述

这个工具用于对JSONL格式的SFT语料数据进行质量评分，通过调用大语言模型API对每条数据进行评估，并添加索引和分数字段。

## 安装依赖

```bash
pip install requests
```

## 使用方法

### 基本用法
```bash
# 设置API密钥环境变量
export API_KEY="your-api-key-here"

# 运行评分工具
python -m app.services.model_inspection data.jsonl
```

### 高级用法
```bash
# 指定输出文件
python -m app.services.model_inspection data.jsonl -o scored_data.jsonl

# 指定API参数
python -m app.services.model_inspection data.jsonl --api-key "your-key" --api-base "https://your-api-endpoint.com" --model "gpt-4"

# 控制批处理大小
python -m app.services.model_inspection data.jsonl --batch-size 5
```

## 示例数据格式

### 输入数据示例 (data.jsonl)
```json
{"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式并做出预测，而无需显式编程。"}
{"instruction": "写一首关于春天的诗", "input": "", "output": "春天来了，花儿开了，鸟儿叫了，真美好！"}
{"instruction": "计算2+2等于多少", "input": "", "output": "2+2=4"}
{"instruction": "介绍Python语言", "input": "", "output": "Python是一种高级编程语言，具有简洁易读的语法，广泛用于Web开发、数据科学和人工智能等领域。"}
{"instruction": "翻译'hello world'到中文", "input": "", "output": "你好世界"}
```

### 输出数据示例 (scored_data.jsonl)
```json
{"instruction": "解释什么是机器学习", "input": "", "output": "机器学习是人工智能的一个分支，它使计算机能够从数据中学习模式并做出预测，而无需显式编程。", "idx": 0, "score": 4.5}
{"instruction": "写一首关于春天的诗", "input": "", "output": "春天来了，花儿开了，鸟儿叫了，真美好！", "idx": 1, "score": 3.2}
{"instruction": "计算2+2等于多少", "input": "", "output": "2+2=4", "idx": 2, "score": 5.0}
{"instruction": "介绍Python语言", "input": "", "output": "Python是一种高级编程语言，具有简洁易读的语法，广泛用于Web开发、数据科学和人工智能等领域。", "idx": 3, "score": 4.8}
{"instruction": "翻译'hello world'到中文", "input": "", "output": "你好世界", "idx": 4, "score": 5.0}
```

## 环境变量配置

| 环境变量 | 描述 | 默认值 |
|---------|------|--------|
| `API_KEY` | API访问密钥 | 无 |
| `API_BASE` | API基础URL | `https://api.openai.com/v1` |
| `MODEL_NAME` | 模型名称 | `gpt-3.5-turbo` |

## 评分标准

工具使用以下标准进行评分：

1. **准确性** (0-5分): output是否正确解决instruction需求，无事实性错误或逻辑矛盾
2. **信息量** (0-5分): 避免冗余回复或过于简略的答案
3. **语法规范** (0-5分): 无拼写错误、标点误用或语义不通顺问题
4. **初步筛选**: 包含敏感词、广告或无效占位符的样本会扣分

## 错误处理

- API请求失败会自动重试3次
- 评分失败的数据会使用默认分数0.0
- 缺少必要字段的数据会跳过评分
- 支持批处理控制以避免API限流

## 注意事项

1. 确保API密钥有效且有足够的额度
2. 大型数据集建议使用较小的批处理大小
3. 输出文件会保留原始数据的所有字段
4. 建议先在小样本上测试后再处理完整数据集