
import pandas as pd
import json
import argparse
import logging
from tqdm import tqdm
from typing import List, Dict, Optional
from pathlib import Path
import os
import openai

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ================= 1. 模板配置 =================

# 基础模板 (用于提供给大模型参考)
BASE_INSTRUCTION_TEMPLATE = """【角色】
您是一位具备金融风控专业知识的智能分析师，擅长结合企业定性及定量的指标数据对企业违约风险进行评估。

【任务指令】
请依据下方的【风险分析框架】和输入中提供的【企业指标数据报告】，按以下步骤对该企业信用风险进行综合判断：
1. 依据【风险分析框架】对企业经营、债务、信用等维度进行分析；
2. 通过指标间的交叉验证，判断是否存在框架未覆盖的风险点；
3. 综合评估企业在未来3个月内的违约风险。

【风险分析框架】
（1）企业经营稳定性：结合企业属性、行业及收支流水，判断经营是否良好及具备偿债能力。
（2）企业债务和流动性：结合短借长投、债务结构，判断是否过度举债。
（3）其它风险信号：关注机器学习评分、行内评级及外部征信数据。

【强制约束】
1. 必须建立双向验证机制：当关键信号与指标数据矛盾时，需启动二次核查；
2. 严格区分事实指标与推测结论。

【输出要求】
根据以上分析，如果认为该企业在未来3个月内不会违约，请输出“否”；如果存在违约风险，请输出“是”。请直接输出结果，不要包含其他分析过程。

【企业指标数据报告】
（动态填充部分）
"""

# 默认的备用模板 (如果大模型调用失败使用此模板)
FALLBACK_INSTRUCTION_TEMPLATE = """【角色】
您是一位具备金融风控专业知识的智能分析师，擅长结合企业定性及定量的指标数据对企业违约风险进行评估。

【任务指令】
请依据下方的【风险分析框架】和输入中提供的【企业指标数据报告】，按以下步骤对该企业信用风险进行综合判断：
1. 依据【风险分析框架】对企业经营、债务、信用等维度进行分析；
2. 通过指标间的交叉验证，判断是否存在框架未覆盖的风险点；
3. 综合评估企业在未来3个月内的违约风险。

【风险分析框架】
（1）企业经营稳定性：结合企业属性（{nature}）、行业（{industry_cd_desc}）及收支流水，判断经营是否良好及具备偿债能力。关注近24个月营业收入（{corp_oper_income_24m}）与支出（{corp_oper_expns_24m}），以及经营净收益的变动（近12个月：{corp_oper_diff_12m}，近24个月：{corp_oper_diff_24m}）。参考纳税等级（{taxes_level_cd}）。
（2）企业债务和流动性：结合短借长投、债务结构，判断是否过度举债。分析当前金融资产余额（{corp_curr_fin_asset_bal}）及其变动（{corp_curr_fin_asset_bal_diff}），以及近3个月平均存款余额（{total_avg_depst_bal_3m}）。关注融资情况（融资机构数：{n_ph_num4}，融资余额：{n_ph_amt4}，本行融资：{n_ph_amt2}）及授信使用（授信额度：{creditamt}，已用：{amtused}）。
（3）其它风险信号：关注贷后检查结果（{check_result}）、风险标识（{risk_flag}）、重要股东风险（{reserve_8}）及押品跌幅（{down30}）。检查逾期情况（近3个月逾期：{rep_overdue_cnt_3m}，近24个月逾期：{rep_overdue_cnt_24m}，当前逾期金额：{corp_curr_lab_amt}）。参考模型评分（{threshold}）和外部规则命中（{n_u0343_hit}, {n_ph_amt1_hit}, {n_ph_amt2_hit}）。

【强制约束】
1. 必须建立双向验证机制：当关键信号与指标数据矛盾时，需启动二次核查；
2. 严格区分事实指标与推测结论。

【输出要求】
根据以上分析，如果认为该企业在未来3个月内不会违约，请输出“否”；如果存在违约风险，请输出“是”。请直接输出结果，不要包含其他分析过程。

【企业指标数据报告】
企业名称：{entityname}
行业：{industry_cd_desc}
分行：{branchname}
首次授信年份：{firstcreditx_year}
（详细数据见Input部分）"""

# 【Input】：包含详细的业务数据
INPUT_DATA_TEMPLATE = """【企业指标数据报告】
基本信息：
- 企业名称：{entityname}
- 企业性质：{nature}
- 首次授信年份：{firstcreditx_year}
- 所属行业：{industry_cd_desc}
- 所属分行：{branchname}

信用评级与风险：
- 信用评级：{cred_level_desc}
- 风险阈值：{threshold}
- 风险标识：{risk_flag}
- 统计算法结果：{tj_result}
- 贷后检查结果：{check_result}

经营状况：
- 近24个月营业收入：{corp_oper_income_24m}
- 近24个月营业支出：{corp_oper_expns_24m}
- 近12个月经营净收益：{corp_oper_diff_12m}
- 近24个月经营净收益：{corp_oper_diff_24m}
- 纳税等级：{taxes_level_cd}

资产与负债：
- 当前金融资产余额：{corp_curr_fin_asset_bal}
- 金融资产余额变动：{corp_curr_fin_asset_bal_diff}
- 近3个月平均存款余额：{total_avg_depst_bal_3m}
- 融资机构数量：{n_ph_num4}
- 融资余额：{n_ph_amt4}
- 融资金额：{n_ph_amt2}
- 授信额度：{creditamt}
- 已用额度：{amtused}

逾期与违约：
- 未结清比例：{unpayoff_mg_ratio}
- 近3个月还款逾期次数：{rep_overdue_cnt_3m}
- 近24个月还款逾期次数：{rep_overdue_cnt_24m}
- 当前逾期金额：{corp_curr_lab_amt}
- 本金/利息逾期标识：{flag_8}

规则命中：
- 规则U0343命中情况：{n_u0343_hit}
- 规则PH_AMT1命中情况：{n_ph_amt1_hit}
- 规则PH_AMT2命中情况：{n_ph_amt2_hit}

信用卡及其他：
- 信用卡额度：{rep_crdt_card_limit}
- 近6个月平均信用卡额度：{rep_crdt_card_avg_limit_6m}
- 信用卡已用额度：{rep_crdt_card_used_limit}
- 信用卡预支使用率：{rep_crdt_card_precent}
- 重要股东风险：{reserve_8}
- 下调30%标识：{down30}"""


# ================= 2. LLM 工具函数 =================

def generate_template_from_llm(features: List[str], api_key: str = None, base_url: str = None, model: str = "gpt-3.5-turbo") -> str:
    """
    调用大模型生成带槽位的 Instruction 模板
    """
    logger.info("正在请求大模型生成新的 Prompt 模板...")
    
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("未检测到 OPENAI_API_KEY，将使用备用模板。")
        return FALLBACK_INSTRUCTION_TEMPLATE

    client = openai.OpenAI(api_key=api_key, base_url=base_url)

    # 构建 Meta-Prompt
    feature_list_str = "\n".join([f"- {f}" for f in features])
    
    system_prompt = "你是一个精通提示词工程（Prompt Engineering）和金融风控的专家。"
    user_prompt = f"""
请根据以下【基础模板】和【特征字段列表】，重写并优化一个用于企业违约风险评估的 Prompt 模板。

【任务要求】
1. 保持基础模板的整体结构（角色、任务指令、风险分析框架、强制约束、输出要求）。
2. 重点优化【风险分析框架】部分：请根据特征字段的业务含义，将它们合理地嵌入到对应的分析维度中（如经营稳定性、债务流动性、其他风险信号）。
3. 嵌入槽位时，请严格使用Python格式化字符串的语法，即 `{{字段名}}`。例如：`营业收入为 {{corp_oper_income_24m}}`。
4. 确保所有重要的特征字段都被利用到。
5. 在模板末尾保留【企业指标数据报告】部分，用于展示一些基础信息（如 entityname, industry_cd_desc 等）。
6. 直接返回优化后的模板内容，不要包含任何解释性文字或Markdown代码块标记（如 ```）。

【基础模板】
{BASE_INSTRUCTION_TEMPLATE}

【特征字段列表】
{feature_list_str}
"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1500
        )
        generated_content = response.choices[0].message.content.strip()
        
        # 简单的清理，防止 LLM 返回 Markdown 代码块
        if generated_content.startswith("```"):
            lines = generated_content.split('\n')
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines[-1].startswith("```"):
                lines = lines[:-1]
            generated_content = "\n".join(lines)
            
        logger.info("大模型成功生成新模板！")
        return generated_content

    except Exception as e:
        logger.error(f"调用大模型失败: {e}。将使用备用模板。")
        return FALLBACK_INSTRUCTION_TEMPLATE


# ================= 3. 数据处理工具函数 =================

def save_as_jsonl(data_list: List[Dict], file_name: str):
    """保存为 JSONL 格式"""
    if not data_list:
        return
    with open(file_name, 'a', encoding='UTF-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def clean_check_result(text: Optional[str]) -> str:
    """清洗贷后检查文本，保留最近两条"""
    if pd.isna(text) or text is None:
        return "无"
    try:
        # 兼容 generate_mock_data 生成的 \n\n 分隔格式
        records = str(text).split("\n\n")
        parsed = []
        for r in records:
            r = r.strip()
            if not r: continue
            # 简单处理，假设开头是日期
            parts = r.split(" ")
            if len(parts) >= 2:
                # 尝试解析日期，这里简单处理
                parsed.append((parts[0], " ".join(parts[1:])))
            else:
                parsed.append(("0000-00-00", r))
        
        # 按日期降序排序 (字符串比较即可)
        parsed.sort(key=lambda x: x[0], reverse=True)
        
        # 格式化为 [Date] Content
        return "\n".join([f"[{t}] {c}" for t, c in parsed[:2]])
    except:
        return str(text)

def format_percentage(row, limit_col, used_col):
    """计算百分比的辅助函数"""
    try:
        limit = pd.to_numeric(row.get(limit_col, 0), errors='coerce')
        used = pd.to_numeric(row.get(used_col, 0), errors='coerce')
        if limit > 0:
            return f"{used / limit:.2%}"
        return "0.00%"
    except:
        return "0.00%"

def process_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """向量化预处理"""
    print("正在进行数据预处理...")
    # 清洗文本列
    if 'check_result' in df.columns:
        df['check_result'] = df['check_result'].apply(clean_check_result)
    # 填充空值
    df = df.fillna("未知")
    return df

def build_alpaca_item(row_series: pd.Series, instruction_template: str) -> Optional[Dict]:
    """
    构建单条 Alpaca 格式数据
    """
    # 1. 准备数据字典
    data_dict = row_series.to_dict()
    
    # 2. 计算动态指标 (信用卡使用率)
    data_dict['rep_crdt_card_precent'] = format_percentage(
        row_series, 'rep_crdt_card_limit', 'rep_crdt_card_used_limit'
    )
    
    # 3. 填充 Input 模板
    try:
        input_content = INPUT_DATA_TEMPLATE.format(**data_dict)
    except KeyError as e:
        # print(f"Warning: Input 模板数据缺失字段 {e}")
        return None
    except Exception as e:
        print(f"Error: Input 模板填充未知错误 {e}")
        return None

    # 4. 填充 Instruction 模板
    try:
        # 这里使用传入的 instruction_template
        instruction_content = instruction_template.format(**data_dict)
    except KeyError as e:
        # 容错：如果生成的模板包含了一些数据中不存在的字段，尝试用 "未知" 填充或忽略
        # print(f"Warning: Instruction 模板字段缺失 {e}，尝试使用默认值填充")
        try:
             # 简单的 fallback 尝试：构建一个 defaultdict 或者 safe format
             # 这里简单处理：如果报错，则跳过这条或者用空字符串替换
             # 为了保证流程不中断，我们返回 None 或者尝试修复
             return None
        except:
             return None
    except Exception as e:
        print(f"Error: Instruction 模板填充失败: {e}")
        return None

    # 5. 获取 GT (label)
    gt_value = str(row_series.get('label', ''))
    if not gt_value or gt_value == 'nan':
        gt_value = "是" if row_series.get('is_risky') else "否"

    # 6. 组装最终 JSON 对象
    return {
        "instruction": instruction_content.strip(),
        "input": input_content.strip(),
        "output": "",  # 置空
        "gt": gt_value
    }

# ================= 4. 主程序 =================

def main():
    parser = argparse.ArgumentParser(description="生成 Alpaca 格式的训练数据")
    parser.add_argument("--input", type=str, default=None, help="输入数据文件路径 (CSV/Parquet)")
    parser.add_argument("--output", type=str, default=None, help="输出 JSONL 文件路径")
    parser.add_argument("--features", type=str, default=None, help="特征描述文件路径 (CSV/TXT)")
    parser.add_argument("--api_key", type=str, default=None, help="OpenAI API Key (可选)")
    parser.add_argument("--base_url", type=str, default=None, help="OpenAI Base URL (可选)")
    parser.add_argument("--model", type=str, default="gpt-3.5-turbo", help="使用的模型名称")
    
    args = parser.parse_args()

    # 配置路径
    current_dir = Path(__file__).parent.absolute()
    data_dir = current_dir / "data"
    
    # 默认路径
    input_path = Path(args.input) if args.input else data_dir / "mock_risk_data.csv"
    output_file = Path(args.output) if args.output else data_dir / "alpaca_dataset.jsonl"
    features_path = Path(args.features) if args.features else data_dir / "feature_description.csv"

    # 1. 读取数据
    if not input_path.exists():
        logger.error(f"未找到数据文件: {input_path}")
        return

    try:
        if input_path.suffix == '.csv':
            df = pd.read_csv(input_path)
        elif input_path.suffix == '.parquet':
            df = pd.read_parquet(input_path)
        else:
            logger.error("不支持的文件格式，请使用 .csv 或 .parquet")
            return
    except Exception as e:
        logger.error(f"读取数据文件失败: {e}")
        return

    logger.info(f"数据加载完成，共 {len(df)} 条。")
    
    # 2. 获取特征列表 (用于给 LLM 生成模板)
    features = []
    if features_path.exists():
        try:
            # 假设 feature_description.csv 有 feature_name 列
            f_df = pd.read_csv(features_path)
            if 'feature_name' in f_df.columns:
                features = f_df['feature_name'].tolist()
            else:
                features = list(df.columns)
        except:
            features = list(df.columns)
    else:
        features = list(df.columns)
        
    # 3. 生成 Prompt 模板 (LLM Step)
    # 如果用户提供了 api_key 参数，或者环境变量里有，就尝试调用
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    
    instruction_template = generate_template_from_llm(
        features=features,
        api_key=api_key,
        base_url=args.base_url,
        model=args.model
    )
    
    logger.info("使用的 Instruction 模板预览：")
    logger.info("-" * 40)
    logger.info(instruction_template[:500] + "...")
    logger.info("-" * 40)

    # 4. 预处理数据
    df_processed = process_dataframe(df)
    
    # 5. 生成 Alpaca 数据
    buffer = []
    BATCH_SIZE = 2000
    
    # 清理旧文件
    if output_file.exists():
        output_file.unlink()
    
    logger.info("开始构建 Alpaca 格式数据...")
    
    count = 0
    success_count = 0
    
    for _, row in tqdm(df_processed.iterrows(), total=len(df_processed)):
        item = build_alpaca_item(row, instruction_template)
        if not item:
            continue
        buffer.append(item)
        success_count += 1
            
        # 批量写入
        if len(buffer) >= BATCH_SIZE:
            save_as_jsonl(buffer, output_file)
            buffer = []
            
    # 写入剩余数据
    if buffer:
        save_as_jsonl(buffer, output_file)
    
    logger.info(f"处理完成！成功生成 {success_count} 条数据。输出文件: {output_file}")

if __name__ == '__main__':
    main()
