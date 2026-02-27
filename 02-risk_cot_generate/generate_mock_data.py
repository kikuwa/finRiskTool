# -*- coding: utf-8 -*-
"""
@File    : generate_mock_data.py
@Desc    : 生成风险评估原始模拟数据 (CSV格式)
           生成的数据将作为 prompt_filling.py 的输入
           同时生成特征含义说明CSV
"""
import pandas as pd
import random
import os
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
class RiskDataGenerator:
    def __init__(self, num_samples=100, output_dir=None):
        self.num_samples = num_samples
        
        # 路径配置
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            current_dir = Path(__file__).parent.absolute()
            self.output_dir = current_dir / "data"
            
        self.output_file = self.output_dir / "mock_risk_data.csv"
        self.feature_desc_file = self.output_dir / "feature_description.csv"
        
        # 基础配置
        self.ownerships = ["国有", "民营", "合资", "外资"]
        self.industries = ["计算机服务业", "制造业", "批发零售", "建筑业", "餐饮业", "交通运输"]
        self.regions = ["深圳分行", "上海分行", "北京分行", "广州分行", "成都分行"]
        # 特征含义字典
        self.feature_descriptions = {
            'is_risky': '是否违约风险样本(布尔值)',
            'label': '违约标签(是/否)',
            'entityname': '企业名称',
            'nature': '企业性质',
            'firstcreditx_year': '首次授信年份',
            'cred_level_desc': '信用评级',
            'threshold': '风险阈值',
            'risk_flag': '风险标识',
            'check_result': '贷后检查结果',
            'corp_oper_income_24m': '近24个月营业收入',
            'corp_oper_expns_24m': '近24个月营业支出',
            'corp_oper_diff_12m': '近12个月经营净收益',
            'corp_oper_diff_24m': '近24个月经营净收益',
            'taxes_level_cd': '纳税等级',
            'corp_curr_fin_asset_bal': '当前金融资产余额',
            'corp_curr_fin_asset_bal_diff': '金融资产余额变动',
            'total_avg_depst_bal_3m': '近3个月平均存款余额',
            'n_ph_num4': '融资机构数量',
            'n_ph_amt4': '融资余额',
            'n_ph_amt2': '融资金额',
            'unpayoff_mg_ratio': '未结清比例',
            'rep_overdue_cnt_3m': '近3个月还款逾期次数',
            'rep_overdue_cnt_24m': '近24个月还款逾期次数',
            'corp_curr_lab_amt': '当前逾期金额',
            'flag_8': '本金/利息逾期标识',
            'creditamt': '授信额度',
            'amtused': '已用额度',
            'n_u0343_hit': '规则U0343命中情况',
            'n_ph_amt1_hit': '规则PH_AMT1命中情况',
            'n_ph_amt2_hit': '规则PH_AMT2命中情况',
            'tj_result': '统计算法结果',
            'rep_crdt_card_limit': '信用卡额度',
            'rep_crdt_card_avg_limit_6m': '近6个月平均信用卡额度',
            'rep_crdt_card_used_limit': '信用卡已用额度',
            'reserve_8': '重要股东风险',
            'down30': '下调30%标识',
            'industry_cd_desc': '行业代码描述',
            'branchname': '分行名称'
        }
    def generate_data(self):
        # 确保目录存在
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        data = []
        logger.info(f"开始生成 {self.num_samples} 条原始数据...")
        
        for i in range(self.num_samples):
            # 决定样本标签 (20% 概率为违约风险)
            is_risky = random.random() < 0.2
            row = self._generate_single_row(i, is_risky)
            data.append(row)
            
        # 转换为 DataFrame 并保存为 CSV
        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False, encoding='utf-8')
        
        # 生成特征说明CSV
        self._generate_feature_description()
        
        logger.info(f"生成完成，已保存至 {self.output_file}")
        logger.info(f"特征说明已保存至 {self.feature_desc_file}")
        logger.info(f"数据维度: {df.shape}")
        logger.info(f"列名示例: {list(df.columns[:5])}...")
    def _generate_feature_description(self):
        """生成特征说明CSV文件"""
        desc_df = pd.DataFrame([
            {'feature_name': k, 'description': v}
            for k, v in self.feature_descriptions.items()
        ])
        desc_df.to_csv(self.feature_desc_file, index=False, encoding='utf-8')
        logger.info(f"特征说明文件已生成: {self.feature_desc_file} (共{len(desc_df)}个特征)")
    def _generate_single_row(self, index, is_risky):
        row = {}
        
        # 0. 标记
        row['is_risky'] = is_risky
        row['label'] = "是" if is_risky else "否"
        # row['isTestSet'] = 1 if random.random() < 0.2 else 0  # 20% 作为测试集
        
        # 1. 基础信息
        row['entityname'] = f"测试企业_{index}_{'风险' if is_risky else '优质'}有限公司"
        row['nature'] = random.choice(self.ownerships)
        row['firstcreditx_year'] = str(random.randint(2010, 2022))
        row['cred_level_desc'] = random.choice(["B", "C", "CC"]) if is_risky else random.choice(["AAA", "AA", "A"])
        row['threshold'] = round(random.uniform(0.5, 0.9), 4) if is_risky else round(random.uniform(0.01, 0.2), 4)
        row['risk_flag'] = "是" if is_risky else "否"
        
        # 贷后检查 (使用 \n\n 分隔多条记录)
        checks = self._gen_post_loan_checks(is_risky)
        row['check_result'] = "\n\n".join(checks)
        
        # 2. 经营情况
        income_24m = random.randint(1000000, 50000000)
        expense_ratio = random.uniform(0.9, 1.5) if is_risky else random.uniform(0.6, 0.9)
        expense_24m = int(income_24m * expense_ratio)
        net_24m = income_24m - expense_24m
        net_12m = net_24m // 2 + random.randint(-100000, 100000)
        
        row['corp_oper_income_24m'] = float(income_24m)
        row['corp_oper_expns_24m'] = float(expense_24m)
        row['corp_oper_diff_12m'] = net_12m
        row['corp_oper_diff_24m'] = net_24m
        row['taxes_level_cd'] = random.choice(["C", "D"]) if is_risky else random.choice(["A", "B"])
        
        # 3. 金融资产
        fin_asset = random.randint(0, 50000) if is_risky else random.randint(1000000, 10000000)
        row['corp_curr_fin_asset_bal'] = fin_asset
        row['corp_curr_fin_asset_bal_diff'] = random.randint(-500000, -10000) if is_risky else random.randint(100000, 2000000)
        row['total_avg_depst_bal_3m'] = random.randint(100, 5000) if is_risky else random.randint(50000, 500000)
        
        # 4. 信用及融资
        row['n_ph_num4'] = random.randint(5, 10) if is_risky else random.randint(1, 4)
        row['n_ph_amt4'] = random.randint(5000000, 50000000)
        row['n_ph_amt2'] = random.randint(500000, 5000000)
        row['unpayoff_mg_ratio'] = f"{random.randint(50, 80)}%" if is_risky else "100%"
        
        row['rep_overdue_cnt_3m'] = random.randint(1, 5) if is_risky else 0
        row['rep_overdue_cnt_24m'] = random.randint(3, 10) if is_risky else 0
        row['corp_curr_lab_amt'] = random.randint(100000, 1000000) if is_risky else 0
        row['flag_8'] = "是" if is_risky else "否" # 本金/利息逾期
        
        # 5. 授信与规则
        limit = random.randint(1000000, 20000000)
        used = int(limit * (random.uniform(0.8, 1.0) if is_risky else random.uniform(0.1, 0.4)))
        row['creditamt'] = limit
        row['amtused'] = used
        
        row['n_u0343_hit'] = "命中" if is_risky and random.random() > 0.3 else "未命中"
        row['n_ph_amt1_hit'] = "命中" if is_risky and random.random() > 0.3 else "未命中"
        row['n_ph_amt2_hit'] = "命中" if is_risky and random.random() > 0.3 else "未命中"
        row['tj_result'] = "高风险" if is_risky else "正常"
        
        # 6. 信用卡
        cc_limit = random.randint(50000, 500000)
        row['rep_crdt_card_limit'] = cc_limit
        row['rep_crdt_card_avg_limit_6m'] = int(cc_limit * (random.uniform(0.8, 0.99) if is_risky else random.uniform(0.1, 0.5)))
        row['rep_crdt_card_used_limit'] = int(cc_limit * (random.uniform(0.8, 0.99) if is_risky else random.uniform(0.1, 0.5))) # 辅助计算使用率
        row['reserve_8'] = "是" if is_risky and random.random() > 0.5 else "否"
        row['down30'] = "是" if is_risky and random.random() > 0.4 else "否"
        row['industry_cd_desc'] = random.choice(self.industries)
        row['branchname'] = random.choice(self.regions)

        return row

    def _gen_post_loan_checks(self, is_risky: bool) -> list:
        """生成贷后检查文本列表"""
        templates = [
            "贷后检查：企业近期经营正常，无异常风险信号。",
            "贷后检查：企业订单量稳定，现金流充裕。",
            "贷后检查：企业存在延迟付息情况，需关注。",
            "贷后检查：企业库存积压，销售回款放缓。",
            "贷后检查：企业涉及诉讼，已冻结部分账户。",
            "贷后检查：企业实际控制人失联，风险陡增。"
        ]
        if is_risky:
            return random.sample(templates[2:], k=random.randint(1, 3))
        else:
            return random.sample(templates[:2], k=random.randint(1, 2))


def parse_args():
    parser = argparse.ArgumentParser(description="生成风险评估模拟数据")
    parser.add_argument("-n", "--num", type=int, default=100, help="生成样本数量，默认100")
    parser.add_argument("-o", "--out", type=str, help="输出目录，默认./data")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    gen = RiskDataGenerator(num_samples=args.num, output_dir=args.out)
    gen.generate_data()