
import pandas as pd
import random
import logging
from datetime import datetime
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RiskDataFactory:
    """
    风险评估原始模拟数据生成工厂
    """
    
    OWNERSHIPS = ["国有", "民营", "合资", "外资"]
    INDUSTRIES = ["计算机服务业", "制造业", "批发零售", "建筑业", "餐饮业", "交通运输"]
    REGIONS = ["深圳分行", "上海分行", "北京分行", "广州分行", "成都分行"]
    
    FEATURE_DESCRIPTIONS = {
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

    @classmethod
    def generate_data(cls, num_samples: int = 100) -> pd.DataFrame:
        """生成模拟数据"""
        data = []
        logger.info(f"开始生成 {num_samples} 条原始数据...")
        
        for i in range(num_samples):
            # 决定样本标签 (20% 概率为违约风险)
            is_risky = random.random() < 0.2
            row = cls._generate_single_row(i, is_risky)
            data.append(row)
            
        return pd.DataFrame(data)

    @classmethod
    def get_feature_descriptions(cls) -> pd.DataFrame:
        """获取特征说明"""
        return pd.DataFrame([
            {'feature_name': k, 'description': v}
            for k, v in cls.FEATURE_DESCRIPTIONS.items()
        ])

    @classmethod
    def _generate_single_row(cls, index: int, is_risky: bool) -> Dict[str, Any]:
        row = {}
        
        # 0. 标记
        row['is_risky'] = is_risky
        row['label'] = "是" if is_risky else "否"
        
        # 1. 基础信息
        row['entityname'] = f"测试企业_{index}_{'风险' if is_risky else '优质'}有限公司"
        row['nature'] = random.choice(cls.OWNERSHIPS)
        row['firstcreditx_year'] = str(random.randint(2010, 2022))
        row['cred_level_desc'] = random.choice(["B", "C", "CC"]) if is_risky else random.choice(["AAA", "AA", "A"])
        row['threshold'] = round(random.uniform(0.5, 0.9), 4) if is_risky else round(random.uniform(0.01, 0.2), 4)
        row['risk_flag'] = "是" if is_risky else "否"
        
        # 贷后检查
        checks = cls._gen_post_loan_checks(is_risky)
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
        row['flag_8'] = "是" if is_risky else "否"
        
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
        row['rep_crdt_card_used_limit'] = int(cc_limit * (random.uniform(0.8, 0.99) if is_risky else random.uniform(0.1, 0.5)))
        row['reserve_8'] = "是" if is_risky and random.random() > 0.5 else "否"
        row['down30'] = "是" if is_risky and random.random() > 0.4 else "否"
        row['industry_cd_desc'] = random.choice(cls.INDUSTRIES)
        row['branchname'] = random.choice(cls.REGIONS)

        return row

    @staticmethod
    def _gen_post_loan_checks(is_risky: bool) -> List[str]:
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
