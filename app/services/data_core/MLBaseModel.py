import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import (
    train_test_split, GridSearchCV, StratifiedKFold
)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, make_scorer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder

# ======================= 1. 配置关键参数（根据你的需求修改） =======================
RECALL_TARGET = 0.5 # 正样本召回率最低目标值（你可根据实际需求调整）
POS_LABEL = 1       # 正样本标签 (0=负样本, 1=正样本)
# RANDOM_SEED = 46  # 固定随机种子保证可复现
RANDOM_SEED = 42    # 固定随机种子保证可复现
N_P = 1000/7
Train_test_split=0.25 # 测试集和训练集的划分策略

# ======================= 2. 数据准备 =======================

def load_and_preprocess_data(X):
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        # 填充特殊字符串
        X[col] = X[col].fillna('Missing')

    # 处理数值特征缺失
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numerical_cols:
        # 填充特殊数值 -999，避免与正常数据冲突
        X[col] = X[col].fillna(-999)

    # -----------------------
    # 3. 类别特征编码
    # -----------------------
    for col in categorical_cols:
        le = LabelEncoder()
        # 处理特殊情况：训练集有缺失值填充后的 'Missing'，直接fit
        X[col] = le.fit_transform(X[col].astype(str))

    return X

# 假设 trainF 已经定义
data = trainF.drop('label', axis=1)
y = trainF['label']
X = load_and_preprocess_data(data)

# 严格分层划分
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=Train_test_split, random_state=RANDOM_SEED, stratify=y
)

# ======================= 3. 数据预处理（规避信息泄露） =======================

# 计算类别权重（解决不平衡问题）
class_weights = compute_class_weight(
    class_weight='balanced', classes=np.unique(y_train), y=y_train
)
weight_dict = {0: class_weights[0], 1: class_weights[1]}

# ======================= 4. 自定义评分器：召回保底+精度优先 =======================

def precision_with_recall_constraint(y_true, y_pred):
    """
    自定义评分函数:
    1. 若正样本召回率 < RECALL_TARGET, 评分为0;
    2. 若召回率 >= RECALL_TARGET, 评分=正样本精度 (越大越好)。
    """
    recall = recall_score(y_true, y_pred, pos_label=POS_LABEL)
    if recall < RECALL_TARGET:
        return 0.0
    precision = precision_score(y_true, y_pred, pos_label=POS_LABEL)
    return precision

# 封装为sklearn可识别的评分器
custom_scroer = make_scorer(precision_with_recall_constraint, greater_is_better=True)

# ========= 尝试负样本采样
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=42, sampling_strategy=0.07)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print(f"调整后比例 (正/负): {np.sum(y_train_rus == 1)/np.sum(y_train_rus == 0):.4f}")

# ======================= 5. LightGBM网格寻参（召回保底+精度优化） =======================
base_model = lgb.LGBMClassifier(
    objective='binary',
    # objective=focal_loss_lgb, # 注释中提到的自定义损失函数，原图未显示具体实现
    metric='None', # 禁用内置指标，用自定义评分器
    # is_unbalance=True,
    scale_pos_weight=N_P,
    random_state=RANDOM_SEED,
    n_jobs=-1,
    reg_alpha=0.1,
    reg_lambda=0.1
)

# 聚焦影响召回和精度的核心参数（避免过拟合）
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_child_samples': [20, 50], # 控制过拟合，提升泛化
    'subsample': [0.8, 0.9],
    'class_weight': [None, weight_dict] # 这里引用了上面计算的weight_dict
}

# 内层分层交叉验证
cv_inner = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)

# 网格搜索：用自定义评分器，优先保证召回≥目标值，再选精度最高的参数
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=cv_inner,
    scoring=custom_scroer,
    refit=True,
    n_jobs=-1,
    verbose=1
)

# 仅在训练集上执行网格搜索
print("参数搜索中...")
grid_search.fit(X_train_rus, y_train_rus)

# ======================= 6. 阈值调优（进一步提升精度，保证召回保底） =======================
print("模型已获取...")
best_model = grid_search.best_estimator_

# 步骤1：获取验证集正样本的预测概率
y_val_proba = best_model.predict_proba(X_val)[:, 1] # 正样本概率

# 步骤2：遍历阈值，找到“召回>=目标值”时精度最高的阈值
thresholds = np.arange(0.1, 0.9, 0.1) # 概率阈值范围
best_threshold = 0.5
best_val_precision = 0.0

for threshold in thresholds:
    y_val_pred = (y_val_proba >= threshold).astype(int)
    val_recall = recall_score(y_val, y_val_pred, pos_label=POS_LABEL)
    if val_recall >= RECALL_TARGET:
        val_precision = precision_score(y_val, y_val_pred, pos_label=POS_LABEL)
        if val_precision > best_val_precision:
            best_val_precision = val_precision
            best_threshold = threshold
    print(f"分类阈值: {threshold:.2f}, 召回率: {val_recall:.2f}, 精准率: {val_precision:.2f}")

打印关键结果
print("="*50)
print(f"最优参数: {grid_search.best_params_}")
print(f"最优分类阈值: {best_threshold:.2f}")