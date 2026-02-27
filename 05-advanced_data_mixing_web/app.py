import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from core.mixer import DataMixer

st.set_page_config(page_title="SFT 数据混合与排序", layout="wide")

st.title("🧩 高级 SFT 数据混合与排序工具")
st.markdown("""
参考 [Data Efficacy for Language Model Training] 的思路，本工具支持：
1. 基于质量、难度、可学习性对样本打分；
2. 按可调节比例混合正负样本；
3. 使用课程学习风格的排序策略（升序、折叠排序等）。
""")

# Initialize Mixer
if 'mixer' not in st.session_state:
    st.session_state.mixer = DataMixer()

# Sidebar: Global Config
st.sidebar.header("⚙️ 全局配置")
pos_multiplier = st.sidebar.slider("正样本采样倍数", 0.1, 5.0, 1.0, 0.1)
neg_multiplier = st.sidebar.slider("负样本采样倍数", 0.1, 5.0, 1.0, 0.1)
total_count = st.sidebar.number_input("样本总数（为 0 时自动最大可用）", min_value=0, value=0)
oversample = st.sidebar.checkbox("允许过采样（不足时重复采样）", value=False)
seed = st.sidebar.number_input("随机种子", value=42)

st.sidebar.markdown("---")
st.sidebar.header("📊 评分与排序")
scoring_method = st.sidebar.selectbox("评分方法", ["heuristic", "random"])
sort_strategy = st.sidebar.selectbox("排序策略", ["random", "ascending", "descending", "folded"])
sort_key = st.sidebar.selectbox("排序依据", ["composite_score", "difficulty", "quality", "learnability"])
num_folds = st.sidebar.number_input("折叠次数（fold 数，针对折叠排序）", min_value=1, value=3)

# Main Area: Data Input
col1, col2 = st.columns(2)

with col1:
    st.subheader("正样本数据")
    pos_path = st.text_input("正样本 JSONL 路径", "04-sft_data_mixing/data/positive.jsonl")
    pos_file = st.file_uploader("上传正样本 JSONL 文件", type=["jsonl"], key="pos_upload")
    
with col2:
    st.subheader("负样本数据")
    neg_path = st.text_input("负样本 JSONL 路径", "04-sft_data_mixing/data/negative.jsonl")
    neg_file = st.file_uploader("上传负样本 JSONL 文件", type=["jsonl"], key="neg_upload")

# Action Button
if st.button("🚀 生成训练数据", type="primary"):
    upload_dir = "05-advanced_data_mixing_web/uploads"
    os.makedirs(upload_dir, exist_ok=True)

    if pos_file is not None:
        pos_path_effective = os.path.join(upload_dir, "pos_uploaded.jsonl")
        with open(pos_path_effective, "wb") as f:
            f.write(pos_file.getbuffer())
    else:
        pos_path_effective = pos_path

    if neg_file is not None:
        neg_path_effective = os.path.join(upload_dir, "neg_uploaded.jsonl")
        with open(neg_path_effective, "wb") as f:
            f.write(neg_file.getbuffer())
    else:
        neg_path_effective = neg_path

    if not os.path.exists(pos_path_effective) or not os.path.exists(neg_path_effective):
        st.error("请上传文件或填写有效的正负样本路径。")
    else:
        with st.spinner("正在处理数据..."):
            try:
                final_total = None if total_count == 0 else total_count
                
                # 这里内部会根据正负样本倍数自动计算有效比例，
                # 我们将基础比例固定为 0.5（即默认正负各一半），
                # 再由倍数进行加权。
                result_data = st.session_state.mixer.process(
                    pos_path_effective, neg_path_effective, 
                    ratio=0.5, 
                    total_count=final_total,
                    scoring_method=scoring_method,
                    sort_strategy=sort_strategy,
                    sort_key=sort_key,
                    num_folds=num_folds,
                    oversample=oversample,
                    pos_multiplier=pos_multiplier,
                    neg_multiplier=neg_multiplier
                )
                
                st.success(f"成功生成 {len(result_data)} 条样本！")
                
                st.subheader("📈 数据分布可视化")
                
                # Convert to DataFrame for easier plotting
                df = pd.DataFrame([
                    {
                        'composite': item['scores']['composite_score'],
                        'difficulty': item['scores']['difficulty'],
                        'quality': item['scores']['quality'],
                        'learnability': item['scores']['learnability']
                    } 
                    for item in result_data
                ])
                
                tab1, tab2, tab3 = st.tabs(["分数分布", "排序曲线", "数据预览"])
                
                with tab1:
                    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
                    df['difficulty'].hist(ax=ax[0], bins=20, alpha=0.7)
                    ax[0].set_title('难度分布')
                    
                    df['quality'].hist(ax=ax[1], bins=20, alpha=0.7, color='orange')
                    ax[1].set_title('质量分布')
                    
                    df['composite'].hist(ax=ax[2], bins=20, alpha=0.7, color='green')
                    ax[2].set_title('综合得分分布')
                    st.pyplot(fig)
                    
                with tab2:
                    st.markdown("**排序后分数变化趋势**")
                    st.line_chart(df[sort_key.split('_')[0] if '_' in sort_key else sort_key])
                    st.caption(f"纵轴：{sort_key}，横轴：最终数据集中样本顺序索引")
                    
                with tab3:
                    st.dataframe(pd.DataFrame(result_data).head(20))
                
                output_path = "05-advanced_data_mixing_web/output/mixed_data.jsonl"
                st.session_state.mixer.save_jsonl(result_data, output_path)
                st.info(f"混合后的数据已保存到：`{output_path}`")
                
                # Download
                with open(output_path, "r") as f:
                    st.download_button("下载 JSONL 文件", f, file_name="mixed_data.jsonl")
                    
            except Exception as e:
                st.error(f"发生错误：{e}")
                st.exception(e)

st.markdown("---")
st.markdown("📝 **说明**：折叠排序会先按分数排序，然后划分为 K 个分段依次拼接，从而在一个静态数据文件中形成多轮“由易到难”的课程式训练顺序。")
