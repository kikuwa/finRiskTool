from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import subprocess
import os
import sys
import time

# 创建Flask应用
app = Flask(__name__)

# 设置静态文件夹和上传文件夹
app.static_folder = '.'
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 确保结果文件夹存在
os.makedirs('result/pu_eval_output', exist_ok=True)
os.makedirs('feature_selection_results', exist_ok=True)

# 检查文件是否允许上传
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 主页路由 - 直接返回guide.html静态文件
@app.route('/')
def index():
    return send_from_directory('ui_page/templates', 'guide.html')

# PU Bagging模块路由 - 直接返回pu_bagging.html静态文件
@app.route('/pu_bagging')
def pu_bagging():
    return send_from_directory('ui_page/templates', 'pu_bagging.html')

# 集成特征选择模块路由 - 直接返回ensemble_feature_selection.html静态文件
@app.route('/ensemble_feature_selection')
def ensemble_feature_selection():
    return send_from_directory('ui_page/templates', 'ensemble_feature_selection.html')

# 上传文件接口
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 保存文件到指定位置，覆盖原有文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train.csv')
        file.save(file_path)
        return jsonify({'success': '文件上传成功'})
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

# 运行模型接口
@app.route('/run_model', methods=['POST'])
def run_model():
    try:
        # 运行PU_bagging.py脚本
        result = subprocess.run([
            "venv/Scripts/python.exe",
            "core/PU_bagging.py"
        ], capture_output=True, text=True, cwd="d:/code/P1")
        
        if result.returncode == 0:
            # 读取预测结果
            predictions_path = "result/pu_eval_output/pu_predictions.csv"
            if os.path.exists(predictions_path):
                df = pd.read_csv(predictions_path)
                
                # 计算结果统计
                top_10 = df.nlargest(10, '违约风险概率')[['违约风险概率']]
                positive_samples = df[df['label'] == 1]
                min_positive_confidence = positive_samples['违约风险概率'].min() if not positive_samples.empty else 0
                high_confidence_count = len(df[df['违约风险概率'] >= 0.9])
                total_samples = len(df)
                
                # 转换为字典格式返回
                top_10_dict = top_10.reset_index().to_dict('records')
                
                return jsonify({
                    'success': True,
                    'log': result.stdout,
                    'stderr': result.stderr,
                    'top_10': top_10_dict,
                    'min_positive_confidence': min_positive_confidence,
                    'high_confidence_count': high_confidence_count,
                    'total_samples': total_samples
                })
            else:
                return jsonify({
                    'success': False,
                    'error': '预测结果文件未找到',
                    'log': result.stdout,
                    'stderr': result.stderr
                })
        else:
            return jsonify({
                'success': False,
                'error': '模型运行失败',
                'log': result.stdout,
                'stderr': result.stderr
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# 下载预测结果接口
@app.route('/download_predictions')
def download_predictions():
    predictions_path = "result/pu_eval_output"
    return send_from_directory(predictions_path, "pu_predictions.csv", as_attachment=True)

# 获取完整预测结果接口
@app.route('/get_full_results')
def get_full_results():
    predictions_path = "result/pu_eval_output/pu_predictions.csv"
    if os.path.exists(predictions_path):
        df = pd.read_csv(predictions_path)
        # 只返回前100行数据，避免数据量过大
        df_sample = df.head(100)
        # 将NaN值转换为null，以便JSON正确解析
        df_sample = df_sample.fillna(value=np.nan)
        # 转换为字典并处理NaN值
        result_dict = df_sample.to_dict('records')
        # 将NaN转换为None（JSON中的null）
        for record in result_dict:
            for key, value in record.items():
                if isinstance(value, float) and np.isnan(value):
                    record[key] = None
        return jsonify(result_dict)
    else:
        return jsonify({'error': '预测结果文件未找到'}), 404

# 集成特征选择 - 上传训练集文件接口
@app.route('/upload_train', methods=['POST'])
def upload_train():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 保存文件到指定位置，覆盖原有文件
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'train.csv')
        file.save(file_path)
        return jsonify({'success': '训练集文件上传成功'})
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

# 集成特征选择 - 上传PU打分文件接口
@app.route('/upload_pu', methods=['POST'])
def upload_pu():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # 确保pu_eval_output文件夹存在
        os.makedirs('pu_eval_output', exist_ok=True)
        # 保存文件到指定位置，覆盖原有文件
        file_path = os.path.join('pu_eval_output', 'pu_predictions.csv')
        file.save(file_path)
        return jsonify({'success': 'PU打分文件上传成功'})
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

# 集成特征选择 - 运行模型接口
@app.route('/run_model_feature_selection', methods=['POST'])
def run_model_feature_selection():
    try:
        # 运行ensemble_feature_selection.py脚本
        result = subprocess.run([
            "venv/Scripts/python.exe",
            "core/ensemble_feature_selection.py"
        ], capture_output=True, text=True, cwd="d:/code/P1")
        
        if result.returncode == 0:
            # 检查结果文件是否生成
            feature_rank_file = "feature_selection_results/feature_rank_comparison.csv"
            success = os.path.exists(feature_rank_file)
            
            return jsonify({
                'success': success,
                'log': result.stdout,
                'stderr': result.stderr,
                'has_results': success
            })
        else:
            return jsonify({
                'success': False,
                'error': '模型运行失败',
                'log': result.stdout,
                'stderr': result.stderr
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

# 集成特征选择 - 下载特征排名结果
@app.route('/download_results')
def download_results():
    results_path = "feature_selection_results"
    return send_from_directory(results_path, "feature_rank_comparison.csv", as_attachment=True)

# 集成特征选择 - 获取特征排名结果数据
@app.route('/get_results_data')
def get_results_data():
    results_path = "feature_selection_results/feature_rank_comparison.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        # 只返回前100行数据，避免数据量过大
        df_sample = df.head(100)
        # 转换为字典格式返回
        result_dict = df_sample.to_dict('records')
        return jsonify(result_dict)
    else:
        return jsonify({'error': '结果文件未找到'}), 404

if __name__ == '__main__':
    print("启动PULearning系统...")
    print("访问地址: http://localhost:5000")
    print("PU Bagging模块地址: http://localhost:5000/pu_bagging")
    print("集成特征选择模块地址: http://localhost:5000/ensemble_feature_selection")
    app.run(debug=True, host='0.0.0.0', port=5000)
