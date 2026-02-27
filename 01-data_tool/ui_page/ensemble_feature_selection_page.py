from flask import Flask, render_template, request, jsonify, send_from_directory
import pandas as pd
import numpy as np
import subprocess
import os
import sys
import time

app = Flask(__name__)

# 设置上传文件夹
UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 确保上传文件夹存在
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 确保结果文件夹存在
os.makedirs('feature_selection_results', exist_ok=True)

# 检查文件是否允许上传
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('ensemble_feature_selection.html')

# 上传训练集文件接口
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

# 上传PU打分文件接口
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

# 运行模型接口
@app.route('/run_model', methods=['POST'])
def run_model():
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

# 下载特征排名结果
@app.route('/download_results')
def download_results():
    results_path = "feature_selection_results"
    return send_from_directory(results_path, "feature_rank_comparison.csv", as_attachment=True)

# 获取特征排名结果数据
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
    print("启动Flask应用...")
    print("访问地址: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)