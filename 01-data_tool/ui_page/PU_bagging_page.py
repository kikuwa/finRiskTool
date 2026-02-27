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
os.makedirs('result/pu_eval_output', exist_ok=True)

# 检查文件是否允许上传
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('pu_bagging.html')

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

if __name__ == '__main__':
    print("启动Flask应用...")
    print("访问地址: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
