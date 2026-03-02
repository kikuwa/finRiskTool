from flask import Blueprint, request, jsonify, current_app, send_from_directory, render_template, Response
import pandas as pd
import numpy as np
import os
import sys
import shutil
import traceback
import json
import chardet
from datetime import datetime
from app.services.data_core.data_loader import DataLoader
from app.services.data_core.PU_bagging import run_pu_learning_pipeline
from app.services.data_core.ensemble_feature_selection import run_feature_selection_pipeline
from app.services.data_core.split_data import split_data
from app.services.data_core.data_analysis import analyze_dataset

def _detect_encoding(file_path: str) -> str:
    """
    使用 chardet 检测文件编码
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(100000))
    return result['encoding']

def _load_csv_robust(file_path: str) -> pd.DataFrame:
    """
    加载 CSV 文件，自动检测编码并包含多种编码回退机制
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    
    detected_encoding = _detect_encoding(file_path)
    encodings_to_try = []
    if detected_encoding:
        encodings_to_try.append(detected_encoding)
    
    common_encodings = [
        'utf-8',
        'gbk',
        'gb18030',
        'big5',
        'latin-1',
        'utf-16',
        'cp1252'
    ]
    
    for enc in common_encodings:
        if enc not in encodings_to_try:
            encodings_to_try.append(enc)
    
    last_error = None
    for encoding in encodings_to_try:
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except (UnicodeDecodeError, LookupError) as e:
            last_error = e
            continue
    
    try:
        return pd.read_csv(file_path, encoding='utf-8', errors='replace', low_memory=False)
    except Exception as e:
        raise ValueError(f"无法读取文件。尝试了以下编码: {encodings_to_try}。错误: {last_error}")

data_tool_bp = Blueprint('data_tool', __name__)

ALLOWED_EXTENSIONS = {'csv'}

def log_exception(e):
    """一个简单的函数，用于在捕获异常时手动记录日志。"""
    error_time = datetime.now()
    error_type = type(e).__name__
    error_message = str(e)
    stack_trace = traceback.format_exc()
    
    # 打印到控制台
    print("="*80, file=sys.stderr)
    print(f"HANDLED EXCEPTION at {error_time.strftime('%Y-%m-%d %H:%M:%S')}", file=sys.stderr)
    print(f"Type: {error_type}", file=sys.stderr)
    print(f"Message: {error_message}", file=sys.stderr)
    print(stack_trace, file=sys.stderr)
    print("="*80, file=sys.stderr)
    
    # 写入日志文件
    log_folder = current_app.config.get('LOG_FOLDER', os.path.join(current_app.config['PROJECT_ROOT'], 'logs'))
    log_file_path = os.path.join(log_folder, f"error_{error_time.strftime('%Y%m%d')}.log")
    log_content = (
        f"Timestamp: {error_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Error Type: {error_type}\n"
        f"Error Message: {error_message}\n"
        f"Stack Trace:\n{stack_trace}\n"
        f"{'-'*80}\n"
    )
    
    try:
        with open(log_file_path, 'a', encoding='utf-8') as f:
            f.write(log_content)
    except Exception as log_e:
        print(f"FATAL: Failed to write to log file: {log_e}", file=sys.stderr)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Pages
@data_tool_bp.route('/')
def dataset_management():
    return render_template('data_tool/dataset.html', active_page='dataset')

@data_tool_bp.route('/pu_bagging')
def pu_bagging():
    return render_template('data_tool/pu_learning.html', active_page='data_engineering')

@data_tool_bp.route('/ensemble_feature_selection')
def ensemble_feature_selection():
    return render_template('data_tool/feature_engineering.html', active_page='feature_engineering')

# APIs
@data_tool_bp.route('/upload_full', methods=['POST'])
def upload_full():
    """上传完整数据集"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    label_col = request.form.get('label_col', 'label')
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, 'full_dataset.csv')
        file.save(file_path)
        
        try:
            loader = DataLoader()
            df = loader._load_csv(file_path)
            preview_data = df.head(5).fillna('').to_dict('records')
            analysis_result = analyze_dataset(df, label_col=label_col)
            
            return jsonify({
                'success': '文件上传成功',
                'columns': df.columns.tolist(),
                'rows': len(df),
                'preview': preview_data,
                'analysis': analysis_result
            })
        except Exception as e:
            log_exception(e)
            return jsonify({'error': str(e)}), 400
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/upload_train', methods=['POST'])
def upload_train():
    """上传训练集"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    label_col = request.form.get('label_col', 'label')
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        data_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, 'train_dataset.csv')
        file.save(file_path)
        shutil.copy2(file_path, os.path.join(data_dir, 'train.csv'))
        
        try:
            loader = DataLoader()
            df = loader._load_csv(file_path)
            preview_data = df.head(5).fillna('').to_dict('records')
            analysis_result = analyze_dataset(df, label_col=label_col)
            
            return jsonify({
                'success': '文件上传成功',
                'columns': df.columns.tolist(),
                'rows': len(df),
                'preview': preview_data,
                'analysis': analysis_result
            })
        except Exception as e:
            log_exception(e)
            return jsonify({'error': str(e)}), 400
            
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/upload_test', methods=['POST'])
def upload_test():
    """上传测试集"""
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    label_col = request.form.get('label_col', 'label')
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        data_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data')
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, 'test_dataset.csv')
        file.save(file_path)
        shutil.copy2(file_path, os.path.join(data_dir, 'test.csv'))
        
        try:
            loader = DataLoader()
            df = loader._load_csv(file_path)
            preview_data = df.head(5).fillna('').to_dict('records')
            analysis_result = analyze_dataset(df, label_col=label_col)
            
            return jsonify({
                'success': '文件上传成功',
                'columns': df.columns.tolist(),
                'rows': len(df),
                'preview': preview_data,
                'analysis': analysis_result
            })
        except Exception as e:
            log_exception(e)
            return jsonify({'error': str(e)}), 400
            
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/split_dataset', methods=['POST'])
def split_dataset():
    """分割全量数据集"""
    try:
        data = request.json or {}
        label_col = data.get('label_col', 'label')
        test_size = data.get('test_size', 0.3)
        
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        data_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data')
        
        full_path = os.path.join(upload_dir, 'full_dataset.csv')
        
        if not os.path.exists(full_path):
            return jsonify({'error': '全量数据集不存在，请先上传'}), 400
            
        train_output = os.path.join(data_dir, 'train.csv')
        test_output = os.path.join(data_dir, 'test.csv')
        
        split_data(full_path, train_output, test_output, label_col=label_col, test_size=test_size)
        
        shutil.copy2(train_output, os.path.join(upload_dir, 'train_dataset.csv'))
        shutil.copy2(test_output, os.path.join(upload_dir, 'test_dataset.csv'))
        
        df = _load_csv_robust(full_path)
        analysis_result = analyze_dataset(df, label_col=label_col)
        
        return jsonify({
            'success': '数据集分割成功',
            'analysis': analysis_result
        })
        
    except Exception as e:
        log_exception(e)
        return jsonify({'error': f'分割失败: {str(e)}'}), 500

@data_tool_bp.route('/download_analysis_report', methods=['POST'])
def download_analysis_report():
    """下载分析报告"""
    try:
        data = request.json or {}
        analysis_result = data.get('analysis_result')
        
        if not analysis_result:
            return jsonify({'error': '没有分析结果'}), 400
            
        # 将分析结果转换为 JSON 字符串
        report_content = json.dumps(analysis_result, indent=4, ensure_ascii=False)
        
        return Response(
            report_content,
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment;filename=dataset_analysis_report.json'}
        )
    except Exception as e:
        log_exception(e)
        return jsonify({'error': str(e)}), 400

@data_tool_bp.route('/run_model', methods=['POST'])
def run_model():
    """运行 PU Learning 模型"""
    try:
        data = request.json or {}
        label_col = data.get('label_col', 'label')
        dataset_type = data.get('dataset_type', 'full') # 'full' or 'split'
        
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        output_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning')
        
        loader = DataLoader(label_col=label_col)
        
        # 加载数据
        if dataset_type == 'full':
            file_path = os.path.join(upload_dir, 'full_dataset.csv')
            if not os.path.exists(file_path):
                default_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'train.csv')
                if os.path.exists(default_path):
                    file_path = default_path
                else:
                    return jsonify({'error': '未找到数据集，请先上传'}), 400
            
            train_df, test_df = loader.load_full_dataset(file_path)
        else:
            train_path = os.path.join(upload_dir, 'train_dataset.csv')
            test_path = os.path.join(upload_dir, 'test_dataset.csv')
            
            if not os.path.exists(train_path) or not os.path.exists(test_path):
                return jsonify({'error': '未找到训练集或测试集，请先上传'}), 400
                
            train_df, test_df = loader.load_train_test_split(train_path, test_path)
            
        train_df = loader.preprocess_data(train_df)
        test_df = loader.preprocess_data(test_df)
        
        result = run_pu_learning_pipeline(
            train_df=train_df,
            test_df=test_df,
            label_col=label_col,
            output_dir=output_dir
        )
        
        predictions_path = result['predictions_path']
        df = _load_csv_robust(predictions_path)
        
        top_10 = df.nlargest(10, 'prob')[['prob']]
        positive_samples = df[df[label_col] == 1]
        min_positive_confidence = positive_samples['prob'].min() if not positive_samples.empty else 0
        high_confidence_count = len(df[df['prob'] >= 0.9])
        total_samples = len(df)
        
        top_10_dict = top_10.reset_index().to_dict('records')
        
        feature_importance_path = result['feature_importance_path']
        feature_importance = []
        if os.path.exists(feature_importance_path):
            fi_df = _load_csv_robust(feature_importance_path)
            feature_importance = fi_df.head(20).to_dict('records')

        return jsonify({
            'success': True,
            'auc': result['auc'],
            'top_10': top_10_dict,
            'feature_importance': feature_importance,
            'min_positive_confidence': min_positive_confidence,
            'high_confidence_count': high_confidence_count,
            'total_samples': total_samples
        })

    except Exception as e:
        log_exception(e)
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_tool_bp.route('/run_model_feature_selection', methods=['POST'])
def run_model_feature_selection():
    """运行特征选择模型"""
    try:
        data = request.json or {}
        label_col = data.get('label_col', 'label')
        dataset_type = data.get('dataset_type', 'full') # 'full' or 'split'
        
        upload_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads')
        output_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection')
        
        loader = DataLoader(label_col=label_col)
        
        if dataset_type == 'full':
            file_path = os.path.join(upload_dir, 'full_dataset.csv')
            if not os.path.exists(file_path):
                default_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'train.csv')
                if os.path.exists(default_path):
                    file_path = default_path
                else:
                    return jsonify({'error': '未找到数据集，请先上传'}), 400
            
            train_df = loader._load_csv(file_path)
            loader.validate_data(train_df)
        else:
            train_path = os.path.join(upload_dir, 'train_dataset.csv')
            if not os.path.exists(train_path):
                return jsonify({'error': '未找到训练集，请先上传'}), 400
            train_df = loader._load_csv(train_path)
            loader.validate_data(train_df)
            
        train_df = loader.preprocess_data(train_df)
        
        result = run_feature_selection_pipeline(
            train_df=train_df,
            label_col=label_col,
            output_dir=output_dir,
            pu_predictions_path=None
        )
        
        return jsonify({
            'success': True,
            'has_results': True,
            'top_features': result['top_features'][:10]
        })

    except Exception as e:
        log_exception(e)
        return jsonify({
            'success': False,
            'error': str(e)
        })

@data_tool_bp.route('/download_predictions')
def download_predictions():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning')
    return send_from_directory(results_path, "test_predictions.csv", as_attachment=True)

@data_tool_bp.route('/download_results')
def download_results():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection')
    return send_from_directory(results_path, "feature_rank_comparison.csv", as_attachment=True)

@data_tool_bp.route('/get_full_results')
def get_full_results():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning', 'test_predictions.csv')
    if os.path.exists(results_path):
        df = _load_csv_robust(results_path)
        df_sample = df.head(100)
        df_sample = df_sample.fillna(value=np.nan)
        result_dict = df_sample.to_dict('records')
        for record in result_dict:
            for key, value in record.items():
                if isinstance(value, float) and np.isnan(value):
                    record[key] = None
        return jsonify(result_dict)
    else:
        return jsonify({'error': '预测结果文件未找到'}), 404

@data_tool_bp.route('/get_results_data')
def get_results_data():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection', 'feature_rank_comparison.csv')
    if os.path.exists(results_path):
        df = _load_csv_robust(results_path)
        df_sample = df.head(100)
        
        df_sample = df_sample.astype(object).where(pd.notnull(df_sample), None)
        
        result_dict = df_sample.to_dict('records')
        return jsonify(result_dict)
    else:
        return jsonify({'error': '结果文件未找到'}), 404

# 兼容旧的上传接口
@data_tool_bp.route('/upload', methods=['POST'])
def upload_file():
    return upload_full()
