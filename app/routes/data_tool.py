from flask import Blueprint, request, jsonify, current_app, send_from_directory, render_template
import pandas as pd
import numpy as np
import subprocess
import os
import sys

data_tool_bp = Blueprint('data_tool', __name__)

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Pages
@data_tool_bp.route('/')
def index():
    return render_template('data_tool/dataset.html', active_page='dataset')

@data_tool_bp.route('/pu_bagging')
def pu_bagging():
    return render_template('data_tool/pu_learning.html', active_page='data_engineering')

@data_tool_bp.route('/ensemble_feature_selection')
def ensemble_feature_selection():
    return render_template('data_tool/feature_engineering.html', active_page='feature_engineering')

# APIs
@data_tool_bp.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # Script expects data/train.csv
        file_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads', 'train.csv')
        # Also copy to data/train.csv as scripts use it
        file.save(file_path)
        import shutil
        shutil.copy(file_path, os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'train.csv'))
        return jsonify({'success': '文件上传成功'})
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/run_model', methods=['POST'])
def run_model():
    try:
        script_path = os.path.join(current_app.config['PROJECT_ROOT'], 'app', 'services', 'data_core', 'PU_bagging.py')
        # Ensure we run in project root so relative paths in script work (or we need to adjust script)
        # The script uses 'data/train.csv' and 'result/...'
        # We need to make sure those paths align with our new structure.
        # Our new structure has 'data' at root.
        
        result = subprocess.run([
            sys.executable,
            script_path
        ], capture_output=True, text=True, cwd=current_app.config['PROJECT_ROOT'])
        
        if result.returncode == 0:
            # The script writes to result/pu_eval_output... relative to CWD.
            # If CWD is PROJECT_ROOT, it writes to PROJECT_ROOT/result/...
            # But we want it in data/results.
            # We might need to adjust the script OR symlink OR just let it write to result/ and serve from there.
            # For "Strictly retain core algorithms", we shouldn't change the script's internal paths if possible.
            # The script writes to: 'result/pu_eval_output/pu_predictions.csv'
            
            # Let's check where it wrote.
            actual_output_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning', 'pu_predictions.csv')
            
            if os.path.exists(actual_output_path):
                df = pd.read_csv(actual_output_path)
                
                top_10 = df.nlargest(10, '违约风险概率')[['违约风险概率']]
                positive_samples = df[df['label'] == 1]
                min_positive_confidence = positive_samples['违约风险概率'].min() if not positive_samples.empty else 0
                high_confidence_count = len(df[df['违约风险概率'] >= 0.9])
                total_samples = len(df)
                
                top_10_dict = top_10.reset_index().to_dict('records')
                
                # Load feature importance if exists
                feature_importance_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning', 'feature_importance.csv')
                feature_importance = []
                if os.path.exists(feature_importance_path):
                    fi_df = pd.read_csv(feature_importance_path)
                    feature_importance = fi_df.head(20).to_dict('records') # Return top 20 features

                return jsonify({
                    'success': True,
                    'log': result.stdout,
                    'stderr': result.stderr,
                    'top_10': top_10_dict,
                    'feature_importance': feature_importance,
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

@data_tool_bp.route('/download_predictions')
def download_predictions():
    # Adjust path to where the script actually writes
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning')
    return send_from_directory(results_path, "pu_predictions.csv", as_attachment=True)

@data_tool_bp.route('/get_full_results')
def get_full_results():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning', 'pu_predictions.csv')
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
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

# ... (Implement other endpoints similarly)
@data_tool_bp.route('/upload_train', methods=['POST'])
def upload_train():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # Script expects data/train.csv
        file_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'uploads', 'train.csv')
        file.save(file_path)
        import shutil
        shutil.copy(file_path, os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'train.csv'))
        return jsonify({'success': '训练集文件上传成功'})
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/upload_pu', methods=['POST'])
def upload_pu():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件上传'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        # Script expects pu_eval_output/pu_predictions.csv (relative to CWD)
        # So we stick to project root structure
        output_dir = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'pu_learning')
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, 'pu_predictions.csv')
        file.save(file_path)
        return jsonify({'success': 'PU打分文件上传成功'})
    
    return jsonify({'error': '只允许上传CSV文件'}), 400

@data_tool_bp.route('/run_model_feature_selection', methods=['POST'])
def run_model_feature_selection():
    try:
        script_path = os.path.join(current_app.config['PROJECT_ROOT'], 'app', 'services', 'data_core', 'ensemble_feature_selection.py')
        result = subprocess.run([
            sys.executable,
            script_path
        ], capture_output=True, text=True, cwd=current_app.config['PROJECT_ROOT'])
        
        if result.returncode == 0:
            feature_rank_file = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection', 'feature_rank_comparison.csv')
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

@data_tool_bp.route('/download_results')
def download_results():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection')
    return send_from_directory(results_path, "feature_rank_comparison.csv", as_attachment=True)

@data_tool_bp.route('/get_results_data')
def get_results_data():
    results_path = os.path.join(current_app.config['PROJECT_ROOT'], 'data', 'results', 'feature_selection', 'feature_rank_comparison.csv')
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        df_sample = df.head(100)
        
        # Replace NaN with None for valid JSON serialization
        # Must cast to object first, otherwise float columns might revert None to NaN
        df_sample = df_sample.astype(object).where(pd.notnull(df_sample), None)
        
        result_dict = df_sample.to_dict('records')
        return jsonify(result_dict)
    else:
        return jsonify({'error': '结果文件未找到'}), 404
