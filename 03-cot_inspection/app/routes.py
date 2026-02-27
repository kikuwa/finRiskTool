import os
import json
import logging
from flask import Blueprint, request, jsonify, render_template, send_file, current_app
from werkzeug.utils import secure_filename
from .services.inspector import inspector
from .services.model_inspection import read_jsonl

main_bp = Blueprint('main', __name__)
logger = logging.getLogger(__name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/inspect', methods=['POST'])
def inspect_data():
    try:
        if 'file' not in request.files:
            return jsonify({'error': '没有上传文件'}), 400
        
        file = request.files['file']
        inspection_type = request.form.get('type', 'model')
        
        if file.filename == '':
            return jsonify({'error': '没有选择文件'}), 400
        
        if not file.filename.endswith('.jsonl'):
            return jsonify({'error': '只支持JSONL文件'}), 400
        
        # 保存上传的文件
        filename = secure_filename(file.filename)
        filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        data = read_jsonl(filepath)
        
        if not data:
            return jsonify({'error': '文件为空或格式错误'}), 400
        
        model_config = {
            "api_key": request.form.get('api_key') or None,
            "api_base": request.form.get('api_base') or None,
            "model_name": request.form.get('model_name') or None,
            "system_prompt": request.form.get('system_prompt') or None,
        }
        enabled_rules_raw = request.form.get('rules') or ''
        enabled_rules = [r for r in enabled_rules_raw.split(',') if r]
        
        if inspection_type == 'model':
            results = inspector.inspect_with_model(data, model_config=model_config)
        else:
            results = inspector.inspect_with_rules(data, enabled_rules=enabled_rules)
        
        # 计算统计信息
        total_samples = len(results)
        processed_samples = total_samples
        scores = [item['score'] for item in results]
        average_score = sum(scores) / len(scores) if scores else 0
        pass_count = sum(1 for score in scores if score >= 6.0)
        pass_rate = (pass_count / total_samples * 100) if total_samples > 0 else 0
        
        # 准备响应数据
        response_data = {
            'total': total_samples,
            'processed': processed_samples,
            'average_score': round(average_score, 2),
            'pass_rate': round(pass_rate, 2),
            'results': results
        }
        
        logger.info(f"质检完成: {total_samples}个样本, 平均分: {average_score:.2f}, 合格率: {pass_rate:.1f}%")
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"质检过程出错: {e}")
        return jsonify({'error': f'质检过程出错: {str(e)}'}), 500

@main_bp.route('/api/health')
def health_check():
    return jsonify({'status': 'healthy', 'message': '质检服务运行正常'})

@main_bp.route('/api/stats')
def get_stats():
    return jsonify({
        'upload_folder': current_app.config['UPLOAD_FOLDER'],
        'results_folder': current_app.config['RESULTS_FOLDER'],
        'max_file_size': current_app.config['MAX_CONTENT_LENGTH']
    })

@main_bp.route('/download_sample')
def download_sample():
    # 示例文件在 app/static/sample_test.jsonl
    # 假设 static_folder 是 app/static
    sample_path = os.path.join(current_app.static_folder, 'sample_test.jsonl')
    if not os.path.exists(sample_path):
        return jsonify({'error': '示例文件不存在，请联系管理员生成'}), 404
    return send_file(sample_path, as_attachment=True, download_name='sample_test.jsonl')
