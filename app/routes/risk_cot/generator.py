from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
from app.services.risk_cot.data_factory import RiskDataFactory
from app.services.risk_cot.prompt_engine import PromptEngine
import os
import uuid
import pandas as pd
import json

generator_bp = Blueprint('generator', __name__, url_prefix='/api/generator')

@generator_bp.route('/upload', methods=['POST'])
def upload_data():
    if 'file' not in request.files:
        return jsonify({'status': 'error', 'message': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected file'}), 400
    
    if file:
        try:
            filename = secure_filename(file.filename)
            # Unique prefix to avoid overwrites
            task_id = str(uuid.uuid4())[:8]
            filename = f"{task_id}_{filename}"
            
            save_path = os.path.join(current_app.config['DATA_FOLDER'], filename)
            file.save(save_path)
            
            # Try to read it to get preview
            if filename.lower().endswith('.csv'):
                df = pd.read_csv(save_path)
            elif filename.lower().endswith(('.xls', '.xlsx')):
                df = pd.read_excel(save_path)
            else:
                 # Clean up if invalid
                 os.remove(save_path)
                 return jsonify({'status': 'error', 'message': 'Unsupported file format. Please upload CSV or Excel.'}), 400
                 
            # Convert NaN to None for JSON compatibility
            preview = df.head(5).where(pd.notnull(df), None).to_dict(orient='records')
            
            return jsonify({
                'status': 'success',
                'message': '文件上传成功',
                'filename': filename,
                'preview': preview
            })
        except Exception as e:
             return jsonify({'status': 'error', 'message': f'Failed to process file: {str(e)}'}), 500

@generator_bp.route('/mock', methods=['POST'])
def generate_mock_data():
    try:
        num_samples = int(request.json.get('num_samples', 100))
        # Unique filename to avoid conflicts
        task_id = str(uuid.uuid4())[:8]
        filename = f"mock_risk_data_{task_id}.csv"
        output_dir = current_app.config['DATA_FOLDER']
        
        # Run generation
        df = RiskDataFactory.generate_data(num_samples=num_samples)
        
        # Save to file
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False, encoding='utf-8-sig')
        
        # Preview data (first 5 rows)
        # Convert NaN to None for JSON compatibility
        preview = df.head(5).where(pd.notnull(df), None).to_dict(orient='records')
        
        return jsonify({
            'status': 'success',
            'message': f'成功生成 {num_samples} 条数据',
            'file_path': str(file_path),
            'filename': filename,
            'preview': preview
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@generator_bp.route('/template/default', methods=['GET'])
def get_default_template():
    return jsonify({
        'status': 'success',
        'template': PromptEngine.BASE_INSTRUCTION_TEMPLATE
    })

@generator_bp.route('/template/generate', methods=['POST'])
def generate_template():
    try:
        api_key = request.json.get('api_key')
        base_url = request.json.get('base_url')
        model = request.json.get('model', 'gpt-3.5-turbo')
        features = request.json.get('features', [])
        
        if not api_key:
             return jsonify({'status': 'error', 'message': 'API Key is required'}), 400
             
        engine = PromptEngine()
        template = engine.generate_template_from_llm(
            features=features,
            api_key=api_key,
            base_url=base_url,
            model=model
        )
        
        return jsonify({
            'status': 'success',
            'template': template
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@generator_bp.route('/alpaca', methods=['POST'])
def generate_alpaca():
    try:
        source_file = request.json.get('source_file')
        template = request.json.get('template')
        
        if not source_file:
            return jsonify({'status': 'error', 'message': 'Source file is required'}), 400
            
        # Full path check
        if not os.path.isabs(source_file):
            source_file = os.path.join(current_app.config['DATA_FOLDER'], source_file)
            
        if not os.path.exists(source_file):
             return jsonify({'status': 'error', 'message': 'Source file not found'}), 404
             
        output_filename = f"alpaca_{os.path.basename(source_file).replace('.csv', '')}.jsonl"
        output_path = os.path.join(current_app.config['DATA_FOLDER'], output_filename)
        
        # Process
        df = pd.read_csv(source_file)
        result_items = PromptEngine.process_data(df, instruction_template=template)
        
        # Write to JSONL
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in result_items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        return jsonify({
            'status': 'success',
            'message': f'成功转换 {len(result_items)} 条数据',
            'output_file': output_filename
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500
