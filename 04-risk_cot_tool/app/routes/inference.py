from flask import Blueprint, request, jsonify, Response, current_app
from app.services import inference_engine
import threading
import os
import json
import time

from werkzeug.utils import secure_filename

inference_bp = Blueprint('inference', __name__, url_prefix='/api/inference')

@inference_bp.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No selected file'}), 400
            
        if file:
            filename = secure_filename(file.filename)
            # Add timestamp to avoid collisions
            timestamp = int(time.time())
            filename = f"{timestamp}_{filename}"
            
            save_path = os.path.join(current_app.config['DATA_FOLDER'], filename)
            file.save(save_path)
            
            return jsonify({
                'status': 'success', 
                'message': 'File uploaded successfully',
                'filename': filename
            })
            
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@inference_bp.route('/run', methods=['POST'])
def run_inference():
    try:
        data = request.json
        input_file = data.get('input_file')
        api_key = data.get('api_key')
        
        if not input_file or not api_key:
            return jsonify({'status': 'error', 'message': 'Missing required parameters'}), 400
            
        # Handle relative paths
        if not os.path.isabs(input_file):
            input_file = os.path.join(current_app.config['DATA_FOLDER'], input_file)
            
        if not os.path.exists(input_file):
            return jsonify({'status': 'error', 'message': 'Input file not found'}), 404
            
        # Generate output filename
        base_name = os.path.basename(input_file).replace('.jsonl', '')
        output_file = os.path.join(current_app.config['RESULTS_FOLDER'], f"{base_name}_inference_result.jsonl")
        
        config = {
            'input_file': input_file,
            'output_file': output_file,
            'api_key': api_key,
            'model': data.get('model', 'deepseek-reasoner'),
            'workers': int(data.get('workers', 5)),
            'base_url': data.get('base_url', "https://api.deepseek.com/chat/completions")
        }
        
        # Check if already running
        status = inference_engine.get_status()
        if status['status'] == 'running':
             return jsonify({'status': 'error', 'message': 'Task is already running'}), 409
        
        # Start in background thread
        thread = threading.Thread(target=inference_engine.run, args=(config,))
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'success', 
            'message': 'Inference task started',
            'output_file': output_file
        })
        
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@inference_bp.route('/stop', methods=['POST'])
def stop_inference():
    try:
        inference_engine.stop()
        return jsonify({'status': 'success', 'message': 'Stop signal sent'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

@inference_bp.route('/status')
def stream_status():
    def generate():
        while True:
            status = inference_engine.get_status()
            yield f"data: {json.dumps(status)}\n\n"
            if status['status'] in ['completed', 'stopped', 'error']:
                break
            time.sleep(1)
    
    return Response(generate(), mimetype='text/event-stream')
