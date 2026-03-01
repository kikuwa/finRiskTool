from flask import Flask, jsonify
import os
import traceback
from datetime import datetime

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')
    app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload
    
    # Path Configuration
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config['PROJECT_ROOT'] = project_root
    app.config['LOG_FOLDER'] = os.path.join(project_root, 'logs')
    
    # Ensure directories exist
    os.makedirs(app.config['LOG_FOLDER'], exist_ok=True)
        
    # Register Blueprints
    from app.routes.main import main_bp
    from app.routes.data_tool import data_tool_bp
    from app.routes.risk_cot.generator import generator_bp
    from app.routes.risk_cot.inference import inference_bp
    from app.routes.risk_cot.inspector import inspector_bp
    from app.routes.risk_cot.views import views_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(data_tool_bp, url_prefix='/data_tool')
    app.register_blueprint(generator_bp)
    app.register_blueprint(inference_bp)
    app.register_blueprint(inspector_bp)
    app.register_blueprint(views_bp, url_prefix='/risk_cot')

    # Register Global Error Handler
    @app.errorhandler(Exception)
    def handle_global_exception(e):
        error_time = datetime.now()
        error_type = type(e).__name__
        error_message = str(e)
        stack_trace = traceback.format_exc()
        
        # 1. 在控制台清晰打印
        print("="*80)
        print(f"CRITICAL ERROR at {error_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Type: {error_type}")
        print(f"Message: {error_message}")
        print(stack_trace)
        print("="*80)
        
        # 2. 直接写入日志文件
        log_file_path = os.path.join(app.config['LOG_FOLDER'], f"error_{error_time.strftime('%Y%m%d')}.log")
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
            print(f"FATAL: Failed to write to log file: {log_e}")

        # 返回一个标准的 JSON 错误响应
        response = {
            'error': 'Internal Server Error',
            'message': 'An unexpected error occurred. The incident has been logged.'
        }
        return jsonify(response), 500
    
    return app
