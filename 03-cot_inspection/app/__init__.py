import os
import logging
from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB限制
    
    # 路径配置
    # 放在根目录的 uploads 和 results 文件夹
    # app.root_path 是 .../app
    project_root = os.path.abspath(os.path.join(app.root_path, '..'))
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'uploads')
    app.config['RESULTS_FOLDER'] = os.path.join(project_root, 'results')
    
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
    
    # 配置日志
    logging.basicConfig(level=logging.INFO)
    
    from .routes import main_bp
    app.register_blueprint(main_bp)
    
    return app
