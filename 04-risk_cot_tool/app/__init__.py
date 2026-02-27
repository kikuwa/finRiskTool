from flask import Flask
import os
import logging
from logging.handlers import RotatingFileHandler

def create_app():
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev')
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload
    
    # Path Configuration
    # Assumes app is in /.../04-risk_cot_tool/app
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app.config['DATA_FOLDER'] = os.path.join(project_root, 'data')
    app.config['UPLOAD_FOLDER'] = os.path.join(project_root, 'data', 'uploads')
    app.config['RESULTS_FOLDER'] = os.path.join(project_root, 'data', 'results')
    app.config['LOG_FOLDER'] = os.path.join(project_root, 'logs')
    
    # Ensure directories exist
    for folder in [app.config['DATA_FOLDER'], app.config['UPLOAD_FOLDER'], 
                   app.config['RESULTS_FOLDER'], app.config['LOG_FOLDER']]:
        os.makedirs(folder, exist_ok=True)
        
    # Logging Configuration
    setup_logging(app)
    
    # Register Blueprints
    from app.routes.main import main_bp
    from app.routes.generator import generator_bp
    from app.routes.inference import inference_bp
    from app.routes.inspector import inspector_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(generator_bp)
    app.register_blueprint(inference_bp)
    app.register_blueprint(inspector_bp)
    
    return app

def setup_logging(app):
    if not app.debug:
        file_handler = RotatingFileHandler(
            os.path.join(app.config['LOG_FOLDER'], 'app.log'),
            maxBytes=1024 * 1024, 
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        app.logger.setLevel(logging.INFO)
        app.logger.info('Risk CoT Tool startup')
