from flask import Blueprint, render_template

main_bp = Blueprint('main', __name__)

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/generator')
def generator():
    return render_template('generator.html')

@main_bp.route('/inference')
def inference():
    return render_template('inference.html')

@main_bp.route('/inspector')
def inspector():
    return render_template('inspector.html')
