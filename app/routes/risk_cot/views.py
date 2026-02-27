from flask import Blueprint, render_template

views_bp = Blueprint('risk_cot_views', __name__)

@views_bp.route('/generator')
def index():
    return render_template('risk_cot/generator.html', active_page='prompt')

@views_bp.route('/inference')
def inference():
    return render_template('risk_cot/inference.html', active_page='cot_synthesis')

@views_bp.route('/inspector')
def inspector():
    return render_template('risk_cot/inspector.html', active_page='cot_inspection')
