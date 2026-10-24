import os
import uuid
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# NumPy 1.x / 2.x Compatibility Shim (fixes Detectron2 & CatVTON np.int and np.float removals)
import numpy as np
try:
    _ = np.int
except (AttributeError, Exception):
    np.int = int
try:
    _ = np.float
except (AttributeError, Exception):
    np.float = float
try:
    _ = np.bool
except (AttributeError, Exception):
    np.bool = bool

# Diffusers & TorchAO compatibility shim for Google Colab Python 3.13
try:
    import torchao.quantization
    if not hasattr(torchao.quantization, 'FqnToConfig'):
        class FqnToConfig:
            pass
        torchao.quantization.FqnToConfig = FqnToConfig
except Exception:
    pass

try:
    import diffusers.utils.import_utils as _diu
    _diu._torchao_available = False
    _diu.is_torchao_available = lambda: False
except Exception:
    pass

from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
from tryon_engine import process_tryon, process_dual_tryon
from garment_extractor import extract_garments
from classifier import classify_garment, classify_garment_detailed

app = Flask(__name__)

try:
    from flask_cors import CORS
    CORS(app, resources={r"/*": {"origins": "*"}})
except Exception as cors_err:
    print(f"CORS setup notice: {cors_err}")

app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['OUTPUT_FOLDER'] = 'static/outputs'
app.config['GARMENT_FOLDER'] = 'static/garments'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024 # 32MB max limit for dual uploads

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)
os.makedirs(app.config['GARMENT_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def _save_or_resolve_upload(file_obj, prefix="upload"):
    if file_obj and allowed_file(file_obj.filename):
        original_cname = secure_filename(file_obj.filename)
        unique_id = str(uuid.uuid4())[:8]
        c_filename = f"{prefix}_{unique_id}_{original_cname}"
        c_path = os.path.join(app.config['UPLOAD_FOLDER'], c_filename)
        file_obj.save(c_path)
        return c_path, c_filename
    return None, None

def _resolve_reference_file(ref_name):
    if not ref_name:
        return None
    ref_name = secure_filename(ref_name)
    for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER'], app.config['GARMENT_FOLDER']]:
        p = os.path.join(folder, ref_name)
        if os.path.exists(p):
            return p
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/status', methods=['GET'])
def health_status():
    import torch
    return jsonify({
        'status': 'online',
        'engine': 'CatVTON Official Repository Engine',
        'cuda_available': torch.cuda.is_available(),
        'device_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'
    })

@app.route('/classify', methods=['POST'])
def classify_endpoint():
    """Classify a garment image into 'upper', 'lower', or 'overall'."""
    file_path = None
    
    # 1. Check direct file upload
    for key in ['file', 'clothing_file', 'image']:
        if key in request.files:
            p, _ = _save_or_resolve_upload(request.files[key], prefix="garment_cls")
            if p:
                file_path = p
                break
                
    # 2. Check reference filename
    if not file_path:
        ref = request.form.get('clothing_reference') or request.form.get('reference_filename')
        if ref:
            file_path = _resolve_reference_file(ref)

    if not file_path or not os.path.exists(file_path):
        return jsonify({'error': 'No valid clothing image provided for classification'}), 400

    try:
        result = classify_garment_detailed(file_path)
        return jsonify({
            'success': True,
            'category': result['category'],
            'cloth_type': result['category'],
            'label': result['label'],
            'confidence': result['confidence'],
            'counts': result['counts']
        })
    except Exception as e:
        return jsonify({'error': str(e), 'category': 'upper', 'label': 'Top / Upper Body'}), 500

def _is_mobile_request(req):
    """Detect if request is from mobile device or requesting fast simple processing."""
    ua = req.headers.get('User-Agent', '').lower()
    mobile_keywords = ['mobile', 'android', 'iphone', 'ipad', 'ipod', 'blackberry', 'iemobile', 'opera mini', 'webos']
    if any(k in ua for k in mobile_keywords):
        return True
    if req.form.get('is_mobile', '').lower() in ['true', '1', 'yes']:
        return True
    if req.form.get('device', '').lower() == 'mobile':
        return True
    if req.form.get('mode', '').lower() in ['mobile', 'simple', 'fast']:
        return True
    if req.args.get('mobile', '').lower() in ['true', '1']:
        return True
    return False

@app.route('/process', methods=['POST'])
def process_image():
    try:
        # Only Try-On is supported
        ai_mode = 'tryon'
        reference_filename = request.form.get('reference_filename')

        is_mobile = _is_mobile_request(request)
        if is_mobile:
            print("[Flask API] [MOBILE] Mobile client detected - simple fast processing profile active")

        filename = None
        input_path = None

        # Handle Re-Run vs New Upload for Person Image
        if reference_filename:
            filename = secure_filename(reference_filename)
            input_path = _resolve_reference_file(filename)
            if not input_path:
                return jsonify({'error': 'Reference person file not found'}), 404
        else:
            if 'file' not in request.files:
                return jsonify({'error': 'No person image file provided'}), 400
            
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No selected person file'}), 400
                
            input_path, filename = _save_or_resolve_upload(file, prefix="person")
            if not input_path:
                return jsonify({'error': 'Invalid file type for person image'}), 400

        # Determine Try-On Mode: 'single' vs 'dual' (Both garments at once)
        cloth_mode = request.form.get('cloth_mode', 'single').lower()
        has_dual = (
            cloth_mode in ['dual', 'both'] or
            'clothing_file_bottom' in request.files or
            'clothing_file_2' in request.files or
            request.form.get('clothing_reference_bottom')
        )

        # ── DUAL GARMENTS TRY-ON (Both clothes at a time) ──
        if has_dual:
            print(f"[Flask API] Processing DUAL GARMENT try-on (Top + Bottom) [{'MOBILE-FAST' if is_mobile else 'DESKTOP'}]...")
            # 1. Resolve Top Garment
            top_path = None
            if 'clothing_file_top' in request.files:
                top_path, _ = _save_or_resolve_upload(request.files['clothing_file_top'], prefix="top")
            elif 'clothing_file' in request.files:
                top_path, _ = _save_or_resolve_upload(request.files['clothing_file'], prefix="top")
            elif request.form.get('clothing_reference_top'):
                top_path = _resolve_reference_file(request.form.get('clothing_reference_top'))
            elif request.form.get('clothing_reference'):
                top_path = _resolve_reference_file(request.form.get('clothing_reference'))

            # 2. Resolve Bottom Garment
            bottom_path = None
            if 'clothing_file_bottom' in request.files:
                bottom_path, _ = _save_or_resolve_upload(request.files['clothing_file_bottom'], prefix="bottom")
            elif 'clothing_file_2' in request.files:
                bottom_path, _ = _save_or_resolve_upload(request.files['clothing_file_2'], prefix="bottom")
            elif request.form.get('clothing_reference_bottom'):
                bottom_path = _resolve_reference_file(request.form.get('clothing_reference_bottom'))
            elif request.form.get('clothing_reference_2'):
                bottom_path = _resolve_reference_file(request.form.get('clothing_reference_2'))

            if not top_path or not bottom_path:
                return jsonify({'error': 'Dual try-on requires both top and bottom clothing images'}), 400

            # Automatic category check & swap if user inverted them
            cat1 = classify_garment(top_path)
            cat2 = classify_garment(bottom_path)
            print(f"[Flask API] Dual classification: Garment 1='{cat1}', Garment 2='{cat2}'")

            if cat1 == 'lower' and cat2 in ['upper', 'overall']:
                print("[Flask API] Auto-detected inverted garments! Swapping top and bottom slots.")
                top_path, bottom_path = bottom_path, top_path

            output_filename = f"dual_{filename}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

            success = process_dual_tryon(input_path, top_path, bottom_path, output_path, is_mobile=is_mobile)
            if success:
                return jsonify({
                    'success': True,
                    'mode': 'dual',
                    'is_mobile': is_mobile,
                    'input_filename': filename,
                    'output_filename': output_filename,
                    'output_url': url_for('static', filename=f'outputs/{output_filename}'),
                    'applied': {
                        'top': 'upper',
                        'bottom': 'lower'
                    }
                })
            else:
                return jsonify({'error': 'Dual try-on failed during synthesis.'}), 500

        # ── SINGLE GARMENT TRY-ON (One at a time) ──
        clothing_input_path = None
        cloth_type = request.form.get('cloth_type', 'auto')

        clothing_reference = request.form.get('clothing_reference')
        if clothing_reference:
            clothing_input_path = _resolve_reference_file(clothing_reference)
        elif 'clothing_file' in request.files:
            clothing_input_path, _ = _save_or_resolve_upload(request.files['clothing_file'], prefix="cloth")

        if not clothing_input_path or not os.path.exists(clothing_input_path):
            return jsonify({'error': 'Missing or invalid source clothing image for Try-On'}), 400

        # Auto-recognition if cloth_type is 'auto' or not specified
        if not cloth_type or cloth_type == 'auto':
            try:
                cloth_type = classify_garment(clothing_input_path)
                print(f"[Flask API] Auto-detected cloth type: '{cloth_type}'")
            except Exception as e:
                print('[Flask API] Classification fallback:', e)
                cloth_type = 'upper'

        output_filename = f"{ai_mode}_{filename}"
        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)

        success = process_tryon(input_path, clothing_input_path, output_path, cloth_type, is_mobile=is_mobile)
        if success:
            return jsonify({
                'success': True,
                'mode': 'single',
                'is_mobile': is_mobile,
                'input_filename': filename,
                'output_filename': output_filename,
                'detected_cloth_type': cloth_type,
                'output_url': url_for('static', filename=f'outputs/{output_filename}')
            })
        else:
            return jsonify({'error': f'{ai_mode.capitalize()} try-on failed.'}), 500

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Try-on error: {str(e)}', 'success': False}), 500


@app.route('/history')
def get_history():
    outputs = []
    if os.path.exists(app.config['OUTPUT_FOLDER']):
        # Sort by modification time to show newest first
        files = [f for f in os.listdir(app.config['OUTPUT_FOLDER']) if allowed_file(f)]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['OUTPUT_FOLDER'], x)), reverse=True)
        outputs = files
        
    history = []
    for f in outputs:
        # Reconstruct input filename by stripping the ai_mode prefix
        parts = f.split('_', 1)
        input_filename = parts[1] if len(parts) > 1 else f
        history.append({
            'output_filename': f,
            'input_filename': input_filename,
            'output_url': url_for('static', filename=f'outputs/{f}')
        })
    return jsonify(history)

@app.route('/extract_garments', methods=['POST'])
def extract_garments_endpoint():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        original_filename = secure_filename(file.filename)
        unique_id = str(uuid.uuid4())[:8]
        filename = f"outfit_{unique_id}_{original_filename}"
        input_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(input_path)
        
        try:
            items = extract_garments(input_path, app.config['GARMENT_FOLDER'])
            for item in items:
                item['url'] = url_for('static', filename=f"garments/{item['filename']}")
            return jsonify({'success': True, 'items': items})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/delete/<filename>', methods=['DELETE'])
def delete_image(filename):
    output_path = os.path.join(app.config['OUTPUT_FOLDER'], secure_filename(filename))
    if os.path.exists(output_path):
        try:
            os.remove(output_path)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'File not found'}), 404

@app.route('/wardrobe')
def get_wardrobe():
    garments = []
    if os.path.exists(app.config['GARMENT_FOLDER']):
        files = [f for f in os.listdir(app.config['GARMENT_FOLDER']) if allowed_file(f)]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(app.config['GARMENT_FOLDER'], x)), reverse=True)
        
        for f in files:
            # Reconstruct type if encoded in filename (e.g. ext_upper_uuid.jpg)
            parts = f.split('_')
            g_type = parts[1] if len(parts) > 1 else 'upper'
            garments.append({
                'type': g_type,
                'filename': f,
                'url': url_for('static', filename=f'garments/{f}')
            })
    return jsonify(garments)

@app.route('/delete_garment/<filename>', methods=['DELETE'])
def delete_garment(filename):
    garment_path = os.path.join(app.config['GARMENT_FOLDER'], secure_filename(filename))
    if os.path.exists(garment_path):
        try:
            os.remove(garment_path)
            return jsonify({'success': True})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)

