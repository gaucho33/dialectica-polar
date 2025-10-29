from flask import Flask, render_template, request, jsonify, send_file, flash, redirect, url_for, session
import os
import cv2
import numpy as np
import pandas as pd
import threading
import time
import json
from datetime import datetime, timedelta
from werkzeug.utils import secure_filename
from collections import defaultdict
from skimage.measure import label, regionprops
import base64
import io
from PIL import Image
import zipfile
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.secret_key = 'forensic_app_secret_key_2025'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['USAGE_FILE'] = 'user_usage.json'

# Crear directorios necesarios
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Configuración de límites
FREE_TRIAL_LIMIT = 5
PREMIUM_PRICE = 29.99

# Extensiones permitidas
# Configuración de archivos permitidos
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB por archivo
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB total

# ==============================================================================
# SISTEMA DE GESTIÓN DE USUARIOS Y LÍMITES
# ==============================================================================

def get_user_id():
    """Obtener ID único del usuario basado en IP y sesión"""
    if 'user_id' not in session:
        # Generar ID único basado en IP y timestamp
        user_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        session['user_id'] = f"user_{hash(user_ip)}_{int(time.time())}"
    return session['user_id']

def load_usage_data():
    """Cargar datos de uso de usuarios"""
    try:
        if os.path.exists(app.config['USAGE_FILE']):
            with open(app.config['USAGE_FILE'], 'r') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_usage_data(data):
    """Guardar datos de uso de usuarios"""
    try:
        with open(app.config['USAGE_FILE'], 'w') as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        print(f"Error saving usage data: {e}")

def get_user_usage(user_id):
    """Obtener uso actual del usuario"""
    usage_data = load_usage_data()
    if user_id not in usage_data:
        usage_data[user_id] = {
            'analyses_count': 0,
            'is_premium': False,
            'first_use': datetime.now().isoformat(),
            'last_use': datetime.now().isoformat()
        }
        save_usage_data(usage_data)
    return usage_data[user_id]

def increment_user_usage(user_id):
    """Incrementar contador de uso del usuario"""
    usage_data = load_usage_data()
    if user_id in usage_data:
        usage_data[user_id]['analyses_count'] += 1
        usage_data[user_id]['last_use'] = datetime.now().isoformat()
        save_usage_data(usage_data)
        return usage_data[user_id]['analyses_count']
    return 0

def can_user_analyze(user_id):
    """Verificar si el usuario puede realizar análisis"""
    user_data = get_user_usage(user_id)
    return user_data['is_premium'] or user_data['analyses_count'] < FREE_TRIAL_LIMIT

def get_remaining_trials(user_id):
    """Obtener pruebas restantes del usuario"""
    user_data = get_user_usage(user_id)
    if user_data['is_premium']:
        return -1  # Ilimitado
    return max(0, FREE_TRIAL_LIMIT - user_data['analyses_count'])

# ==============================================================================
# ALGORITMOS FORENSES (ADAPTADOS PARA WEB)
# ==============================================================================

def allowed_file(filename):
    """Verificar si el archivo tiene una extensión permitida"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file):
    """Validar el tamaño del archivo"""
    file.seek(0, 2)  # Ir al final del archivo
    size = file.tell()
    file.seek(0)  # Volver al inicio
    return size <= MAX_FILE_SIZE

def get_file_size_mb(size_bytes):
    """Convertir bytes a MB con formato"""
    return round(size_bytes / (1024 * 1024), 1)

def rle_encode(mask):
    """Codifica una máscara NumPy binaria a RLE (Run-Length Encoding)"""
    if mask is None or not np.any(mask):
        return "authentic"
    
    if mask.ndim == 3:
        mask = mask.squeeze()
        
    pixels = mask.T.flatten().astype(int) 
    pixels = np.concatenate([[0], pixels, [0]])
    
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] = runs[1::2] - runs[:-1:2]
    
    return ' '.join(str(x) for x in runs)

def extract_noise_continent(image, bbox, padding=10):
    """Extrae características de ruido para análisis forense"""
    min_row, min_col, max_row, max_col = bbox
    h, w = image.shape

    cont_min_r = max(0, min_row - padding)
    cont_max_r = min(h, max_row + padding)
    cont_min_c = max(0, min_col - padding)
    cont_max_c = min(w, max_col + padding)

    region_of_interest = image[cont_min_r:cont_max_r, cont_min_c:cont_max_c]

    # Kernel de alta frecuencia para extraer el ruido residual
    kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
    noise_map = cv2.filter2D(region_of_interest, -1, kernel)
    
    rel_min_r, rel_min_c = min_row - cont_min_r, min_col - cont_min_c
    rel_max_r, rel_max_c = max_row - cont_min_r, max_col - cont_min_c
    
    noise_patch = noise_map[rel_min_r:rel_max_r, rel_min_c:rel_max_c] 
    
    noise_continent_mask = np.ones(noise_map.shape, dtype=bool)
    noise_continent_mask[rel_min_r:rel_max_r, rel_min_c:rel_max_c] = False
    
    noise_continent = noise_map[noise_continent_mask]
    
    if noise_continent.size == 0:
        return {
            'M_C_Ruido': np.array([np.mean(np.abs(noise_patch)), np.std(noise_patch)]),
            'M_Co': np.array([0.0, 0.0])
        }

    return {
        'M_C_Ruido': np.array([np.mean(np.abs(noise_patch)), np.std(noise_patch)]), 
        'M_Co': np.array([np.mean(np.abs(noise_continent)), np.std(noise_continent)])
    }

def verify_and_segment_forgeries_optimized(gray_image, params):
    """Pipeline forense optimizado para web"""
    h, w = gray_image.shape
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        sift = cv2.ORB_create() 
        
    kp, des = sift.detectAndCompute(gray_image, None)
    
    if des is None or len(des) < 2: 
        return None

    try:
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des, des, k=2)
    except Exception:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des, des, k=2)

    candidate_locations = defaultdict(list) 
    
    # FASE 1: Detección de duplicaciones
    for m, n in matches:
        if m.distance < params['sift_distance_ratio'] * n.distance:
            pt1 = np.int32(kp[m.queryIdx].pt)
            pt2 = np.int32(kp[m.trainIdx].pt)
            
            if np.linalg.norm(pt1 - pt2) > params['min_distance_pixels']:
                candidate_locations[m.queryIdx].append(pt1)
                candidate_locations[m.queryIdx].append(pt2)
    
    if not candidate_locations: 
        return None

    # FASE 2: Verificación forense
    suspicious_mask = np.zeros((h, w), dtype=bool)
    patch_size = params['block_size'] // 2
    
    for _, locations in candidate_locations.items():
        if len(locations) >= 2: 
            clusters_data = []
            
            for pt in locations:
                r, c = pt[1], pt[0]
                min_r, max_r = max(0, r - patch_size), min(h, r + patch_size)
                min_c, max_c = max(0, c - patch_size), min(w, c + patch_size)
                bbox = (min_r, min_c, max_r, max_c)
                
                noise_results = extract_noise_continent(gray_image, bbox, padding=params['padding_noise'])
                clusters_data.append({'bbox': bbox, 'M_C_Ruido': noise_results['M_C_Ruido'], 'M_Co': noise_results['M_Co']})
                
            for i in range(len(clusters_data)):
                cluster_A = clusters_data[i]
                m_r_local_A = np.linalg.norm(cluster_A['M_C_Ruido'] - cluster_A['M_Co'])
                is_locally_inconsistent_A = m_r_local_A > params['forensic_threshold_local']
                
                for j in range(i + 1, len(clusters_data)):
                    cluster_B = clusters_data[j]
                    
                    m_r_global = np.linalg.norm(cluster_A['M_Co'] - cluster_B['M_Co'])
                    m_r_local_B = np.linalg.norm(cluster_B['M_C_Ruido'] - cluster_B['M_Co'])
                    is_locally_inconsistent_B = m_r_local_B > params['forensic_threshold_local']
                    
                    is_forgery = (m_r_global > params['forensic_threshold_global']) or \
                                 is_locally_inconsistent_A or is_locally_inconsistent_B
                    
                    if is_forgery:
                        for cluster in [cluster_A, cluster_B]:
                            min_r, min_c, max_r, max_c = cluster['bbox']
                            suspicious_mask[min_r:max_r, min_c:max_c] = 1

    # Agrupamiento final
    final_mask = np.zeros_like(gray_image, dtype=np.uint8)
    labeled_mask = label(suspicious_mask)
    for region in regionprops(labeled_mask):
        if region.area >= params['min_region_size']:
             final_mask[region.coords[:, 0], region.coords[:, 1]] = 1
             
    return final_mask.astype(np.uint8) if np.any(final_mask) else None

# Parámetros del modelo forense
MODEL_PARAMS = {
    'block_size': 16, 
    'sift_distance_ratio': 0.9,             
    'min_distance_pixels': 20,
    'padding_noise': 10,
    'min_region_size': 50,
    'forensic_threshold_global': 0.25,       
    'forensic_threshold_local': 0.25         
}

# ==============================================================================
# RUTAS FLASK
# ==============================================================================

@app.route('/')
def index():
    """Página principal"""
    user_id = get_user_id()
    user_data = get_user_usage(user_id)
    remaining_trials = get_remaining_trials(user_id)
    
    return render_template('index.html', 
                         user_data=user_data, 
                         remaining_trials=remaining_trials,
                         premium_price=PREMIUM_PRICE)

@app.route('/pricing')
def pricing():
    """Página de precios"""
    user_id = get_user_id()
    user_data = get_user_usage(user_id)
    remaining_trials = get_remaining_trials(user_id)
    
    return render_template('pricing.html', 
                         user_data=user_data, 
                         remaining_trials=remaining_trials,
                         premium_price=PREMIUM_PRICE)

@app.route('/upload', methods=['POST'])
def upload_files():
    """Subir archivos para análisis con validación completa"""
    if 'files' not in request.files:
        return jsonify({'error': 'No files selected'}), 400
    
    files = request.files.getlist('files')
    uploaded_files = []
    errors = []
    total_size = 0
    
    # Validar cada archivo
    for file in files:
        if file and file.filename:
            # Validar extensión
            if not allowed_file(file.filename):
                ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else 'desconocida'
                errors.append(f"Formato no soportado: {file.filename} (.{ext})")
                continue
            
            # Validar tamaño individual
            if not validate_file_size(file):
                file.seek(0, 2)
                size_mb = get_file_size_mb(file.tell())
                file.seek(0)
                errors.append(f"Archivo muy grande: {file.filename} ({size_mb}MB). Máximo 10MB.")
                continue
            
            # Calcular tamaño total
            file.seek(0, 2)
            file_size = file.tell()
            file.seek(0)
            total_size += file_size
            
            # Guardar archivo válido
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append({
                'filename': filename,
                'size_mb': get_file_size_mb(file_size)
            })
    
    # Validar tamaño total
    if total_size > MAX_TOTAL_SIZE:
        total_mb = get_file_size_mb(total_size)
        errors.append(f"Tamaño total muy grande: {total_mb}MB. Máximo 50MB total.")
        # Eliminar archivos ya subidos
        for file_info in uploaded_files:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_info['filename'])
            if os.path.exists(filepath):
                os.remove(filepath)
        return jsonify({'error': errors}), 400
    
    # Verificar si hay errores
    if errors and not uploaded_files:
        return jsonify({'error': errors}), 400
    
    # Respuesta con información detallada
    response_data = {
        'success': True, 
        'files': [f['filename'] for f in uploaded_files],
        'total_size_mb': get_file_size_mb(total_size),
        'file_count': len(uploaded_files)
    }
    
    if errors:
        response_data['warnings'] = errors
    
    return jsonify(response_data)

@app.route('/analyze', methods=['POST'])
def analyze_images():
    """Analizar imágenes subidas"""
    try:
        user_id = get_user_id()
        
        # Verificar si el usuario puede realizar análisis
        if not can_user_analyze(user_id):
            return jsonify({
                'error': 'Has alcanzado el límite de 5 pruebas gratuitas. Actualiza a Premium para análisis ilimitados.',
                'upgrade_required': True,
                'remaining_trials': 0
            }), 403
        
        # Incrementar contador de uso
        increment_user_usage(user_id)
        remaining = get_remaining_trials(user_id)
        
        # Obtener lista de archivos subidos
        upload_dir = app.config['UPLOAD_FOLDER']
        image_files = [f for f in os.listdir(upload_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif'))]
        
        if not image_files:
            return jsonify({'error': 'No images found for analysis'}), 400
        
        results = []
        
        for filename in image_files:
            filepath = os.path.join(upload_dir, filename)
            
            # Cargar y procesar imagen
            img = cv2.imread(filepath)
            if img is None:
                results.append({'filename': filename, 'result': 'error', 'encoded_pixels': 'error'})
                continue
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Aplicar algoritmo forense
            final_mask = verify_and_segment_forgeries_optimized(gray, MODEL_PARAMS)
            encoded_pixels = rle_encode(final_mask)
            
            # Determinar resultado
            result_type = 'authentic' if encoded_pixels == 'authentic' else 'manipulated'
            
            results.append({
                'filename': filename,
                'result': result_type,
                'encoded_pixels': encoded_pixels,
                'dimensions': f"{img.shape[1]}x{img.shape[0]}"
            })
        
        # Generar CSV de resultados
        df = pd.DataFrame([{'ImageId': r['filename'], 'EncodedPixels': r['encoded_pixels']} 
                          for r in results])
        
        csv_path = os.path.join(app.config['RESULTS_FOLDER'], 'forensic_results.csv')
        df.to_csv(csv_path, index=False)
        
        return jsonify({
            'success': True,
            'results': results,
            'total_images': len(results),
            'authentic_count': sum(1 for r in results if r['result'] == 'authentic'),
            'manipulated_count': sum(1 for r in results if r['result'] == 'manipulated'),
            'csv_available': True,
            'remaining_trials': remaining,
            'is_premium': get_user_usage(user_id)['is_premium']
        })
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/download-results')
def download_results():
    """Descargar archivo CSV de resultados"""
    csv_path = os.path.join(app.config['RESULTS_FOLDER'], 'forensic_results.csv')
    if os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True, download_name='forensic_analysis_results.csv')
    else:
        flash('No results file available for download.', 'error')
        return redirect(url_for('index'))

@app.route('/clear')
def clear_data():
    """Limpiar archivos subidos y resultados"""
    try:
        # Limpiar uploads
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        
        # Limpiar resultados
        for filename in os.listdir(app.config['RESULTS_FOLDER']):
            filepath = os.path.join(app.config['RESULTS_FOLDER'], filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        
        return jsonify({'success': True, 'message': 'All data cleared successfully'})
    except Exception as e:
        return jsonify({'error': f'Failed to clear data: {str(e)}'}), 500

@app.route('/status')
def status():
    """Estado de la aplicación"""
    user_id = get_user_id()
    user_data = get_user_usage(user_id)
    remaining_trials = get_remaining_trials(user_id)
    
    upload_count = len([f for f in os.listdir(app.config['UPLOAD_FOLDER']) 
                       if os.path.isfile(os.path.join(app.config['UPLOAD_FOLDER'], f))])
    
    return jsonify({
        'status': 'running',
        'uploaded_files': upload_count,
        'model_params': MODEL_PARAMS,
        'user_data': user_data,
        'remaining_trials': remaining_trials,
        'is_premium': user_data['is_premium']
    })

@app.route('/upgrade', methods=['POST'])
def upgrade_to_premium():
    """Simular upgrade a premium (en producción conectar con procesador de pagos)"""
    user_id = get_user_id()
    usage_data = load_usage_data()
    
    if user_id in usage_data:
        usage_data[user_id]['is_premium'] = True
        usage_data[user_id]['premium_date'] = datetime.now().isoformat()
        save_usage_data(usage_data)
        
        return jsonify({
            'success': True,
            'message': 'Upgrade a Premium exitoso! Ahora tienes análisis ilimitados.',
            'is_premium': True
        })
    
    return jsonify({'error': 'Usuario no encontrado'}), 400

@app.route('/download-desktop')
def download_desktop_app():
    """Descargar aplicación de escritorio"""
    user_id = get_user_id()
    user_data = get_user_usage(user_id)
    
    # Solo usuarios premium pueden descargar la app de escritorio
    if not user_data['is_premium']:
        return jsonify({
            'error': 'Necesitas ser usuario Premium para descargar la aplicación de escritorio.',
            'upgrade_required': True
        }), 403
    
    # Crear un zip con la aplicación de escritorio
    try:
        zip_path = os.path.join(app.config['RESULTS_FOLDER'], 'ForensicApp_Desktop.zip')
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            # Agregar archivos de la aplicación de escritorio
            if os.path.exists('app_completa.py'):
                zipf.write('app_completa.py', 'ForensicApp/app_completa.py')
            if os.path.exists('run_app.bat'):
                zipf.write('run_app.bat', 'ForensicApp/run_app.bat')
            if os.path.exists('run_app.ps1'):
                zipf.write('run_app.ps1', 'ForensicApp/run_app.ps1')
            if os.path.exists('requirements.txt'):
                zipf.write('requirements.txt', 'ForensicApp/requirements.txt')
            
            # Crear archivo README para la descarga
            readme_content = """# Aplicación Forense de Escritorio - Versión Premium

¡Gracias por ser usuario Premium!

## Instalación:
1. Instalar Python 3.11
2. pip install -r requirements.txt
3. Ejecutar run_app.ps1 o run_app.bat

## Características Premium:
- Análisis ilimitados
- Sin conexión a internet requerida
- Procesamiento local de imágenes
- Todas las funcionalidades forenses avanzadas

Soporte: forensic.support@empresa.com
"""
            zipf.writestr('ForensicApp/README_PREMIUM.txt', readme_content)
        
        return send_file(zip_path, as_attachment=True, download_name='ForensicApp_Desktop_Premium.zip')
        
    except Exception as e:
        return jsonify({'error': f'Error creating download: {str(e)}'}), 500

@app.route('/reset-trial')
def reset_trial():
    """Reset trial para testing (remover en producción)"""
    user_id = get_user_id()
    usage_data = load_usage_data()
    
    if user_id in usage_data:
        usage_data[user_id]['analyses_count'] = 0
        usage_data[user_id]['is_premium'] = False
        save_usage_data(usage_data)
    
    return jsonify({'success': True, 'message': 'Trial reset successfully'})

if __name__ == '__main__':
    print("🌐 Iniciando aplicación forense web con sistema freemium...")
    print("📍 Accede a: http://localhost:5000")
    print("🔍 Funcionalidades: 5 pruebas gratuitas + Upgrade Premium")
    print("💰 Premium: Análisis ilimitados + Descarga de app de escritorio")
    app.run(debug=True, host='0.0.0.0', port=5000)