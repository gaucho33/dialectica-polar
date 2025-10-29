# Configuración de la Aplicación Forense

## Configuración Flask
DEBUG = True
SECRET_KEY = 'your-secret-key-change-in-production'
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'

## Configuración Freemium
FREE_TRIAL_LIMIT = 5
PREMIUM_PRICE = 29.99

## Configuración de Archivos
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_TOTAL_SIZE = 50 * 1024 * 1024  # 50MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp'}

## Configuración de Análisis Forense
ANALYSIS_ALGORITHMS = ['SIFT', 'ORB', 'NOISE_DETECTION']
OUTPUT_FORMAT = 'CSV'

## Configuración de Seguridad (Producción)
# Cambiar estos valores para producción:
# - SECRET_KEY: Generar clave aleatoria segura
# - DEBUG: Cambiar a False
# - Configurar HTTPS
# - Configurar dominio personalizado