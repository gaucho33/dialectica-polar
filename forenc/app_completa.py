import os
import cv2
import numpy as np
import pandas as pd
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import customtkinter as ctk

# Asegúrate de que las bibliotecas estén instaladas en tu máquina local:
# pip install customtkinter opencv-python-headless numpy pandas scikit-image matplotlib

from skimage.measure import label, regionprops
from collections import defaultdict
import matplotlib.pyplot as plt 
import warnings
warnings.filterwarnings('ignore') # Ignorar warnings de NumPy y Pandas

# Configuración de apariencia de CustomTkinter
ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue") 


# ==============================================================================
# 1. FUNCIONES CENTRALES DEL ALGORITMO (FORENSE)
# ==============================================================================

def rle_encode(mask):
    """
    Codifica una máscara NumPy binaria a RLE (Run-Length Encoding), en Orden F y Base 1. 
    Devuelve "authentic" si la máscara es nula o vacía.
    """
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
    """
    Calcula los descriptores estadísticos de ruido para el parche sospechoso (M_C_Ruido) 
    y su entorno circundante (M_Co).
    """
    min_row, min_col, max_row, max_col = bbox
    h, w = image.shape

    # Define el continente de ruido con padding
    cont_min_r = max(0, min_row - padding)
    cont_max_r = min(h, max_row + padding)
    cont_min_c = max(0, min_col - padding)
    cont_max_c = min(w, max_col + padding)

    region_of_interest = image[cont_min_r:cont_max_r, cont_min_c:cont_max_c]

    # Kernel de alta frecuencia para extraer el ruido residual
    kernel = np.array([[-1, -1, -1], [-1,  8, -1], [-1, -1, -1]])
    noise_map = cv2.filter2D(region_of_interest, -1, kernel)
    
    # Coordenadas relativas del parche dentro del continente
    rel_min_r, rel_min_c = min_row - cont_min_r, min_col - cont_min_c
    rel_max_r, rel_max_c = max_row - cont_min_r, max_col - cont_min_c
    
    # Extrae el ruido del parche (M_C_Ruido)
    noise_patch = noise_map[rel_min_r:rel_max_r, rel_min_c:rel_max_c] 
    
    # Extrae el ruido del continente/entorno (M_Co)
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
    """Implementa el pipeline forense en dos fases: SIFT y Verificación de ruido."""
    h, w = gray_image.shape
    try:
        sift = cv2.SIFT_create()
    except AttributeError:
        sift = cv2.ORB_create() 
        
    kp, des = sift.detectAndCompute(gray_image, None)
    
    if des is None or len(des) < 2: return None

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    
    try:
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des, des, k=2)
    except Exception:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des, des, k=2)


    candidate_locations = defaultdict(list) 
    
    # FASE 1: Detección Rápida de M_C (Contenido) - Filtrado de SIFT
    for m, n in matches:
        if m.distance < params['sift_distance_ratio'] * n.distance:
            pt1 = np.int32(kp[m.queryIdx].pt)
            pt2 = np.int32(kp[m.trainIdx].pt)
            
            if np.linalg.norm(pt1 - pt2) > params['min_distance_pixels']:
                candidate_locations[m.queryIdx].append(pt1)
                candidate_locations[m.queryIdx].append(pt2)
    
    if not candidate_locations: return None

    # FASE 2: Verificación Forense (M_R_Local/Global) - Análisis de Ruido
    suspicious_mask = np.zeros((h, w), dtype=bool)
    patch_size = params['block_size'] // 2
    
    for _, locations in candidate_locations.items():
        if len(locations) >= 2: 
            clusters_data = []
            
            # 1. Extraer características de ruido
            for pt in locations:
                r, c = pt[1], pt[0]
                min_r, max_r = max(0, r - patch_size), min(h, r + patch_size)
                min_c, max_c = max(0, c - patch_size), min(w, c + patch_size)
                bbox = (min_r, min_c, max_r, max_c)
                
                noise_results = extract_noise_continent(gray_image, bbox, padding=params['padding_noise'])
                
                clusters_data.append({'bbox': bbox, 'M_C_Ruido': noise_results['M_C_Ruido'], 'M_Co': noise_results['M_Co']})
                
            # 2. Comparación Local y Global
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

    # Agrupamiento geométrico final 
    final_mask = np.zeros_like(gray_image, dtype=np.uint8)
    labeled_mask = label(suspicious_mask)
    for region in regionprops(labeled_mask):
        if region.area >= params['min_region_size']:
             final_mask[region.coords[:, 0], region.coords[:, 1]] = 1
             
    return final_mask.astype(np.uint8) if np.any(final_mask) else None

# ------------------------------------------------------------------------------

def visualize_forgery(image, mask, image_id):
    """Visualiza la imagen original y superpone la máscara de detección usando Matplotlib."""
    plt.figure(figsize=(10, 8))
    
    if mask is None or not np.any(mask):
        img_display = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(img_display)
        plt.title(f"Resultado: {image_id} (AUTÉNTICA)")
    else:
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask_color = np.zeros_like(img_rgb, dtype=np.uint8)
        mask_color[mask == 1] = [255, 0, 0] # Color rojo para la falsificación
        
        # Superpone la máscara roja con transparencia
        masked_img = cv2.addWeighted(img_rgb, 0.7, mask_color, 0.3, 0)
        plt.imshow(masked_img)
        plt.title(f"Resultado: {image_id} (FALSIFICACIÓN DETECTADA)")
    
    plt.axis('off')
    plt.tight_layout()
    plt.show() # Muestra la ventana de Matplotlib

# ==============================================================================
# 2. FUNCIÓN PRINCIPAL DE EJECUCIÓN DEL PIPELINE (Integrada con el Logging)
# ==============================================================================

def run_submission_pipeline(app_instance, image_dir, sample_path, output_filename, params):
    """
    Ejecuta el pipeline de procesamiento forense, enviando la salida a la GUI.
    """
    
    app_instance.log_message(f"Iniciando el procesamiento desde: {image_dir}")
    
    # Búsqueda robusta de imágenes en la ruta base o subcarpetas comunes
    test_image_dir = image_dir
    
    # Primero busca en la ruta base
    image_paths = [os.path.join(test_image_dir, f) 
                   for f in os.listdir(test_image_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))] 

    if not image_paths:
        # Busca en subcarpetas comunes (ej: 'test_images', 'images', 'data')
        app_instance.log_message("No se encontraron imágenes en la ruta principal. Buscando en subcarpetas...")
        for subdir in ['test_images', 'images', 'data', os.path.basename(image_dir)]:
            potential_dir = os.path.join(image_dir, subdir)
            if os.path.isdir(potential_dir):
                 image_paths = [os.path.join(potential_dir, f) 
                                for f in os.listdir(potential_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
                 if image_paths:
                    test_image_dir = potential_dir
                    break
    
    if not image_paths:
        app_instance.log_message(f"❌ ERROR CRÍTICO: No se encontraron imágenes válidas en {image_dir} ni subcarpetas. Verifique la ruta.", is_error=True)
        return False

    app_instance.log_message(f"Procesando {len(image_paths)} imágenes desde {test_image_dir}...")

    # === Carga y Corrección de Encabezados ===
    try:
        sample_df = pd.read_csv(sample_path)
        submission_df = sample_df.copy()
    except Exception as e:
        app_instance.log_message(f"❌ ERROR CRÍTICO: No se pudo leer el CSV de muestra en {sample_path}.", is_error=True)
        app_instance.log_message(f"Detalle: {e}", is_error=True)
        return False
    
    # Normalizar los nombres de columna (robustez para diferentes formatos)
    column_mapping = {}
    for col in submission_df.columns:
        clean_col = col.strip().lower() 
        if 'imageid' in clean_col or 'case_id' in clean_col or 'id' == clean_col:
            column_mapping[col] = 'ImageId'
        elif 'encodedpixels' in clean_col or 'annotation' in clean_col:
            column_mapping[col] = 'EncodedPixels'

    submission_df = submission_df.rename(columns=column_mapping)
    
    if 'ImageId' not in submission_df.columns or 'EncodedPixels' not in submission_df.columns:
        app_instance.log_message(f"❌ ERROR CRÍTICO: Faltan encabezados esenciales ('ImageId' o 'EncodedPixels').", is_error=True)
        return False
    
    id_to_index = {row['ImageId']: idx for idx, row in submission_df.iterrows()}
    
    total_images_found = len(image_paths)

    # === Bucle Principal de Procesamiento Forense ===
    for i, image_path in enumerate(image_paths):
        image_id = os.path.basename(image_path)
        image_id_no_ext = os.path.splitext(image_id)[0]
        
        # Mapeo robusto:
        df_key = None
        if image_id in id_to_index:
            df_key = image_id
        elif image_id_no_ext in id_to_index:
            df_key = image_id_no_ext
        else:
            continue # Ignora archivos no listados en el CSV
        
        idx = id_to_index[df_key]

        if (i + 1) % 50 == 0 or i == 0:
            app_instance.log_message(f"Procesando imagen {i+1}/{total_images_found}: {image_id}")
        
        img = cv2.imread(image_path)
        
        if img is None: 
            submission_df.loc[idx, 'EncodedPixels'] = "authentic"
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        final_mask = verify_and_segment_forgeries_optimized(gray, params)
        prediction_string = rle_encode(final_mask)
        
        submission_df.loc[idx, 'EncodedPixels'] = prediction_string
        submission_df.loc[idx, 'ImageId'] = df_key 

        # --- VISUALIZACIÓN CONDICIONAL ---
        if params['visualize_results'] and (final_mask is not None and np.any(final_mask)):
             app_instance.log_message(f"Detección de falsificación en: {image_id}. Mostrando visualización.")
             visualize_forgery(img, final_mask, image_id)


    # === GUARDAR Y ASEGURAR FORMATO FINAL ===
    final_submission = submission_df[['ImageId', 'EncodedPixels']].copy()
    final_submission['EncodedPixels'] = final_submission['EncodedPixels'].fillna('authentic')

    # Asegurarse de que el directorio de salida exista
    output_dir = os.path.dirname(output_filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        final_submission.to_csv(output_filename, index=False, encoding='utf-8')
        app_instance.log_message(f"\n✅ ¡PROCESAMIENTO COMPLETO! Archivo de envío generado en: {output_filename}", is_success=True)
        app_instance.log_message(f"Filas totales en el archivo de envío: {len(final_submission)}")
    except Exception as e:
        app_instance.log_message(f"❌ ERROR al guardar el CSV: {e}", is_error=True)
        return False
        
    return True


# ==============================================================================
# 3. INTERFAZ GRÁFICA (CUSTOMTKINTER)
# ==============================================================================

class ForensicsApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("🔬 Herramienta Forense de Detección de Copy-Move")
        self.geometry("800x850")
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(3, weight=1) # Fila del log

        self.DEFAULT_PARAMS = {
            'block_size': 16, 
            'sift_distance_ratio': 0.9,             
            'min_distance_pixels': 20,
            'padding_noise': 10,
            'min_region_size': 50,       
            'forensic_threshold_global': 0.25,       
            'forensic_threshold_local': 0.25,
            'visualize_results': True              
        }

        # --- 1. Marco de Rutas ---
        self.path_frame = ctk.CTkFrame(self)
        self.path_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=(20, 10), sticky="ew")
        self.path_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(self.path_frame, text="Configuración de Rutas", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, pady=(10, 5))

        # Directorio de Imágenes (Usa el directorio de trabajo actual como default)
        self.dir_path_var = tk.StringVar(value=os.getcwd())
        ctk.CTkLabel(self.path_frame, text="Directorio de Imágenes:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(self.path_frame, textvariable=self.dir_path_var, width=500).grid(row=2, column=0, padx=10, pady=5, sticky="ew")
        ctk.CTkButton(self.path_frame, text="Seleccionar Carpeta", command=self.select_directory).grid(row=2, column=1, padx=10, pady=5, sticky="e")

        # CSV de Muestra
        self.sample_csv_var = tk.StringVar(value="sample_submission.csv")
        ctk.CTkLabel(self.path_frame, text="Nombre del CSV de Muestra:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(self.path_frame, textvariable=self.sample_csv_var).grid(row=4, column=0, padx=10, pady=5, sticky="ew")

        # CSV de Salida
        self.output_csv_var = tk.StringVar(value="submission_forense.csv")
        ctk.CTkLabel(self.path_frame, text="Nombre del CSV de Salida:").grid(row=3, column=1, padx=10, pady=5, sticky="w")
        ctk.CTkEntry(self.path_frame, textvariable=self.output_csv_var).grid(row=4, column=1, padx=10, pady=5, sticky="ew")


        # --- 2. Marco de Parámetros ---
        self.params_frame = ctk.CTkFrame(self)
        self.params_frame.grid(row=1, column=0, padx=20, pady=(10, 10), sticky="nsew")
        self.params_frame.grid_columnconfigure((0, 1), weight=1)
        
        ctk.CTkLabel(self.params_frame, text="Ajustes del Algoritmo Forense", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=2, pady=(10, 5))

        # Variables de parámetros
        self.block_size_var = tk.IntVar(value=self.DEFAULT_PARAMS['block_size'])
        self.sift_ratio_var = tk.DoubleVar(value=self.DEFAULT_PARAMS['sift_distance_ratio'])
        self.min_dist_var = tk.IntVar(value=self.DEFAULT_PARAMS['min_distance_pixels'])
        self.min_region_var = tk.IntVar(value=self.DEFAULT_PARAMS['min_region_size'])
        self.global_thresh_var = tk.DoubleVar(value=self.DEFAULT_PARAMS['forensic_threshold_global'])
        self.local_thresh_var = tk.DoubleVar(value=self.DEFAULT_PARAMS['forensic_threshold_local'])

        self.create_slider(self.params_frame, "Tamaño de Bloque (px):", 1, self.block_size_var, 8, 32, 4)
        self.create_slider(self.params_frame, "Mín. Distancia Copy-Move (px):", 2, self.min_dist_var, 5, 50, 5)
        self.create_slider(self.params_frame, "Mín. Tamaño Región (px²):", 3, self.min_region_var, 10, 100, 10)
        self.create_slider(self.params_frame, "Ratio SIFT (0.7-1.0):", 4, self.sift_ratio_var, 0.7, 1.0, 0.05, fmt='%.2f')
        self.create_slider(self.params_frame, "Umbral Ruido Global:", 5, self.global_thresh_var, 0.01, 1.0, 0.05, fmt='%.2f')
        self.create_slider(self.params_frame, "Umbral Ruido Local:", 6, self.local_thresh_var, 0.01, 1.0, 0.05, fmt='%.2f')

        # --- 3. Botón y Log ---
        self.run_button = ctk.CTkButton(self, text="▶ INICIAR ANÁLISIS FORENSE", command=self.start_analysis_thread, height=50, font=ctk.CTkFont(size=16, weight="bold"))
        self.run_button.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 10), sticky="ew")

        ctk.CTkLabel(self, text="Registro de Ejecución (Log):", font=ctk.CTkFont(weight="bold")).grid(row=3, column=0, columnspan=2, padx=20, pady=(0, 5), sticky="w")
        
        self.log_widget = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=15, state='disabled')
        self.log_widget.grid(row=4, column=0, columnspan=2, padx=20, pady=(0, 20), sticky="nsew")
        
        # Etiquetas de color para el log
        self.log_widget.tag_config('error', foreground='red')
        self.log_widget.tag_config('success', foreground='green', font='Arial 10 bold')
        self.log_widget.tag_config('default', foreground='gray')
        
    def create_slider(self, parent, label_text, row, var, min_val, max_val, step, fmt='%.0f'):
        """Crea un label, slider y entrada de valor en una fila."""
        label = ctk.CTkLabel(parent, text=label_text, width=200, anchor="w")
        label.grid(row=row, column=0, padx=(10, 5), pady=5, sticky="w")

        slider = ctk.CTkSlider(parent, from_=min_val, to=max_val, number_of_steps=int((max_val - min_val) / step), variable=var)
        slider.grid(row=row, column=1, padx=(5, 10), pady=5, sticky="ew")
        
        # Mostrar el valor actual del slider
        value_label = ctk.CTkLabel(parent, text=fmt % var.get(), width=50)
        value_label.grid(row=row, column=2, padx=(0, 10), pady=5)
        
        def update_label(value):
            value_label.configure(text=fmt % value)
        
        var.trace_add("write", lambda name, index, mode: update_label(var.get()))


    def select_directory(self):
        """Abre un diálogo para seleccionar el directorio de imágenes."""
        directory = filedialog.askdirectory(title="Seleccionar Carpeta de Imágenes")
        if directory:
            self.dir_path_var.set(directory)

    def log_message(self, message, is_error=False, is_success=False):
        """Escribe un mensaje en el widget de log de la GUI."""
        self.log_widget.configure(state='normal')
        
        tag = 'default'
        if is_error:
            tag = 'error'
        elif is_success:
            tag = 'success'
            
        self.log_widget.insert(tk.END, message + "\n", tag)
        self.log_widget.see(tk.END) # Auto-scroll
        self.log_widget.configure(state='disabled')
        self.update_idletasks() # Asegura que la GUI se actualice

    def get_current_params(self):
        """Recoge los parámetros del modelo desde los widgets."""
        return {
            'block_size': self.block_size_var.get(), 
            'sift_distance_ratio': self.sift_ratio_var.get(),             
            'min_distance_pixels': self.min_dist_var.get(),
            'padding_noise': self.DEFAULT_PARAMS['padding_noise'], # Valor fijo
            'min_region_size': self.min_region_var.get(),       
            'forensic_threshold_global': self.global_thresh_var.get(),       
            'forensic_threshold_local': self.local_thresh_var.get(),
            'visualize_results': self.DEFAULT_PARAMS['visualize_results'] 
        }

    def start_analysis_thread(self):
        """Inicia el análisis en un hilo separado para no bloquear la GUI."""
        self.log_widget.configure(state='normal')
        self.log_widget.delete('1.0', tk.END)
        self.log_widget.configure(state='disabled')
        self.log_message("Preparando el análisis. Por favor, espere...")
        self.run_button.configure(state="disabled", text="Procesando...")

        # Ejecutar la función run_analysis_wrapper en un hilo
        self.analysis_thread = threading.Thread(target=self.run_analysis_wrapper)
        self.analysis_thread.start()

    def run_analysis_wrapper(self):
        """Función wrapper para ejecutar el análisis y manejar el estado del botón."""
        try:
            image_dir = self.dir_path_var.get()
            sample_csv_name = self.sample_csv_var.get()
            output_csv_name = self.output_csv_var.get()

            # Rutas completas
            sample_path = os.path.join(image_dir, sample_csv_name)
            output_filename = os.path.join(image_dir, output_csv_name)
            
            params = self.get_current_params()
            
            self.log_message(f"\n--- Parámetros de Ejecución ---")
            self.log_message(f"Dir Imágenes: {image_dir}")
            self.log_message(f"CSV Muestra: {sample_path}")
            self.log_message(f"CSV Salida: {output_filename}")
            self.log_message(f"Modelo: {params}")
            self.log_message("-" * 40)

            run_submission_pipeline(self, image_dir, sample_path, output_filename, params)
            
        except Exception as e:
            self.log_message(f"❌ ERROR INESPERADO: {e}", is_error=True)
            messagebox.showerror("Error de Ejecución", f"Ocurrió un error inesperado:\n{e}")
        finally:
            # Habilitar el botón de nuevo al finalizar
            self.run_button.configure(state="normal", text="▶ INICIAR ANÁLISIS FORENSE")
            self.log_message("\n--- Análisis Finalizado ---", is_success=True)


if __name__ == "__main__":
    app = ForensicsApp()
    app.mainloop()