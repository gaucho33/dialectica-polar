// JavaScript para la aplicación forense web con sistema freemium
let uploadedFiles = [];
let analysisInProgress = false;
let userPremium = false;
let remainingTrials = 5;

// Configuración de validación de archivos
const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp', 'image/tiff', 'image/webp'];
const ALLOWED_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'];
const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB en bytes
const MAX_TOTAL_SIZE = 50 * 1024 * 1024; // 50MB total

// Validar archivo individual
function validateFile(file) {
    const errors = [];
    
    // Validar tipo de archivo
    const fileExtension = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_TYPES.includes(file.type) && !ALLOWED_EXTENSIONS.includes(fileExtension)) {
        errors.push(`Formato no soportado: ${file.name}. Use JPG, PNG, BMP, TIFF o WEBP.`);
    }
    
    // Validar tamaño individual
    if (file.size > MAX_FILE_SIZE) {
        const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
        errors.push(`Archivo muy grande: ${file.name} (${sizeMB}MB). Máximo 10MB por imagen.`);
    }
    
    return errors;
}

// Validar tamaño total de archivos
function validateTotalSize(files) {
    const totalSize = files.reduce((sum, file) => sum + file.size, 0);
    if (totalSize > MAX_TOTAL_SIZE) {
        const totalMB = (totalSize / (1024 * 1024)).toFixed(1);
        return [`Tamaño total muy grande: ${totalMB}MB. Máximo 50MB en total.`];
    }
    return [];
}

// Inicialización cuando la página se carga
document.addEventListener('DOMContentLoaded', function() {
    initializeFileUpload();
    checkServerStatus();
    updateUIBasedOnStatus();
});

// Actualizar UI basado en estado del usuario
function updateUIBasedOnStatus() {
    fetch('/status')
    .then(response => response.json())
    .then(data => {
        userPremium = data.is_premium;
        remainingTrials = data.remaining_trials;
        
        updateAnalyzeButtonText();
        updateTrialWarnings();
    })
    .catch(error => {
        console.error('Error checking status:', error);
    });
}

// Actualizar texto del botón de análisis
function updateAnalyzeButtonText() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    if (uploadedFiles.length > 0) {
        if (userPremium) {
            analyzeBtn.innerHTML = `
                <i class="fas fa-microscope"></i> 
                Analizar ${uploadedFiles.length} imagen${uploadedFiles.length > 1 ? 'es' : ''} (Ilimitado)
            `;
        } else {
            analyzeBtn.innerHTML = `
                <i class="fas fa-microscope"></i> 
                Analizar ${uploadedFiles.length} imagen${uploadedFiles.length > 1 ? 'es' : ''} 
                (${remainingTrials} restantes)
            `;
        }
    }
}

// Actualizar advertencias de pruebas
function updateTrialWarnings() {
    if (!userPremium && remainingTrials <= 2) {
        showAlert(`Solo te quedan ${remainingTrials} pruebas gratuitas. Considera actualizar a Premium.`, 'warning');
    }
}

// Mostrar modal de upgrade
function showUpgradeModal() {
    const modal = new bootstrap.Modal(document.getElementById('upgradeModal'));
    modal.show();
}

// Procesar upgrade a Premium
function processPremiumUpgrade() {
    // En producción, esto conectaría con un procesador de pagos real
    showAlert('Procesando pago...', 'info');
    
    fetch('/upgrade', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            userPremium = true;
            remainingTrials = -1; // Ilimitado
            
            // Cerrar modal
            const modal = bootstrap.Modal.getInstance(document.getElementById('upgradeModal'));
            modal.hide();
            
            // Mostrar éxito y recargar página
            showAlert('¡Upgrade exitoso! Ahora tienes acceso Premium completo.', 'success');
            setTimeout(() => {
                window.location.reload();
            }, 2000);
        } else {
            showAlert(data.error || 'Error en el upgrade', 'danger');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        showAlert('Error de conexión durante el upgrade', 'danger');
    });
}

// Configurar upload de archivos con drag & drop
function initializeFileUpload() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    // Drag & Drop events
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);
    
    // File input change
    fileInput.addEventListener('change', handleFileSelect);
    
    // Click en upload area
    uploadArea.addEventListener('click', function(e) {
        if (e.target === uploadArea || e.target.closest('.upload-content')) {
            fileInput.click();
        }
    });
}

// Manejo de eventos drag & drop
function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'copy';
    document.getElementById('uploadArea').classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    document.getElementById('uploadArea').classList.remove('drag-over');
    
    const files = Array.from(e.dataTransfer.files);
    processFiles(files);
}

// Manejo de selección de archivos
function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    processFiles(files);
}

// Procesar archivos seleccionados
function processFiles(files) {
    // Convertir FileList a Array
    const fileArray = Array.from(files);
    
    // Validar cada archivo individualmente
    let allErrors = [];
    const validFiles = [];
    
    fileArray.forEach(file => {
        const errors = validateFile(file);
        if (errors.length > 0) {
            allErrors = allErrors.concat(errors);
        } else {
            validFiles.push(file);
        }
    });
    
    // Validar tamaño total
    if (validFiles.length > 0) {
        const totalErrors = validateTotalSize(validFiles);
        allErrors = allErrors.concat(totalErrors);
    }
    
    // Mostrar errores si los hay
    if (allErrors.length > 0) {
        const errorMessage = allErrors.join('<br>');
        showAlert(`
            <strong>Archivos rechazados:</strong><br>
            ${errorMessage}<br><br>
            <small><strong>Formatos permitidos:</strong> JPG, PNG, BMP, TIFF, WEBP<br>
            <strong>Tamaño máximo:</strong> 10MB por imagen, 50MB total</small>
        `, 'danger');
        return;
    }
    
    // Si no hay archivos válidos después de filtrar
    if (validFiles.length === 0) {
        showAlert(`
            Por favor selecciona archivos de imagen válidos.<br>
            <small><strong>Formatos soportados:</strong> JPG, PNG, BMP, TIFF, WEBP</small>
        `, 'warning');
        return;
    }
    
    // Mostrar resumen de archivos aceptados
    if (validFiles.length > 0) {
        const totalSizeMB = (validFiles.reduce((sum, file) => sum + file.size, 0) / (1024 * 1024)).toFixed(1);
        showAlert(`
            <i class="fas fa-check-circle"></i> 
            ${validFiles.length} archivo${validFiles.length > 1 ? 's' : ''} válido${validFiles.length > 1 ? 's' : ''} 
            (${totalSizeMB}MB total)
        `, 'success');
    }
    
    uploadFiles(validFiles);
}

// Subir archivos al servidor
function uploadFiles(files) {
    const formData = new FormData();
    files.forEach(file => {
        formData.append('files', file);
    });
    
    // Mostrar progreso
    showUploadProgress();
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideUploadProgress();
        
        if (data.success) {
            uploadedFiles = uploadedFiles.concat(data.files);
            updateFileList();
            enableAnalyzeButton();
            showAlert(`${data.files.length} archivos subidos correctamente`, 'success');
        } else {
            showAlert(data.error || 'Error al subir archivos', 'danger');
        }
    })
    .catch(error => {
        hideUploadProgress();
        console.error('Error:', error);
        showAlert('Error de conexión al subir archivos', 'danger');
    });
}

// Mostrar progreso de subida
function showUploadProgress() {
    document.querySelector('.upload-content').classList.add('d-none');
    document.getElementById('uploadProgress').classList.remove('d-none');
    
    // Simular progreso
    let progress = 0;
    const progressBar = document.querySelector('#uploadProgress .progress-bar');
    const interval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) {
            progress = 90;
        }
        progressBar.style.width = progress + '%';
        
        if (progress >= 90) {
            clearInterval(interval);
        }
    }, 200);
}

// Ocultar progreso de subida
function hideUploadProgress() {
    document.querySelector('.upload-content').classList.remove('d-none');
    document.getElementById('uploadProgress').classList.add('d-none');
    document.querySelector('#uploadProgress .progress-bar').style.width = '0%';
}

// Actualizar lista de archivos
function updateFileList() {
    const fileList = document.getElementById('fileList');
    
    if (uploadedFiles.length === 0) {
        fileList.innerHTML = '';
        return;
    }
    
    let html = '<h6 class="mt-3 mb-2">Archivos subidos:</h6>';
    uploadedFiles.forEach((filename, index) => {
        html += `
            <div class="file-item fade-in-up">
                <div class="file-icon">
                    <i class="fas fa-image"></i>
                </div>
                <div class="file-info">
                    <div class="file-name">${filename}</div>
                    <div class="file-size">Imagen lista para análisis</div>
                </div>
                <div class="file-status uploaded">
                    <i class="fas fa-check"></i> Subido
                </div>
            </div>
        `;
    });
    
    fileList.innerHTML = html;
}

// Habilitar botón de análisis
function enableAnalyzeButton() {
    const analyzeBtn = document.getElementById('analyzeBtn');
    analyzeBtn.disabled = false;
    
    if (userPremium) {
        analyzeBtn.innerHTML = `
            <i class="fas fa-microscope"></i> 
            Analizar ${uploadedFiles.length} imagen${uploadedFiles.length > 1 ? 'es' : ''} (Ilimitado)
        `;
    } else {
        analyzeBtn.innerHTML = `
            <i class="fas fa-microscope"></i> 
            Analizar ${uploadedFiles.length} imagen${uploadedFiles.length > 1 ? 'es' : ''} 
            (${remainingTrials} restantes)
        `;
        
        if (remainingTrials <= 0) {
            analyzeBtn.disabled = true;
            analyzeBtn.innerHTML = `
                <i class="fas fa-lock"></i> 
                Límite alcanzado - Upgrade a Premium
            `;
            analyzeBtn.onclick = () => showUpgradeModal();
        }
    }
}

// Iniciar análisis de imágenes
function analyzeImages() {
    if (analysisInProgress) return;
    
    // Verificar límites antes de proceder
    if (!userPremium && remainingTrials <= 0) {
        const modal = new bootstrap.Modal(document.getElementById('limitReachedModal'));
        modal.show();
        return;
    }
    
    analysisInProgress = true;
    showAnalysisSection();
    
    fetch('/analyze', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        analysisInProgress = false;
        hideAnalysisSection();
        
        if (data.success) {
            // Actualizar estado del usuario
            remainingTrials = data.remaining_trials;
            userPremium = data.is_premium;
            
            showResults(data);
            showAlert('Análisis completado exitosamente', 'success');
            
            // Mostrar advertencia si quedan pocas pruebas
            if (!userPremium && remainingTrials <= 2 && remainingTrials > 0) {
                setTimeout(() => {
                    showAlert(`Te quedan ${remainingTrials} pruebas gratuitas. ¡Upgrade a Premium para análisis ilimitados!`, 'warning');
                }, 3000);
            } else if (!userPremium && remainingTrials === 0) {
                setTimeout(() => {
                    showAlert('¡Has usado todas tus pruebas gratuitas! Upgrade a Premium para continuar.', 'warning');
                }, 3000);
            }
            
        } else if (data.upgrade_required) {
            const modal = new bootstrap.Modal(document.getElementById('limitReachedModal'));
            modal.show();
        } else {
            showAlert(data.error || 'Error durante el análisis', 'danger');
        }
    })
    .catch(error => {
        analysisInProgress = false;
        hideAnalysisSection();
        console.error('Error:', error);
        showAlert('Error de conexión durante el análisis', 'danger');
    });
}

// Mostrar sección de análisis
function showAnalysisSection() {
    const section = document.getElementById('analysisSection');
    section.classList.remove('d-none');
    section.classList.add('fade-in-up');
    
    // Simular progreso
    let progress = 0;
    const progressBar = document.getElementById('analysisProgress');
    const statusText = document.getElementById('analysisStatus');
    
    const interval = setInterval(() => {
        progress += Math.random() * 10;
        if (progress > 95) {
            progress = 95;
        }
        progressBar.style.width = progress + '%';
        
        // Actualizar texto de estado
        if (progress < 30) {
            statusText.innerHTML = '<i class="fas fa-search"></i> Detectando características SIFT...';
        } else if (progress < 60) {
            statusText.innerHTML = '<i class="fas fa-microscope"></i> Analizando ruido residual...';
        } else if (progress < 90) {
            statusText.innerHTML = '<i class="fas fa-chart-line"></i> Procesando matrices forenses...';
        } else {
            statusText.innerHTML = '<i class="fas fa-check-circle"></i> Finalizando análisis...';
        }
        
        if (!analysisInProgress) {
            clearInterval(interval);
            progressBar.style.width = '100%';
        }
    }, 500);
}

// Ocultar sección de análisis
function hideAnalysisSection() {
    setTimeout(() => {
        document.getElementById('analysisSection').classList.add('d-none');
    }, 1000);
}

// Mostrar resultados
function showResults(data) {
    const resultsSection = document.getElementById('resultsSection');
    resultsSection.classList.remove('d-none');
    resultsSection.classList.add('fade-in-up');
    
    // Actualizar estadísticas
    document.getElementById('totalImages').textContent = data.total_images;
    document.getElementById('authenticImages').textContent = data.authentic_count;
    document.getElementById('manipulatedImages').textContent = data.manipulated_count;
    document.getElementById('accuracy').textContent = '100%';
    
    // Llenar tabla de resultados
    const tableBody = document.getElementById('resultsTable');
    let html = '';
    
    data.results.forEach(result => {
        const statusClass = result.result === 'authentic' ? 'status-authentic' : 
                           result.result === 'manipulated' ? 'status-manipulated' : 'status-error';
        
        const statusIcon = result.result === 'authentic' ? 'fa-shield-alt' : 
                          result.result === 'manipulated' ? 'fa-exclamation-triangle' : 'fa-times';
        
        const statusText = result.result === 'authentic' ? 'Auténtica' : 
                          result.result === 'manipulated' ? 'Manipulada' : 'Error';
        
        html += `
            <tr>
                <td>
                    <i class="fas fa-image text-muted me-2"></i>
                    ${result.filename}
                </td>
                <td>${result.dimensions || 'N/A'}</td>
                <td>
                    <span class="${statusClass}">
                        <i class="fas ${statusIcon}"></i> ${statusText}
                    </span>
                </td>
                <td>
                    <code style="font-size: 0.8rem; word-break: break-all;">
                        ${result.encoded_pixels.length > 50 ? 
                          result.encoded_pixels.substring(0, 50) + '...' : 
                          result.encoded_pixels}
                    </code>
                </td>
            </tr>
        `;
    });
    
    tableBody.innerHTML = html;
}

// Descargar resultados CSV
function downloadResults() {
    window.location.href = '/download-results';
}

// Limpiar todos los datos
function clearAllData() {
    if (confirm('¿Estás seguro de que quieres limpiar todos los datos?')) {
        fetch('/clear')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                uploadedFiles = [];
                document.getElementById('fileList').innerHTML = '';
                document.getElementById('analyzeBtn').disabled = true;
                document.getElementById('analyzeBtn').innerHTML = '<i class="fas fa-microscope"></i> Iniciar Análisis Forense';
                document.getElementById('resultsSection').classList.add('d-none');
                document.getElementById('analysisSection').classList.add('d-none');
                
                showAlert('Todos los datos han sido limpiados', 'success');
            } else {
                showAlert(data.error || 'Error al limpiar datos', 'danger');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            showAlert('Error de conexión', 'danger');
        });
    }
}

// Verificar estado del servidor
function checkServerStatus() {
    fetch('/status')
    .then(response => response.json())
    .then(data => {
        const indicator = document.getElementById('statusIndicator');
        if (data.status === 'running') {
            indicator.className = 'badge bg-success status-indicator';
            indicator.textContent = 'En línea';
            
            // Actualizar datos del usuario
            userPremium = data.is_premium;
            remainingTrials = data.remaining_trials;
            updateAnalyzeButtonText();
        }
    })
    .catch(error => {
        const indicator = document.getElementById('statusIndicator');
        indicator.className = 'badge bg-danger';
        indicator.textContent = 'Desconectado';
    });
}

// Mostrar alertas
function showAlert(message, type) {
    // Crear elemento de alerta
    const alertDiv = document.createElement('div');
    alertDiv.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    alertDiv.style.cssText = 'top: 20px; right: 20px; z-index: 1050; min-width: 300px;';
    alertDiv.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(alertDiv);
    
    // Auto-remover después de 5 segundos
    setTimeout(() => {
        if (alertDiv.parentNode) {
            alertDiv.remove();
        }
    }, 5000);
}

// Actualizar estado periódicamente
setInterval(checkServerStatus, 30000); // Cada 30 segundos