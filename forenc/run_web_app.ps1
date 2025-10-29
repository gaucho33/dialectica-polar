# Script PowerShell para ejecutar la aplicación forense web
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "🌐 Iniciando Aplicacion Forense Web..." -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Funcionalidades:" -ForegroundColor Yellow
Write-Host "- Algoritmos SIFT + Analisis de Ruido" -ForegroundColor White
Write-Host "- Interfaz web moderna con Bootstrap" -ForegroundColor White
Write-Host "- Upload drag & drop de imagenes" -ForegroundColor White
Write-Host "- Descarga de resultados CSV" -ForegroundColor White
Write-Host ""
Write-Host "📍 URL: http://localhost:5000" -ForegroundColor Magenta
Write-Host ""

Set-Location "c:\Users\2016a\OneDrive\Escritorio\forenc"
& .\.venv311\Scripts\python.exe web_app.py