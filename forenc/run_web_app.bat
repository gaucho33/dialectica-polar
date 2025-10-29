@echo off
REM Script para ejecutar la aplicacion forense WEB
echo =========================================
echo 🌐 Iniciando Aplicacion Forense Web...
echo =========================================
echo.
echo Funcionalidades:
echo - Algoritmos SIFT + Analisis de Ruido
echo - Interfaz web moderna con Bootstrap
echo - Upload drag & drop de imagenes
echo - Descarga de resultados CSV
echo.
echo 📍 URL: http://localhost:5000
echo.
cd /d "c:\Users\2016a\OneDrive\Escritorio\forenc"
.\.venv311\Scripts\python.exe web_app.py
pause