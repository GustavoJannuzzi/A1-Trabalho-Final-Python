@echo off
echo ========================================
echo   Iniciando Aplicacao de ML
echo ========================================
echo.

echo [1/2] Iniciando Backend Flask...
start cmd /k "python backend\api.py"

timeout /t 3 /nobreak >nul

echo [2/2] Iniciando Frontend Streamlit...
start cmd /k "streamlit run frontend\app.py"

echo.
echo ========================================
echo   Aplicacao iniciada com sucesso!
echo ========================================
echo.
echo Backend (Flask):    http://localhost:5000
echo Frontend (Streamlit): http://localhost:8501
echo.
echo Pressione qualquer tecla para fechar...
pause >nul
