@echo off
echo Starting RetailAI...

echo Starting PostgreSQL + pgAdmin...
docker compose up -d
timeout /t 5 /nobreak >nul

echo Starting MLflow...
start cmd /k "python -m mlflow server --port 5000"
timeout /t 3 /nobreak >nul

echo Starting FastAPI...
start cmd /k "uvicorn src.api.main:app --reload"
timeout /t 3 /nobreak >nul

echo Starting Streamlit...
start cmd /k "streamlit run dashboard/app.py"

echo.
echo All services started!
echo   PostgreSQL:  localhost:5432
echo   pgAdmin:     http://localhost:5050
echo   MLflow:      http://localhost:5000
echo   FastAPI:     http://localhost:8000
echo   Streamlit:   http://localhost:8501
echo.
pause
