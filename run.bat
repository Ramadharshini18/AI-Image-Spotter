@echo off
echo =========================================================
echo       AI IMAGE SPOTTER: DECOUPLED WEB APPLICATION       
echo =========================================================
echo.
echo [1/2] Starting FastAPI Backend Server (Port 8000)...
start /b python server.py

echo [2/2] Starting React + Vite Frontend Server (Port 5173)...
start /b cmd /c "npm run dev --prefix frontend"

echo.
echo Waiting 5 seconds for servers to initialize...
timeout /t 5 >nul

echo.
echo [OK] Launching browser to http://localhost:5173 ...
start http://localhost:5173

echo.
echo Servers are running in the background. 
echo Press Ctrl+C in this terminal window to stop both servers.
echo.
pause
