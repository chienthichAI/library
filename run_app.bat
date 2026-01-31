@echo off
echo ==================================================
echo   KHOI DONG HE THONG SMARTLIB KIOSK
echo ==================================================

echo.
echo [1/2] Dang khoi dong Backend API...
start "SmartLib Backend" cmd /c "cd backend && run.bat"

echo.
echo [2/2] Dang khoi dong Frontend...
cd frontend
if not exist "node_modules" (
    echo   - Phat hien lan chay dau tien. Dang cai dat thu vien npm install...
    echo   - Viec nay co the mat 1-2 phut...
    call npm install
)

echo   - Dang start server...
npm run dev
