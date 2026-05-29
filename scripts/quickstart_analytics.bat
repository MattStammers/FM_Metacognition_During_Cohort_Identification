@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Foolproof analytics quickstart. Tries Docker first (fully isolated,
rem zero local Python setup required). Falls back to the native venv
rem runner if Docker is unavailable.
rem
rem Usage:
rem   scripts\quickstart_analytics.bat
rem   scripts\quickstart_analytics.bat path\to\analysis_config.json
rem ---------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "ROOT_DIR=%CD%"
popd >nul

set "USER_CONFIG=%~1"

where docker 1>nul 2>nul
if not errorlevel 1 (
  docker info 1>nul 2>nul
  if not errorlevel 1 (
    echo [quickstart] Docker detected; running analytics in container.
    if "%USER_CONFIG%"=="" (
      docker compose -f "%SCRIPT_DIR%docker-compose.analytics.yml" run --rm analytics
    ) else (
      docker compose -f "%SCRIPT_DIR%docker-compose.analytics.yml" run --rm ^
        -e ANALYTICS_CONFIG=/workspace/%USER_CONFIG:\=/% analytics
    )
    exit /b %errorlevel%
  )
)

echo [quickstart] Docker not available; falling back to native venv runner.
if "%USER_CONFIG%"=="" (
  call "%SCRIPT_DIR%run_analytics_pipeline.bat"
) else (
  call "%SCRIPT_DIR%run_analytics_pipeline.bat" "%USER_CONFIG%"
)
exit /b %errorlevel%
