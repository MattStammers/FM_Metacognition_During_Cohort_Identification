@echo off
setlocal EnableExtensions

rem ---------------------------------------------------------------------------
rem Run the analytics_code pipeline against the bundled all-models dummy
rem analysis config. The analytics suite is
rem evaluated at three truth-label resolutions, executed sequentially:
rem
rem   * patient-level (default)             -- run-all
rem   * document-level (sibling)             -- run-document-level-all
rem        writes to <output_root>_document_level
rem   * document-level complete-marker arm   -- run-document-level-complete-all
rem        writes to <output_root>_document_level_complete
rem        (sensitivity variant: restricts to rows whose relevant
rem        document markers are all present)
rem   * validation-view endpoint runs        -- run-validation-views-all
rem        writes to <output_root>\{Document,Cumulative,Final,Doc2Patient}
rem
rem Usage:
rem   scripts\run_analytics_pipeline.bat                        (uses bundled dummy config)
rem   scripts\run_analytics_pipeline.bat path\to\analysis.json  (custom config)
rem
rem To avoid the OneDrive / long-path / venv pain entirely, prefer:
rem   docker compose -f scripts\docker-compose.analytics.yml run --rm analytics
rem ---------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "ROOT_DIR=%CD%"
popd >nul

set "USER_CONFIG_PATH=%~1"
if "%USER_CONFIG_PATH%"=="" set "USER_CONFIG_PATH=%ROOT_DIR%\analytics_code\configs\all_models_dummy_analysis.json"

if not exist "%USER_CONFIG_PATH%" (
  echo [analytics] Analytics config not found: %USER_CONFIG_PATH%>&2
  exit /b 1
)

rem ---------------------------------------------------------------------------
rem Map the (potentially long, OneDrive-resident) repo path to a short drive
rem letter so that pipeline outputs do not blow past the 260-character
rem Windows MAX_PATH limit. We try a handful of unused letters.
rem ---------------------------------------------------------------------------
set "SHORT_DRIVE="
set "SHORT_ROOT="
call :acquire_short_root
if not defined SHORT_ROOT (
  echo [analytics] Could not allocate a short drive letter; running on raw path.>&2
  set "SHORT_ROOT=%ROOT_DIR%"
)

set "PYTHON_BIN=%SHORT_ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON_BIN%" (
  echo [analytics] Creating Python venv at %SHORT_ROOT%\.venv
  py -3 -m venv "%SHORT_ROOT%\.venv" 1>nul 2>nul
  if errorlevel 1 (
    python -m venv "%SHORT_ROOT%\.venv" 1>nul 2>nul
  )
)
if not exist "%PYTHON_BIN%" (
  echo [analytics] Failed to create Python venv. Please install Python 3.12+ or run via Docker.>&2
  call :release_short_root
  exit /b 1
)

call :install_deps
if errorlevel 1 (
  call :release_short_root
  exit /b 1
)

rem Rewrite the analysis config so its paths sit under the short drive.
call :materialize_short_config
if errorlevel 1 (
  call :release_short_root
  exit /b 1
)

rem ---------------------------------------------------------------------------
rem Run the analytics CLI. We invoke each command as a separate line (no &
rem chaining inside parens) to keep the cmd.exe parser happy on all locales.
rem ---------------------------------------------------------------------------
pushd "%SHORT_ROOT%\analytics_code" >nul

"%PYTHON_BIN%" -m analytics_code validate-config --config "%EFFECTIVE_CONFIG_PATH%"
if errorlevel 1 goto :run_failed

"%PYTHON_BIN%" -m analytics_code run-all --config "%EFFECTIVE_CONFIG_PATH%"
if errorlevel 1 goto :run_failed

"%PYTHON_BIN%" -m analytics_code run-document-level-all --config "%EFFECTIVE_CONFIG_PATH%"
if errorlevel 1 goto :run_failed

"%PYTHON_BIN%" -m analytics_code run-document-level-complete-all --config "%EFFECTIVE_CONFIG_PATH%"
if errorlevel 1 goto :run_failed

"%PYTHON_BIN%" -m analytics_code run-validation-views-all --config "%EFFECTIVE_CONFIG_PATH%"
if errorlevel 1 goto :run_failed

popd >nul

echo.
echo [analytics] Pipeline completed.
echo [analytics] Patient-level outputs:                    ^<output_root^>
echo [analytics] Document-level outputs:                   ^<output_root^>_document_level
echo [analytics] Document-level (complete-marker) outputs: ^<output_root^>_document_level_complete
echo [analytics] Validation-view outputs:                  ^<output_root^>\{Document,Cumulative,Final,Doc2Patient^}
call :cleanup_temp_config
call :release_short_root
exit /b 0

:run_failed
popd >nul
echo [analytics] Pipeline FAILED. See log output above.>&2
call :cleanup_temp_config
call :release_short_root
exit /b 1


rem ---------------------------------------------------------------------------
rem Helper: install deps without polluting stdout unless something fails.
rem ---------------------------------------------------------------------------
:install_deps
"%PYTHON_BIN%" -m pip install --upgrade --quiet pip
if errorlevel 1 exit /b 1
"%PYTHON_BIN%" -m pip install --quiet -r "%SHORT_ROOT%\analytics_code\requirements.txt"
if errorlevel 1 exit /b 1
"%PYTHON_BIN%" -m pip install --quiet -e "%SHORT_ROOT%\analytics_code"
if errorlevel 1 exit /b 1
exit /b 0

rem ---------------------------------------------------------------------------
rem Helper: pick a free drive letter and substitute the repo onto it.
rem ---------------------------------------------------------------------------
:acquire_short_root
for %%D in (X W V U T S R Q) do (
  if not exist "%%D:\" (
    subst %%D: "%ROOT_DIR%" 1>nul 2>nul
    if exist "%%D:\" (
      set "SHORT_DRIVE=%%D:"
      set "SHORT_ROOT=%%D:"
      exit /b 0
    )
  )
)
exit /b 0

rem ---------------------------------------------------------------------------
rem Helper: write a temporary copy of the analysis config whose paths refer
rem to the short drive. Falls back gracefully if the original is already
rem rooted under the short drive.
rem ---------------------------------------------------------------------------
:materialize_short_config
set "EFFECTIVE_CONFIG_PATH=%USER_CONFIG_PATH%"
if "%SHORT_ROOT%"=="%ROOT_DIR%" exit /b 0
set "TEMP_CONFIG_PATH=%TEMP%\analytics_pipeline_shortpaths.json"
set "MATERIALIZE_SCRIPT=%SCRIPT_DIR%_materialize_short_config.py"
"%PYTHON_BIN%" "%MATERIALIZE_SCRIPT%" "%USER_CONFIG_PATH%" "%ROOT_DIR%" "%SHORT_ROOT%" "%TEMP_CONFIG_PATH%"
if errorlevel 1 exit /b 1
set "EFFECTIVE_CONFIG_PATH=%TEMP_CONFIG_PATH%"
exit /b 0

:cleanup_temp_config
if defined TEMP_CONFIG_PATH if exist "%TEMP_CONFIG_PATH%" del /q "%TEMP_CONFIG_PATH%" 1>nul 2>nul
exit /b 0

:release_short_root
if defined SHORT_DRIVE (
  subst %SHORT_DRIVE% /d 1>nul 2>nul
)
exit /b 0
