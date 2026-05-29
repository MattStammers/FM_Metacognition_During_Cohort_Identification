@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem Phased all-models dummy chronology matrix runner (Windows companion to .sh)
rem ---------------------------------------------------------------------------

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%.." >nul
set "ROOT_DIR=%CD%"
popd >nul

set "PYTHON_BIN=%ROOT_DIR%\.venv\Scripts\python.exe"
set "CLIENT_DIR=%ROOT_DIR%\python_client"
set "COMPOSE_DIR=%ROOT_DIR%\gradio_api"

if not exist "%ROOT_DIR%\.env" (
  echo Missing %ROOT_DIR%\.env with HF_ACCESS_TOKEN set.>&2
  exit /b 1
)

call :ensure_env || exit /b 1

call :run_phase "mixtral"    "mixtral"    "%CLIENT_DIR%\configs\all_models_dummy_mixtral.json"    "http://127.0.0.1:9001" || exit /b 1
call :run_phase "mixtral_t1" "mixtral_t1" "%CLIENT_DIR%\configs\all_models_dummy_mixtral_t1.json" "http://127.0.0.1:9011" || exit /b 1
call :run_phase "m42"        "m42"        "%CLIENT_DIR%\configs\all_models_dummy_m42.json"        "http://127.0.0.1:9002" || exit /b 1
call :run_phase "m42_t1"     "m42_t1"     "%CLIENT_DIR%\configs\all_models_dummy_m42_t1.json"     "http://127.0.0.1:9012" || exit /b 1
call :run_phase "deepseek14"     "deepseek14"     "%CLIENT_DIR%\configs\all_models_dummy_deepseek14.json"      "http://127.0.0.1:9003" || exit /b 1
call :run_phase "deepseek14_t05" "deepseek14_t05" "%CLIENT_DIR%\configs\all_models_dummy_deepseek14_t050.json" "http://127.0.0.1:9013" || exit /b 1
call :run_phase "deepseek32"     "deepseek32"     "%CLIENT_DIR%\configs\all_models_dummy_deepseek32.json"      "http://127.0.0.1:9004" || exit /b 1
call :run_phase "qwen32"         "qwen32"         "%CLIENT_DIR%\configs\all_models_dummy_qwen32.json"          "http://127.0.0.1:9005" || exit /b 1
call :run_phase "gemma4_31b"     "gemma4_31b"     "%CLIENT_DIR%\configs\all_models_dummy_gemma4_31b.json"      "http://127.0.0.1:9006" || exit /b 1

echo Completed multi-phase all-model dummy matrix run.
echo Outputs: python_client\outputs\chronology_runs\all_models_realistic_dummy_v1
exit /b 0

rem ---------------------------------------------------------------------------
:ensure_env
pushd "%ROOT_DIR%" >nul
if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv || (popd & exit /b 1)
)
call ".venv\Scripts\activate.bat" || (popd & exit /b 1)
python -m pip install --upgrade pip || (popd & exit /b 1)
python -m pip install -r python_client\requirements.txt || (popd & exit /b 1)
python -m pip install -e python_client || (popd & exit /b 1)
popd >nul
exit /b 0

rem ---------------------------------------------------------------------------
:run_phase
set "PHASE=%~1"
set "PROFILE=%~2"
set "CONFIG_PATH=%~3"
set "READY_URL=%~4"

echo ==^> Phase: %PHASE%

pushd "%COMPOSE_DIR%" >nul
docker compose --env-file "%ROOT_DIR%\.env" --profile "%PROFILE%" up --build -d || (popd & exit /b 1)
popd >nul

echo Waiting for %READY_URL%
call :wait_for_endpoint "%READY_URL%" || exit /b 1

pushd "%CLIENT_DIR%" >nul
"%PYTHON_BIN%" -m python_client validate-config --config "%CONFIG_PATH%" || (popd & exit /b 1)
"%PYTHON_BIN%" -m python_client run --config "%CONFIG_PATH%" || (popd & exit /b 1)
popd >nul

pushd "%COMPOSE_DIR%" >nul
docker compose --env-file "%ROOT_DIR%\.env" --profile "%PROFILE%" down
popd >nul
exit /b 0

rem ---------------------------------------------------------------------------
:wait_for_endpoint
set "URL=%~1"
if "%RUN_READY_TIMEOUT_SECONDS%"=="" set "RUN_READY_TIMEOUT_SECONDS=7200"
set /a ATTEMPTS=%RUN_READY_TIMEOUT_SECONDS%/5
for /L %%I in (1,1,%ATTEMPTS%) do (
  "%PYTHON_BIN%" -c "from gradio_client import Client; c=Client('%URL%'); c.close()" >nul 2>&1 && exit /b 0
  timeout /t 5 /nobreak >nul
)
echo Service at %URL% did not become ready within %RUN_READY_TIMEOUT_SECONDS%s.>&2
exit /b 1
