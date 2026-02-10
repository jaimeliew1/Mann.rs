@echo off
setlocal enabledelayedexpansion

echo Checking for Rust installation...
where rustc >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing Rust toolchain via rustup...
    curl -sSf https://sh.rustup.rs -o rustup-init.exe
    rustup-init.exe -y
    del rustup-init.exe
    call "%USERPROFILE%\.cargo\env.bat"
) else (
    echo Rust is already installed.
)


echo Checking for uv installation...
where uv >nul 2>nul
if %errorlevel% neq 0 (
    echo Installing uv...
    powershell -NoProfile -ExecutionPolicy Bypass ^
        -Command "irm https://astral.sh/uv/install.ps1 | iex"

    :: Ensure uv is on PATH for this session
    set "Path=%USERPROFILE%\.local\bin;%Path%"
) else (
    echo uv is already installed.
)


:: Change to parent directory of script
cd /d "%~dp0\.."

:: Define Python versions
set PY_VERSIONS=3.9 3.10 3.11 3.12 3.13

echo Building wheels for Python versions: %PY_VERSIONS%
for %%V in (%PY_VERSIONS%) do (
    echo Building for python%%V...
    uv build --python python%%V
)

:: Copy source distributions to wheelhouse
echo Copying source distributions...
if not exist wheelhouse mkdir wheelhouse
copy dist\* wheelhouse\

echo Build complete. Files are in .\wheelhouse\

:: ----------------------------------------
:: Test all built wheels with pytest using uv (no gotos)
:: ----------------------------------------
echo.
echo Testing built wheels with pytest using uv...

for %%V in (%PY_VERSIONS%) do (
    set "VER=%%V"
    rem remove the dot from the version to match wheel tags (e.g. 3.10 -> 310)
    set "PYP=cp!VER:.=!"
    set "WHEEL="
    for %%W in (wheelhouse\*!PYP!* ) do set "WHEEL=%%~fW"
    if defined WHEEL (
        echo Testing !WHEEL! with python%%V...
        uv run --python python%%V --with pytest --with "!WHEEL!" pytest tests/ -v
        if errorlevel 1 exit /b 1
        echo Tests passed for python%%V
    ) else (
        echo No wheel found for !PYP!
        exit /b 1
    )
)

echo Build and test complete. Files are in .\wheelhouse\
