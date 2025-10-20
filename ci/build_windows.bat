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

:: Change to parent directory of script
cd /d "%~dp0\.."

:: Define Python versions
set PY_VERSIONS=3.8 3.9 3.10 3.11 3.12 3.13

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
