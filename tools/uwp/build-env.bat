@echo off
REM ============================================================
REM  eden-UWP - ambiente de build UWP (x64, Store CRT)
REM  Uso:
REM    tools\uwp\build-env.bat                  (abre um shell com o ambiente pronto)
REM    tools\uwp\build-env.bat cmake --preset uwp-x64
REM  Rode no cmd ou PowerShell, nao no Git Bash.
REM ============================================================

set "VSROOT=C:\Program Files\Microsoft Visual Studio\2022\Community"
set "PERL_BIN=C:\Strawberry\perl\bin"
set "NASM_BIN=C:\GitHub\tools\nasm"
set "GLSLANG_BIN=C:\VulkanSDK\1.4.341.1\Bin"

REM 1) vswhere precisa estar no PATH, senao o vcvarsall cai em um ambiente desktop sem avisar
set "PATH=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer;%PATH%"

REM 2) ambiente UWP (Store CRT)
call "%VSROOT%\VC\Auxiliary\Build\vcvarsall.bat" x64 uwp
if errorlevel 1 exit /b 1

REM 3) CMake e Ninja do Visual Studio na frente de qualquer outro; depois perl, nasm e glslang.
REM    Nao adicionamos C:\Strawberry\c\bin (traz um CMake antigo).
set "PATH=%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%VSROOT%\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;%PERL_BIN%;%NASM_BIN%;%GLSLANG_BIN%;%PATH%"

REM 4) verificacoes: cada erro de ambiente aqui falha em silencio mais tarde
echo %LIB% | findstr /i "x64\\store" >nul
if errorlevel 1 (
  echo [ERRO] LIB nao aponta para x64\store: o build NAO seria UWP.
  exit /b 1
)
if /i not "%VSCMD_ARG_app_plat%"=="UWP" (
  echo [ERRO] VSCMD_ARG_app_plat nao e UWP.
  exit /b 1
)
echo [OK] Ambiente UWP x64 ativo.
echo --- primeira ocorrencia de cada ferramenta ---
where cmake
where ninja
where perl
where nasm
where glslangValidator

if "%~1"=="" (
  cmd /k
) else (
  %*
)
