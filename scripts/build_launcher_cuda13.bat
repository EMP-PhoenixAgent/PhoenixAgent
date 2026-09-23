@echo off
rem Build PhoenixAgent-V0.8.6b-CUDA13.exe — the encapsulated launcher against
rem the CUDA 13.3 toolkit. Toolkit-switch guard included (cargo does not
rem fingerprint WHICH nvcc built candle-kernels). Output lands in
rem N:\Phoenix Agent\ALPHA\Launchers\PhoenixAgent-V0.8.6b-CUDA13.exe

set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;%PATH%
set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3

call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set CUDA_COMPUTE_CAP=86
set CL=/Zc:preprocessor /std:c++17
rem link.exe resolves cuda.lib/cublas.lib via LIB (vcvars rewrites it, set AFTER the call)
set LIB=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\lib\x64;%LIB%
set PATH=C:\Strawberry\perl\bin;C:\Strawberry\c\bin;%PATH%

cd /d "N:\Phoenix Agent\Phoenix Agent"
rem NB: no quotes in `set` - cmd would keep them in the value.
set CARGO_TARGET_DIR=N:\Phoenix Agent\phoenix-agent\target
rem TOOLKIT SWITCH GUARD: remove candle-kernels/cudarc artifacts directly so
rem the build regenerates the PTX with THIS toolkit.
powershell -NoProfile -Command "Remove-Item -Recurse -Force 'N:\Phoenix Agent\phoenix-agent\target\release\build\candle-kernels-*','N:\Phoenix Agent\phoenix-agent\target\release\.fingerprint\candle-kernels-*','N:\Phoenix Agent\phoenix-agent\target\release\deps\candle_kernels*','N:\Phoenix Agent\phoenix-agent\target\release\deps\libcandle_kernels*','N:\Phoenix Agent\phoenix-agent\target\release\build\cudarc-*','N:\Phoenix Agent\phoenix-agent\target\release\.fingerprint\cudarc-*','N:\Phoenix Agent\phoenix-agent\target\release\deps\cudarc*','N:\Phoenix Agent\phoenix-agent\target\release\deps\libcudarc*' -ErrorAction SilentlyContinue"
cargo tauri build --no-bundle --features ambercore-cuda
if errorlevel 1 exit /b 1
copy /Y "N:\Phoenix Agent\phoenix-agent\target\release\phoenix-agent.exe" "N:\Phoenix Agent\ALPHA\Launchers\PhoenixAgent-V0.8.6b-CUDA13.exe"
echo LAUNCHER READY: N:\Phoenix Agent\ALPHA\Launchers\PhoenixAgent-V0.8.6b-CUDA13.exe
