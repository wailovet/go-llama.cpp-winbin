set INCLUDE=%~dp0..\llama.cpp\;%~dp0..\llama.cpp\examples\;%~dp0;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\include
set LIB=%~dp0;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.4\lib\x64
@REM call "F:\VisualStudio\VC\Auxiliary\Build\vcvarsall.bat" amd64
call "D:\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
 
nvcc -c -o ggml-cuda.obj %~dp0..\llama.cpp\ggml-cuda.cu

cl.exe /EHsc /arch:AVX2 /Ot /Ox /Gs /DGGML_USE_CUBLAS -c %~dp0..\llama.cpp\ggml.c %~dp0..\llama.cpp\llama.cpp %~dp0..\llama.cpp\examples\common.cpp json11.cpp binding.cpp

link -dll -out:..\llama.cpp.cuda.v3.dll cublas.lib cuda.lib cudart.lib cudart_static.lib ggml-cuda.obj ggml.obj llama.obj common.obj json11.obj binding.obj 
 
echo "ok" 
pause