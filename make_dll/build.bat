set INCLUDE=%~dp0..\llama.cpp\;%~dp0..\llama.cpp\examples\;%~dp0 
@REM call "F:\VisualStudio\VC\Auxiliary\Build\vcvarsall.bat" amd64
call "D:\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvarsall.bat" amd64
 

cl.exe /EHsc /arch:AVX2 /Ot /Ox /Gs -c %~dp0..\llama.cpp\ggml.c %~dp0..\llama.cpp\llama.cpp %~dp0..\llama.cpp\examples\common.cpp json11.cpp binding.cpp
link -dll -out:..\llama.cpp.AVX2.dll ggml.obj llama.obj common.obj json11.obj binding.obj 

cl.exe /EHsc /arch:AVX512 /Ot /Ox /Gs -c %~dp0..\llama.cpp\ggml.c %~dp0..\llama.cpp\llama.cpp %~dp0..\llama.cpp\examples\common.cpp json11.cpp binding.cpp
link -dll -out:..\llama.cpp.AVX512.dll ggml.obj llama.obj common.obj json11.obj binding.obj 
   
echo "ok" 
pause