package llama

import (
	_ "embed"
	"os"
	"path/filepath"

	"github.com/wailovet/easycgo"
	"golang.org/x/sys/cpu"
)

var sharedLibrary *easycgo.EasyCgo
var llaMA_load_model *easycgo.EasyCgoProc
var llaMA_free_model *easycgo.EasyCgoProc
var llaMA_allocate_params *easycgo.EasyCgoProc
var llaMA_predict *easycgo.EasyCgoProc
var llaMA_free_params *easycgo.EasyCgoProc
var new_chars *easycgo.EasyCgoProc

//go:embed llama.cpp.AVX2.dll
var dllFileAVX2 []byte

//go:embed llama.cpp.AVX512.dll
var dllFileAVX512 []byte

func Install() {
	tmpDir := os.TempDir()
	//判断是否支持AVX512
	if isSupportAVX512() {
		os.WriteFile(filepath.Join(tmpDir, "llama.cpp.AVX512.dll"), dllFileAVX512, 0666)
		LoadDll(filepath.Join(tmpDir, "llama.cpp.AVX512.dll"))
		return
	}

	os.WriteFile(filepath.Join(tmpDir, "llama.cpp.AVX2.dll"), dllFileAVX2, 0666)
	LoadDll(filepath.Join(tmpDir, "llama.cpp.AVX2.dll"))
}

func isSupportAVX512() bool {
	return cpu.X86.HasAVX512
}

func LoadDll(dllFile string) {
	sharedLibrary = easycgo.MustLoad(dllFile)
	llaMA_load_model = sharedLibrary.MustFind("load_model")
	llaMA_free_model = sharedLibrary.MustFind("llama_free_model")
	llaMA_allocate_params = sharedLibrary.MustFind("llama_allocate_params")
	llaMA_predict = sharedLibrary.MustFind("llama_predict")
	llaMA_free_params = sharedLibrary.MustFind("llama_free_params")
}

func LlaMA_load_model(modelPath string, contextSize int, parts int, seed int, f16Memory bool, mLock bool) easycgo.ValueInf {
	f16MemoryInt := 0
	if f16Memory {
		f16MemoryInt = 1
	}

	mLockInt := 0
	if mLock {
		mLockInt = 1
	}

	ret := llaMA_load_model.Call(modelPath, contextSize, parts, seed, f16MemoryInt, mLockInt)

	ptr, ok := ret.Value().(uintptr)
	if !ok || ptr == 0 {
		return nil
	}

	return ret
}

func LlaMA_free_model(p easycgo.ValueInf) {
	llaMA_free_model.Call(p)
}

func LlaMA_free_params(p easycgo.ValueInf) {
	llaMA_free_params.Call(p)
}

func LlaMA_allocate_params(input string, seed int, threads int, tokens int, topK int, topP float64, temperature float64, penalty float64, repeat int, ignoreEOS bool, f16KV bool) easycgo.ValueInf {
	ignoreEOSInt := 0
	if ignoreEOS {
		ignoreEOSInt = 1
	}

	f16KVInt := 0
	if f16KV {
		f16KVInt = 1
	}

	ret := llaMA_allocate_params.Call(input, seed, threads, tokens, topK, topP, temperature, penalty, repeat, ignoreEOSInt, f16KVInt)

	ptr, ok := ret.Value().(uintptr)
	if !ok || ptr == 0 {
		return nil
	}

	return ret
}

func LlaMA_predict(p easycgo.ValueInf, model easycgo.ValueInf) int {
	ret := llaMA_predict.Call(p.Value().(uintptr), model.Value().(uintptr))
	return ret.ToInt()
}
