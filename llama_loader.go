package llama

import (
	_ "embed"
	"os"
	"path/filepath"

	"github.com/wailovet/easycgo"
	"golang.org/x/sys/cpu"
)

var sharedLibrary *easycgo.EasyCgo

var llaMA_load_model = map[string]*easycgo.EasyCgoProc{}

var llaMA_free_model = map[string]*easycgo.EasyCgoProc{}

var llaMA_allocate_params = map[string]*easycgo.EasyCgoProc{}

var llaMA_predict = map[string]*easycgo.EasyCgoProc{}

var llaMA_free_params = map[string]*easycgo.EasyCgoProc{}

//go:embed llama.cpp.AVX2.dll
var dllFileAVX2 []byte

//go:embed llama.cpp.AVX2.v3.dll
var dllFileAVX2V3 []byte

//go:embed llama.cpp.cuda.dll
var dllFileCuda []byte

//go:embed llama.cpp.cuda.v3.dll
var dllFileCudaV3 []byte

func InstallCuda() {
	tmpDir := os.TempDir()
	os.WriteFile(filepath.Join(tmpDir, "llama.cpp.cuda.dll"), dllFileCuda, 0666)
	LoadDll(filepath.Join(tmpDir, "llama.cpp.cuda.dll"), "1")

	os.WriteFile(filepath.Join(tmpDir, "llama.cpp.cuda.v3.dll"), dllFileCudaV3, 0666)
	LoadDll(filepath.Join(tmpDir, "llama.cpp.cuda.v3.dll"), "3")
}

func Install() {
	tmpDir := os.TempDir()

	os.WriteFile(filepath.Join(tmpDir, "llama.cpp.AVX2.dll"), dllFileAVX2, 0666)
	LoadDll(filepath.Join(tmpDir, "llama.cpp.AVX2.dll"), "1")

	os.WriteFile(filepath.Join(tmpDir, "llama.cpp.AVX2.v3.dll"), dllFileAVX2V3, 0666)
	LoadDll(filepath.Join(tmpDir, "llama.cpp.AVX2.v3.dll"), "3")
}

func isSupportAVX512() bool {
	return cpu.X86.HasAVX512
}

func LoadDll(dllFile string, version string) {
	sharedLibrary = easycgo.MustLoad(dllFile)
	llaMA_load_model[version] = sharedLibrary.MustFind("load_model")
	llaMA_free_model[version] = sharedLibrary.MustFind("llama_free_model")
	llaMA_allocate_params[version] = sharedLibrary.MustFind("llama_allocate_params")
	llaMA_predict[version] = sharedLibrary.MustFind("llama_predict")
	llaMA_free_params[version] = sharedLibrary.MustFind("llama_free_params")
}

func LlaMA_load_model(version string, modelPath string, contextSize int, parts int, seed int, f16Memory bool, mLock bool, nGPULayers int) easycgo.ValueInf {
	f16MemoryInt := 0
	if f16Memory {
		f16MemoryInt = 1
	}

	mLockInt := 0
	if mLock {
		mLockInt = 1
	}

	ret := llaMA_load_model[version].Call(modelPath, contextSize, parts, seed, f16MemoryInt, mLockInt, nGPULayers)

	if ret.IsNil() {
		return nil
	}

	return ret
}

func LlaMA_free_model(version string, p easycgo.ValueInf) {
	llaMA_free_model[version].Call(p.Value().(uintptr))
}

func LlaMA_free_params(version string, p easycgo.ValueInf) {
	llaMA_free_params[version].Call(p.Value().(uintptr))
}

func LlaMA_allocate_params(version string, input string, seed int, threads int, tokens int, topK int, topP float64, temperature float64, penalty float64, repeat int, ignoreEOS bool, f16KV bool, batch int) easycgo.ValueInf {

	ret := llaMA_allocate_params[version].Call(input, seed, threads, tokens, topK,
		float32(topP), float32(temperature), float32(penalty), repeat, ignoreEOS, f16KV, batch)

	if ret.IsNil() {
		return nil
	}

	return ret
}

func LlaMA_predict(version string, p easycgo.ValueInf, model easycgo.ValueInf) int {
	ret := llaMA_predict[version].Call(p, model)
	return ret.ToInt()
}
