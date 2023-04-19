package llama

import (
	"math"
	"runtime"
)

type ModelOptions struct {
	ContextSize int
	Parts       int
	Seed        int
	F16Memory   bool
	MLock       bool
	Embedding   bool
}

type PredictOptions struct {
	Seed, Threads, Tokens, TopK, Repeat int
	TopP, Temperature, Penalty          float64
	F16KV                               bool
	IgnoreEOS                           bool
	Stream                              func(outputText string) (stop bool)
}

type PredictOption func(p *PredictOptions)
type ModelOption func(p *ModelOptions)

var DefaultModelOptions ModelOptions = ModelOptions{
	ContextSize: 512,
	Seed:        0,
	F16Memory:   false,
	MLock:       false,
	Embedding:   false,
}

var defaultThreadNum = int(math.Min(1, float64(runtime.NumCPU()/2)))

var DefaultOptions PredictOptions = PredictOptions{
	Seed:        -1,
	Threads:     defaultThreadNum,
	Tokens:      128,
	TopK:        10000,
	TopP:        0.7,
	Temperature: 0.7,
	Penalty:     1,
	Repeat:      64 * 4,
}

// SetContext sets the context size.
func SetContext(c int) ModelOption {
	return func(p *ModelOptions) {
		p.ContextSize = c
	}
}

func SetModelSeed(c int) ModelOption {
	return func(p *ModelOptions) {
		p.Seed = c
	}
}

func SetParts(c int) ModelOption {
	return func(p *ModelOptions) {
		p.Parts = c
	}
}

var EnableF16Memory ModelOption = func(p *ModelOptions) {
	p.F16Memory = true
}

var EnableF16KV PredictOption = func(p *PredictOptions) {
	p.F16KV = true
}

var EnableMLock ModelOption = func(p *ModelOptions) {
	p.MLock = true
}

var EnableEmbedding ModelOption = func(p *ModelOptions) {
	p.Embedding = true
}

// Create a new PredictOptions object with the given options.
func NewModelOptions(opts ...ModelOption) ModelOptions {
	p := DefaultModelOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}

var IgnoreEOS PredictOption = func(p *PredictOptions) {
	p.IgnoreEOS = true
}

// SetSeed sets the random seed for sampling text generation.
func SetSeed(seed int) PredictOption {
	return func(p *PredictOptions) {
		p.Seed = seed
	}
}

// SetStreamFn sets the stream callback for text generation.
func SetStreamFn(stream func(outputText string) (stop bool)) PredictOption {
	return func(p *PredictOptions) {
		p.Stream = stream
	}
}

// SetThreads sets the number of threads to use for text generation.
func SetThreads(threads int) PredictOption {
	return func(p *PredictOptions) {
		p.Threads = threads
	}
}

// SetTokens sets the number of tokens to generate.
func SetTokens(tokens int) PredictOption {
	return func(p *PredictOptions) {
		p.Tokens = tokens
	}
}

// SetTopK sets the value for top-K sampling.
func SetTopK(topk int) PredictOption {
	return func(p *PredictOptions) {
		p.TopK = topk
	}
}

// SetTopP sets the value for nucleus sampling.
func SetTopP(topp float64) PredictOption {
	return func(p *PredictOptions) {
		p.TopP = topp
	}
}

// SetTemperature sets the temperature value for text generation.
func SetTemperature(temp float64) PredictOption {
	return func(p *PredictOptions) {
		p.Temperature = temp
	}
}

// SetPenalty sets the repetition penalty for text generation.
func SetPenalty(penalty float64) PredictOption {
	return func(p *PredictOptions) {
		p.Penalty = penalty
	}
}

// SetRepeat sets the number of times to repeat text generation.
func SetRepeat(repeat int) PredictOption {
	return func(p *PredictOptions) {
		p.Repeat = repeat
	}
}

// Create a new PredictOptions object with the given options.
func NewPredictOptions(opts ...PredictOption) PredictOptions {
	p := DefaultOptions
	for _, opt := range opts {
		opt(&p)
	}
	return p
}
