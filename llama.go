package llama

import (
	"bufio"
	"fmt"

	"github.com/natefinch/npipe"
	"github.com/tidwall/gjson"
	"github.com/wailovet/easycgo"
	"github.com/wailovet/nuwa"
)

type LLama struct {
	state easycgo.ValueInf
}

func New(model string, opts ...ModelOption) (*LLama, error) {
	mo := NewModelOptions(opts...)

	result := LlaMA_load_model(model, mo.ContextSize, mo.Parts, mo.Seed, mo.F16Memory, mo.MLock)

	if result == nil {
		return nil, fmt.Errorf("failed loading model")
	}

	return &LLama{state: result}, nil
}

func (l *LLama) Free() {
	LlaMA_free_model(l.state)
}

const pipeName = `\\.\pipe\llama_pipe`

func (l *LLama) Predict(text string, opts ...PredictOption) (string, error) {

	po := NewPredictOptions(opts...)
	ln, err := npipe.Listen(pipeName)
	if err != nil {
		// handle error
		return "", err
	}
	var content = ""
	go func() {
		defer func() {
			recover()
			ln.Close()
		}()

		conn, err := ln.Accept()
		if err != nil {
			// handle error
			return
		}

		for {
			msg, err := bufio.NewReader(conn).ReadString('\n')
			if err != nil {
				// handle error
				continue
			}
			nuwa.Helper().WriteFileContent("log.txt", msg+"\r\n")
			content = gjson.Get(msg, "content").String()
			if po.Stream != nil {
				if po.Stream(content) {
					conn.Close()
					return
				}
			}
		}
	}()

	input := text
	if po.Tokens == 0 {
		po.Tokens = 99999999
	}

	params := LlaMA_allocate_params(input, po.Seed, po.Threads, po.Tokens, po.TopK, po.TopP, po.Temperature, po.Penalty, po.Repeat, po.IgnoreEOS, po.F16KV)
	defer LlaMA_free_params(params)
	ret := LlaMA_predict(params, l.state)
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}

	return content, nil
}
