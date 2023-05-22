package llama

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"log"
	"os"

	"github.com/natefinch/npipe"
	"github.com/tidwall/gjson"
	"github.com/wailovet/easycgo"
)

type LLama struct {
	version string
	state   easycgo.ValueInf
}

func getVersion(filename string) uint32 {
	//open a file
	f, err := os.Open(filename)
	if err != nil {
		return 0
	}
	defer f.Close()

	// take the first byte,type is u32
	var b uint32
	err = binary.Read(f, binary.LittleEndian, &b)
	if err != nil {
		log.Println("binary.Read failed:", err)
		return 0
	}

	var c uint32
	err = binary.Read(f, binary.LittleEndian, &c)
	if err != nil {
		log.Println("binary.Read failed:", err)
		return 0
	}

	return c
}

func New(model string, opts ...ModelOption) (*LLama, error) {
	mo := NewModelOptions(opts...)
	version := getVersion(model)
	versionStr := "1"
	if version > 1 {
		versionStr = "3"
	}
	result := LlaMA_load_model(versionStr, model, mo.ContextSize, mo.Parts, mo.Seed, mo.F16Memory, mo.MLock, mo.NGPULayers)

	if result == nil {
		return nil, fmt.Errorf("failed loading model")
	}

	return &LLama{
		version: versionStr,
		state:   result,
	}, nil
}

func (l *LLama) Free() {
	LlaMA_free_model(l.version, l.state)
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
				conn.Close()
				return
			}
			// nuwa.Helper().WriteFileContent("log.txt", msg+"\r\n")
			content = gjson.Get(msg, "content").String()
			tokensSize := int(gjson.Get(msg, "tokens_size").Int())
			if po.Stream != nil {
				if po.Stream(content) {
					conn.Close()
					return
				}
			}
			if tokensSize >= po.Tokens {
				conn.Close()
				return
			}
		}
	}()

	input := text
	if po.Tokens == 0 {
		po.Tokens = 512
	}

	if po.Temperature <= 0 {
		po.Temperature = 0.8
	}

	params := LlaMA_allocate_params(l.version, input, po.Seed, po.Threads, po.Tokens, po.TopK, po.TopP, po.Temperature, po.Penalty, po.Repeat, po.IgnoreEOS, po.F16KV, po.BatchSize)
	defer LlaMA_free_params(l.version, params)
	ret := LlaMA_predict(l.version, params, l.state)
	if ret != 0 {
		return "", fmt.Errorf("inference failed")
	}

	return content, nil
}
