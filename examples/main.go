package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	llama "github.com/wailovet/go-llama.cpp-winbin"
)

var (
	threads = 4
	tokens  = 128
)

func jsonEncode(i interface{}) string {
	jsonBytes, err := json.Marshal(i)
	if err != nil {
		return ""
	}
	return string(jsonBytes)
}

func main() {
	llama.LoadDll("llama.cpp.cuda.dll")
	var model string

	flags := flag.NewFlagSet(os.Args[0], flag.ExitOnError)
	flags.StringVar(&model, "m", "./ggml-alpaca-7b-native-q4.bin", "path to q4_0.bin model file to load")
	flags.IntVar(&threads, "t", 8, "number of threads to use during computation")
	flags.IntVar(&tokens, "n", 256, "number of tokens to predict")

	err := flags.Parse(os.Args[1:])
	if err != nil {
		fmt.Printf("Parsing program arguments failed: %s", err)
		os.Exit(1)
	}
	l, err := llama.New(model, llama.EnableEmbedding, llama.SetContext(2048), llama.SetParts(-1))
	if err != nil {
		fmt.Println("Loading the model failed:", err.Error())
		os.Exit(1)
	}
	// embdata := l.Embedding("heyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyheyhey")
	// fmt.Println(jsonEncode(embdata))
	// return
	fmt.Printf("Model loaded successfully.\n")
	reader := bufio.NewReader(os.Stdin)

	for {
		text := readMultiLineInput(reader)
		fmt.Print("### Assistant:")
		sendText := fmt.Sprintf("你是一个AI助手,你需要回答用户的问题\n\n### Human: %s\n### Assistant: ", text)

		data, err := l.Predict(sendText, llama.SetTokens(tokens), llama.SetThreads(threads), llama.SetTopK(90), llama.SetTopP(0.86), llama.SetStreamFn(func(s string) (stop bool) {
			if strings.HasSuffix(s, "##") {
				return true
			}
			return false
		}))
		if err != nil {
			panic(err)
		}
		fmt.Print(data)
		fmt.Printf("\n\n")
	}
}

// readMultiLineInput reads input until an empty line is entered.
func readMultiLineInput(reader *bufio.Reader) string {
	var lines []string
	fmt.Print("### Human:")

	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			if err == io.EOF {
				os.Exit(0)
			}
			fmt.Printf("Reading the prompt failed: %s", err)
			os.Exit(1)
		}

		if len(strings.TrimSpace(line)) == 0 {
			break
		}

		lines = append(lines, line)
	}

	text := strings.Join(lines, "")
	// fmt.Println("Sending", text)
	return text
}
