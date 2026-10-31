package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"fmt"
	"html/template"
	"os"
)

//go:embed report.html
var reportTemplate string

type reportSample struct {
	Temperature float64 `json:"temperature"`
	Sales       float64 `json:"sales"`
}
type snapshot struct {
	Epoch    int     `json:"epoch"`
	Slope    float64 `json:"slope"`
	Bias     float64 `json:"bias"`
	TrainMSE float64 `json:"trainMSE"`
	TestMSE  float64 `json:"testMSE"`
}
type report struct {
	Seed         int64          `json:"seed"`
	Rate         float64        `json:"rate"`
	Noise        float64        `json:"noise"`
	Temperature  float64        `json:"temperature"`
	Train        []reportSample `json:"train"`
	Test         []reportSample `json:"test"`
	History      []snapshot     `json:"history"`
	BaselineRMSE float64        `json:"baselineRMSE"`
}

func reportSamples(data []sample) []reportSample {
	result := make([]reportSample, len(data))
	for i, s := range data {
		result[i] = reportSample{s.temperature, s.sales}
	}
	return result
}

// Goで計算した結果を埋め込み、ネット接続なしで開けるHTMLを保存する。
func writeReport(path string, r report) error {
	data, err := json.Marshal(r)
	if err != nil {
		return fmt.Errorf("結果のJSON変換: %w", err)
	}
	t, err := template.New("report").Parse(reportTemplate)
	if err != nil {
		return err
	}
	var output bytes.Buffer
	// json.MarshalがHTML特殊文字をエスケープしたJSONのみを渡す。
	if err := t.Execute(&output, template.JS(data)); err != nil {
		return err
	}
	return os.WriteFile(path, output.Bytes(), 0644)
}
