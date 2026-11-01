package main

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestReportEmbedsActualResults(t *testing.T) {
	train, test := dataset(42, 15)
	m := step(model{}, train, 0.1)
	want := report{Seed: 42, Rate: 0.1, Noise: 15, Temperature: 25, Train: reportSamples(train), Test: reportSamples(test), History: []snapshot{{1, m.weight / 10, m.bias, mse(m, train), mse(m, test)}}}
	path := filepath.Join(t.TempDir(), "report-linear-regression.html")
	if err := writeReport(path, want); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	_, rest, ok := strings.Cut(string(data), `<script id="report-data" type="application/json">`)
	if !ok {
		t.Fatal("埋め込みデータが見つからない")
	}
	payload, _, ok := strings.Cut(rest, "</script>")
	if !ok {
		t.Fatal("scriptの終端が見つからない")
	}
	var got report
	if err := json.Unmarshal([]byte(payload), &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatal("HTMLの値が実行結果と一致しない")
	}
}

func TestReportRejectsNonFiniteBeforeWriting(t *testing.T) {
	path := filepath.Join(t.TempDir(), "report-linear-regression.html")
	if err := os.WriteFile(path, []byte("existing"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := writeReport(path, report{Rate: math.Inf(1)}); err == nil {
		t.Fatal("非有限値を拒否していない")
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "existing" {
		t.Fatal("変換失敗時に既存ファイルを破壊した")
	}
}
