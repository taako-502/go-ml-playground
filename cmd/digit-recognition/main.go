package main

import (
	"bytes"
	_ "embed"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"math"
	"math/rand"
	"os"
	"time"
)

//go:embed report.html
var htmlSource string

type example struct {
	Pixels     []byte    `json:"pixels"`
	Label      int       `json:"label"`
	Prediction int       `json:"prediction"`
	Scores     []float64 `json:"scores"`
}

type report struct {
	Initial         network     `json:"initial"`
	Model           network     `json:"model"`
	History         []metrics   `json:"history"`
	Test            metrics     `json:"test"`
	Confusion       [10][10]int `json:"confusion"`
	Samples         []example   `json:"samples"`
	Mistakes        []example   `json:"mistakes"`
	TrainCount      int         `json:"trainCount"`
	ValidationCount int         `json:"validationCount"`
	TestCount       int         `json:"testCount"`
	Seed            int64       `json:"seed"`
	Rate            float64     `json:"rate"`
	Batch           int         `json:"batch"`
	Seconds         float64     `json:"seconds"`
}

func (r *report) inspect(data dataset) {
	x, h, p := make([]float64, inputs), make([]float64, r.Model.Hidden), make([]float64, classes)
	for i, label := range data.Labels {
		decode(data.image(i), x)
		r.Model.forward(x, h, p)
		prediction := argmax(p)
		r.Confusion[label][prediction]++
		// 評価セットの先頭100件と、最初の誤分類36件を恣意的に選ばず表示する。
		e := example{data.image(i), int(label), prediction, append([]float64(nil), p...)}
		if i < 100 {
			r.Samples = append(r.Samples, e)
		}
		if prediction != int(label) && len(r.Mistakes) < 36 {
			r.Mistakes = append(r.Mistakes, e)
		}
	}
}

func run() error {
	dir := flag.String("data", "data/mnist", "MNISTの保存先（初回は約11MBをダウンロード）")
	output := flag.String("html", "report-digits.html", "学習結果と描画体験のHTML出力先")
	epochs := flag.Int("epochs", 5, "学習回数")
	count := flag.Int("train", 20000, "学習画像数（1〜55000）")
	hidden := flag.Int("hidden", 64, "隠れ層のニューロン数（1〜256）")
	rate := flag.Float64("lr", 0.1, "学習率")
	batch := flag.Int("batch", 32, "ミニバッチサイズ（1〜1024）")
	seed := flag.Int64("seed", 42, "乱数シード")
	flag.Parse()
	if *epochs < 1 || *count < 1 || *count > 55000 || *hidden < 1 || *hidden > 256 || *batch < 1 || *batch > 1024 || math.IsNaN(*rate) || math.IsInf(*rate, 0) || *rate <= 0 {
		return fmt.Errorf("epochs・lrは正、trainは1〜55000、hiddenは1〜256、batchは1〜1024を指定してください")
	}
	if err := os.MkdirAll(*dir, 0755); err != nil {
		return err
	}
	fmt.Println("MNISTを読み込みます（未取得のファイルは公開ミラーからダウンロード）。")
	training, err := loadData(*dir, "train")
	if err != nil {
		return err
	}
	testing, err := loadData(*dir, "t10k")
	if err != nil {
		return err
	}
	if len(training.Labels) != 60000 || len(testing.Labels) != 10000 {
		return fmt.Errorf("MNISTの件数が想定と異なる")
	}
	// 最後の5000件を検証専用に固定し、テスト10000件は学習後だけ評価する。
	validation := training.slice(55000, 60000)
	training = training.slice(0, *count)
	rng := rand.New(rand.NewSource(*seed))
	n := newNetwork(*hidden, rng)
	r := report{Initial: n.clone(), Model: n, TrainCount: *count, ValidationCount: 5000, TestCount: 10000, Seed: *seed, Rate: *rate, Batch: *batch}
	fmt.Printf("ニューラルネット: 784 → %d (ReLU) → 10 (softmax) / 学習 %d枚・検証5000枚・テスト10000枚\n", *hidden, *count)
	start := time.Now()
	initial := n.evaluate(validation)
	r.History = append(r.History, initial)
	fmt.Printf("epoch=0 検証正解率=%.2f%% 損失=%.4f\n", initial.Accuracy*100, initial.Loss)
	for epoch := 1; epoch <= *epochs; epoch++ {
		n.trainEpoch(training, rng, *rate, *batch)
		result := n.evaluate(validation)
		result.Epoch = epoch
		if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
			return fmt.Errorf("学習が発散しました。lrを小さくしてください")
		}
		r.History = append(r.History, result)
		fmt.Printf("epoch=%d 検証正解率=%.2f%% 損失=%.4f\n", epoch, result.Accuracy*100, result.Loss)
	}
	r.Test = n.evaluate(testing)
	r.inspect(testing)
	r.Seconds = time.Since(start).Seconds()
	payload, err := json.Marshal(r)
	if err != nil {
		return err
	}
	t, err := template.New("digits").Parse(htmlSource)
	if err != nil {
		return err
	}
	var out bytes.Buffer
	// json.MarshalでHTML特殊文字をエスケープしたJSONのみを埋め込む。
	if err := t.Execute(&out, template.JS(payload)); err != nil {
		return err
	}
	if err := os.WriteFile(*output, out.Bytes(), 0644); err != nil {
		return err
	}
	fmt.Printf("最終テスト正解率=%.2f%%（10000枚） / %.1f秒\nHTML: %s\n", r.Test.Accuracy*100, r.Seconds, *output)
	return nil
}
func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}
