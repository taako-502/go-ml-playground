package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
)

type sample struct{ temperature, sales float64 }
type model struct{ weight, bias float64 }

// 気温を10で割り、勾配が大きくなりすぎるのを防ぐ。
func (m model) predict(temperature float64) float64 { return m.weight*(temperature/10) + m.bias }

func mse(m model, data []sample) float64 {
	var sum float64
	for _, s := range data {
		err := m.predict(s.temperature) - s.sales
		sum += err * err
	}
	return sum / float64(len(data))
}

// 全学習データの勾配を平均し、両パラメータを同時に更新する。
func step(m model, data []sample, rate float64) model {
	var dw, db float64
	for _, s := range data {
		err := m.predict(s.temperature) - s.sales
		dw += 2 * err * (s.temperature / 10)
		db += 2 * err
	}
	n := float64(len(data))
	return model{m.weight - rate*dw/n, m.bias - rate*db/n}
}

func dataset(seed int64, noise float64) ([]sample, []sample) {
	r := rand.New(rand.NewSource(seed))
	data := make([]sample, 100)
	for i := range data {
		temperature := r.Float64() * 35
		// データ生成専用の式。学習処理には係数を渡さない。
		data[i] = sample{temperature, 8*temperature + 40 + r.NormFloat64()*noise}
	}
	r.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	return data[:80], data[80:]
}

func main() {
	epochs := flag.Int("epochs", 2000, "学習回数")
	rate := flag.Float64("lr", 0.1, "学習率")
	noise := flag.Float64("noise", 15, "販売数ノイズの標準偏差（個）")
	seed := flag.Int64("seed", 42, "乱数シード")
	temperature := flag.Float64("temperature", 25, "予測する気温（℃）")
	htmlPath := flag.String("html", "", "可視化HTMLの出力先（例: report.html）")
	flag.Parse()
	finite := func(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
	if *epochs <= 0 || !finite(*rate) || *rate <= 0 || !finite(*noise) || *noise < 0 || !finite(*temperature) {
		fmt.Fprintln(os.Stderr, "epochs・lrは正の値、noiseは0以上、temperatureは有限値を指定してください")
		os.Exit(1)
	}
	train, test := dataset(*seed, *noise)
	m := model{}
	var history []snapshot
	record := func(epoch int) {
		if *htmlPath != "" {
			history = append(history, snapshot{epoch, m.weight / 10, m.bias, mse(m, train), mse(m, test)})
		}
	}
	record(0)
	fmt.Printf("架空のアイスクリームの販売数 / 学習 %d件・評価 %d件 / seed=%d\n", len(train), len(test), *seed)
	fmt.Printf("epoch=%4d  学習MSE=%.2f\n", 0, mse(m, train))
	interval := max(*epochs/10, 1)
	for epoch := 1; epoch <= *epochs; epoch++ {
		m = step(m, train, *rate)
		loss := mse(m, train)
		if !finite(loss) {
			fmt.Fprintln(os.Stderr, "学習が発散しました。lrを小さくしてください")
			os.Exit(1)
		}
		if epoch%((*epochs-1)/1000+1) == 0 || epoch == *epochs {
			record(epoch)
		}
		if epoch%interval == 0 || epoch == *epochs {
			fmt.Printf("epoch=%4d  学習MSE=%.2f\n", epoch, loss)
		}
	}
	fmt.Printf("\n学習した式: 販売数（個）= %.2f × 気温（℃）+ %.2f\n", m.weight/10, m.bias)
	fmt.Printf("学習RMSE: %.2f個\n評価RMSE: %.2f個\n", math.Sqrt(mse(m, train)), math.Sqrt(mse(m, test)))
	var mean float64
	for _, s := range train {
		mean += s.sales / float64(len(train))
	}
	fmt.Printf("平均販売数だけで予測する基準モデルの評価RMSE: %.2f個\n", math.Sqrt(mse(model{bias: mean}, test)))
	fmt.Println("\n評価データの先頭5件: 気温 / 正解 / 予測（個）")
	for _, s := range test[:5] {
		fmt.Printf("%5.1f℃ / %7.1f / %7.1f\n", s.temperature, s.sales, m.predict(s.temperature))
	}
	fmt.Printf("\n%.1f℃の予測販売数: %.1f個\n", *temperature, m.predict(*temperature))
	if *htmlPath != "" {
		r := report{Seed: *seed, Rate: *rate, Noise: *noise, Temperature: *temperature, Train: reportSamples(train), Test: reportSamples(test), History: history, BaselineRMSE: math.Sqrt(mse(model{bias: mean}, test))}
		if err := writeReport(*htmlPath, r); err != nil {
			fmt.Fprintln(os.Stderr, "HTML出力に失敗しました:", err)
			os.Exit(1)
		}
		fmt.Printf("\n可視化HTML: %s\n", *htmlPath)
	}
}
