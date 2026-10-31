package main

import (
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
)

type sample struct{ area, price float64 }
type model struct{ weight, bias float64 }

// 面積を100で割り、勾配が大きくなりすぎるのを防ぐ。
func (m model) predict(area float64) float64 { return m.weight*(area/100) + m.bias }

func mse(m model, data []sample) float64 {
	var sum float64
	for _, s := range data {
		err := m.predict(s.area) - s.price
		sum += err * err
	}
	return sum / float64(len(data))
}

// 全学習データの勾配を平均し、両パラメータを同時に更新する。
func step(m model, data []sample, rate float64) model {
	var dw, db float64
	for _, s := range data {
		err := m.predict(s.area) - s.price
		dw += 2 * err * (s.area / 100)
		db += 2 * err
	}
	n := float64(len(data))
	return model{m.weight - rate*dw/n, m.bias - rate*db/n}
}

func dataset(seed int64, noise float64) ([]sample, []sample) {
	r := rand.New(rand.NewSource(seed))
	data := make([]sample, 100)
	for i := range data {
		area := 20 + r.Float64()*100
		// データ生成専用の式。学習処理には係数を渡さない。
		data[i] = sample{area, 30*area + 500 + r.NormFloat64()*noise}
	}
	r.Shuffle(len(data), func(i, j int) { data[i], data[j] = data[j], data[i] })
	return data[:80], data[80:]
}

func main() {
	epochs := flag.Int("epochs", 2000, "学習回数")
	rate := flag.Float64("lr", 0.1, "学習率")
	noise := flag.Float64("noise", 150, "価格ノイズの標準偏差（万円）")
	seed := flag.Int64("seed", 42, "乱数シード")
	area := flag.Float64("area", 75, "予測する面積（平方メートル）")
	flag.Parse()
	finite := func(x float64) bool { return !math.IsNaN(x) && !math.IsInf(x, 0) }
	if *epochs <= 0 || !finite(*rate) || *rate <= 0 || !finite(*noise) || *noise < 0 || !finite(*area) || *area <= 0 {
		fmt.Fprintln(os.Stderr, "epochs・lr・areaは正の値、noiseは0以上の有限値を指定してください")
		os.Exit(1)
	}
	train, test := dataset(*seed, *noise)
	m := model{}
	fmt.Printf("架空の住宅価格 / 学習 %d件・評価 %d件 / seed=%d\n", len(train), len(test), *seed)
	fmt.Printf("epoch=%4d  学習MSE=%.2f\n", 0, mse(m, train))
	interval := max(*epochs/10, 1)
	for epoch := 1; epoch <= *epochs; epoch++ {
		m = step(m, train, *rate)
		loss := mse(m, train)
		if !finite(loss) {
			fmt.Fprintln(os.Stderr, "学習が発散しました。lrを小さくしてください")
			os.Exit(1)
		}
		if epoch%interval == 0 || epoch == *epochs {
			fmt.Printf("epoch=%4d  学習MSE=%.2f\n", epoch, loss)
		}
	}
	fmt.Printf("\n学習した式: 価格（万円）= %.2f × 面積（㎡）+ %.2f\n", m.weight/100, m.bias)
	fmt.Printf("学習RMSE: %.2f万円\n評価RMSE: %.2f万円\n", math.Sqrt(mse(m, train)), math.Sqrt(mse(m, test)))
	var mean float64
	for _, s := range train {
		mean += s.price / float64(len(train))
	}
	fmt.Printf("平均価格だけで予測する基準モデルの評価RMSE: %.2f万円\n", math.Sqrt(mse(model{bias: mean}, test)))
	fmt.Println("\n評価データの先頭5件: 面積 / 正解 / 予測（万円）")
	for _, s := range test[:5] {
		fmt.Printf("%5.1f㎡ / %7.1f / %7.1f\n", s.area, s.price, m.predict(s.area))
	}
	fmt.Printf("\n%.1f㎡の予測価格: %.1f万円\n", *area, m.predict(*area))
}
