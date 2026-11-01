package main

import (
	"math"
	"math/rand"
)

const inputs = 28 * 28
const classes = 10

// networkは画素→ReLU隠れ層→softmax出力の小さなニューラルネットワーク。
// 重みは出力ニューロンごとに連続して格納する。
type network struct {
	Hidden int       `json:"hidden"`
	W1     []float64 `json:"w1"`
	B1     []float64 `json:"b1"`
	W2     []float64 `json:"w2"`
	B2     []float64 `json:"b2"`
}

func newNetwork(hidden int, rng *rand.Rand) network {
	n := network{hidden, make([]float64, hidden*inputs), make([]float64, hidden), make([]float64, classes*hidden), make([]float64, classes)}
	for i := range n.W1 {
		n.W1[i] = rng.NormFloat64() * math.Sqrt(2.0/inputs)
	}
	for i := range n.W2 {
		n.W2[i] = rng.NormFloat64() * math.Sqrt(2.0/float64(hidden))
	}
	return n
}

func (n network) clone() network {
	return network{n.Hidden, append([]float64(nil), n.W1...), append([]float64(nil), n.B1...), append([]float64(nil), n.W2...), append([]float64(nil), n.B2...)}
}

// softmaxの指数計算前に最大値を引き、オーバーフローを防ぐ。
func (n network) forward(x, hidden, probabilities []float64) {
	for j := range hidden {
		sum := n.B1[j]
		w := n.W1[j*inputs : (j+1)*inputs]
		for i, v := range x {
			sum += w[i] * v
		}
		hidden[j] = math.Max(0, sum)
	}
	maximum := math.Inf(-1)
	for k := range probabilities {
		sum := n.B2[k]
		for j, v := range hidden {
			sum += n.W2[k*n.Hidden+j] * v
		}
		probabilities[k] = sum
		maximum = math.Max(maximum, sum)
	}
	sum := 0.0
	for k, v := range probabilities {
		probabilities[k] = math.Exp(v - maximum)
		sum += probabilities[k]
	}
	for k := range probabilities {
		probabilities[k] /= sum
	}
}

func argmax(values []float64) int {
	best := 0
	for i, v := range values {
		if v > values[best] {
			best = i
		}
	}
	return best
}

func decode(pixels []byte, x []float64) {
	for i, v := range pixels {
		x[i] = float64(v) / 255
	}
}

// 勾配はパラメータ更新前の重みで計算し、ミニバッチ内で加算する。
func (n network) gradients(x []float64, label int, hidden, probabilities, delta []float64, g network) {
	n.forward(x, hidden, probabilities)
	clear(delta)
	for k, p := range probabilities {
		if k == label {
			p--
		}
		g.B2[k] += p
		for j, v := range hidden {
			g.W2[k*n.Hidden+j] += p * v
			delta[j] += n.W2[k*n.Hidden+j] * p
		}
	}
	for j, v := range hidden {
		if v <= 0 {
			continue
		}
		d := delta[j]
		g.B1[j] += d
		w := g.W1[j*inputs : (j+1)*inputs]
		for i, v := range x {
			w[i] += d * v
		}
	}
}

func (n network) trainEpoch(data dataset, rng *rand.Rand, rate float64, batch int) {
	order := rng.Perm(len(data.Labels))
	g := network{n.Hidden, make([]float64, len(n.W1)), make([]float64, len(n.B1)), make([]float64, len(n.W2)), make([]float64, len(n.B2))}
	x, h, p, delta := make([]float64, inputs), make([]float64, n.Hidden), make([]float64, classes), make([]float64, n.Hidden)
	for start := 0; start < len(order); start += batch {
		end := min(start+batch, len(order))
		for _, index := range order[start:end] {
			decode(data.image(index), x)
			n.gradients(x, int(data.Labels[index]), h, p, delta, g)
		}
		scale := rate / float64(end-start)
		for _, pair := range [][2][]float64{{n.W1, g.W1}, {n.B1, g.B1}, {n.W2, g.W2}, {n.B2, g.B2}} {
			for i, v := range pair[1] {
				pair[0][i] -= scale * v
				pair[1][i] = 0
			}
		}
	}
}

type metrics struct {
	Epoch    int     `json:"epoch"`
	Loss     float64 `json:"loss"`
	Accuracy float64 `json:"accuracy"`
}

func (n network) evaluate(data dataset) metrics {
	x, h, p := make([]float64, inputs), make([]float64, n.Hidden), make([]float64, classes)
	loss, correct := 0.0, 0
	for i, label := range data.Labels {
		decode(data.image(i), x)
		n.forward(x, h, p)
		loss -= math.Log(math.Max(p[label], 1e-15))
		if argmax(p) == int(label) {
			correct++
		}
	}
	return metrics{Loss: loss / float64(len(data.Labels)), Accuracy: float64(correct) / float64(len(data.Labels))}
}
