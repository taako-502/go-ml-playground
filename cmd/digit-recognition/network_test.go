package main

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestBackpropagationMatchesNumericalGradient(t *testing.T) {
	n := newNetwork(3, rand.New(rand.NewPCG(1, 0)))
	// ReLUの折れ点から離し、両層の重みとバイアスを検証する。
	for i := range n.B1 {
		n.B1[i] = 2
	}
	x := make([]float64, inputs)
	x[0], x[21], x[400] = 0.8, 0.2, 0.4
	g := network{n.Hidden, make([]float64, len(n.W1)), make([]float64, len(n.B1)), make([]float64, len(n.W2)), make([]float64, len(n.B2))}
	h, p, delta := make([]float64, n.Hidden), make([]float64, classes), make([]float64, n.Hidden)
	n.gradients(x, 4, h, p, delta, g)
	loss := func() float64 { n.forward(x, h, p); return -math.Log(p[4]) }
	for _, pair := range [][2][]float64{{n.W1, g.W1}, {n.B1, g.B1}, {n.W2, g.W2}, {n.B2, g.B2}} {
		for i := range pair[0] {
			v := pair[0][i]
			eps := 1e-5
			pair[0][i] = v + eps
			plus := loss()
			pair[0][i] = v - eps
			minus := loss()
			pair[0][i] = v
			numeric := (plus - minus) / (2 * eps)
			if math.Abs(numeric-pair[1][i]) > 1e-6 {
				t.Fatalf("勾配が不一致 index=%d numerical=%g analytic=%g", i, numeric, pair[1][i])
			}
		}
	}
}

func TestTrainingLearnsPixelPatterns(t *testing.T) {
	data := dataset{Images: make([]byte, 20*inputs), Labels: make([]byte, 20)}
	for i := range data.Labels {
		label := byte(i % 2)
		data.Labels[i] = label
		for j := range 30 {
			data.image(i)[int(label)*100+j] = 255
		}
	}
	rng := rand.New(rand.NewPCG(42, 0))
	n := newNetwork(8, rng)
	initial := n.evaluate(data)
	for range 20 {
		n.trainEpoch(data, rng, 0.1, 4)
	}
	final := n.evaluate(data)
	if final.Accuracy != 1 || final.Loss >= initial.Loss*0.1 {
		t.Fatalf("画素パターンを学習できない: before=%+v after=%+v", initial, final)
	}
}

func TestSoftmaxIsFiniteAndNormalized(t *testing.T) {
	n := newNetwork(2, rand.New(rand.NewPCG(1, 0)))
	for i := range n.B2 {
		n.B2[i] = 10000 + float64(i)
	}
	p := make([]float64, classes)
	n.forward(make([]float64, inputs), make([]float64, 2), p)
	sum := 0.0
	for _, v := range p {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < 0 || v > 1 {
			t.Fatalf("不正なsoftmax: %v", p)
		}
		sum += v
	}
	if math.Abs(sum-1) > 1e-12 {
		t.Fatalf("確率の和=%g", sum)
	}
}
