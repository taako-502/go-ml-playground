package main

import (
	"math"
	"testing"
)

func TestLearnKnownLine(t *testing.T) {
	train, test := dataset(42, 0)
	m := model{}
	for i := 0; i < 2000; i++ {
		m = step(m, train, 0.1)
	}
	if math.Abs(m.weight/100-30) > 0.01 || math.Abs(m.bias-500) > 0.1 || mse(m, test) > 0.01 {
		t.Fatalf("既知の直線を復元できない: %+v, test MSE=%g", m, mse(m, test))
	}
}

func TestGradientMatchesFiniteDifference(t *testing.T) {
	data := []sample{{20, 1000}, {70, 2700}, {110, 3600}}
	m := model{120, 80}
	next := step(m, data, 1)
	eps := 0.001
	dw := (mse(model{m.weight + eps, m.bias}, data) - mse(model{m.weight - eps, m.bias}, data)) / (2 * eps)
	db := (mse(model{m.weight, m.bias + eps}, data) - mse(model{m.weight, m.bias - eps}, data)) / (2 * eps)
	if math.Abs((m.weight-next.weight)-dw) > 0.001 || math.Abs((m.bias-next.bias)-db) > 0.001 {
		t.Fatal("解析勾配と数値微分が一致しない")
	}
}
