// Copyright 2025 The Gaussian Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"compress/bzip2"
	"embed"
	"encoding/gob"
	"flag"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strings"

	"github.com/pointlander/gradient/tf32"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

const (
	// B1 exponential decay of the rate for the first moment estimates
	B1 = 0.8
	// B2 exponential decay rate for the second-moment estimates
	B2 = 0.89
	// Eta is the learning rate
	Eta = 1.0e-3
	// Eta1
	Eta1 = 1.0e-1
)

const (
	// StateM is the state for the mean
	StateM = iota
	// StateV is the state for the variance
	StateV
	// StateTotal is the total number of states
	StateTotal
)

//go:embed books/*
var Data embed.FS

// Vector is a vector
type Vector struct {
	Vector [256]float32
	Symbol byte
}

// Statistics are the per symbol statistics
type Statistics struct {
	Count      int
	Average    [256]float32
	Covariance [256][256]float32
	A          [256][256]float32
	AI         [256][256]float32
}

// Set is a set of statistics
type Set []Statistics

func (s Set) Calculate() {
	file, err := Data.Open("books/10.txt.utf-8.bz2")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	reader := bzip2.NewReader(file)
	data, err := io.ReadAll(reader)
	if err != nil {
		panic(err)
	}

	vectors := make([]Vector, len(data))
	m := NewMixer()
	m.Add(0)
	for i, v := range data {
		m.Mix(&vectors[i].Vector)
		s[v].Count++
		for j, vv := range vectors[i].Vector {
			s[v].Average[j] += vv
		}
		vectors[i].Symbol = v
		m.Add(v)
	}
	for i := range s {
		count := float32(s[i].Count)
		if count == 0 {
			continue
		}
		for j := range s[i].Average {
			s[i].Average[j] /= count
		}
	}
	for _, vector := range vectors {
		stats := &s[vector.Symbol]
		for i, v := range vector.Vector {
			for ii, vv := range vector.Vector {
				diff1 := stats.Average[i] - v
				diff2 := stats.Average[ii] - vv
				stats.Covariance[i][ii] += diff1 * diff2
			}
		}
	}
	for k := range s {
		stats := &s[k]
		count := float32(stats.Count)
		if count == 0 {
			continue
		}
		for i := range stats.Covariance {
			for j := range stats.Covariance[i] {
				stats.Covariance[i][j] = stats.Covariance[i][j] / count
			}
		}
	}
}

var (
	// FlagCheck checks the model
	FlagCheck = flag.String("check", "", "check a model")
	// FlagInfer infers text from a query
	FlagInfer = flag.String("infer", "", "infer text based on a model")
	// FlagQuery is the query
	FlagQuery = flag.String("query", "What is the meaning of life?", "query")
)

func main() {
	flag.Parse()

	if *FlagCheck != "" {
		fmt.Println("checking...")
		total := float32(0.0)
		var set Set
		input, err := os.Open(*FlagCheck)
		if err != nil {
			panic(err)
		}
		decoder := gob.NewDecoder(input)
		err = decoder.Decode(&set)
		if err != nil {
			panic(err)
		}
		for i := range set {
			if set[i].Count == 0 {
				continue
			}
			a := NewMatrix(256, 256)
			for r := 0; r < a.Rows; r++ {
				for c := 0; c < a.Cols; c++ {
					a.Data = append(a.Data, set[i].A[r][c])
				}
			}
			ai := NewMatrix(256, 256)
			for r := 0; r < ai.Rows; r++ {
				for c := 0; c < ai.Cols; c++ {
					ai.Data = append(ai.Data, set[i].AI[r][c])
				}
			}
			ii := NewMatrix(256, 256)
			for r := 0; r < ii.Rows; r++ {
				for c := 0; c < ii.Cols; c++ {
					if r == c {
						ii.Data = append(ii.Data, 1)
					} else {
						ii.Data = append(ii.Data, 0)
					}
				}
			}
			diff := ai.MulT(a).Sub(ii)
			sum := float32(0.0)
			for _, v := range diff.Data {
				if v < 0 {
					v = -v
				}
				sum += v
			}
			fmt.Println(i, sum/float32(diff.Rows*diff.Cols))
			total += sum / float32(diff.Rows*diff.Cols)
		}
		fmt.Println("total", total)
		return
	} else if *FlagInfer != "" {
		var set Set
		input, err := os.Open(*FlagInfer)
		if err != nil {
			panic(err)
		}
		decoder := gob.NewDecoder(input)
		err = decoder.Decode(&set)
		if err != nil {
			panic(err)
		}

		m := NewMixer()
		for _, v := range []byte(*FlagQuery) {
			m.Add(v)
		}

		for s := 0; s < 33; s++ {
			var vector [256]float32
			m.Mix(&vector)
			vec := NewMatrix(256, 1)
			vec.Data = append(vec.Data, vector[:]...)
			min, symbol := float32(math.MaxFloat32), 0
			for i := range set {
				if set[i].Count == 0 {
					continue
				}
				u := NewMatrix(256, 1)
				for _, v := range set[i].Average {
					u.Data = append(u.Data, v)
				}
				ai := NewMatrix(256, 256)
				for r := 0; r < ai.Rows; r++ {
					for c := 0; c < ai.Cols; c++ {
						ai.Data = append(ai.Data, set[i].AI[r][c])
					}
				}
				diff := ai.MulT(vec.Sub(u))
				sum := float32(0.0)
				for _, v := range diff.Data {
					if v < 0 {
						v = -v
					}
					sum += v
				}
				if sum < min {
					min, symbol = sum, i
				}
			}
			fmt.Println(min, symbol)
			fmt.Printf("%c\n", byte(symbol))
			m.Add(byte(symbol))
		}
		return
	}

	rng := rand.New(rand.NewSource(1))
	statistics := make(Set, 256)
	statistics.Calculate()

	for s := range statistics {
		stats := &statistics[s]
		if stats.Count == 0 {
			continue
		}

		{
			set := tf32.NewSet()
			set.Add("A", 256, 256)

			for i := range set.Weights {
				w := set.Weights[i]
				if strings.HasPrefix(w.N, "b") {
					w.X = w.X[:cap(w.X)]
					w.States = make([][]float32, StateTotal)
					for i := range w.States {
						w.States[i] = make([]float32, len(w.X))
					}
					continue
				}
				factor := math.Sqrt(2.0 / float64(w.S[0]))
				for i := 0; i < cap(w.X); i++ {
					w.X = append(w.X, float32(rng.NormFloat64()*factor))
				}
				w.States = make([][]float32, StateTotal)
				for i := range w.States {
					w.States[i] = make([]float32, len(w.X))
				}
			}

			others := tf32.NewSet()
			others.Add("E", 256, 256)
			E := others.ByName["E"]
			for i := range stats.Covariance {
				for j := range stats.Covariance[i] {
					E.X = append(E.X, stats.Covariance[i][j])
				}
			}

			loss := tf32.Sum(tf32.Quadratic(others.Get("E"), tf32.Mul(set.Get("A"), set.Get("A"))))

			points := make(plotter.XYs, 0, 8)
			for i := 0; i < 1024; i++ {
				pow := func(x float32) float32 {
					y := math.Pow(float64(x), float64(i+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return float32(y)
				}

				set.Zero()
				others.Zero()
				cost := tf32.Gradient(loss).X[0]
				if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
					fmt.Println(i, cost)
					break
				}

				norm := float32(0.0)
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := float32(1.0)
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					for l, d := range w.D {
						g := d * scaling
						m := B1*w.States[StateM][l] + (1-B1)*g
						v := B2*w.States[StateV][l] + (1-B2)*g*g
						w.States[StateM][l] = m
						w.States[StateV][l] = v
						mhat := m / (1 - b1)
						vhat := v / (1 - b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[l] -= Eta * mhat / (sqrt(vhat) + 1e-8)
					}
				}
				points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
				fmt.Println(i, cost)
			}

			p := plot.New()

			p.Title.Text = "epochs vs cost"
			p.X.Label.Text = "epochs"
			p.Y.Label.Text = "cost"

			scatter, err := plotter.NewScatter(points)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			p.Add(scatter)

			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs/epochs%d_A.png", s))
			if err != nil {
				panic(err)
			}

			index := 0
			for r := range stats.A {
				for c := range stats.A {
					stats.A[r][c] = set.ByName["A"].X[index]
					index++
				}
			}
		}

		{
			others := tf32.NewSet()
			others.Add("A", 256, 256)
			others.Add("I", 256, 256)
			A := others.ByName["A"]
			I := others.ByName["I"]
			for i := range stats.A {
				for j := range stats.A[i] {
					A.X = append(A.X, stats.A[i][j])
					if i == j {
						I.X = append(I.X, 1)
					} else {
						I.X = append(I.X, 0)
					}
				}
			}

			min, best := float32(math.MaxFloat32), tf32.Set{}
			for t := 0; t < 32; t++ {
				set := tf32.NewSet()
				set.Add("AI", 256, 256)
				for i := range set.Weights {
					w := set.Weights[i]
					if strings.HasPrefix(w.N, "b") {
						w.X = w.X[:cap(w.X)]
						w.States = make([][]float32, StateTotal)
						for i := range w.States {
							w.States[i] = make([]float32, len(w.X))
						}
						continue
					}
					factor := math.Sqrt(2.0 / float64(w.S[0]))
					for i := 0; i < cap(w.X); i++ {
						w.X = append(w.X, float32(rng.NormFloat64()*factor))
					}
					w.States = make([][]float32, StateTotal)
					for i := range w.States {
						w.States[i] = make([]float32, len(w.X))
					}
				}

				loss := tf32.Sum(tf32.Quadratic(others.Get("I"), tf32.Mul(set.Get("AI"), others.Get("A"))))

				set.Zero()
				others.Zero()
				cost := tf32.Gradient(loss).X[0]

				if cost < min {
					min, best = cost, set
				}
			}

			set := best

			loss := tf32.Sum(tf32.Quadratic(others.Get("I"), tf32.Mul(set.Get("AI"), others.Get("A"))))

			points := make(plotter.XYs, 0, 8)
			for i := 0; i < 8*1024; i++ {
				pow := func(x float32) float32 {
					y := math.Pow(float64(x), float64(i+1))
					if math.IsNaN(y) || math.IsInf(y, 0) {
						return 0
					}
					return float32(y)
				}

				set.Zero()
				others.Zero()
				cost := tf32.Gradient(loss).X[0]
				if math.IsNaN(float64(cost)) || math.IsInf(float64(cost), 0) {
					fmt.Println(i, cost)
					break
				}

				norm := float32(0.0)
				for _, p := range set.Weights {
					for _, d := range p.D {
						norm += d * d
					}
				}
				norm = sqrt(norm)
				b1, b2 := pow(B1), pow(B2)
				scaling := float32(1.0)
				if norm > 1 {
					scaling = 1 / norm
				}
				for _, w := range set.Weights {
					for l, d := range w.D {
						g := d * scaling
						m := B1*w.States[StateM][l] + (1-B1)*g
						v := B2*w.States[StateV][l] + (1-B2)*g*g
						w.States[StateM][l] = m
						w.States[StateV][l] = v
						mhat := m / (1 - b1)
						vhat := v / (1 - b2)
						if vhat < 0 {
							vhat = 0
						}
						w.X[l] -= Eta1 * mhat / (sqrt(vhat) + 1e-8)
					}
				}
				points = append(points, plotter.XY{X: float64(i), Y: float64(cost)})
				fmt.Println(i, cost)
			}

			p := plot.New()

			p.Title.Text = "epochs vs cost"
			p.X.Label.Text = "epochs"
			p.Y.Label.Text = "cost"

			scatter, err := plotter.NewScatter(points)
			if err != nil {
				panic(err)
			}
			scatter.GlyphStyle.Radius = vg.Length(1)
			scatter.GlyphStyle.Shape = draw.CircleGlyph{}
			p.Add(scatter)

			err = p.Save(8*vg.Inch, 8*vg.Inch, fmt.Sprintf("epochs/epochs%d_AI.png", s))
			if err != nil {
				panic(err)
			}

			index := 0
			for r := range stats.AI {
				for c := range stats.AI {
					stats.AI[r][c] = set.ByName["AI"].X[index]
					index++
				}
			}
		}
	}

	output, err := os.Create("data.bin")
	if err != nil {
		panic(err)
	}
	defer output.Close()

	encoder := gob.NewEncoder(output)
	err = encoder.Encode(statistics)
	if err != nil {
		panic(err)
	}
}
