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

	//"github.com/alixaxel/pagerank"
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
	Symbol State
}

// Statistics are the per symbol statistics
type Statistics struct {
	Count      int
	Average    [256]float32
	Covariance [256][256]float32
	A          [256][256]float32
}

// State is a markov state
type State [2]byte

// Set is a set of statistics
type Set map[State]Statistics

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
	for i, v := range data[:len(data)-1] {
		state := [2]byte{v, data[i+1]}
		ss := s[state]
		m.Mix(&vectors[i].Vector)
		ss.Count++
		for j, vv := range vectors[i].Vector {
			ss.Average[j] += vv
		}
		vectors[i].Symbol = state
		s[state] = ss
		m.Add(v)
	}
	for i := range s {
		si := s[i]
		count := float32(si.Count)
		if count == 0 {
			continue
		}
		for j := range si.Average {
			si.Average[j] /= count
		}
		s[i] = si
	}
	for _, vector := range vectors {
		stats := s[vector.Symbol]
		for i, v := range vector.Vector {
			for ii, vv := range vector.Vector {
				diff1 := stats.Average[i] - v
				diff2 := stats.Average[ii] - vv
				stats.Covariance[i][ii] += diff1 * diff2
			}
		}
		s[vector.Symbol] = stats
	}
	for k := range s {
		stats := s[k]
		count := float32(stats.Count)
		if count == 0 {
			continue
		}
		for i := range stats.Covariance {
			for j := range stats.Covariance[i] {
				stats.Covariance[i][j] = stats.Covariance[i][j] / count
			}
		}
		s[k] = stats
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
			cov := NewMatrix(256, 256)
			for r := 0; r < cov.Rows; r++ {
				for c := 0; c < cov.Cols; c++ {
					cov.Data = append(cov.Data, set[i].Covariance[r][c])
				}
			}
			a := NewMatrix(256, 256)
			for r := 0; r < a.Rows; r++ {
				for c := 0; c < a.Cols; c++ {
					a.Data = append(a.Data, set[i].A[r][c])
				}
			}
			diff := a.MulT(a).Sub(cov)
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
		rng := rand.New(rand.NewSource(1))

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
			var samples [256]float32
			var vector [256]float32
			var counts [256]float32
			m.Mix(&vector)
			for i := range set {
				if set[i].Count == 0 || i[0] >= 128 {
					continue
				}
				counts[i[0]]++
				u := NewMatrix(256, 1)
				for _, v := range set[i].Average {
					u.Data = append(u.Data, v)
				}
				a := NewMatrix(256, 256)
				for r := 0; r < a.Rows; r++ {
					for c := 0; c < a.Cols; c++ {
						a.Data = append(a.Data, set[i].A[r][c])
					}
				}
				rng := rand.New(rand.NewSource(1))
				vectors := make([]Matrix, 32)
				for j := range vectors {
					vec := NewMatrix(256, 1)
					for k := 0; k < vec.Cols; k++ {
						vec.Data = append(vec.Data, float32(rng.NormFloat64()))
					}
					vectors[j] = a.MulT(vec).Add(u)
					cs := CS(vector[:], vectors[j].Data)
					samples[i[0]] += cs
				}
				/*graph := pagerank.NewGraph()
				for a := range vectors {
					for b := range vectors {
						cs := CS(vectors[a].Data, vectors[b].Data)
						if cs < 0 {
							cs = -cs
						}
						graph.Link(uint32(a), uint32(b), float64(cs))
					}
				}
				for a := range vectors {
					cs := CS(vector[:], vectors[a].Data)
					if cs < 0 {
						cs = -cs
					}
					graph.Link(uint32(a), uint32(32), float64(cs))
					graph.Link(uint32(32), uint32(a), float64(cs))
				}
				ranks := make([]float32, 33)
				graph.Rank(.8, 1e-3, func(node uint32, rank float64) {
					ranks[node] = float32(rank)
				})
				samples[i] = ranks[32]*/

			}
			sum := float32(0.0)
			for i := range samples {
				if counts[i] == 0 {
					continue
				}
				samples[i] /= counts[i]
			}
			for _, v := range samples {
				sum += v
			}
			total, selected, symbol := float32(0.0), rng.Float32(), 0
			for i, v := range samples {
				if v == 0 {
					continue
				}
				total += v / sum
				if selected < total {
					symbol = i
					break
				}
			}
			/*max, symbol := float32(0.0), 0
			for i, v := range samples {
				if v > max {
					max, symbol = v, i
				}
			}*/
			fmt.Printf("%d %c\n", symbol, byte(symbol))
			m.Add(byte(symbol))
		}
		return
	}

	rng := rand.New(rand.NewSource(1))
	statistics := make(Set)
	statistics.Calculate()

	for s := range statistics {
		stats := statistics[s]
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
		statistics[s] = stats
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
