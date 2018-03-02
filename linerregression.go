package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
)

type recordStruct struct {
	x float64
	y float64
}

func readCsv(filename string) []recordStruct {
	filePoint, _ := os.Open(filename)
	r := csv.NewReader(bufio.NewReader(filePoint))
	listrecords := []recordStruct{}
	for {
		records, err := r.Read()
		// fmt.Print(records)
		if err == io.EOF {
			break
		}
		x, _ := strconv.ParseFloat(records[0], 64)
		y, _ := strconv.ParseFloat(records[1], 64)
		listrecords = append(listrecords, recordStruct{x, y})
	}
	return listrecords

}

func step_gradient(b_current float64, m_current float64, points []recordStruct, learningRate float64) (float64, float64) {
	b_gradient := 0.0
	m_gradient := 0.0
	lengthval := len(points)
	N := float64(lengthval)
	for i := 0; i < lengthval; i++ {
		b_gradient += -((2.0 / N) * (points[i].y - ((m_current * points[i].x) + b_current)))
		m_gradient += -(2.0 / N) * points[i].x * (points[i].y - ((m_current * points[i].x) + b_current))
	}
	new_b := b_current - (learningRate * b_gradient)
	new_m := m_current - (learningRate * m_gradient)
	return new_b, new_m

}

func gradient_descent_runner(point []recordStruct, starting_b float64, starting_m float64, learing_rate float64, num_iteartion int) (float64, float64) {
	b := starting_b
	m := starting_m
	for i := 0; i < num_iteartion; i++ {
		b, m = step_gradient(b, m, point, learing_rate)
	}
	return b, m
}

func compute_error_for_line_given_points(b float64, m float64, points []recordStruct) {
	totalerror := 0.0
	lenval := len(points)
	for i := 0; i < lenval; i++ {
		calVal := (points[i].y - (m*points[i].x + b))
		totalerror += math.Exp2(calVal)
	}
	fmt.Println("error rate::>>", totalerror/float64(lenval))
}

func percentage(predval float64, exacval float64) {
	errorrate := (((exacval - predval) * 100.0) / exacval)
	fmt.Println("error rate::", predval, " ", exacval, " >>", errorrate)
}

func predict(points []recordStruct, cal_b float64, cal_m float64) {
	lenval := len(points)
	for i := 0; i < lenval; i++ {
		y_pre := cal_m*points[i].x + cal_b
		percentage(y_pre, points[i].y)

	}
}

func main() {
	fmt.Println("hello world")
	trainpoint := readCsv("train.csv")
	b, m := gradient_descent_runner(trainpoint, 0.0, 0.0, 0.000001, 10000)
	fmt.Println("pred_b=", b, " pre_m=", m)
	compute_error_for_line_given_points(b, m, trainpoint)
	testpoint := readCsv("test.csv")
	predict(testpoint, b, m)
}
