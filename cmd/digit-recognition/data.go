package main

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"time"
)

type dataset struct {
	Images []byte
	Labels []byte
}

func (d dataset) image(i int) []byte { return d.Images[i*inputs : (i+1)*inputs] }
func (d dataset) slice(start, end int) dataset {
	return dataset{d.Images[start*inputs : end*inputs], d.Labels[start:end]}
}

// MNISTの配布ファイルをキャッシュする。途中で失敗したダウンロードは採用しない。
func download(dir, name string) (string, error) {
	path := filepath.Join(dir, name)
	if _, err := os.Stat(path); err == nil {
		return path, nil
	}
	client := http.Client{Timeout: 2 * time.Minute}
	resp, err := client.Get("https://storage.googleapis.com/cvdf-datasets/mnist/" + name)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("%s: HTTP %d", name, resp.StatusCode)
	}
	f, err := os.CreateTemp(dir, "download-*")
	if err != nil {
		return "", err
	}
	defer os.Remove(f.Name())
	_, copyErr := io.Copy(f, io.LimitReader(resp.Body, 20<<20))
	closeErr := f.Close()
	if copyErr != nil {
		return "", copyErr
	}
	if closeErr != nil {
		return "", closeErr
	}
	if err := os.Rename(f.Name(), path); err != nil {
		return "", err
	}
	return path, nil
}

// IDXヘッダー・サイズ・ラベルを検証してから学習へ渡す。
func readIDX(path string, magic uint32) ([]byte, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, 0, err
	}
	defer gz.Close()
	data, err := io.ReadAll(io.LimitReader(gz, 50<<20))
	if err != nil {
		return nil, 0, err
	}
	if len(data) < 8 || binary.BigEndian.Uint32(data[:4]) != magic {
		return nil, 0, fmt.Errorf("%s: IDX形式が不正", path)
	}
	count := int(binary.BigEndian.Uint32(data[4:8]))
	offset, size := 8, 1
	if count < 1 || count > 60000 {
		return nil, 0, fmt.Errorf("%s: 件数が不正", path)
	}
	if magic == 2051 {
		if len(data) < 16 || binary.BigEndian.Uint32(data[8:12]) != 28 || binary.BigEndian.Uint32(data[12:16]) != 28 {
			return nil, 0, fmt.Errorf("%s: 28×28画像ではない", path)
		}
		offset, size = 16, inputs
	}
	if len(data) != offset+count*size {
		return nil, 0, fmt.Errorf("%s: IDXサイズが不正", path)
	}
	return data[offset:], count, nil
}

func loadData(dir, prefix string) (dataset, error) {
	var result dataset
	imagePath, err := download(dir, prefix+"-images-idx3-ubyte.gz")
	if err != nil {
		return result, err
	}
	labelPath, err := download(dir, prefix+"-labels-idx1-ubyte.gz")
	if err != nil {
		return result, err
	}
	images, n, err := readIDX(imagePath, 2051)
	if err != nil {
		return result, err
	}
	labels, m, err := readIDX(labelPath, 2049)
	if err != nil {
		return result, err
	}
	if n != m {
		return result, fmt.Errorf("画像とラベルの件数が異なる")
	}
	for _, v := range labels {
		if v >= classes {
			return result, fmt.Errorf("数字ラベルの範囲外: %d", v)
		}
	}
	return dataset{images, labels}, nil
}
