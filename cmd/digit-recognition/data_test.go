package main

import (
	"compress/gzip"
	"encoding/binary"
	"os"
	"path/filepath"
	"testing"
)

func TestReadIDX(t *testing.T) {
	valid := make([]byte, 16+inputs)
	binary.BigEndian.PutUint32(valid[0:4], 2051)
	binary.BigEndian.PutUint32(valid[4:8], 1)
	binary.BigEndian.PutUint32(valid[8:12], 28)
	binary.BigEndian.PutUint32(valid[12:16], 28)
	valid[16] = 123
	badSize := append([]byte(nil), valid...)
	binary.BigEndian.PutUint32(badSize[8:12], 27)
	for _, tc := range []struct {
		name      string
		data      []byte
		wantError bool
	}{{"valid", valid, false}, {"truncated", valid[:len(valid)-1], true}, {"dimensions", badSize, true}, {"header", []byte{0, 1}, true}} {
		t.Run(tc.name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "images.gz")
			f, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			gz := gzip.NewWriter(f)
			if _, err := gz.Write(tc.data); err != nil {
				t.Fatal(err)
			}
			if err := gz.Close(); err != nil {
				t.Fatal(err)
			}
			if err := f.Close(); err != nil {
				t.Fatal(err)
			}
			pixels, count, err := readIDX(path, 2051)
			if tc.wantError {
				if err == nil {
					t.Fatal("不正な入力を受理した")
				}
				return
			}
			if err != nil {
				t.Fatal(err)
			}
			if count != 1 || len(pixels) != inputs || pixels[0] != 123 {
				t.Fatal("IDXの画素を復元できない")
			}
		})
	}
}
