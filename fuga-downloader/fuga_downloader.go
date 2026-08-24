// fuga-downloader: сетевой стример лексиконов для Rust-ядра.
// Go качает и распаковывает многогигабайтные дампы (kaikki.org JSONL.gz,
// частотные списки) горутинами, дедуплицирует на лету и отдаёт чистый
// поток словоформ в файл/stdout — без нагрузки на Python/Rust.
package main

import (
	"bufio"
	"compress/gzip"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
)

type result struct {
	words chan string
	errs  chan error
}

func isGzip(url string) bool {
	return strings.HasSuffix(url, ".gz") || strings.HasSuffix(url, ".gzip")
}

func streamURL(client *http.Client, url string, out chan<- string, errs chan<- error) {
	resp, err := client.Get(url)
	if err != nil {
		errs <- fmt.Errorf("%s: %v", url, err)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode != 200 {
		errs <- fmt.Errorf("%s: HTTP %d", url, resp.StatusCode)
		return
	}
	var reader io.Reader = resp.Body
	if isGzip(url) {
		gz, err := gzip.NewReader(resp.Body)
		if err != nil {
			errs <- fmt.Errorf("%s: gzip: %v", url, err)
			return
		}
		reader = gz
	}
	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, 1<<20), 1<<24)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		// Форматы: "слово" | JSON {"word": ...} | "слово частота".
		word := extractWord(line)
		if word != "" {
			out <- word
		}
	}
	if err := scanner.Err(); err != nil {
		errs <- fmt.Errorf("%s: scan: %v", url, err)
	}
}

func extractWord(line string) string {
	if strings.HasPrefix(line, "{") {
		const key = `"word"`
		i := strings.Index(line, key)
		if i < 0 {
			return ""
		}
		rest := line[i+len(key):]
		rest = strings.TrimLeft(rest, " :")
		if len(rest) < 2 || rest[0] != '"' {
			return ""
		}
		end := strings.IndexByte(rest[1:], '"')
		if end < 0 {
			return ""
		}
		return rest[1 : 1+end]
	}
	fields := strings.Fields(line)
	if len(fields) == 0 {
		return ""
	}
	return fields[0]
}

func main() {
	urlsFlag := flag.String("urls", "", "comma-separated source URLs")
	outPath := flag.String("out", "datasets/wiktionary/lexicon.txt", "output file (one word per line)")
	workers := flag.Int("workers", 4, "parallel fetch workers")
	maxWords := flag.Int64("max", 0, "stop after N unique words (0 = unlimited)")
	flag.Parse()

	if *urlsFlag == "" {
		fmt.Fprintln(os.Stderr, "usage: fuga_downloader -urls URL1,URL2 [-out lexicon.txt] [-workers 4] [-max N]")
		os.Exit(2)
	}

	var urls []string
	for _, u := range strings.Split(*urlsFlag, ",") {
		if u = strings.TrimSpace(u); u != "" {
			urls = append(urls, u)
		}
	}

	res := result{words: make(chan string, 1<<16), errs: make(chan error, len(urls))}
	var wg sync.WaitGroup
	client := &http.Client{}
	jobs := make(chan string, len(urls))
	for _, u := range urls {
		jobs <- u
	}
	close(jobs)

	for i := 0; i < *workers; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for u := range jobs {
				streamURL(client, u, res.words, res.errs)
			}
		}()
	}
	go func() {
		wg.Wait()
		close(res.words)
		close(res.errs)
	}()

	if err := os.MkdirAll(dirOf(*outPath), 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "mkdir:", err)
		os.Exit(1)
	}
	f, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "create:", err)
		os.Exit(1)
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	defer w.Flush()

	seen := make(map[string]struct{})
	var count int64
	for word := range res.words {
		if _, dup := seen[word]; dup {
			continue
		}
		seen[word] = struct{}{}
		w.WriteString(word)
		w.WriteByte('\n')
		count++
		if *maxWords > 0 && count >= *maxWords {
			break
		}
		if count%500000 == 0 {
			fmt.Printf("[go-fetcher] %d unique words...\n", count)
		}
	}
	for e := range res.errs {
		fmt.Fprintln(os.Stderr, "[go-fetcher]", e)
	}
	fmt.Printf("[go-fetcher] done: %d unique words -> %s\n", count, *outPath)
}

func dirOf(path string) string {
	if i := strings.LastIndexByte(path, '/'); i > 0 {
		return path[:i]
	}
	return "."
}
