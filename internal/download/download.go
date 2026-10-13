// Package download fetches files over HTTP with resume and integrity verification.
package download

import (
	"context"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"hash"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"
)

// Digest identifies expected content. Exactly one field should be set.
type Digest struct {
	SHA256 string // file sha256 (Hugging Face LFS oid, GitHub release digest)
	GitOID string // git blob sha1 (Hugging Face non-LFS files)
}

func (d Digest) key() string {
	if d.SHA256 != "" {
		return "sha256-" + d.SHA256
	}
	return "git-" + d.GitOID
}

// Request is one file to fetch.
type Request struct {
	URL    string
	Dest   string // final path; written only after verification succeeds
	Size   int64  // expected size; 0 = unknown
	Digest Digest
	Header http.Header // extra headers (e.g. Authorization for gated Hugging Face repos)
}

// Progress reports bytes done of total (total may be 0 if unknown).
type Progress func(done, total int64)

// Client downloads into a partials directory and resumes from it.
type Client struct {
	HTTP     *http.Client
	PartsDir string // where *.part files live (inside KEPLER_HOME)
	Retries  int
}

// New returns a client with sensible defaults.
func New(partsDir string) *Client {
	return &Client{HTTP: &http.Client{Timeout: 0}, PartsDir: partsDir, Retries: 3}
}

// ErrDigestMismatch means the downloaded bytes do not match the expected digest.
var ErrDigestMismatch = errors.New("digest mismatch")

// PartPath is where the in-progress bytes for r live. Callers record it in the ledger
// before calling Fetch so a crash mid-download is cleaned up.
func (c *Client) PartPath(r Request) string {
	return filepath.Join(c.PartsDir, r.Digest.key()+".part")
}

// Fetch downloads r.URL to r.Dest, resuming a previous partial download, verifying size
// and digest, then renaming into place. On digest mismatch the partial is discarded
// and the download retried from scratch once.
func (c *Client) Fetch(ctx context.Context, r Request, progress Progress) error {
	if r.Digest.SHA256 == "" && r.Digest.GitOID == "" {
		return fmt.Errorf("%s: refusing to download without a digest", r.URL)
	}
	if ok, _ := verifyFile(r.Dest, r); ok {
		if progress != nil {
			progress(r.Size, r.Size)
		}
		return nil
	}
	if err := os.MkdirAll(c.PartsDir, 0o755); err != nil {
		return err
	}
	part := c.PartPath(r)

	var lastErr error
	restarted := false
	for attempt := 0; attempt <= c.Retries; attempt++ {
		if attempt > 0 {
			select {
			case <-ctx.Done():
				return ctx.Err()
			case <-time.After(time.Duration(attempt) * time.Second):
			}
		}
		lastErr = c.fetchOnce(ctx, r, part, progress)
		if lastErr == nil {
			break
		}
		if errors.Is(lastErr, ErrDigestMismatch) {
			os.Remove(part)
			if restarted {
				return lastErr
			}
			restarted = true
			continue
		}
		if ctx.Err() != nil {
			return ctx.Err()
		}
	}
	if lastErr != nil {
		return lastErr
	}
	if err := os.MkdirAll(filepath.Dir(r.Dest), 0o755); err != nil {
		return err
	}
	return os.Rename(part, r.Dest)
}

func (c *Client) fetchOnce(ctx context.Context, r Request, part string, progress Progress) error {
	var offset int64
	if st, err := os.Stat(part); err == nil {
		offset = st.Size()
	}
	if r.Size > 0 && offset > r.Size {
		os.Remove(part)
		offset = 0
	}
	if r.Size > 0 && offset == r.Size {
		return verifyPart(part, r)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, r.URL, nil)
	if err != nil {
		return err
	}
	for k, vs := range r.Header {
		for _, v := range vs {
			req.Header.Add(k, v)
		}
	}
	if offset > 0 {
		req.Header.Set("Range", "bytes="+strconv.FormatInt(offset, 10)+"-")
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	flags := os.O_CREATE | os.O_WRONLY
	switch {
	case resp.StatusCode == http.StatusPartialContent && offset > 0:
		flags |= os.O_APPEND
	case resp.StatusCode == http.StatusOK:
		offset = 0 // server ignored Range; start over
		flags |= os.O_TRUNC
	case resp.StatusCode == http.StatusRequestedRangeNotSatisfiable:
		os.Remove(part)
		return fmt.Errorf("%s: range not satisfiable; restarting", r.URL)
	case resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden:
		return fmt.Errorf("%s: HTTP %d (gated or private; set HF_TOKEN)", r.URL, resp.StatusCode)
	default:
		return fmt.Errorf("%s: HTTP %d", r.URL, resp.StatusCode)
	}

	f, err := os.OpenFile(part, flags, 0o644)
	if err != nil {
		return err
	}
	total := r.Size
	if total == 0 && resp.ContentLength > 0 {
		total = offset + resp.ContentLength
	}
	w := &progressWriter{w: f, done: offset, total: total, fn: progress}
	_, copyErr := io.Copy(w, resp.Body)
	closeErr := f.Close()
	if copyErr != nil {
		return copyErr
	}
	if closeErr != nil {
		return closeErr
	}
	return verifyPart(part, r)
}

func verifyPart(part string, r Request) error {
	ok, err := verifyFile(part, r)
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("%s: %w", r.URL, ErrDigestMismatch)
	}
	return nil
}

// verifyFile reports whether path exists with the expected size and digest.
func verifyFile(path string, r Request) (bool, error) {
	st, err := os.Stat(path)
	if err != nil {
		return false, err
	}
	if r.Size > 0 && st.Size() != r.Size {
		return false, nil
	}
	f, err := os.Open(path)
	if err != nil {
		return false, err
	}
	defer f.Close()
	var h hash.Hash
	var want string
	if r.Digest.SHA256 != "" {
		h, want = sha256.New(), r.Digest.SHA256
	} else {
		h, want = sha1.New(), r.Digest.GitOID
		fmt.Fprintf(h, "blob %d\x00", st.Size())
	}
	if _, err := io.Copy(h, f); err != nil {
		return false, err
	}
	return hex.EncodeToString(h.Sum(nil)) == want, nil
}

type progressWriter struct {
	w           io.Writer
	done, total int64
	fn          Progress
	last        time.Time
}

func (p *progressWriter) Write(b []byte) (int, error) {
	n, err := p.w.Write(b)
	p.done += int64(n)
	if p.fn != nil && (time.Since(p.last) > 200*time.Millisecond || (p.total > 0 && p.done >= p.total)) {
		p.fn(p.done, p.total)
		p.last = time.Now()
	}
	return n, err
}
