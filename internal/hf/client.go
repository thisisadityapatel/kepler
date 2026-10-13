package hf

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"
)

// DefaultEndpoint is the Hugging Face base URL.
const DefaultEndpoint = "https://huggingface.co"

// Client talks to the Hugging Face Hub API.
type Client struct {
	Endpoint string
	Token    string
	HTTP     *http.Client
}

// NewClient builds a client from HF_ENDPOINT / HF_TOKEN (endpoint argument wins if set).
func NewClient(endpoint string) *Client {
	if endpoint == "" {
		endpoint = os.Getenv("HF_ENDPOINT")
	}
	if endpoint == "" {
		endpoint = DefaultEndpoint
	}
	return &Client{
		Endpoint: strings.TrimRight(endpoint, "/"),
		Token:    os.Getenv("HF_TOKEN"),
		HTTP:     &http.Client{Timeout: 30 * time.Second},
	}
}

// ErrNotFound means the repo, revision, or file does not exist (or is private).
var ErrNotFound = errors.New("not found on Hugging Face")

// ErrGated means the repo needs an accepted license / token.
var ErrGated = errors.New("gated or private repository")

// ModelInfo is the subset of /api/models/{repo} Kepler uses.
type ModelInfo struct {
	ID          string   `json:"id"`
	SHA         string   `json:"sha"`
	Tags        []string `json:"tags"`
	Gated       any      `json:"gated"` // false | "auto" | "manual"
	Private     bool     `json:"private"`
	LibraryName string   `json:"library_name"`
	PipelineTag string   `json:"pipeline_tag"`
	CardData    struct {
		BaseModel any `json:"base_model"` // string or []string
	} `json:"cardData"`
}

// IsGated reports whether the repo requires accepting terms.
func (m ModelInfo) IsGated() bool {
	switch g := m.Gated.(type) {
	case bool:
		return g
	case string:
		return g != "" && g != "false"
	}
	return false
}

// TreeEntry is one item from /api/models/{repo}/tree/{rev}.
type TreeEntry struct {
	Type string `json:"type"` // file | directory
	Path string `json:"path"`
	Size int64  `json:"size"`
	OID  string `json:"oid"` // git blob sha1
	LFS  *struct {
		OID  string `json:"oid"` // sha256
		Size int64  `json:"size"`
	} `json:"lfs"`
}

// File is a repo file with integrity info.
type File struct {
	Path   string `json:"path"`
	Size   int64  `json:"size_bytes"`
	SHA256 string `json:"sha256,omitempty"`  // LFS files
	GitOID string `json:"git_oid,omitempty"` // non-LFS files
}

func (c *Client) get(ctx context.Context, path string, out any) (http.Header, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.Endpoint+path, nil)
	if err != nil {
		return nil, err
	}
	c.authorize(req)
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, fmt.Errorf("Hugging Face API: %w", err)
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusUnauthorized, http.StatusForbidden:
		return nil, ErrGated
	case http.StatusNotFound:
		return nil, ErrNotFound
	default:
		b, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("Hugging Face API %s: HTTP %d: %s", path, resp.StatusCode, strings.TrimSpace(string(b)))
	}
	return resp.Header, json.NewDecoder(resp.Body).Decode(out)
}

func (c *Client) authorize(req *http.Request) {
	if c.Token != "" {
		req.Header.Set("Authorization", "Bearer "+c.Token)
	}
}

// AuthHeader returns headers for file downloads.
func (c *Client) AuthHeader() http.Header {
	h := http.Header{}
	if c.Token != "" {
		h.Set("Authorization", "Bearer "+c.Token)
	}
	return h
}

// Info fetches model metadata at revision ("" = main).
func (c *Client) Info(ctx context.Context, repo, revision string) (ModelInfo, error) {
	path := "/api/models/" + repo
	if revision != "" {
		path += "/revision/" + url.PathEscape(revision)
	}
	var m ModelInfo
	_, err := c.get(ctx, path, &m)
	if err != nil {
		return m, fmt.Errorf("%s: %w", repo, err)
	}
	if m.SHA == "" {
		return m, fmt.Errorf("%s: Hugging Face returned no commit sha", repo)
	}
	return m, nil
}

var nextLinkRe = regexp.MustCompile(`<([^>]+)>;\s*rel="next"`)

// Files lists every file in the repo at a pinned commit, following pagination.
func (c *Client) Files(ctx context.Context, repo, sha string) ([]File, error) {
	path := "/api/models/" + repo + "/tree/" + sha + "?recursive=true"
	var out []File
	for page := 0; path != "" && page < 100; page++ {
		var entries []TreeEntry
		h, err := c.get(ctx, path, &entries)
		if err != nil {
			return nil, fmt.Errorf("%s: list files: %w", repo, err)
		}
		for _, e := range entries {
			if e.Type != "file" {
				continue
			}
			f := File{Path: e.Path, Size: e.Size}
			if e.LFS != nil {
				f.SHA256 = e.LFS.OID
				if e.LFS.Size > 0 {
					f.Size = e.LFS.Size
				}
			} else {
				f.GitOID = e.OID
			}
			out = append(out, f)
		}
		path = ""
		if m := nextLinkRe.FindStringSubmatch(h.Get("Link")); m != nil {
			path = strings.TrimPrefix(m[1], c.Endpoint)
		}
	}
	return out, nil
}

// FileURL is the download URL of a file at a pinned commit.
func (c *Client) FileURL(repo, sha, path string) string {
	return c.Endpoint + "/" + repo + "/resolve/" + sha + "/" + path
}

// ReadRange returns a reader for bytes [offset, ∞) of a file; the caller closes it.
func (c *Client) ReadRange(ctx context.Context, repo, sha, path string, offset int64) (io.ReadCloser, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.FileURL(repo, sha, path), nil)
	if err != nil {
		return nil, err
	}
	c.authorize(req)
	req.Header.Set("Range", fmt.Sprintf("bytes=%d-", offset))
	client := *c.HTTP
	client.Timeout = 0 // streaming; the caller's context bounds it
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	switch resp.StatusCode {
	case http.StatusOK, http.StatusPartialContent:
		return resp.Body, nil
	case http.StatusUnauthorized, http.StatusForbidden:
		resp.Body.Close()
		return nil, fmt.Errorf("%s/%s: %w", repo, path, ErrGated)
	default:
		resp.Body.Close()
		return nil, fmt.Errorf("%s/%s: HTTP %d", repo, path, resp.StatusCode)
	}
}

// GetJSON downloads and decodes a small JSON file from the repo (e.g. config.json).
func (c *Client) GetJSON(ctx context.Context, repo, sha, path string, out any) error {
	rc, err := c.ReadRange(ctx, repo, sha, path, 0)
	if err != nil {
		return err
	}
	defer rc.Close()
	return json.NewDecoder(io.LimitReader(rc, 8<<20)).Decode(out)
}
