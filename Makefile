VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
LDFLAGS := -s -w -X github.com/thisisadityapatel/kepler/internal/cli.Version=$(VERSION)

.PHONY: build test race vet cross clean

build:
	CGO_ENABLED=0 go build -ldflags '$(LDFLAGS)' -o bin/kepler ./cmd/kepler

test:
	go test ./...

race:
	go test -race ./...

vet:
	gofmt -l . | (! grep .) && go vet ./...

# v1 ships darwin/arm64; linux and windows must keep compiling so ports stay additive.
cross:
	GOOS=darwin GOARCH=arm64 go build ./...
	GOOS=linux GOARCH=amd64 go build ./...
	GOOS=windows GOARCH=amd64 go build ./...

clean:
	rm -rf bin
