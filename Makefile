NIX     := $(shell which nix 2>/dev/null)
BINARY  := ./build/kepler
ARGS    ?=

.PHONY: setup build run clean

setup:
ifndef NIX
	@echo "==> Installing Nix (Determinate Systems)..."
	curl --proto '=https' --tlsv1.2 -sSf -L https://install.determinate.systems/nix | sh -s -- install
	@echo ""
	@echo "  Nix installed. Open a new terminal, then run: make build"
else
	@$(MAKE) build
endif

build:
	nix develop --command bash -c "cmake -B build -G Ninja && cmake --build build"

run: $(BINARY)
	nix develop --command $(BINARY) $(ARGS)

clean:
	rm -rf build

$(BINARY):
	@$(MAKE) build
