.PHONY: build build-c run benchmark test clean help docker-build docker-run

# Configuration Docker
DOCKER_COMPOSE = docker-compose

# ============================================================
# AIDE
# ============================================================
help:
	@echo "=== Descripteurs de Fourier avec OpenBLAS (C + Python) ==="
	@echo ""
	@echo "Architecture:"
	@echo "  - C/OpenBLAS: Logique des algorithmes (c_src/)"
	@echo "  - Python: Wrappers et visualisation (python/)"
	@echo ""
	@echo "Commandes Docker:"
	@echo "  make docker-build   - Construire l'image Docker"
	@echo "  make docker-run     - Exécuter le programme principal"
	@echo "  make docker-shell   - Ouvrir un shell dans le container"
	@echo "  make docker-test    - Exécuter les tests"
	@echo ""
	@echo "Commandes locales (Linux avec OpenBLAS installé):"
	@echo "  make build-c        - Compiler le code C"
	@echo "  make test-c         - Tester le code C"
	@echo "  make run            - Exécuter le programme"
	@echo "  make benchmark      - Lancer les benchmarks"
	@echo "  make clean          - Nettoyer"
	@echo ""

# ============================================================
# DOCKER
# ============================================================
docker-build:
	$(DOCKER_COMPOSE) build

docker-run:
	$(DOCKER_COMPOSE) up fourier

docker-shell:
	$(DOCKER_COMPOSE) run --rm fourier /bin/bash

docker-test:
	$(DOCKER_COMPOSE) run --rm fourier bash -c "cd c_src && make test && cd .. && pytest python/tests/ -v"

docker-benchmark:
	$(DOCKER_COMPOSE) run --rm fourier python python/benchmarks/run_benchmarks.py

# ============================================================
# LOCAL (Linux uniquement)
# ============================================================
build-c:
	cd c_src && make all

test-c:
	cd c_src && make test

run:
	python python/main.py

benchmark:
	python python/benchmarks/run_benchmarks.py

clean:
	cd c_src && make clean
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -f python/lib/*.so 2>/dev/null || true
