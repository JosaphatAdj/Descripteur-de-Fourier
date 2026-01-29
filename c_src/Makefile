# Makefile pour la bibliothèque C avec OpenBLAS
# ==============================================

# Compilateur et flags
CC = gcc
CFLAGS = -Wall -Wextra -O3 -fPIC -march=native -fopenmp
DEBUG_FLAGS = -g -O0 -DDEBUG

# Bibliothèques OpenBLAS
LDFLAGS = -lopenblas -lm -fopenmp

# Répertoires
SRC_DIR = .
INC_DIR = include
OBJ_DIR = obj
LIB_DIR = ../python/lib

# Fichiers sources
SOURCES = fourier_naive.c fourier_openblas.c contour.c utils.c batch.c
OBJECTS = $(patsubst %.c,$(OBJ_DIR)/%.o,$(SOURCES))

# Bibliothèque partagée
SHARED_LIB = $(LIB_DIR)/libfourier.so

# Exécutable de test
TEST_EXEC = test_fourier

# ==============================================
# RÈGLES PRINCIPALES
# ==============================================

.PHONY: all clean test debug dirs

all: dirs $(SHARED_LIB) $(TEST_EXEC)
	@echo "✓ Compilation terminée: $(SHARED_LIB)"

dirs:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(LIB_DIR)

# Bibliothèque partagée pour Python
$(SHARED_LIB): $(OBJECTS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)
	@echo "✓ Bibliothèque partagée créée: $@"

# Exécutable de test
$(TEST_EXEC): $(OBJECTS) test_main.c
	$(CC) $(CFLAGS) -o $@ test_main.c $(OBJECTS) $(LDFLAGS)
	@echo "✓ Exécutable de test créé: $@"

# Compilation des objets
$(OBJ_DIR)/%.o: %.c $(INC_DIR)/fourier.h
	$(CC) $(CFLAGS) -I$(INC_DIR) -c $< -o $@

# ==============================================
# DEBUG
# ==============================================

debug: CFLAGS += $(DEBUG_FLAGS)
debug: clean all
	@echo "✓ Compilation en mode debug"

# ==============================================
# TESTS
# ==============================================

test: $(TEST_EXEC)
	@echo "=== Exécution des tests C ==="
	./$(TEST_EXEC)

# Vérification mémoire avec Valgrind
valgrind: $(TEST_EXEC)
	valgrind --leak-check=full --show-leak-kinds=all ./$(TEST_EXEC)

# ==============================================
# NETTOYAGE
# ==============================================

clean:
	rm -rf $(OBJ_DIR)
	rm -f $(SHARED_LIB)
	rm -f $(TEST_EXEC)
	@echo "✓ Nettoyage terminé"

# ==============================================
# INFORMATIONS
# ==============================================

info:
	@echo "=== Configuration du Build ==="
	@echo "Compilateur: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "LDFLAGS: $(LDFLAGS)"
	@echo "Sources: $(SOURCES)"
	@echo ""
	@echo "=== Vérification OpenBLAS ==="
	@pkg-config --cflags --libs openblas 2>/dev/null || echo "OpenBLAS: configuration manuelle"
