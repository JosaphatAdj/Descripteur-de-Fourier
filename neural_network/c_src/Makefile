# Makefile pour Neural Network avec OpenBLAS

CC = gcc
CFLAGS = -Wall -O3 -march=native -Iinclude
LDFLAGS = -lopenblas -lm

# Dossiers
SRC_DIR = .
OBJ_DIR = obj
LIB_DIR = lib
INC_DIR = include

# Fichiers source
SOURCES = matrix.c nn.c mnist.c
OBJECTS = $(SOURCES:%.c=$(OBJ_DIR)/%.o)

# Bibliothèque
LIBRARY = $(LIB_DIR)/libnn.so

# Cible par défaut
all: $(LIBRARY)

# Créer dossiers
$(OBJ_DIR):
	@mkdir -p $(OBJ_DIR)

$(LIB_DIR):
	@mkdir -p $(LIB_DIR)

# Compilation objets
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c | $(OBJ_DIR)
	@echo "  CC $<"
	@$(CC) $(CFLAGS) -fPIC -c $< -o $@

# Lier bibliothèque partagée
$(LIBRARY): $(OBJECTS) | $(LIB_DIR)
	@echo "  LD $(LIBRARY)"
	@$(CC) -shared $(OBJECTS) $(LDFLAGS) -o $(LIBRARY)
	@echo "✓ Bibliothèque compilée : $(LIBRARY)"

# Nettoyage
clean:
	@rm -rf $(OBJ_DIR) $(LIB_DIR)
	@echo "✓ Nettoyage effectué"

# Programme de test
test: $(LIBRARY)
	@$(CC) $(CFLAGS) test_nn.c -L$(LIB_DIR) -lnn $(LDFLAGS) -o test_nn
	@echo "✓ Test compilé"

.PHONY: all clean test
