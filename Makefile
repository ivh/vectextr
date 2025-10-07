CC = gcc
CXX = g++
CFLAGS = -Wall -Wextra -g -O2
CXXFLAGS = -Wall -Wextra -g -O2 -std=c++11
LDFLAGS = -lm

# Directories
SRC_DIR = .
TEST_DIR = .
BUILD_DIR = build
PYTHON_DIR = .
LIB_DIR = $(BUILD_DIR)/lib

# Source files
SRC_FILES = $(SRC_DIR)/extract.c
TEST_SRC = $(TEST_DIR)/extract_test.c
TEST_TARGET = $(BUILD_DIR)/extract_test

# Python wrapper
PYTHON_WRAPPER = $(PYTHON_DIR)/extract_wrapper.cpp

# Library target
LIB_TARGET = $(LIB_DIR)/libcharslit.a
OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SRC_FILES))
TEST_OBJECTS = $(patsubst $(TEST_DIR)/%.c,$(BUILD_DIR)/%.o,$(TEST_SRC))

# Debug flags
DEBUG_FLAGS = -DDEBUG -fsanitize=address

# Default target
all: directories $(LIB_TARGET) $(TEST_TARGET)

# Create build directories
directories:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(LIB_DIR)

# Compile static library
$(LIB_TARGET): $(OBJECTS)
	@echo "Creating library $@"
	@ar rcs $@ $^

# Compile object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# Compile test program
$(TEST_TARGET): $(TEST_OBJECTS) $(LIB_TARGET)
	@echo "Linking test program $@"
	@$(CC) $(CFLAGS) -o $@ $(TEST_OBJECTS) -L$(LIB_DIR) -lcharslit $(LDFLAGS)

# Compile test object files
$(BUILD_DIR)/%.o: $(TEST_DIR)/%.c
	$(CC) $(CFLAGS) -c $< -o $@

# Run tests
test: $(TEST_TARGET)
	@echo "Running tests..."
	@$(TEST_TARGET)

# Debug build with sanitizers
debug: CFLAGS += $(DEBUG_FLAGS)
debug: CXXFLAGS += $(DEBUG_FLAGS)
debug: clean all test

# Python wrapper build
python-wrapper:
	@echo "Building Python wrapper..."
	@cd $(PYTHON_DIR) && \
	if [ -d ".venv" ]; then \
		echo "Using existing virtual environment"; \
	else \
		echo "Creating new virtual environment"; \
		uv venv; \
	fi && \
	source .venv/bin/activate && \
	uv pip install -e . && \
	uv pip install pytest

# Python test
python-test: python-wrapper
	@echo "Running Python tests..."
	@cd $(PYTHON_DIR) && source .venv/bin/activate && uv run -m pytest -v tests/

# Clean build files
clean:
	@echo "Cleaning build files..."
	@rm -rf $(BUILD_DIR)
	@rm -rf __pycache__ .pytest_cache
	@rm -rf *.egg-info
	@rm -rf _skbuild
	@find . -name "*.pyc" -delete

# Very clean (including virtual environment)
distclean: clean
	@echo "Cleaning all generated files..."
	@rm -rf .venv

.PHONY: all clean test debug python-wrapper python-test distclean directories
