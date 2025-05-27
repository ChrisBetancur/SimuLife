# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -g -O0 -std=c++17 -Wall -Iinclude \
            -I$(shell brew --prefix armadillo)/include \
            -I$(shell brew --prefix sdl2)/include/SDL2 \
            -I$(shell brew --prefix sdl2_ttf)/include/SDL2

# Linker flags
LDFLAGS := -L$(shell brew --prefix armadillo)/lib -larmadillo \
		   -L$(shell brew --prefix sdl2)/lib -lSDL2 \
           -L$(shell brew --prefix sdl2_ttf)/lib -lSDL2_ttf \
           -Llib -lnn_api -Wl,-rpath,@loader_path/../lib

# Directories
SRCDIR := src
OBJDIR := obj
BINDIR := bin

# Target executable name
TARGET := $(BINDIR)/life

# Find all source files in src/
SOURCES := $(wildcard $(SRCDIR)/*.cpp)

# Create corresponding object files in obj/
OBJECTS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR)/%.o,$(SOURCES))

# Default target
all: $(TARGET)

# Link object files to create the final executable
$(TARGET): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS) -lnn_api

# Compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean up generated files
clean:
	rm -rf $(OBJDIR) $(BINDIR)

.PHONY: all clean

