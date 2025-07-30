# Compiler
CXX := g++

# Compiler flags
CXXFLAGS := -std=c++17 -Wall -g -O0 -Iinclude -I$(shell brew --prefix armadillo)/include

# Linker flags
LDFLAGS := -L$(shell brew --prefix armadillo)/lib -larmadillo

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

# Library name and paths
LIB_TARGET := $(BINDIR)/libnn_api.dylib
OBJDIR_LIB := obj-lib
LIB_SOURCES := $(filter-out $(SRCDIR)/main.cpp, $(SOURCES))  # Exclude main.cpp
LIB_OBJECTS := $(patsubst $(SRCDIR)/%.cpp,$(OBJDIR_LIB)/%.o,$(LIB_SOURCES))
LIB_CXXFLAGS := $(CXXFLAGS) -fPIC  # Add Position-Independent Code flag



# Default target
all: $(TARGET)
lib: $(LIB_TARGET)   # Build shared library

# Link object files to create the final executable
$(TARGET): $(OBJECTS)
	@mkdir -p $(BINDIR)
	$(CXX) $(OBJECTS) -o $(TARGET) $(LDFLAGS)

# Compile source files into object files
$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# --- Shared Library Build ---
$(LIB_TARGET): $(LIB_OBJECTS)
	@mkdir -p $(BINDIR)
	$(CXX) -dynamiclib -o $@ $(LIB_OBJECTS) $(LDFLAGS) -Wl,-install_name,@rpath/libnn_api.dylib
	@mkdir -p ../game/lib
	cp $@ ../game/lib/

$(OBJDIR_LIB)/%.o: $(SRCDIR)/%.cpp
	@mkdir -p $(OBJDIR_LIB)
	$(CXX) $(LIB_CXXFLAGS) -c $< -o $@

# Clean both builds
clean:
	rm -rf $(OBJDIR) $(OBJDIR_LIB) $(BINDIR)

.PHONY: all lib clean

