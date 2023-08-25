CXX ?= g++
OUT ?= build/libflash_attention.so
OBJDIR ?= $(dir $(OUT))obj

TESTS ?= build/tests
TESTSOBJDIR ?= $(dir $(TESTS))obj/test

CXXFLAGS = -Wall -Wextra -Werror -Wno-unused-function -std=c++14 -O2 -g -fPIC -fdiagnostics-color=always -DONNX_NAMESPACE=onnx
LIBS = -lpopart -lpopnn -lpoplin -lpopops -lpoprand -lpoputil -lpoplar

OBJECTS = $(OBJDIR)/vanilla_attention.o $(OBJDIR)/flash_attention_qkv_packed.o
TESTOBJECTS = $(TESTSOBJDIR)/catch.o $(TESTSOBJDIR)/test_flash_attention.o

all: $(OUT) $(TESTS)

$(OBJECTS): $(OBJDIR)/%.o: flash_attention_ipu/cpp/%.cpp
	@mkdir -p $(@D)
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OUT): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -shared $^ -o $@ -Wl,--no-undefined $(LIBS)

$(TESTSOBJDIR)/catch.o: third_party/catch2/extras/catch_amalgamated.cpp
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -Ithird_party/catch2/extras -c third_party/catch2/extras/catch_amalgamated.cpp -o $(TESTSOBJDIR)/catch.o

$(TESTSOBJDIR)/test_flash_attention.o: tests/cpp/test_flash_attention.cpp
	$(CXX) $(CXXFLAGS) -Iflash_attention_ipu/cpp -Ithird_party/catch2/extras -c tests/cpp/test_flash_attention.cpp -o $(TESTSOBJDIR)/test_flash_attention.o
	
$(TESTS): $(OBJECTS) $(TESTOBJECTS)
	$(CXX) $(CXXFLAGS) $(OBJECTS) $(TESTOBJECTS) -o $(TESTS) -Wl,--no-undefined $(LIBS)

clean:
	rm -rf $(OBJDIR)
