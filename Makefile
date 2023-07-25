CXX ?= c++

all: serialised_attention

serialised_attention: serialised_attention.cpp 
		$(CXX) -std=c++14 serialised_attention.cpp -lpopnn -lpoplin -lpopops -lpoprand -lpoputil -lpoplar -o serialised_attention

clean:
		rm -f serialised_attention