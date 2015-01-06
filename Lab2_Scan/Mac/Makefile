CXX = g++
CXXFLAGS = -c -std=c++11 -Wall -W
LDFLAGS = -framework OpenCL
INCPATH = -I.
SOURCES = main.cpp 
OBJECTS = $(SOURCES:.cpp=.o)
EXECUTABLE = scan

all: $(SOURCES) $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(CXX) $(LDFLAGS) $(OBJECTS) -o $@

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCPATH) $< -o $@

clean:
	rm -f *.o scan
