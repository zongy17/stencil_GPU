
CC = nvcc 
DIAG = --ptxas-options=-v
OPT = -Xptxas -O3 -arch=compute_70 -code=compute_70
MACRO = #-DTILE_X=$(TX) -DTILE_Y=$(TY)
CFLAGS = $(OPT) $(MACRO)
LDFLAGS =
LDLIBS = $(LDFLAGS)

targets = benchmark-naive benchmark-optimized
objects = check.o benchmark.o stencil-naive.o stencil-optimized.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

benchmark-naive : check.o benchmark.o stencil-naive.o
	$(CC) -o $@ $^ $(LDLIBS)

benchmark-optimized : check.o benchmark.o stencil-optimized.o
	$(CC) -o $@ $^ $(LDLIBS)

%.o : %.cu common.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)