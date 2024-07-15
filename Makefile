CC=gcc
CFLAGS=-O3 -march=native -mfpmath=sse -funroll-loops -lm
DEPS = clebsch_gordan.h e3nn.h tp.h

CFLAGS += -Wall -Wextra -Wpedantic \
		  -Wformat=2 -Wno-unused-parameter -Wshadow \
          -Wwrite-strings -Wstrict-prototypes -Wold-style-definition \
          -Wredundant-decls -Wnested-externs -Wmissing-include-dirs

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

example: clebsch_gordan.o e3nn.o example.o tp.o
	$(CC) -o $@ $^ $(CFLAGS)

benchmark_c.c: extra/benchmark_c_codegen.py extra/benchmark_python.py
	python extra/benchmark_c_codegen.py > benchmark_c.c

benchmark_c: clebsch_gordan.o e3nn.o benchmark_c.o tp.o
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: benchmark
benchmark: benchmark_c
	rm benchmark.txt
	./benchmark_c >> benchmark.txt
	python extra/benchmark_python.py >> benchmark.txt
	python extra/plot_benchmark.py

# -fPIC makes the code considerably slower, but is needed to call the c code
# from python; for this reason compiling separately
# Should also change L_MAX lower but for some reason -DL_MAX doesn't work
# so I typically change it manually to test
.PHONY: test
test:
	$(CC) -DL_MAX=6 -shared -o e3nn.so e3nn.c clebsch_gordan.h clebsch_gordan.c tp.h tp.c $(CFLAGS) -fPIC
	$(CC) -DL_MAX=6 -shared -o clebsch_gordan.so clebsch_gordan.c $(CFLAGS) -fPIC
	python -m pytest tests/

.PHONY: clean
clean:
	rm -f *.o *.so benchmark_c example benchmark_c.c
