CC ?= gcc

CFLAGS ?= -std=gnu99 -O2 -g -Wall -Wextra
VALGRIND_CFLAGS = -I. -std=gnu99 -O2 -g -Wall -Wno-unknown-pragmas
CPPFLAGS ?= -I.
LDFLAGS = -lm

OPENMP_FLAG = -fopenmp
COMPILER_SUPPORTS_OPENMP := $(shell echo | $(CC) $(OPENMP_FLAG) -E - >/dev/null 2>&1 && echo YES || echo NO)

ifeq ($(COMPILER_SUPPORTS_OPENMP), YES)
    CFLAGS += $(OPENMP_FLAG)
    LDFLAGS += $(OPENMP_FLAG)
endif

COMMON_HEADERS = BHaH_defines.h BHaH_function_prototypes.h NRPYELL_binary_format.h interpolation.h interpolation_lagrange_uniform.h

READER_SRC_FILES = \
    interpolation_3d_general__uniform_src_grid.c \
    Cart_to_xx_and_nearest_i0i1i2.c \
    SinhSymTP/Cart_to_xx_and_nearest_i0i1i2__rfm__SinhSymTP.c \
    read_NRPYELL_binary.c \
    main.c

READER_OBJ_FILES = $(READER_SRC_FILES:.c=.o)

all: nrpyell_reader

%.o: %.c $(COMMON_HEADERS)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

nrpyell_reader: $(READER_OBJ_FILES)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

valgrind: clean
	$(MAKE) CFLAGS="$(VALGRIND_CFLAGS)" all
	valgrind --track-origins=yes --leak-check=full -s ./nrpyell_reader

clean:
	$(RM) *.o */*.o *~ */*~ nrpyell_reader

.PHONY: all clean valgrind
