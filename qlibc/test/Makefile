CC=x86_64-linux-gnu-g++
LD=x86_64-linux-gnu-g++
AR=x86_64-linux-gnu-ar
CFLAGS=-I. -I../inc -I../lib -march=native -Wall -Weffc++ -O2

OPTOBJ=.libs/cmdopt.o
LIBOPT=.libs/libcmdopt.a
LIB_DIR=../../build/temp.linux-x86_64-3.7/qlibc/lib
LIBQLIBC=$(LIB_DIR)/libqlibc.so
TSIDXOBJ=.libs/tsidx.o
TSIDX=tsidx

default: $(TSIDX)

.libs/%.o:%.c
	@echo CC $(<F)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIBQLIBC):
	@echo BUILD LIB
	cd ../../; make

$(LIBOPT):$(OPTOBJ)
	@echo AR libcmdopt
	$(AR) rc $@ $<

$(TSIDX):$(TSIDXOBJ) $(LIBOPT) $(LIBQLIBC)
	@echo LD $(TSIDX)
	$(LD) -L $(LIB_DIR) -lqlibc -o $@ $^

clean:
	find .libs -type f |xargs rm
	rm $(TSIDX)

libclean:
	cd ../../; make clean