CC=x86_64-linux-gnu-g++
LD=x86_64-linux-gnu-g++
AR=x86_64-linux-gnu-ar
CFLAGS=-I. -I../inc -I../lib -march=native -Wall -Weffc++ -O2

OPTOBJ=.libs/cmdopt.o
LIBOPT=.libs/libcmdopt.a
QUOTEOBJ=.libs/quote.o
ORDEROBJ=.libs/order.o
TSIDXOBJ=.libs/tsidx.o
QUOTE=quote_test
ORDER=order_test
TSIDX=tsidx_test

default:$(QUOTE) $(ORDER) $(TSIDX)

.libs/%.o:%.c
	@echo CC $(<F)
	$(CC) $(CFLAGS) -c $< -o $@

$(LIBOPT):$(OPTOBJ)
	@echo AR libcmdopt
	$(AR) rc $@ $<

$(QUOTE):$(QUOTEOBJ) $(LIBOPT) $(LIBQLIBC)
	@echo LD $(QUOTE)
	$(LD) -o $@ $^

$(ORDER):$(ORDEROBJ) $(LIBOPT) $(LIBQLIBC)
	@echo LD $(ORDER)
	$(LD) -o $@ $^


$(TSIDX):$(TSIDXOBJ) $(LIBOPT) $(LIBQLIBC)
	@echo LD $(TSIDX)
	$(LD) -o $@ $^

clean:
	find .libs -type f |xargs rm
	rm $(QUOTE) $(ORDER) $(TSIDX)