.PHONY: build dist redist install clean

ROOT=$(PWD)
CC=x86_64-linux-gnu-g++
LD=x86_64-linux-gnu-g++
AR=x86_64-linux-gnu-ar
CYTHON=cython
CYTHON_FLAGS=--cplus
# CFLAGS=-pthread -DNDEBUG -D_DEBUG  -fwrapv -g  -Wall -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC
CFLAGS=-pthread -DNDEBUG  -fwrapv -O3  -Wall -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2 -fPIC
LDFLAGS=-pthread -shared -Wl,-O2 -Wl,-Bsymbolic-functions -Wl,-Bsymbolic-functions -Wl,-z,relro -Wl,-Bsymbolic-functions -Wl,-z,relro -fstack-protector --param=ssp-buffer-size=4 -Wformat -Werror=format-security -D_FORTIFY_SOURCE=2
LIB_QLIBC_FLAGS=-fPIC -DPIC -shared
TARGET=$(VIRTUAL_ENV)/lib/python$(PYTHON_VERSION)/site-packages/pyxqlib-0.0.1-py$(PYTHON_VERSION)-linux-x86_64.egg/pyxqlib.cpython-$(PYTHONVERSION)m-x86_64-linux-gnu.so

LIB_QLIBC_LIB=$(ROOT)/qlibc/lib
LIB_QLIBC_INCLUDE=$(ROOT)/qlibc/inc
LIB_QLIBC_OBJECT_DIR=$(ROOT)/build/temp.linux-x86_64-$(PYTHON_VERSION)/qlibc/lib
LIB_QLIBC_OBJECT:=$(LIB_QLIBC_OBJECT_DIR)/quote.o $(LIB_QLIBC_OBJECT_DIR)/order.o $(LIB_QLIBC_OBJECT_DIR)/tsidx.o
LIB_QLIBC_CFLAGS=$(CFLAGS) -DQLIBC_USE_SSE4 -msse4 -DQLIBC_USE_POPCNT -mpopcnt -I$(LIB_QLIBC_LIB) -I$(LIB_QLIBC_INCLUDE)
LIB_QLIBC_OUTPUT_DIR=$(ROOT)/build/temp.linux-x86_64-$(PYTHON_VERSION)
LIB_QLIBC_S=libqlibc.a
LIB_QLIBC_D=libqlibc.so

PYTHON_HEAD=$(VIRTUAL_ENV)/include
NUMPY_HEAD=$(VIRTUAL_ENV)/lib/python$(PYTHON_VERSION)/site-packages/numpy/core/include
EXT_CFLAGS=$(CFLAGS) -I$(LIB_QLIBC_INCLUDE) -I$(PYTHON_HEAD) -I $(NUMPY_HEAD)
EXT_SRC_DIR=$(ROOT)/src
EXT_OBJECT_DIR=$(ROOT)/build/temp.linux-x86_64-$(PYTHON_VERSION)/src
EXT_OBJECT:=$(EXT_OBJECT_DIR)/quote.o $(EXT_OBJECT_DIR)/order.o
EXT_OUTPUT_DIR=$(ROOT)/build/lib.linux-x86_64-$(PYTHON_VERSION)
EXT=pyxqlib.cpython-$(PYTHONVERSION)m-x86_64-linux-gnu.so

default:ext

$(LIB_QLIBC_OBJECT_DIR)/%.o:$(LIB_QLIBC_LIB)/%.cc
	@echo CC lib $(<F)
	$(CC) $(LIB_QLIBC_CFLAGS) -c $< -o $@

lib_qlibc:$(LIB_QLIBC_OBJECT)
	@echo AR lib
	$(AR) rc $(LIB_QLIBC_OUTPUT_DIR)/$(LIB_QLIBC_S) $(LIB_QLIBC_OBJECT)
	$(LD) $(LIB_QLIBC_FLAGS) $(LIB_QLIBC_OBJECT) -o $(LIB_QLIBC_OUTPUT_DIR)/$(LIB_QLIBC_D)

$(EXT_SRC_DIR)/quote.cpp:$(EXT_SRC_DIR)/quote.pxd
	@echo CYTHON $(<F)
	$(CYTHON) $< $(CYTHON_FLAGS) -o $@

$(EXT_SRC_DIR)/order.cpp:$(EXT_SRC_DIR)/order.pxd
	@echo CYTHON $(<F)
	$(CYTHON) $< $(CYTHON_FLAGS) -o $@

$(EXT_SRC_DIR)/pyxqlib.cpp:$(EXT_SRC_DIR)/pyxqlib.pyx
	@echo CYTHON $(<F)
	$(CYTHON) $< $(CYTHON_FLAGS) -o $@

$(EXT_OBJECT_DIR)/%.o:$(EXT_SRC_DIR)/%.cpp
	@echo CC ext $(<F)
	$(CC) $(EXT_CFLAGS) -c $< -o $@

ext:$(EXT_OBJECT) lib_qlibc
	@echo LD ext
	$(LD) $(LDFLAGS) $(EXT_OBJECT) -L$(LIB_QLIBC_OUTPUT_DIR) -lqlibc -lpthread  -o $(EXT_OUTPUT_DIR)/$(EXT)
	cp $(EXT_OUTPUT_DIR)/$(EXT) $(TARGET)

build:
	python ./setup.py build

dist:
	python ./setup.py sdist bdist_wheel

redist: clean dist

install:
	pip install .

clean:
	$(RM) -r build dist src/*.egg-info
	find . -name __pycache__ -exec rm -r {} +
	find ./build -type f |xargs rm