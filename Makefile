
ROOT_DIR:=$(PWD)
GLPK_DIST ?= glpk-4.47.tar.gz  
PYTHON_GLPK_DIST ?= python-glpk-0.4.43.tar.gz

GLPK_SRC_DIR := glpk-4.47
PYTHON_GLPK_SRC_DIR := python-glpk-0.4.43
PYTHON_INC=$(shell python -c "from distutils.sysconfig import get_python_inc; print get_python_inc(plat_specific=0)")
PYTHON_LIB=$(shell python -c "from distutils.sysconfig import get_python_lib; print get_python_lib(plat_specific=0)")
CPATH := "$(PYTHON_INC):$(PYTHON_LIB)/../config"
LIBRARY_PATH := $(ROOT_DIR)/build/glpk_build/lib

all: glpk python-glpk glass

glpk:
	(cd lib \
	&& tar xvzf $(GLPK_DIST) \
	&& cd $(GLPK_SRC_DIR)\
	&& ./configure --prefix=$(ROOT_DIR)/build/glpk_build \
	&& make \
	&& make install)


python-glpk:
	echo $(CPATH)

	(cd lib \
	&& tar xvzf $(PYTHON_GLPK_DIST) \
	&& cd $(PYTHON_GLPK_SRC_DIR)/src \
	&& export CPATH=$(CPATH) \
	&& export LIBRARY_PATH=$(LIBRARY_PATH) \
	&& make -C swig all \
	&& cp swig/glpkpi.py python/glpkpi.py \
	&& python setup.py build --build-base=$(ROOT_DIR)/build/python-glpk)

glass:
	python setup.py build
