.PHONY: build dist redist install clean

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
