.PHONY: format yapf isort

format: yapf isort

yapf:
	yapf -i -r .

isort:
	isort .
