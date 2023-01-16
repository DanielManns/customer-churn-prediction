COMPONENT = backend
DOCKER_BUILD = dev
VENV = $(COMPONENT)/venv
PYTHON = $(VENV)/Scripts/python
REQUIREMENTS = $(COMPONENT)/requirements.txt
ENTRYPOINT = $(COMPONENT)/src/main.py
WSL = C:\Windows\sysnative\wsl.exe

run: $(VENV)/Scripts/activate
	$(PYTHON) $(ENTRYPOINT)

venv: $(VENV)/Scripts/activate

$(VENV)/Scripts/activate: $(REQUIREMENTS)
	python -m venv $(VENV)
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -r $(REQUIREMENTS)

#	. $(VENV)/Scripts/activate
	

run_docker:
	$(WSL) docker run -it --rm $(COMPONENT):$(DOCKER_BUILD)

build_run_docker: build_docker
	$(WSL) docker run -it --rm $(COMPONENT):$(DOCKER_BUILD)

build_docker: $(COMPONENT)/Dockerfile
	$(WSL) docker build --tag $(COMPONENT):$(DOCKER_BUILD) --target $(DOCKER_BUILD) $(COMPONENT)

clean:
	$(WSL) rm -rf $(VENV)