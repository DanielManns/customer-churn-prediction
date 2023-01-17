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
	

docker_run:
	$(WSL) docker run -it --rm -p 8000:8000 $(COMPONENT):$(DOCKER_BUILD)

docker_build_run: docker_build docker_run

docker_build: $(COMPONENT)/Dockerfile
	$(WSL) docker build --tag $(COMPONENT):$(DOCKER_BUILD) --target $(DOCKER_BUILD) $(COMPONENT)

clean:
	$(WSL) rm -rf $(VENV)