ARG PYTHON_VERSION=3.10.7
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app
ENV PATH="/venv/bin:${PATH}" \
    VIRTUAL_ENV="/venv" \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1


FROM base AS builder
# set DEV environment variable to install development dependencies
ARG DEV
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# use virtualenv to leverage multi-stage builds
RUN python -m venv $VIRTUAL_ENV
RUN python -m pip install --upgrade pip
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

FROM base AS prod
COPY --from=builder /venv /venv
COPY ./src ./src
COPY ./data ./data
COPY ./experiments ./experiments
ENTRYPOINT ["python", "src/main.py"]

FROM base AS dev
COPY --from=builder /venv /venv
COPY . .
CMD ["bash"]
