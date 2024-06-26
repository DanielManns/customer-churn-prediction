ARG PYTHON_VERSION=3.10.7
FROM python:${PYTHON_VERSION}-slim AS base

WORKDIR /app
ENV PATH="/venv/bin:/usr/bin/dot:${PATH}" \
    VIRTUAL_ENV="/venv/" \
    PYTHONFAULTHANDLER=1 \
    PYTHONUNBUFFERED=1


FROM base AS builder
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# use virtualenv to leverage multi-stage builds
RUN python -m venv $VIRTUAL_ENV
RUN python -m pip install --upgrade pip
COPY ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY . .

FROM base AS dev
RUN apt-get update && apt-get install -y graphviz
COPY --from=builder /venv /venv
COPY . .
EXPOSE 7860
CMD ["bash"]

FROM base AS prod
RUN apt-get update && apt-get install -y graphviz
COPY --from=builder /venv /venv
COPY . .
EXPOSE 7860
ENTRYPOINT ["python", "src/main.py"]


