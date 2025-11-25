FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Change the working directory to the `workspace` directory
WORKDIR /workspace

ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin
ENV UV_PYTHON=3.12

RUN --mount=type=cache,target=/root/.cache/uv \
    uv python install 3.12

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

ENV PATH="/root/.local/share/uv/python/latest/bin:$PATH"

ENV HF_HOME=/root/.cache/huggingface