FROM python:3.12-slim

WORKDIR /app

# Install uv and git (needed for git dependencies)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY prompt.md .
COPY src/ src/

# Install dependencies (pondside comes from GitHub)
RUN uv pip install --system .

EXPOSE 8100

CMD ["uvicorn", "intro.app:app", "--host", "0.0.0.0", "--port", "8100"]
