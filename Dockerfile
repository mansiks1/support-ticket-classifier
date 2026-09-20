FROM python:3.12-slim@sha256:2f17fc044b579bab302c2e8054d3a686e2cb9a83de48e70534b94cd8ebbe06a9
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
WORKDIR /app
COPY requirements.lock pyproject.toml ./
COPY requirements-transformer.lock ./
RUN pip install --no-cache-dir -r requirements.lock
ARG WITH_TRANSFORMER=0
RUN if [ "$WITH_TRANSFORMER" = "1" ]; then pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements-transformer.lock; fi
COPY src ./src
COPY api ./api
RUN pip install --no-deps . && useradd --uid 10001 --create-home appuser && mkdir -p /app/models
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
