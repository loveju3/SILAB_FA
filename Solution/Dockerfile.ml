FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
COPY Solution/requirements-ml.txt /app/requirements-ml.txt
RUN pip install --no-cache-dir -r /app/requirements-ml.txt
COPY data/devices.json data/error_codes.json /app/data/
COPY Solution/triage/ /app/Solution/triage/
COPY Solution/config/ /app/Solution/config/
COPY Solution/models/ /app/Solution/models/
COPY Solution/predict.py /app/Solution/predict.py
COPY Solution/scripts/check_limits.py /app/Solution/scripts/check_limits.py
ENTRYPOINT ["python3"]
CMD ["Solution/predict.py", "/input/tickets.jsonl", "/results/predictions.jsonl", "--method", "tfidf-lr", "--metrics", "/results/metrics.json"]
