FROM python:3.13-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    OMP_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    MKL_NUM_THREADS=2
COPY data/ /app/data/
COPY Solution/triage/ /app/Solution/triage/
COPY Solution/config/ /app/Solution/config/
COPY Solution/predict.py /app/Solution/predict.py
COPY Solution/scripts/check_limits.py /app/Solution/scripts/check_limits.py
ENTRYPOINT ["python3"]
CMD ["Solution/predict.py", "data/tickets.jsonl", "/results/tickets_predictions.jsonl", "--metrics", "/results/metrics.json"]
