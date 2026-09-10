# Engine image. Paper by default; arming is by environment only — see README.

FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=UTC

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

# No CRYPTO_LIVE, no EQUITY_LIVE, no KRAKEN_LIVE, no TRADER_MODE, no secrets.
# Host/port/db come from fly.toml [env]. ENGINE_API_TOKEN is a Fly secret
# the owner sets out of band; the API refuses to boot without it.
EXPOSE 8080
CMD ["python3", "-m", "runner"]
