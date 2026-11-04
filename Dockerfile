FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml README.md README.en.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .
RUN useradd -m agent && mkdir -p /data && chown agent /data
USER agent
ENV DB_PATH=/data/voice_agent_ar.db PORT=8080
EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8080/health'); sys.exit(0)"
CMD ["voice-agent-ar", "serve", "--host", "0.0.0.0", "--port", "8080"]
