FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home --uid 10001 whisper \
    && mkdir -p /home/whisper/.cache/huggingface \
    && chown -R whisper:whisper /home/whisper/.cache
COPY server.py .
USER whisper
ENV HF_HOME=/home/whisper/.cache/huggingface
EXPOSE 9000
CMD ["python", "-u", "server.py"]
