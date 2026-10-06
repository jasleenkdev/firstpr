# Same API as the serverless deployment, for a Hugging Face Docker Space (port 7860) or any host.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt "uvicorn==0.*"
COPY serve serve
COPY artifacts artifacts
COPY api api
ENV FIRSTPR_ARTIFACTS=/app/artifacts
EXPOSE 7860
CMD ["uvicorn", "api.index:app", "--host", "0.0.0.0", "--port", "7860", "--no-access-log"]
