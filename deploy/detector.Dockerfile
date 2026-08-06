FROM python:3.12-slim
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir "transformers<5"
COPY scripts/detector_common.py scripts/detector_server.py /app/
# Bake the model into the image so pods start without network access.
ENV HF_HOME=/opt/hf
RUN python -c "import sys; sys.path.insert(0, '/app'); \
    from detector_common import load_model; load_model()" \
    && chmod -R a+rX /opt/hf
USER 65534:65534
EXPOSE 9310
ENTRYPOINT ["python", "/app/detector_server.py", "0.0.0.0:9310"]
