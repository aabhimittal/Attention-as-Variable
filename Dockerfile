# Hugging Face Space (Docker SDK) — serves API + frontend on :7860
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/hf-cache
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

COPY --chown=user backend/ backend/
COPY --chown=user frontend/ frontend/

# Bake the model into the image so cold starts don't re-download it
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
    AutoTokenizer.from_pretrained('distilgpt2'); \
    AutoModelForCausalLM.from_pretrained('distilgpt2')"

EXPOSE 7860
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
