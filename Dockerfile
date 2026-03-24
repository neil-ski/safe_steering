FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

ENV PIP_BREAK_SYSTEM_PACKAGES=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    bitsandbytes \
    torchmetrics \
    google-cloud-storage \
    transformers \
    accelerate \
    datasets \
    scikit-learn

COPY . .

CMD ["sleep", "infinity"]