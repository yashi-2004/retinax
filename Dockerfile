FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    MPLCONFIGDIR=/tmp/retinax_mpl \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build Matplotlib font cache and initialize clearly-labelled synthetic demo data
RUN python scripts/prepare_data.py --dataset synthetic_primary --synthetic-demo --num-primary 60 --num-external 30 \
    && python scripts/prepare_data.py --dataset synthetic_external \
    && python -c "import matplotlib.pyplot as plt; plt.figure(); plt.close()"

EXPOSE 7860

CMD ["python", "run.py"]
