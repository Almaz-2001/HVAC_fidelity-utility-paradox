FROM sailugr/sinergym:v2.5.2

WORKDIR /app

# Speeds up Python dependency installation
ENV PIP_NO_CACHE_DIR=1

# Install only what may be missing from the base image
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Project code
COPY . /app

CMD ["python", "main.py"]
