FROM sailugr/sinergym:v2.5.2

WORKDIR /app

# (Опционально) ускоряет установку питон-зависимостей
ENV PIP_NO_CACHE_DIR=1

# Ставим только то, чего может не быть в образе
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Код проекта
COPY . /app

CMD ["python", "main.py"]
