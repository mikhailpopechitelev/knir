# Базовый образ
FROM python:3.9-slim

# Установка зависимостей для скачивания и распаковки
RUN apt-get update && \
    apt-get install -y wget unzip && \
    rm -rf /var/lib/apt/lists/*

# Рабочая директория в контейнере
WORKDIR /app

# Переменные для файла
ENV FILE_ID=1G5AhD8hSnTURRK2U0TeARdh2ag9oOdSh
ENV FILE_NAME=chest_xray.zip

# Шаг 1: Получение подтверждения загрузки
#RUN https://drive.google.com/open?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM
#RUN wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0B7EVK8r0v71pZjFTYXZWM3FlRnM" -O img_align_celeba.zip && rm -rf /tmp/cookies.txt
# Шаг 2: Скачивание файла с подтверждением

RUN pip install gdown

RUN gdown 1G5AhD8hSnTURRK2U0TeARdh2ag9oOdSh

# Разархивирование файла с помощью 7z
# Используем 7z для разархивирования и удаления архива
RUN unzip chest_xray.zip && \
    rm chest_xray.zip
# Копирование файлов проекта
COPY . .

# Установка зависимостей Python
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install scipy
# Команда по умолчанию
CMD ["python", "test.py"]