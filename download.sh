#!/bin/sh
FILE_ID="1kvOp-KOck6-44vnTSd8g4pnk8O1DrOk4"
FILENAME="chest_xray.zip"

# Скачивание файла с подтверждением токена
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${FILE_ID}" > /dev/null
CONFIRM=$(awk '/download/ {print $NF}' ./cookie)
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -o ${FILENAME}

# Распаковка и удаление архива
unzip ${FILENAME}
rm ${FILENAME} ./cookie