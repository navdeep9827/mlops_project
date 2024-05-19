FROM python:3.8-slim

WORKDIR /app
COPY ISN-2-custom-resnet.pth .
COPY model2.py .
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 5000

CMD ["python", "app.py"]
