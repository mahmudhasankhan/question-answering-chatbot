FROM python:3.10-slim-bullseye
 
ENV HOST=0.0.0.0
 
ENV LISTEN_PORT 8000
 
EXPOSE 8000
 
RUN apt-get update && apt-get install -y git
 
WORKDIR app/

COPY . .
 
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt
 
CMD ["python", "main.py"]
