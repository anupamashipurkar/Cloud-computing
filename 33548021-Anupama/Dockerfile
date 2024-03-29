FROM python:3.9.9-slim-bullseye  
RUN mkdir app
WORKDIR /app
ADD . /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
CMD ["gunicorn", "--timeout", "60", "-b", ":5000", "app:app"]