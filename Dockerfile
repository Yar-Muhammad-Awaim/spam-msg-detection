FROM python:3.14-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/

EXPOSE 80

CMD [ "fastapi", "run", "app/main.py", "--port", "80" ]