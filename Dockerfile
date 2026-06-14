FROM python:3.14-slim

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . /code/

EXPOSE 80

ENV PYTHONPATH=/code

RUN python -m nltk.downloader stopwords punkt punkt_tab wordnet

CMD [ "fastapi", "run", "app/main.py", "--port", "80" ]