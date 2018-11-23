FROM vitaflow-base
LABEL maintainer="Sampath Kumar M"

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN python -m spacy download en_core_web_md

#CMD "sleep 1000"