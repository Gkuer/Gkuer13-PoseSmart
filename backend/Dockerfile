FROM python:3
ENV PYTHONUNBUFFERED 1
RUN mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip install cmake
RUN pip install -r requirements.txt
RUN pip install gunicorn
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install libglib2.0-0
COPY . /code/

RUN python manage.py migrate

CMD ["gunicorn", "posesmart.wsgi", "--bind", "0.0.0.0:8000"]