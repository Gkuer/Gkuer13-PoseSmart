# ./Dockerfile Settings
FROM python:3
WORKDIR /usr/src/app

# Installation
COPY requirements.txt ./
RUN pip install cmake
RUN pip install -r requirements.txt
RUN pip install gunicorn
RUN apt-get update
RUN apt-get -y install libgl1-mesa-glx
RUN apt-get install libglib2.0-0

# Copy src files
COPY . .

# Migrate
RUN python manage.py migrate

# Run 8000 port
EXPOSE 8000

# Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "posesmart.wsgi:application"]