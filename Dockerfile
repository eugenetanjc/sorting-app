FROM python:3.8
WORKDIR /app/prod
COPY . /app/prod
#RUN apt-get -y update && apt-get -y install nginx
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt
EXPOSE 5000

VOLUME /app/mnt/sortingpinning/Countries

CMD ["python", "app.py"]