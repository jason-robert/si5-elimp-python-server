FROM python:3.8
ADD requirements.txt /requirements.txt
ADD main.py /main.py
ADD okteto-stack.yaml /okteto-stack.yaml
RUN pip install scipy==1.4.1
RUN pip install -r requirements.txt
EXPOSE 8080
COPY ./app app
COPY ./csv csv
COPY ./models models
CMD ["python3", "main.py"]