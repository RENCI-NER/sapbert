FROM nvcr.io/nvidia/nemo:22.01

COPY ./requirements.txt requirements.txt
RUN pip install --upgrade pip

RUN pip install --no-cache-dir -r requirements.txt

