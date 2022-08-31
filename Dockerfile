FROM debian:bullseye

WORKDIR /src

RUN apt-get update && \
apt-get install python3-pip -y \
libgl1 libglib2.0-0

COPY requirements.txt /src/

RUN pip3 install -r requirements.txt

COPY stable_diffusion.py /src/
COPY data/ /src/data/
# download models and save them to image
RUN python3 stable_diffusion.py --prompt "test" --num-inference-steps 1 --output /tmp/test.jpg
ENTRYPOINT ["python3", "stable_diffusion.py"]
