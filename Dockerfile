FROM python:3.9.9-bullseye

WORKDIR /src

RUN apt-get update && \
  apt-get install -y \
  libgl1 libglib2.0-0

COPY requirements.txt /src/

RUN pip3 install -r requirements.txt

COPY stable_diffusion_engine.py demo.py demo_web.py /src/
COPY data/ /src/data/

# download models
RUN python3 demo.py --num-inference-steps 1 --prompt "test" --output /tmp/test.jpg

ENTRYPOINT ["python3", "demo.py"]
