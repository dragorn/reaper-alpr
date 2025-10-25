FROM python:3.12.12-slim-trixie

ADD model-cache/fast-plate-ocr.tar.gz /root/.cache/
ADD model-cache/open-image-models.tar.gz /root/.cache/
COPY fast-alpr /opt/fast-alpr
COPY reaper-alpr.py /opt/reaper-alpr

RUN	apt-get update && \
    apt-get install -y \
        build-essential \
        libglib2.0-0t64 \
        libgl1 && \
    pip install onnxruntime && \
	pip install /opt/fast-alpr && \
    pip install mjpeg-streamer --prefer-binary

CMD /opt/reaper-alpr

