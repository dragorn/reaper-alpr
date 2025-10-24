FROM python:3.12.12-slim-trixie

ADD model-cache/fast-plate-ocr.tar.gz /root/.cache/
ADD model-cache/open-image-models.tar.gz /root/.cache/
COPY fast-alpr /opt/fast-alpr
COPY reaper-alpr.py /opt/reaper-alpr

RUN	pip install onnxruntime && \
	pip install /opt/fast-alpr

CMD /opt/reaper-alpr

