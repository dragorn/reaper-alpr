#!/usr/bin/env python

# Reaper - Fast-ALPR
# based on fast-alpr  https://github.com/ankandrew/fast-alpr.git
#
# git submodule update --init --recrusive
# docker build -t reaperml .
# REAPER=reaper-hostname docker run -v $(pwd):/data -E REAPER -it reaperml

from fast_alpr import ALPR, ALPRResult
import numpy as np
import cv2
import os
import socket
import statistics
from mjpeg_streamer import MjpegServer, Stream

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="cct-xs-v1-global-model",
)

# fps decimation (1 in X frames processed)
fpsDecimation = 30
reaperHost = "192.168.3.100"
reaperPort = 2000

# toggle between IR and color
s_imgIR = True
s_FrameNum = -1

stream = Stream("reaper", size=(1640, 922), quality=50, fps=15)

def mark_image(img: np.ndarray, alpr_results: list[ALPRResult]) -> np.ndarray:
    for result in alpr_results:
        detection = result.detection
        ocr_result = result.ocr
        bbox = detection.bounding_box
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
        # Draw the bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (36, 255, 12), 2)
        if ocr_result is None or not ocr_result.text or not ocr_result.confidence:
            continue
        # Remove padding symbols if any
        plate_text = ocr_result.text
        confidence: float = (
            statistics.mean(ocr_result.confidence)
            if isinstance(ocr_result.confidence, list)
            else ocr_result.confidence
        )
        display_text = f"{plate_text} {confidence * 100:.2f}%"
        font_scale = 1.25
        # Draw black background for better readability
        cv2.putText(
            img=img,
            text=display_text,
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(0, 0, 0),
            thickness=6,
            lineType=cv2.LINE_AA,
        )
        # Draw white text
        cv2.putText(
            img=img,
            text=display_text,
            org=(x1, y1 - 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv2.LINE_AA,
        )

    return img


def handle_image(image):
    global s_imgIR
    global s_FrameNum

    if not s_imgIR:
        s_imgIR = True
        return

    s_FrameNum = s_FrameNum + 1
    if s_FrameNum % fpsDecimation != 0:
        return

    cvimage = cv2.imdecode(np.asarray(bytearray(image),
                                      dtype=np.uint8), cv2.IMREAD_COLOR)
    alpr_results = alpr.predict(cvimage)

    for i, r in enumerate(alpr_results):
        print("#", i, r.ocr.text)

    cvmarked = mark_image(cvimage, alpr_results)
    cv2.imwrite("/data/reaper-plate.jpg", cvmarked)

    stream.set_frame(cvmarked)


def network_feed(host, port):
    client_socket = socket.socket()
    client_socket.connect((host, port))

    inImage = False
    curImg = bytearray()
    chunk = bytearray()
    sPos = 0
    lPos = 0

    while True:
        recv = client_socket.recv(1024*256)

        chunk.extend(recv)

        epos = 0

        if not inImage:
            sPos = chunk.find(b'\xFF\xD8\xFF')
            if sPos >= 0:
                inImage = True
            else:
                continue

        epos = chunk[lPos:].find(b'\xFF\xD9')
        if epos >= 0:
            curImg = bytearray(chunk[sPos:lPos+epos+2])

            chunk = chunk[lPos+epos+2:]

            handle_image(curImg)

            inImage = False
            lPos = 0

    client_socket.close()


if __name__ == '__main__':
    if "REAPER" in os.environ:
        reaperHost = os.environ["REAPER"]
    else:
        print("Missing REAPER variable")
        exit(1)

    if "FPS" in os.environ:
        fpsDecimation = int(os.environ["FPS"])

    server = MjpegServer("0.0.0.0", 8080)
    server.add_stream(stream)
    server.start()

    network_feed(reaperHost, reaperPort)
