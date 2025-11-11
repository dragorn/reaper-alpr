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
import struct
from mjpeg_streamer import MjpegServer, Stream

alpr = ALPR(
    detector_model="yolo-v9-t-384-license-plate-end2end",
    ocr_model="global-plates-mobile-vit-v2-model",
)

# fps decimation (1 in X frames processed)
fpsDecimation = 15
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

    s_imgIR = False

    s_FrameNum = s_FrameNum + 1
    if s_FrameNum % fpsDecimation != 0:
        return

    cvimage = cv2.imdecode(np.asarray(bytearray(image),
                                      dtype=np.uint8), cv2.IMREAD_COLOR)
    alpr_results = alpr.predict(cvimage)

    for i, r in enumerate(alpr_results):
        print("#", i, r.ocr.text)

    cvmarked = mark_image(cvimage, alpr_results)
    # cv2.imwrite("/data/reaper-plate.jpg", cvmarked)

    stream.set_frame(cvmarked)


def network_feed(host, port):
    client_socket = socket.socket()
    client_socket.connect((host, port))

    chunk = bytearray()

    while True:
        while len(chunk) < 4:
            recv = client_socket.recv(1024*256)
            chunk.extend(recv)

        vals = struct.unpack('<I', chunk[:4])

        if vals[0] == 3001:
            while len(chunk) < 40:
                recv = client_socket.recv(1024*256)
                chunk.extend(recv)

            vals = struct.unpack('<IIIIIIIIII', chunk[:40])

            # frametype = vals[0]
            # cameraid = vals[2]
            # framenum = vals[8]
            jpeglen = vals[9]

            # print("IR frametype", frametype, "camera", cameraid, "frame", framenum, "jpeg", jpeglen)
            while len(chunk) < 40 + jpeglen:
                recv = client_socket.recv(jpeglen)
                chunk.extend(recv)

            handle_image(chunk[40:40+jpeglen])

            #print(chunk[40:48].hex(), "...", chunk[40+jpeglen-8:40+jpeglen].hex(), " | ", chunk[40+jpeglen:40+jpeglen+8].hex())
            chunk = chunk[40+jpeglen:]

        elif vals[0] == 1640:
            while len(chunk) < 28:
                recv = client_socket.recv(1024*256)
                chunk.extend(recv)

            vals = struct.unpack('<IIIIIII', chunk[:28])

            # frametype = vals[0]
            # cameraid = vals[2]
            jpeglen = vals[6]

            # print("COLOR frametype", frametype, "camera", cameraid, "jpeg", jpeglen)

            while len(chunk) < 28 + jpeglen:
                recv = client_socket.recv(jpeglen)
                chunk.extend(recv)

            # print(chunk[28:36].hex(), "...", chunk[28+jpeglen-8:28+jpeglen].hex(), " | ", chunk[28+jpeglen:28+jpeglen+8].hex())

            chunk = chunk[28+jpeglen:]

        elif vals[0] == 1032:
            # Marker frame of some sort, no jpeg data
            while len(chunk) < 4:
                recv = client_socket.recv(1024)
                chunk.extend(recv)

            chunk = chunk[4:]

        else:
            print("unknown frame", vals[0])

            while len(chunk) < 128:
                recv = client_socket.recv(1024)
                chunk.extend(recv)

            print(chunk[:128].hex())
            return

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
