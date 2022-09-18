import cv2
import time
from threading import Thread


class Camera:

    def __init__(self, src=0, wCam=640, hCam=480):
        self.stream = cv2.VideoCapture(src)

        self.stream.set(3, wCam)
        self.stream.set(4, hCam)

        self.ret, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.ret:
                self.stop()
                self.stream.release()
                print("[#] Stop video stream")
            else:
                self.ret, self.frame = self.stream.read()

    def stop(self):
        self.stopped = True

