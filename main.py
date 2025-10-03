import cv2
import numpy as np
from tinytag import TinyTag
import datetime
import ultralytics
import mediapipe as mp

yoloModel = ultralytics.YOLO(r".\models\yolo\yolo11n.pt")

mpHolistic = mp.solutions.pose
holistic = mpHolistic.Pose( static_image_mode=False,
               model_complexity=2,
               smooth_landmarks=True,
               enable_segmentation=False,
               smooth_segmentation=True,
               min_detection_confidence=0.5,
               min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils # Drawing utilities
mp_drawing_styles = mp.solutions.drawing_styles


if __name__ == '__main__':

    pathToVideo = r'.\data\person-cows.mp4'

    tags = TinyTag.get(pathToVideo)

    cap = cv2.VideoCapture(pathToVideo)

    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    dt = 1 / fps

    numberFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    print(rf'fps = {fps}, numberFrames = {numberFrames}')

    w, h = 0.3*w, 0.3*h
    k = 0

    out = cv2.VideoWriter(r'.\data\test-0.mp4',
                          cv2.VideoWriter_fourcc(*'MP4V'),  # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                          int(fps),
                          (int(w), int(h)))

    startDate = datetime.datetime.now()

    while True:
        success, originalFrame = cap.read()

        if not success:
            break

        frame = cv2.resize(originalFrame, (int(w), int(h)))
        frameForMediapipe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        resultsYolo = yoloModel.predict(frame, conf=0.5)
        annotator = ultralytics.utils.plotting.Annotator(frame)

        for r in resultsYolo:
            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
                c = box.cls

                colorObject = (0, 255, 0) if yoloModel.names[int(c)] == 'cow' else (0, 0, 255)

                annotator.box_label(b, yoloModel.names[int(c)],
                                    color=colorObject, txt_color=(255, 0, 0))

        frame = annotator.result()

        resultsMediapipe = holistic.process(frameForMediapipe)
        mp_drawing.draw_landmarks(frame, resultsMediapipe.pose_landmarks, mpHolistic.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=4),
                                  mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2))

        frame = cv2.putText(frame,
                            f'date = {startDate + datetime.timedelta(seconds=k*dt)}',
                            (int(0.5 * w), int(0.97 * h)),
                            cv2.FONT_HERSHEY_PLAIN,
                            1,
                            (0, 0, 255),
                            2
        )

        # cv2.imshow(f'{pathToVideo}', frame)
        #
        # if cv2.waitKey(1) & 0xff == ord('q'):
        #     break

        k += 1

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()