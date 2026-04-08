import time
import cv2
import numpy as np
import tensorflow as tf
import os

# Reduce CPU usage
os.environ["OMP_NUM_THREADS"] = "2"
tf.config.set_visible_devices([], 'GPU')

# Load model
model = tf.keras.models.load_model("asl_classifier_finetuned.h5")
class_mapping = np.load("class_mapping.npy", allow_pickle=True).item()


def preprocess_frame(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    return np.expand_dims(frame, axis=0)


def get_gesture_text():
    # 🔥 Kill any stuck camera process
    os.system("fuser -k /dev/video0 2>/dev/null")

    cap = cv2.VideoCapture(0)

    sentence = ""
    last_time = 0

    prev_letter = ""
    stable_count = 0
    STABILITY_THRESHOLD = 3

    COOLDOWN = 0.3

    last_added_letter = ""
    ready_for_next = True   # 🔥 key improvement

    print("Press 'q' to finish input")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("❌ Camera not working")
            break

        frame = cv2.flip(frame, 1)

        # Frame size
        h, w = frame.shape[:2]

        # ROI box
        x1, y1 = w//2 - 150, h//2 - 150
        x2, y2 = w//2 + 150, h//2 + 150

        # Draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, "Place hand inside box", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Crop ROI
        center = frame[y1:y2, x1:x2]

        if center is None or center.size == 0:
            continue

        processed = preprocess_frame(center)

        predictions = model.predict(processed, verbose=0)[0]
        top_idx = np.argmax(predictions)
        confidence = predictions[top_idx]

        letter = class_mapping[top_idx]

        # 🔹 Show UI info
        cv2.putText(frame, f"Letter: {letter}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # 🔹 Stability tracking
        if letter == prev_letter:
            stable_count += 1
        else:
            prev_letter = letter
            stable_count = 1
            ready_for_next = True   # reset when gesture changes

        # 🔹 Reset if confidence drops (hand moved)
        if confidence < 0.80:
            ready_for_next = True

        # 🔹 Accept letter ONLY when stable + ready
        if (
            confidence > 0.95 and
            stable_count >= STABILITY_THRESHOLD and
            time.time() - last_time > COOLDOWN and
            ready_for_next
        ):
            sentence += letter
            last_added_letter = letter
            print("Current:", sentence)

            last_time = time.time()
            stable_count = 0
            ready_for_next = False   # block until next gesture

        # 🔹 Always show text
        cv2.putText(frame, f"Text: {sentence}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 🔹 Instructions
        cv2.putText(frame, "q: quit | c: clear | space: space", (10, 450),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

        # Show frame
        cv2.imshow("Gesture Input", frame)

        key = cv2.waitKey(1)

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            return sentence

        elif key == ord('c'):
            sentence = ""
            last_added_letter = ""
            ready_for_next = True

        elif key == ord(' '):
            sentence += " "
            last_added_letter = ""
            ready_for_next = True

    cap.release()
    cv2.destroyAllWindows()
    cv2.waitKey(1)

    return sentence
