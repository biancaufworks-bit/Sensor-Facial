import cv2
import mediapipe as mp
import os
import time

model_path = r'C:\Bianca\programinhas\face_landmarker.task'

# --- STEP 1: LOAD MULTIPLE IMAGES ---
# We store them in a dictionary so we can switch easily
images = {
    "neutralH": cv2.imread(r'C:\Bianca\programinhas\neutralH.png', cv2.IMREAD_UNCHANGED),
    "neutralW": cv2.imread(r'C:\Bianca\programinhas\neutralW.png', cv2.IMREAD_UNCHANGED),
    "sassy": cv2.imread(r'C:\Bianca\programinhas\sassy.png', cv2.IMREAD_UNCHANGED),
    "happy": cv2.imread(r'C:\Bianca\programinhas\happy.png', cv2.IMREAD_UNCHANGED),
    "sad": cv2.imread(r'C:\Bianca\programinhas\sad.png', cv2.IMREAD_UNCHANGED),
    "sus": cv2.imread(r'C:\Bianca\programinhas\sus.png', cv2.IMREAD_UNCHANGED),
    "angry": cv2.imread(r'C:\Bianca\programinhas\angry.png', cv2.IMREAD_UNCHANGED)

}
sex = input("Você é mulher ou homem? (digite M ou H maiúsculo)")
if sex == "M":
    resul = "neutralW"
else:
    resul = "neutralH"

if not os.path.exists(model_path):
    print(f"❌ ERROR: Model missing at {model_path}")
else:
    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

    # IMPORTANT: We added output_face_blendshapes=True here!
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        output_face_blendshapes=True)

    detector = FaceLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(0)


    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        timestamp = int(time.time() * 1000)
        results = detector.detect_for_video(mp_image, timestamp)




        if results.face_landmarks:
            # --- STEP 2: DETECT EXPRESSION ---
            current_img = images[resul]  # Default

            if results.face_blendshapes:
                # We turn the results into a simple dictionary for easy reading
                scores = {b.category_name: b.score for b in results.face_blendshapes[0]}
                smiling = scores['mouthSmileLeft'] > 0.5 or scores['mouthSmileRight'] > 0.5
                angry = scores['browDownLeft'] > 0.3 or scores['browDownRight'] > 0.3
                sad = scores['mouthFrownLeft'] > 0.15 or scores['mouthFrownRight'] > 0.15
                sassy = (scores['browOuterUpLeft'] > 0.3 or scores['browOuterUpRight'] > 0.3) and (scores['mouthSmileLeft'] > 0.5 or scores['mouthSmileRight'] > 0.5)
                sus = scores['browOuterUpLeft'] > 0.3 or scores['browOuterUpRight'] > 0.3

                # Logic: If smile score is over 0.5, switch to happy
                if sassy == True:
                    current_img = images["sassy"]
                elif smiling == True:
                    current_img = images["happy"]
                elif angry == True:
                    current_img = images["angry"]
                elif sad == True:
                    current_img = images["sad"]
                elif sus == True:
                    current_img = images["sus"]


            # --- STEP 3: DRAW CHOSEN IMAGE ---
            nose = results.face_landmarks[0][1]
            nx, ny = int(nose.x * w), int(nose.y * h)
            overlay_size = 240

            # Make sure the current image exists before drawing
            if current_img is not None:
                img_to_draw = cv2.resize(current_img, (overlay_size, overlay_size))
                y1, y2 = ny - overlay_size // 2, ny + overlay_size // 2
                x1, x2 = nx - overlay_size // 2, nx + overlay_size // 2

                if y1 > 0 and y2 < h and x1 > 0 and x2 < w:
                    if img_to_draw.shape[2] == 4:
                        alpha_s = img_to_draw[:, :, 3] / 255.0
                        alpha_l = 1.0 - alpha_s
                        for c in range(0, 3):
                            frame[y1:y2, x1:x2, c] = (alpha_s * img_to_draw[:, :, c] +
                                                      alpha_l * frame[y1:y2, x1:x2, c])
                    else:
                        frame[y1:y2, x1:x2] = img_to_draw[:, :, :3]

        cv2.imshow('Expression Filter', frame)
        if cv2.waitKey(1) & 0xFF == 27: break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()