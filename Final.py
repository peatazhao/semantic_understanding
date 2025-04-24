import pyrealsense2 as rs
import numpy as np
import cv2
import math
from ultralytics import YOLO
from datetime import datetime
import time
import io
import os
import threading
from playsound import playsound
from gtts import gTTS
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from ultralytics import YOLO
import pyrealsense2 as rs
import cv2
import numpy as np
# ------------------------------------ play mp3 function ----------------------------------------
def play_mp3(mp3_file_path):
    
    playsound(mp3_file_path)

    if os.path.exists(mp3_file_path):
        os.remove(mp3_file_path)
        print(f"Removed temporary file: {mp3_file_path}")
# ------------------------------------ play mp3 function ----------------------------------------








# -------------------------------------- Generating clip phrases ---------------------------------
def generate_clip_phrases(object1, object2):

    print(f'Generating CLIP phrases for: {object1} and {object2}')
    
    to_sit_on = ["bicycle", "chair", "couch", "bed", "bench", "toilet", "stool"] 

    to_put_on = ["table", "desk", "shelf", "counter", "bed", "chair", "couch", "suitcase", "backpack", "bench", "dining table", "sink", "refrigerator"]

    small_objects = ["person", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana",
                     "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
                     "cake", "laptop", "mouse", "remote", "keyboard", "cell phone",
                     "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
                     "backpack", "handbag", "tie", "suitcase", "frisbee", "sports ball", "kite",
                     "baseball bat", "baseball glove", "tennis racket", "wine glass", "potted plant"] 
    
    to_hold = ["person", "bottle", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
               "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
               "laptop", "mouse", "remote", "keyboard", "cell phone", "book", "clock", "vase",
               "scissors", "teddy bear", "hair drier", "toothbrush", "backpack", "umbrella",
               "handbag", "tie", "suitcase", "frisbee", "sports ball", "kite", "baseball bat",
               "baseball glove", "skateboard", "surfboard", "tennis racket", "wine glass", "potted plant",
               "stop sign", "parking meter", "cat", "dog"] 

    captions = []

    # Logic for person if they're sitting on smth
    if object1 == "person" and object2 in to_sit_on:
        captions.append(f"a person sitting on a {object2}")
    elif object2 == "person" and object1 in to_sit_on:
        captions.append(f"a person sitting on a {object1}")

    # Logic for if an obj is sitting on smth
    if object1 in to_put_on and object2 in small_objects:
        captions.append(f"a {object2} on a {object1}")
    elif object2 in to_put_on and object1 in small_objects:
        captions.append(f"a {object1} on a {object2}")

    # Logic for if person if holding an obj
    if object1 == "person" and object2 in to_hold:
        if object2 != "person":
            captions.append(f"a person holding a {object2}")
    elif object2 == "person" and object1 in to_hold:
        if object1 != "person":
            captions.append(f"a person holding a {object1}")

    # Our catch all case
    if not captions:
         captions.append(f"a {object1} near a {object2}")
         captions.append(f"a {object2} near a {object1}")

    print(f"Generated captions: {captions}")
    return captions
# -------------------------------------- Generating clip phrases ---------------------------------




global clip_model
global clip_processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# -------------------------------------- Describing clip now ---------------------------------
def describe_with_clip(image_for_clip, captions):

    if not captions or captions == ["No specific relations found"]:
         print("No valid captions provided to CLIP. Try another frame.")
         return None

    # Convert BGR to RGB cuz that's how its read
    image = Image.fromarray(cv2.cvtColor(image_for_clip, cv2.COLOR_BGR2RGB))

    # Pass into clip
    inputs = clip_processor(text=captions, images=image, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image  
        probs = logits_per_image.softmax(dim=1)

    # If it's blank or empty then...
    if probs.numel() == 0:
        print("CLIP probabilities tensor is empty.")
        return None

    top_idx = probs.argmax().item()
    best_caption = captions[top_idx]
    confidence = probs[0, top_idx].item()
    print(f"CLIP Best Caption: '{best_caption}' with confidence {confidence:.4f}")

    return best_caption
# -------------------------------------- Describing clip now ---------------------------------






YOLO_MODEL_PATH = 'yolo12n.pt' # We're using 12 instead of 8 - prof said
model = YOLO(YOLO_MODEL_PATH)
classNames = model.names

def calculate_distance(p1, p2):
    return math.dist(p1, p2)

def speaking(img, result_list, H, W, depth_frame):
    individual_descriptions = []
    detected_objects_info = []

    if not result_list:
        print("No YOLO results passed to speaking function.")
        description = "I don't see anything."
    else:
        valid_detection_found = False
        for result in result_list:
            for box in result.boxes:
                confidence = box.conf[0].item()
                if confidence < 0.5:
                     continue

                valid_detection_found = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                class_name = classNames[cls_id]

                centerX = (x1 + x2) // 2
                centerY = (y1 + y2) // 2
                safe_cx = max(0, min(W - 1, centerX))
                safe_cy = max(0, min(H - 1, centerY))

                # Get Depth - NEW: added try catch so it doesn't keep bre
                depth = depth_frame.get_distance(safe_cx, safe_cy)
                if depth <= 0 or depth > 30:
                    depth_phrase = "at an unknown distance"
                else:
                    depth_phrase = f"about {depth:.2f} meters away"

                # Determining the location by dividing into 3x3 grids
                # (Left/Center/Right, Top/Mid/Bottom)
                if centerX <= W / 3:
                    W_pos = "on the left"
                elif centerX <= (W / 3 * 2):
                    W_pos = "in the center"
                else:
                    W_pos = "on the right"

                if centerY <= H / 3:
                    H_pos = "at the top"
                elif centerY <= (H / 3 * 2):
                    H_pos = "in the middle"
                else:
                    H_pos = "at the bottom"

                detected_objects_info.append({
                    "name": class_name,
                    "center": (centerX, centerY),
                    "bbox": (x1, y1, x2, y2)
                 })
                individual_descriptions.append(f"A {class_name} {H_pos} {W_pos}, {depth_phrase}")

        num_detections = len(detected_objects_info)
        description = ""
        clip_description = None

        if num_detections == 0:
            print("No objects detected above confidence threshold. Try again or adjust parameters.")
            description = "Adjust parameters since nothing detected?"
        elif num_detections == 1:
            print("1 object detected.")
            description = individual_descriptions[0]
        else:
            print("Multiple objects detected.")
            person_detections = [obj for obj in detected_objects_info if obj['name'] == 'person']
            non_person_detections = [obj for obj in detected_objects_info if obj['name'] != 'person']

            object1_info = None
            object2_info = None

            if person_detections and non_person_detections:
                person_info = person_detections[0]
                closest_obj = None
                min_dist = float('inf')

                for other_obj in non_person_detections:
                    dist = calculate_distance(person_info['center'], other_obj['center'])
                    if dist < min_dist:
                        min_dist = dist
                        closest_obj = other_obj

                if closest_obj:
                    object1_info = person_info
                    object2_info = closest_obj

            elif num_detections >= 2:
                 print("Trying to find the closest objects.")
                 closest_pair = None
                 min_dist = float('inf')

                 for i in range(num_detections):
                     for j in range(i + 1, num_detections):
                         obj_i = detected_objects_info[i]
                         obj_j = detected_objects_info[j]
                         dist = calculate_distance(obj_i['center'], obj_j['center'])
                         if dist < min_dist:
                             min_dist = dist
                             closest_pair = (obj_i, obj_j)

                 if closest_pair:
                    if closest_pair[0]['center'][0] <= closest_pair[1]['center'][0]:
                         object1_info = closest_pair[0]
                         object2_info = closest_pair[1]
                    else:
                         object1_info = closest_pair[1]
                         object2_info = closest_pair[0]

            if object1_info and object2_info:
                print(f"Results of CLIP: {object1_info['name']} and {object2_info['name']}")
                captions = generate_clip_phrases(object1_info['name'], object2_info['name'])
                clip_description = describe_with_clip(img, captions)
            else:
                 print("No closest pairs. Try again.")

    if clip_description and clip_description != "No specific relations found":
        description += f" I also see {clip_description}."

    if not description:
         description = "Unsure. Please try again."

    print("\n--- Final Description ---")
    print(description)
    print("-------------------------\n")

    # --- Text-to-Speech Stuff ---
    print("Creating MP3...")
    myobj = gTTS(text=description, lang="en", slow=False)
    mp3_file = io.BytesIO()
    myobj.write_to_fp(mp3_file)
    mp3_file.seek(0)
    print("MP3 data generated.")
    now = datetime.now()
    current_time = now.strftime("speech_%Y%m%d_%H%M%S.mp3")
    temp_mp3_path = os.path.join(os.getcwd(), current_time)

    with open(temp_mp3_path, "wb") as f:
        f.write(mp3_file.read())
    print(f"MP3 file saved: {temp_mp3_path}")
    tts_thread = threading.Thread(target=play_mp3, args=(temp_mp3_path,), daemon=True)
    tts_thread.start()



















# Main file basically
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
print("\n--- Configuring the Intel RealSense D435i camera ----\n")
pipeline = rs.pipeline()
config = rs.config()
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
config.enable_stream(rs.stream.depth, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.z16, 30)
config.enable_stream(rs.stream.color, CAMERA_WIDTH, CAMERA_HEIGHT, rs.format.bgr8, 30)
profile = pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)
print("\n --- Setup successful ---\n")
print("--- Make sure to switch window to the cv2 imshow frame. ---")

# Int main basically
while True:
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    H, W = color_image.shape[:2]

    # Pass into YOLO model now obv
    display_results = model(color_image, stream=True, verbose=False)
    display_image = color_image.copy()

    # For everything the YOLO model predicted...
    for r in display_results:
        for box in r.boxes:
            # Might need to tune on test day!
            if box.conf[0] > 0.3:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                label = classNames[cls_id]
                conf = box.conf[0].item()

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                safe_cx = max(0, min(W - 1, cx))
                safe_cy = max(0, min(H - 1, cy))
                depth_m = depth_frame.get_distance(safe_cx, safe_cy)
                if depth_m > 0:
                    label_text = f'{label} {depth_m:.1f}m'
                else:
                    label_text = f'{label}'
                
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 0, 255), 2)
                cv2.putText(display_image, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        # Inner Loop
    # Outer Loop
    cv2.imshow('Cam', display_image)

    # If user presses a key...
    pressed_key = cv2.waitKey(1) & 0xFF

    # If q (quit)
    if pressed_key == ord('q'):
        print("\nUser pressed q for quit.")
        break

    # If user presses s, then speak
    if pressed_key == ord('s'):
        print("\nUser pressed s for speech.")
        speak_results = model(color_image, stream=True, verbose=False)
        speaking(color_image, list(speak_results), H, W, depth_frame)
# While true loop
pipeline.stop()
cv2.destroyAllWindows()
print("Cleaning up finished.")



