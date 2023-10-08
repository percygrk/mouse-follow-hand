import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Mirror the frame
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB and process it with MediaPipe Hands
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    # Convert RGB image back to BGR for rendering with OpenCV
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if result.multi_hand_landmarks:
        for hand_landmarks, hand_info in zip(result.multi_hand_landmarks, result.multi_handedness):
            handedness = hand_info.classification[0].label  # 'Left' or 'Right'
            print(f'Handedness: {handedness}')
            finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            finger_mcp = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]




            finger_coords = (int(finger_tip.x * frame.shape[1]), int(finger_tip.y * frame.shape[0]))
            middle_coords = (int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0]))
            finger_mcp_coords = (int(finger_mcp.x * frame.shape[1]), int(finger_mcp.y * frame.shape[0]))
            thumb_coords = (int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0]))
            pinky_coords = (int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0]))

            distance = ((finger_coords[0] - thumb_coords[0]) ** 2 + (finger_coords[1] - thumb_coords[1]) ** 2) ** 0.5
            distance_pinky = ((thumb_coords[0] - pinky_coords[0]) ** 2 + (
                        thumb_coords[1] - pinky_coords[1]) ** 2) ** 0.5
            distance_middle = ((thumb_coords[0] - middle_coords[0]) ** 2 + (
                        thumb_coords[1] - middle_coords[1]) ** 2) ** 0.5
            distance_mcp = ((finger_mcp_coords[0] - thumb_coords[0]) ** 2 + (
                        finger_mcp_coords[1] - thumb_coords[1]) ** 2) ** 0.5

            if distance < 30:
                pyautogui.click()
            if distance_middle < 30:
                pyautogui.press("backspace")
            print(distance_mcp)
            if distance_pinky < 20:
                pyautogui.rightClick()
            if distance_mcp > 80 and thumb_coords[0] > finger_mcp_coords[0]:
                pyautogui.drag()
            #if check_thumb_up(hand_landmarks, distance_mcp):
            #    pyautogui.press("enter")

            # Move the mouse to the position of the finger tip
            pyautogui.moveTo(finger_coords[0], finger_coords[1])

            # Draw hand landmarks on the image
            mp_drawing.draw_landmarks(image_bgr, hand_landmarks, mp_hands.HAND_CONNECTIONS)


        # Get the coordinates of the tip of the index finger and thumb

    # Display the mirrored frame
    cv2.imshow('Frame', image_bgr)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Cleanup
hands.close()
cap.release()
cv2.destroyAllWindows()
