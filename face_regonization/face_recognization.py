import cv2
import face_recognition

# Load an image
image = face_recognition.load_image_file("images/with_my_cutipie.png")

# Detect face locations
face_locations = face_recognition.face_locations(image)

# Get face encodings
face_encodings = face_recognition.face_encodings(image, face_locations)

# Draw rectangle around faces
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
for top, right, bottom, left in face_locations:
    cv2.rectangle(image_bgr, (left, top), (right, bottom), (0, 255, 0), 2)

# Show result
cv2.imshow("Detected Faces", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()