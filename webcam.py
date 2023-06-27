import cv2

# Create a VideoCapture object to read from the default camera
cap = cv2.VideoCapture(0)
# Check if camera opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
# Define the window name
window_name = "Webcam Live Video"
# Create a new window
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Load model
model = cv2.dnn.readNetFromCaffe('pose_deploy.prototxt', 'pose_iter_584000.caffemodel')

# Loop through the frames from the camera
while True:
    # Read a new frame from the camera
    ret, frame = cap.read()     # frame: np array with shape (480, 640, 3)

    # Break the loop if we have reached the end of the video
    if not ret:
        break
    print(type(frame))
    print(frame.shape)
    # Input the frame to the model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (368, 368), (127.5, 127.5, 127.5), swapRB=True, crop=False)
    model.setInput(blob)
    output = model.forward()    # Output: np array with shape (1, 78, 46, 46)
    
    # Display the frame in the new window
    cv2.imshow(window_name, frame)

    # Wait for a key press and check if the user wants to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all windows
cap.release()
cv2.destroyAllWindows()