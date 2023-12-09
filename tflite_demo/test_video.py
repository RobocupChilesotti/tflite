import cv2

# Create a VideoCapture object
cap = cv2.VideoCapture('vids/test.mp4')

while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break


# Release the VideoCapture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
