import cv2
from initialize_tf import labels, height, width


def draw_bbox(frame, object_name='', score=0, y_min=0, x_min=0, y_max=0, x_max=0):
        # Draw a rectangle on the image
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 255), 2)

        # Draw the label
        label = '%s: %d%%' % (object_name, int(score * 100))  # Example: 'person: 72%'
        # Get font details
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        # Make sure not to draw label too close to top of window
        label_ymin = max(y_min, labelSize[1] + 10)
        # Draw label text
        cv2.putText(frame, label, (x_min, label_ymin - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 0, 255), 2)
        
        cx = int((x_min + x_max) / 2)
        cy = int((y_min + y_max) / 2)
        # Draw the center point
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)