while catched < 3:
    # Acquire frame and resize to expected shape [1xHxWx3]
    ret, frame = cap.read()

    # Resize frame as required by the model
    frame = cv2.resize(frame, (width, height))

    rects, classes, scores = inf(frame)

    # The enumerate() function is used to get both the index and the value of each item in the list
    for index, score in enumerate(scores[0]):
        if score > conf_thresh:
            # Get the labels
            object_name = labels[int(classes[0][index])]

            # Get the bounding box coordinates
            y_min = int(max(1, (rects[0][index][0] * height)))
            x_min = int(max(1, (rects[0][index][1] * width)))
            y_max = int(min(height, (rects[0][index][2] * height)))
            x_max = int(min(width, (rects[0][index][3] * width)))

            # Get the coordinates of the center
            # cx = x_min + (x_max - x_min) / 2 = (2 * x_min + x_max - x_min) / 2 = (x_min + x_max) / 2
            cx = int((x_min + x_max) / 2)
            cy = int((y_min + y_max) / 2)

            # Get the area of the bounding box

            balls[object_name] = (cx, cy)

            if display:
                draw_bbox(frame, y_min, x_min, y_max, x_max, cx, cy, object_name, score)

    
    # Check if balls have been detected
    if balls:
        if False:
            print('Placeholder')
    else:
        # Il robot deve girare finchè non trova qualcosa
        # Ogni volta che entra in questo else gira di x gradi




        if False:
            print('Just to keep the else')



    '''
    # Compare each center point from the current frame with each center point with the previous frame
    for pt in cur_center_pts:
        for pt2 in prev_center_pts:
            # Calculate the distance
            distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

            if distance < dist_thresh:
                tracking_objects[track_id] = pt
                track_id += 1

    for object_id, pt in tracking_objects.items():
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)
    '''

    # Break the loop on 'q' key press
    if cv2.waitKey(50) == ord('q'):
        break


# Esci dalla stanza


# Clean up
cap.release()
cv2.destroyAllWindows()
