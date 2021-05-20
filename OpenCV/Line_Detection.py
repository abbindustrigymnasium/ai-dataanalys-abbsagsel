import cv2
import numpy as np
import matplotlib.pylab as plt

# def regionOfInterest(img, vertices):
#     mask = np.zeros_like(img)
#     # channel_count = img.shape[2]
#     match_mask_color = 255
#     cv2.fillPoly(mask, vertices, match_mask_color)
#     masked_image = cv2.bitwise_and(img, mask)
#     return masked_image

def drawTheLines(img, lines):
    cimg = np.copy(img)
    blank_image = np.zeros((cimg.shape[0], cimg.shape[1], 3), dtype=np.uint8)

    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(blank_image, (x1, y1), (x2, y2), (36, 215, 255), thickness=5)
    cimg = cv2.addWeighted(cimg, 0.8, blank_image, 1, 0.0)
    return cimg

def process(image):


    height = image.shape[0]
    width = image.shape[1]

    region_of_interest_vertices = [
        (0, height),
        (width/2, height/2),
        (width, height)
    ]

    grey_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    canny_image = cv2.Canny(grey_image, 100, 200)

    lines = cv2.HoughLinesP(canny_image, rho=6, theta=np.pi/60,
                            threshold=300, lines=100, minLineLength=40, maxLineGap=25)

    image_with_lines = drawTheLines(image,lines)
    return image_with_lines

cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret, frame = cap.read()
    frame = process(frame)
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.DestroyAllWindows()