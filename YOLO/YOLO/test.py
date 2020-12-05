import cv2
win_name = 'Image detection'
img = cv2.imread("test.jpg")
cv2.imshow(win_name,img)
cv2.waitKey(0)
cv2.destroyAllWindows()