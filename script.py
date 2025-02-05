import cv2, glob

class Face_detection():
    def image_face_detection():
        # select all images
        images = glob.glob("*.png")

        # mention the classifier to detect a particular thing
        detect_face = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        # loop through all images
        for image in images:
            img = cv2.imread(image)
            grayScale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detect_face.detectMultiScale(grayScale_img, 1.3, 5)

            for (x, y, w, h) in faces:
                updated_image = cv2.rectangle(img, (x,y), (x+w, y+h), (255,255,0), 2)

            cv2.imshow("Face Detection",img) # to show the image
            cv2.waitKey(800) # 0 to close image when we want instead of automatically after speicified seconds
            cv2.destroyAllWindows()


if __name__ == "__main__":
    Face_detection.image_face_detection()