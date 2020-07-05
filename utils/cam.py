import cv2

def get_image(path = None) :
    if path == None :
        #Make object
        cam = cv2.VideoCapture(0)
        #Create named window
        cv2.namedWindow("test")
        #Counter no of images
        img_counter = 0
        while True:
            ret, frame = cam.read() #read frame
            cv2.imshow("test", frame) #show frame
            if not ret:
                break
            k = cv2.waitKey(1)

            if k%256 == 27:
                # ESC pressed to close window
                print("Escape hit, closing...")
                break
            elif k%256 == 32:
                # SPACE pressed to capture Image
                #img_name = "opencv_frame_{}.jpg".format(img_counter) #relative image path
                #cv2.imwrite(img_name, frame) #Save image
                #print("{} written!".format(img_name))
                frame = frame
                img_counter += 1
                break
        #Closing camera
        cam.release()
        cv2.destroyAllWindows()

        # if not os.path.exists('captured'):
        #     os.makedirs('captured')
        # cv2.imwrite('cam.jpg',frame)
        # image_path = get_images('captured/')
        # return image_path

        return frame


        
# cap_image = get_image()
# cv2.imshow('captured', cap_image)
# cv2.waitKey(5000) #time in milliseconds
