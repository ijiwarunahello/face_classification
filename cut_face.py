import cv2
import os
import glob


# constant
in_dir = "./data/*"
out_dir = "./face_data/"
in_jpg = glob.glob(in_dir)
in_jpg_number = []
for i in range(len(in_jpg)):
    in_jpg_number.append(glob.glob(in_jpg[i] + "/*"))
in_file_name = os.listdir("./data")

for member_num in range(len(in_file_name)):
    output_path = out_dir + "/" + in_file_name[member_num]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for num in range(len(in_jpg_number[member_num])):
        image = cv2.imread(in_jpg_number[member_num][num])
        if image is None:
            print("Not open")
            continue
        image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cascade = cv2.CascadeClassifier("/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml")
        face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1, minNeighbors=2, minSize=(64, 64))

        count = 0
        if len(face_list) > 0:
            for rect in face_list:
                count += 1
                x, y, width, height = rect
                print(x, y, width, height)
                image_face = image[y:y+height, x:x+width]
                if image_face.shape[0] < 64:
                    continue
                image_face = cv2.resize(image_face, (64, 64))
                file_name = os.path.join(
                    out_dir + str(in_jpg_number[member_num][num][7:-4]) + "_" + str(count) + ".jpg")
                print(file_name)
                cv2.imwrite(str(file_name), image_face)
        else:
            print("no face")
            continue
