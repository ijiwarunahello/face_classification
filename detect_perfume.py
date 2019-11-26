import torch
from network import Net
import cv2
import os
import glob
from torchsummary import summary
import torchvision.transforms as transforms


# constant
cascade_file_path = "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml"


def detect_who(image_path, model):
    image = cv2.imread(image_path)
    if image is None:
        print("Do not open")
    image_gs = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(cascade_file_path)
    face_list = cascade.detectMultiScale(image_gs, scaleFactor=1.1,
                                         minNeighbors=2, minSize=(64, 64))

    count = 0
    if len(face_list) > 0:
        for rect in face_list:
            count += 1
            x, y, width, height = rect
            print("X: {}, Y: {}, width: {}, height: {}".format(x, y, width, height))
            image_face = image[y: y+height, x:x+width]
            if image_face.shape[0] < 64:
                continue
            image_face = cv2.resize(image_face, (64, 64))
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5),
                                                                 (0.5, 0.5, 0.5))])
            image_face = transform(image_face)
            image_face = image_face.view(1, 3, 64, 64)

            output = model(image_face)

            member_label = output.argmax(dim=1, keepdim=True)
            name = label2name(member_label)

            print("output: {}".format(output))
            cv2.rectangle(image, (x, y), (x+width, y+height), (255, 0, 0), thickness=3)
            cv2.putText(image, name, (x, y+height+20),
                        cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2)
    else:
        print("no face")
    return image


def label2name(member_label):
    if member_label == 0:
        name = "a-chan"
    elif member_label == 1:
        name = "kashiyuka"
    elif member_label == 2:
        name = "nocchi"
    return name


def main():
    model = Net()
    model.load_state_dict(torch.load("perfume_cnn.pt"))
    model.eval()

    summary(model, (3, 64, 64))

    path_list = glob.glob("who_is_this_member/*")
    print("path_list: {}".format(path_list))

    out_dir = "./her_name_is"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for path in path_list:
        image_name = path.split("/")[-1]
        who = detect_who(path, model)
        save_path = os.path.join(out_dir + "/" + image_name)
        cv2.imwrite(save_path, who)


if __name__ == '__main__':
    main()
