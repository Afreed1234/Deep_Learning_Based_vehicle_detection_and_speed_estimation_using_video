import customtkinter as ctk
import random
import cv2
import tkinter as tk
import PIL.Image, PIL.ImageTk
from PIL import ImageTk, Image
from customtkinter import filedialog
# import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from objTracker import *
import time
from math import dist
# limit=80

tracker = EuclideanDistTracker()
def open_image(image_path,speed):
    new_window = ctk.CTkToplevel(root)
    new_window.title("Vehicle Speed Exceeded")

    # Load the image
    # image_path = "C:\Projects\DeepLearning Based Vehicle Detection and Speed Estimation Using Video\Training\TrafficRecord\ExceededVehicles_id_2_speed_101.14962338925554.jpg"  # Replace "path_to_your_image.jpg" with the path to your image
    img = Image.open(image_path)
    img = img.resize((300, 300), Image.Resampling.LANCZOS)  # Resize the image with anti-aliasing

    # Display the image
    img = ImageTk.PhotoImage(img)
    label = ctk.CTkLabel(new_window, image=img,text=str(speed)+"Kmph!!")
    label.image = img  # fKeep a reference to the image
    label.pack()
#
# def capture(self, img, x, y, h, w, sp, id):
#     if (self.capf[id] == 0):
#         self.capf[id] = 1
#         self.f[id] = 0
#         crop_img = img[y:y + h, x:x + w]
#         n = "_id_" + str(id) + "_speed_" + str(sp)
#         file = 'C://Projects//DeepLearning Based Vehicle Detection and Speed Estimation Using Video//Training//TrafficRecord//' + n + '.jpg'
#         cv2.imwrite(file, crop_img)
#         self.count += 1
#         fileTR = open(
#             "C://Projects//DeepLearning Based Vehicle Detection and Speed Estimation Using Video//Training//TrafficRecord//SpeedRecord.txt",
#             "a")
#         if (sp > limit):
#             fileEX = 'C://Projects//DeepLearning Based Vehicle Detection and Speed Estimation Using Video//Training//TrafficRecord//ExceededVehicles//' + n + '.jpg'
#             cv2.imwrite(fileEX, crop_img)
#             fileTR.write(str(id) + " \t " + str(sp) + " <-- exceeded\n")
#             self.exceeded += 1
#         else:
#             fileTR.write(str(id) + " \t " + str(sp) + "\n")
#         fileTR.close()
#         self.ids_DATA.append((id))
#         self.spd_DATA.append((sp))
#

# Global variables to store PhotoImage objects
photo_images = []
# global file_path
# Function to open file dialog and select video file
def open_file():
    # print(photo_images)
    # photo_images.clear()
    file_path = ctk.filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi;*.mkv")])
    if file_path:
        play_video(file_path)

# Function to play the video
def play_video(file_path):
    # for widget in frame1.winfo_children():
    #     if widget !='!ctklabel' or widget!='!ctkbutton':
    #         widget.destroy()
    widgets_to_keep=[video_label,browse_button]
    for widget in frame1.winfo_children():
        if widget not in widgets_to_keep:
            widget.destroy()
    print(file_path)
    entr = tk.StringVar()
    entry = ctk.CTkEntry(frame1,placeholder_text="Enter Speed Limit")
    entry.pack(side="left", expand=True, fill=None)

    # print("Limit1",type(limit))
    detect_button = ctk.CTkButton(frame1, text="Detect And Estimate", command=lambda: detect_estimate(file_path,entry))
    detect_button.pack(side="left",expand=True,fill=None)

    cap = cv2.VideoCapture(file_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the target width and height for display
    target_width = 640
    target_height = 480

    # Create a canvas to display video frames
    canvas = ctk.CTkCanvas(frame1, width=target_width, height=target_height)
    canvas.pack()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Resize the frame to fit the target width and height
        frame = cv2.resize(frame, (target_width, target_height))
        # Convert frame to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert frame to ImageTk format
        img = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_rgb))
        # Store the PhotoImage object in the list
        photo_images.append(img)
        # Update canvas with the new frame
        canvas.create_image(0, 0, anchor=ctk.NW, image=img)
        # Delay to display the frame
        canvas.update()
    cap.release()



# Function to generate 5 random labels
def generate_random_labels():
    for i in range(5):
        label_text = "Random Label {}".format(i+1)
        label = ctk.CTkLabel(frame2, text=label_text)
        label.pack()

def detect_estimate(file_path,entry):
    widget_to_keep=[frame2_label]
    for widget in frame2.winfo_children():
        if widget not in widget_to_keep:
            widget.destroy()
    model = YOLO('yolov8n.pt')
    limit = entry.get()
    tracker.setLimit(int(limit))
    def RGB(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            colorsBGR = [x, y]
            print(colorsBGR)

    cv2.namedWindow('RGB')
    cv2.setMouseCallback('RGB', RGB)

    cap = cv2.VideoCapture(file_path)

    my_file = open("coco.txt", "r")
    data = my_file.read()
    class_list = data.split("\n")
    # print(class_list)

    count = 0


    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    cy1 = 322
    cy2 = 368

    offset = 6

    vh_down = {}
    counter = []
    limit=tracker.getLimit()
    vh_up = {}
    counter1 = []
    frame_skip = 3
    frame_count = 0
    exceed = 0
    while True:

        print("Limit2", limit)
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % 3 != 0:
            continue
        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame)
        #   print(results)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")
        #    print(px)
        list = []

        for index, row in px.iterrows():
            #        print(row)

            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]
            if str(c) == 'car' or str(c) == 'truck':
                list.append([x1, y1, x2, y2])
                print("list", list)
                # list.append([x1, y1, x2, y2])
                # print("-------------------------------------------------------------------------------------------------List",list)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)
            # elif 'truck' in c:
            #     list.append([x1, y1, x2, y2])
            # elif 'bus' in c:
            #     list.append([x1, y1, x2, y2])
            # elif 'motorcycle' in c:
            #     list.append([x1, y1, x2, y2])
        bbox_id = tracker.update(list)

        for bbox in bbox_id:
            x3, y3, x4, y4, id = bbox
            cx = int(x3 + x4) // 2
            cy = int(y3 + y4) // 2
            #
            # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

            if cy1 < (cy + offset) and cy1 > (cy - offset):
                vh_down[id] = time.time()
            if id in vh_down:
                if cy2 < (cy + offset) and cy2 > (cy - offset):
                    elapsed_time = time.time() - vh_down[id]
                    if counter.count(id) == 0:
                        counter.append(id)
                        distance = 10  # meters
                        a_speed_ms = distance / elapsed_time
                        a_speed_kh = a_speed_ms * 3.6
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        # cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                        if (a_speed_kh < limit):
                            cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                        (255, 0, 0), 2)
                            # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)

                            print(
                                "Speed:------------------------------------------------------------------------------>",
                                a_speed_kh)
                        else:
                            tracker.capture(frame, x3, y3, x4, y4, a_speed_kh, id)
                            im=tracker.getFileList()
                            print("im", im)
                            cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                        (0, 0, 255), 2)

                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                            # # im = tracker.capture()
                            # label = ctk.CTkButton(frame2, text="Speed Limit Exceeded: Speed->" + str(a_speed_kh),
                            #                       command=lambda: open_image(im[-1],round(a_speed_kh,2)))
                            # label.pack()
                            exceed+=1
                            print(
                                "Speed Exceeded:------------------------------------------------------------------------------>",
                                a_speed_kh)

                        # cv2.putText(frame, str(int(a_speed_kh)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                        #             (0, 255, 255), 2)

            #####going UP#####
            if cy2 < (cy + offset) and cy2 > (cy - offset):
                vh_up[id] = time.time()
            if id in vh_up:

                if cy1 < (cy + offset) and cy1 > (cy - offset):
                    elapsed1_time = time.time() - vh_up[id]

                    if counter1.count(id) == 0:
                        counter1.append(id)
                        distance1 = 10  # meters
                        a_speed_ms1 = distance1 / elapsed1_time
                        a_speed_kh1 = a_speed_ms1 * 3.6
                        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                        # cv2.putText(frame, str(id), (x3, y3), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                        if (a_speed_kh1 < limit):
                            cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                        (255, 0, 0), 2)
                            # cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)

                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)
                            print(
                                "Speed:------------------------------------------------------------------------------>",
                                a_speed_kh1)
                        else:
                            tracker.capture(frame, x3, y3, x4, y4, a_speed_kh1, id)

                            im = tracker.getFileList()
                            print("im", im)
                            cv2.putText(frame, str(int(a_speed_kh1)) + 'Km/h', (x4, y4), cv2.FONT_HERSHEY_COMPLEX, 0.8,
                                        (0, 0, 255), 2)

                            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 0, 255), 2)
                            # im=tracker.capture()
                            # label = ctk.CTkButton(frame2, text="Speed Limit Exceeded: Speed->"+str(a_speed_kh1),command=lambda:open_image(im[-1],round(a_speed_kh1,2)))
                            # label.pack()
                            exceed+=1
                            print(
                                "Speed Exceeded:------------------------------------------------------------------------------>",
                                a_speed_kh1)

        cv2.line(frame, (0, cy1), (1020, cy1), (255, 255, 255), 3)
        cv2.line(frame, (0, cy2), (1020, cy2), (255, 255, 255), 3)
        d = (len(counter))
        u = (len(counter1))
        cv2.putText(frame, ('goingdown:-') + str(d), (60, 90), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, ('goingup:-') + str(u), (60, 130), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow("RGB", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    widgets_to_keep = [frame2_label]
    for widget in frame2.winfo_children():
        if widget not in widgets_to_keep:
            widget.destroy()
    for (im,sp) in zip(tracker.getFileList(),tracker.getSpeedList()):
    # for im,sp in :
        random_label_button = ctk.CTkButton(frame2, text=sp, command=lambda im=im,sp=sp: open_image(im, sp))
        print("im--------------------------------------------------",im)
        random_label_button.pack()
    tracker.clearList()
def set_background(root, image_path):
    canvas = tk.Canvas(root, width=900, height=600)
    canvas.pack(fill="both", expand=True)

    # Load the image
    img = Image.open(image_path)
    img = img.resize((900, 600), Image.Resampling.LANCZOS)
    bg_image = ImageTk.PhotoImage(img)

    # Set the background image
    canvas.create_image(0, 0, anchor="nw", image=bg_image)

    # Ensure the image sticks even when the window is resized
    canvas.image = bg_image


# Create the main window
root = ctk.CTk()
root.title("Vehicle detection And Speed Estimation")
root.minsize( 900,600)
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
# Add video player and file browser to frame 1
canvas = tk.Canvas(root, width=900, height=600)
canvas.pack(fill="both", expand=True)
image_path="background.jpg"
    # Load the image
img = Image.open(image_path)
img = img.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
bg_image = ImageTk.PhotoImage(img)

    # Set the background image
canvas.create_image(0, 0, anchor="nw", image=bg_image)

    # Ensure the image sticks even when the window is resized
canvas.image = bg_image

video_label = ctk.CTkLabel(canvas, text="Deep Learning Based Vehicle Detection And Speed Estimation in Traffic Using Videos",font = ("Corbel",36),padx=20,pady = 20,justify="center")
video_label.pack(side="top")
# Create frames
frame1 = ctk.CTkFrame(canvas)
frame1.pack(side="left", expand=True, fill=None)

frame2 = ctk.CTkFrame(canvas)
frame2.pack(side="left", expand=True, fill=None)

# background_image_path = "../WhatsApp Image 2024-03-28 at 2.35.03 PM.jpeg"
# set_background(frame1, background_image_path)
# video_label.place(anchor="center")

browse_button = ctk.CTkButton(frame1, text="Browse Video File", command=open_file,anchor="center")
browse_button.pack(side="top")


# Add 5 random labels to frame 2
frame2_label=ctk.CTkLabel(frame2,text="Vehicles Exceeded Speed Limit",justify="right")
frame2_label.pack()


# random_label_button = ctk.CTkButton(frame2, text="Generate Random Labels", command=generate_random_labels)
# random_label_button.pack()

root.mainloop()
