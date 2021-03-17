"""
script for streamlit web app
"""

import streamlit as st
import cv2 as cv
cascade_classifier = cv.CascadeClassifier(cv.data.haarcascades + \
    'haarcascade_frontalcatface_extended.xml')
import tempfile

import pandas as pd
import numpy as np
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, log_loss





# header = st.beta_container()
# dataset = st.beta_container()
# features = st.beta_container()
# model = st.beta_container()

# with header:
#     st.title('Welcome to Picture Purrfect!')



def new_video_pick(f):
    # First empty data folder if a new file needs to be uploaded
    mypath='data/'
    filelist = [file for file in os.listdir(mypath)]
    for file in filelist:
        os.remove(os.path.join(mypath, file))

    # st.subheader('Picking your purrfect picture...')
    my_slot1 = st.empty()
    my_slot1.subheader('Picking your purrfect picture...')

    video_name = f.name
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    cap = cv.VideoCapture(tfile.name)
    stframe = st.empty()

    
    # YOLO detection
    # Load Yolo
    net = cv.dnn.readNet("./YOLO_detection/yolov3_custom_last.weights", "./YOLO_detection/yolov3_testing.cfg")

    # Name custom object
    # classes = ["Koala"]
    classes = ["cat_eye", "cat_ear", "cat_nose"]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))


    # Read video frame by frame and extract info in each frame
    i=0
    while True:
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if ret:
            frame_filename = video_name+'_'+str(i)+'.jpg'
            print(frame_filename)
            frame_display = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            detections = cascade_classifier.detectMultiScale(frame, 1.3, 5)
            if (len(detections)>0):
                (x,y,w,h) = detections[0]
                # frame = cv.rectangle(frame,(x,y),(x+w,y+h),(151, 191, 157),2) # Comment out on 3/13/2021 Save frame without bounding box
                
                if i%10 ==0:
                    # frame_filename = video_name+'_'+str(i)+'.jpg'
                    cv.imwrite('data/'+frame_filename,frame)

                    (x,y,w,h) = detections[0]

                    img = cv.imread('data/'+frame_filename)
                    img = frame

                    cropped = img[y:y+h,x:x+w]
                    kernel_size=3;
                    laplacian_var_roi = cv.Laplacian(cropped, cv.CV_64F,kernel_size).var()
                    scale=img.shape[1]//w

                    resized = cv.resize(cropped, (w*scale,h*scale), interpolation=cv.INTER_CUBIC)
                    lp_cat = cv.Laplacian(resized, cv.CV_64F,kernel_size).var()

                    lp_all = cv.Laplacian(img, cv.CV_64F,kernel_size).var()
                    lp_ratio = lp_cat/lp_all

                    # Check if blurry
                    if lp_cat/lp_all < 0.03:
                        blur = 1
                        # blur.append(1)
                        # print("Image blurry")
                    else:
                        blur = 0
                        # blur.append(0)
                        # print("Image okay")


                    # Canny edges
                    canny = cv.Canny(img,100,200)
                    # Crop canny
                    cropped_canny = canny[y:y+h,x:x+w]
                    kernel_size = 3;
                    lp_cat_canny = cv.Laplacian(cropped_canny, cv.CV_64F,kernel_size).var()
                    lp_all_canny = cv.Laplacian(canny, cv.CV_64F,kernel_size).var()
                    lp_ratio_canny = lp_cat_canny/lp_all_canny



                    # Cat face info
                    img_size = img.shape[0]*img.shape[1]
                    # print('Size of image:',img_size)
                    cat_size = w*h
                    # print('Size of cat face:',cat_size)
                    size_ratio = cat_size/img_size
                    # print('Size ratio:', size_ratio)

                    img_ctr = (img.shape[1]/2, img.shape[0]/2)
                    cat_ctr = (x,y)

                    import math
                    to_ctr = math.dist(img_ctr,cat_ctr)
                    # print('Distance to center:', to_ctr)
                    cat_x = x/img.shape[1]
                    cat_y = y/img.shape[0]

                    
                    # YOLO detection
                    # Loading image
                    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
                    height, width, channels = img.shape

                    # Detecting objects
                    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

                    net.setInput(blob)
                    outs = net.forward(output_layers)

                    # Showing informations on the screen
                    class_ids = []
                    confidences = []
                    boxes = []
                    for out in outs:
                        for detection in out:
                            scores = detection[5:]
                            class_id = np.argmax(scores)
                            confidence = scores[class_id]
                            if confidence > 0.1:
                                # Object detected
                                # print(class_id)
                                center_x = int(detection[0] * width)
                                center_y = int(detection[1] * height)
                                w = int(detection[2] * width)
                                h = int(detection[3] * height)

                                # Rectangle coordinates
                                x = int(center_x - w / 2)
                                y = int(center_y - h / 2)

                                boxes.append([x, y, w, h])
                                confidences.append(float(confidence))
                                class_ids.append(class_id)

                    indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
                    
                    # print(indexes)
                    font = cv.FONT_HERSHEY_SIMPLEX

                    eyes = []
                    ears = []
                    nose = []

                    for j in range(len(boxes)):
                        if j in indexes:
                            x, y, w, h = boxes[j]
                            label = str(classes[class_ids[j]])
                            # print(label,x,y,w,h)
                            if label == "cat_eye":
                                eyes.append((x,y,w,h))
                            elif label == "cat_ear":
                                ears.append([x,y,w,h])
                            elif label == "cat_nose":
                                nose.append([x,y,w,h])


                    rows.append([frame_filename,lp_cat,lp_all,lp_ratio,lp_cat_canny,lp_all_canny,lp_ratio_canny,blur, to_ctr,cat_x,cat_y,cat_size,size_ratio,eyes,ears,nose])
                    print(len(rows))
                # cv.waitKey(1)


            # Display frame by frame
            # stframe.image(frame_display)

            i+=1
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        else:
            break


    print(len(rows))
    # Save the results as csv file
    import csv

    save_filename = 'TEST_face_features_st.csv'
    with open('data/'+save_filename, 'w') as f:
        write = csv.writer(f)
        write.writerow(headers)
        write.writerows(rows)


    # Read .csv as DataFrame
    test = pd.read_csv('data/'+save_filename)

    # Feature engineering of YOLO detected facial features
    import ast
    # Convert to actual list and count number of eyes/ears/nose detected
    test['eyes'] = test['eyes'].apply(lambda x: ast.literal_eval(x))
    test['num_eye'] = test['eyes'].apply(lambda x: len(x))
    test['ears'] = test['ears'].apply(lambda x: ast.literal_eval(x))
    test['num_ear'] = test['ears'].apply(lambda x: len(x))
    test['nose'] = test['nose'].apply(lambda x: ast.literal_eval(x))
    test['num_nose'] = test['nose'].apply(lambda x: len(x))

    # eye position
    test['eye_x1'] = test['eyes'].apply(lambda x: x[0][0] if (len(x)>0) else 0)
    test['eye_y1'] = test['eyes'].apply(lambda x: x[0][1] if (len(x)>0) else 0)
    test['eye_x2'] = test['eyes'].apply(lambda x: x[1][0] if (len(x)>1) else 0)
    test['eye_y2'] = test['eyes'].apply(lambda x: x[1][1] if (len(x)>1) else 0)
    # eye width
    test['eye_w1'] = test['eyes'].apply(lambda x: x[0][2] if (len(x)>0) else 0)
    test['eye_w2'] = test['eyes'].apply(lambda x: x[1][2] if (len(x)>1) else 0)
    test['eye_w'] = test['eyes'].apply(lambda x: sum([x[i][2] for i in range(len(x))])/len(x) if len(x)!=0 else 0)
    # eye height
    test['eye_h1'] = test['eyes'].apply(lambda x: x[0][3] if (len(x)>0) else 0)
    test['eye_h2'] = test['eyes'].apply(lambda x: x[1][3] if (len(x)>1) else 0)
    test['eye_h'] = test['eyes'].apply(lambda x: sum([x[i][3] for i in range(len(x))])/len(x) if len(x)!=0 else 0)
    # eye size and ratio to face size
    test['eye_size'] = test['eye_w']*test['eye_h']
    test['eye_ratio'] = test['eye_size'] / test['face_size']

    # nose position
    test['nose_x'] = test['nose'].apply(lambda x: x[0][0] if (len(x)>0) else 0)
    test['nose_y'] = test['nose'].apply(lambda x: x[0][1] if (len(x)>0) else 0)

    # ear width
    test['ear_w1'] = test['ears'].apply(lambda x: x[0][2] if (len(x)>0) else 0)
    test['ear_w2'] = test['ears'].apply(lambda x: x[1][2] if (len(x)>1) else 0)
    test['ear_w'] = test['ears'].apply(lambda x: sum([x[i][2] for i in range(len(x))])/len(x) if len(x)!=0 else 0)
    # ear height
    test['ear_h1'] = test['ears'].apply(lambda x: x[0][3] if (len(x)>0) else 0)
    test['ear_h2'] = test['ears'].apply(lambda x: x[1][3] if (len(x)>1) else 0)
    test['ear_h'] = test['ears'].apply(lambda x: sum([x[i][3] for i in range(len(x))])/len(x) if len(x)!=0 else 0)


    X_test_df = test.copy()
    # Treat features

    # Continuous variables
    features = ['lp_cat', 'lp_all', 'lp_ratio', 'lp_cat_canny', 'lp_all_canny', 'lp_ratio_canny',\
                'face_size', 'size_ratio', 'to_ctr',\
                'num_eye','num_ear', 'num_nose', 
                'eye_w1', 'eye_w2', 'eye_w', 'eye_h1', 'eye_h2', 'eye_h',
                'eye_size', 'eye_ratio',
                'ear_w1', 'ear_w2', 'ear_w', 'ear_h1', 'ear_h2', 'ear_h',
                ]
    X_test_con = X_test_df[features]

    # Since we're using more than one feature, let's scale our features
    scaler = StandardScaler()
    X_test_con_scaled = scaler.fit_transform(X_test_df[features])
    cont_columns = X_test_con.columns
    X_test_con_df = pd.DataFrame(X_test_con_scaled, columns=cont_columns, index=X_test_con.index)

    # Combine Con and Cat
    X_test = pd.concat([X_test_con_df,X_test_df[['blur']]], axis='columns')
    # Try without 'blur'
    X_test = X_test_con_df




    # Load model
    import pickle
    model = pickle.load(open('randomforest.pkl','rb'))
    # Prediction
    test['prob'] = model.predict_proba(X_test)[:, 1]
    # Save the DataFrame
    df = test.sort_values('prob',ascending=False)
    df.to_csv('./data/df_predict.csv')


    # Display best frame
    my_slot1.subheader('Your purrfect picture!')
    best_frame = df.frame_name.iloc[0]
    stframe = st.empty()
    stframe.image('data/' + best_frame)
 



    print('Done')






st.title('Welcome to Picture Purrfect!')

menu = ['Home','Try it out','Analysis','About']
choice = st.sidebar.selectbox("Menu", menu)





# Saved as .csv in data folder
# Headers (=column names for df)
headers = ['frame_name','lp_cat','lp_all','lp_ratio','lp_cat_canny','lp_all_canny','lp_ratio_canny','blur',\
        'to_ctr','cat_x','cat_y','face_size','size_ratio',\
        'eyes','ears','nose']
# Empty list to record each row (file)
# Will be a list of lists
rows = []
print(rows)

if choice == "Try it out":
    st.subheader("Your Purrfect Picture")

    # Check if data folder has images already
    import os

    mypath='data/'
    filelist = [f for f in os.listdir(mypath)]

    if 'df_predict.csv' in filelist:
        try:
            df = pd.read_csv('./data/df_predict.csv')
            # Display best frame
            best_frame = df.frame_name.iloc[0]
            stframe = st.empty()
            stframe.image('data/' + best_frame)
            # Display the DataFrame
            show_df = st.checkbox('Show Frame Data')
            if show_df:
                st.write(df.iloc[:,1:])


            # Option to upload a new video
            st.subheader('Upload a new video')
            f = st.file_uploader('Upload your shot video here:',type=['MOV','MP4'],key='another video')
            if f is not None:
            # if f.name != video_name:
                # Run the function to pick best frame
                new_video_pick(f)
                show_df = st.checkbox('Show Frame Data',key='new video')
                if show_df:
                    df = pd.read_csv('./data/df_predict.csv')
                    st.write(df.iloc[:,1:])
        except:
            f = st.file_uploader('Upload your video here:',type=['MOV','MP4'],key='new video')
            if f is not None:
                new_video_pick(f)
                video_name = f.name
                # Display the DataFrame
                show_df = st.checkbox('Show Frame Data',key='new video')
                if show_df:
                    st.write(df.iloc[:,1:])


    else:        
        # Upload video for the first time
        f = st.file_uploader('Upload your video here:',type=['MOV','MP4'],key='new video')
        if f is not None:
            new_video_pick(f)
            video_name = f.name
            # Display the DataFrame
            show_df = st.checkbox('Show Frame Data',key='new video')
            if show_df:
                st.write(df.iloc[:,1:])

    

if choice == "Home":

    st.text('This will appear first')
    # Appends some text to the app.

    my_slot1 = st.empty()
    # Appends an empty slot to the app. We'll use this later.

    my_slot2 = st.empty()
    # Appends another empty slot.

    st.text('This will appear last')
    # Appends some more text to the app.

    my_slot1.text('This will appear second')
    # Replaces the first empty slot with a text string.

    my_slot2.line_chart(np.random.randn(20, 2))
    # Replaces the second empty slot with a chart.


    import numpy as np
    import time

    # Get some data.
    data = np.random.randn(10, 2)

    # Show the data as a chart.
    chart = st.line_chart(data)

    # Wait 1 second, so the change is clearer.
    time.sleep(1)

    # Grab some more data.
    data2 = np.random.randn(10, 2)

    # Append the new data to the existing chart.
    chart.add_rows(data2)


    my_slot1 = st.empty()
    my_slot1.subheader('Doing it')
    time.sleep(1)
    my_slot1.subheader('DONE!')
