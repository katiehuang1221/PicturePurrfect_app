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
from altair import *

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import precision_score, recall_score, precision_recall_curve,f1_score, fbeta_score, log_loss






# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# def remote_css(url):
#     st.markdown(f'<link href="{url}" rel="stylesheet">', unsafe_allow_html=True)    

# def icon(icon_name):
#     st.markdown(f'<i class="material-icons">{icon_name}</i>', unsafe_allow_html=True)

# local_css("style.css")
# remote_css('https://fonts.googleapis.com/icon?family=Material+Icons')

# icon("search")
# selected = st.text_input("", "Search...")
# button_clicked = st.button("OK")

# st.markdown(
#     """
# <style>
# .reportview-container .markdown-text-container {
#     font-family: monospace;
# }
# .sidebar .sidebar-content {
#     background-image: url("https://i.pinimg.com/originals/80/38/5d/80385d6fa0e6cafef67442020b096b85.jpg");
#     background-size: cover;
#     color: #158c8c;
# }
# .Widget>label {
#     color: #c2a9c2;
#     font-family: monospace;
# }
# [class^="st-b"]  {
#     color: white;
#     font-family: monospace;
# }
# .st-bb {
#     background-color: transparent;
# }

# .st-bb {
#     background-color: #c2a9c2;
# }

# footer {
#     font-family: monospace;
# }
# .reportview-container .main footer, .reportview-container .main footer a {
#     color: #158c8c;
# }
# header .decoration {
#     background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
# }

# </style>
# """,
#     unsafe_allow_html=True,
# )


import time
# my_bar = st.progress(0)
# for percent_complete in range(100):
#     time.sleep(0.1)
#     my_bar.progress(percent_complete + 1)

# st.spinner()
# with st.spinner(text='Picking your purrfect picture'):
#     st.balloons()
#     time.sleep(5)
#     st.success('Done')
# st.balloons()


st.title('Welcome to Picture Purrfect!')
my_slot_header_all = st.empty()

st.sidebar.header('MENU')
menu = ['Home','Your purrfect pic','Customize picker', 'Analysis','About']
# ['Test','Home','Your purrfect pic','Customize picker', 'Analysis','About']
choice = st.sidebar.selectbox("", menu)

page_bg_img = '''
<style>
body {
background-image: url("https://i.pinimg.com/originals/80/38/5d/80385d6fa0e6cafef67442020b096b85.jpg");

}
</style>
'''
st.markdown(page_bg_img, unsafe_allow_html=True)


# Saved as .csv in data folder
# Headers (=column names for df)
headers = ['frame_name','lp_cat','lp_all','lp_ratio','lp_cat_canny','lp_all_canny','lp_ratio_canny','blur',\
        'cat_x','cat_y','face_size','size_ratio',\
        'eyes','ears','nose',
        'img_ctr']
# Empty list to record each row (file)
rows = []







import streamlit as st
import streamlit_theme as stt

stt.set_theme({'primary': '#1b3388',
                'secondary': '#439140'})



def paginator(label, items, items_per_page=10, on_sidebar=True):
        """Lets the user paginate a set of items.
        Parameters
        ----------
        label : str
            The label to display over the pagination widget.
        items : Iterator[Any]
            The items to display in the paginator.
        items_per_page: int
            The number of items to display per page.
        on_sidebar: bool
            Whether to display the paginator widget on the sidebar.
            
        Returns
        -------
        Iterator[Tuple[int, Any]]
            An iterator over *only the items on that page*, including
            the item's index.
        Example
        -------
        This shows how to display a few pages of fruit.
        >>> fruit_list = [
        ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
        ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
        ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
        ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
        ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
        ... ]
        ...
        ... for i, fruit in paginator("Select a fruit page", fruit_list):
        ...     st.write('%s. **%s**' % (i, fruit))
        """

        # Figure out where to display the paginator
        if on_sidebar:
            location = st.sidebar.empty()
        else:
            location = st.empty()

        # Display a pagination selectbox in the specified location.
        items = list(items)
        n_pages = len(items)
        n_pages = (len(items) - 1) // items_per_page + 1
        page_format_func = lambda i: "Page %s" % i
        page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

        # Iterate over the items in the page to let the user display them.
        min_index = page_number * items_per_page
        max_index = min_index + items_per_page
        import itertools
        return itertools.islice(enumerate(items), min_index, max_index)
        


def new_video_pick(f,fine):

    # First empty data folder if a new file needs to be uploaded
    mypath='data/'
    filelist = [file for file in os.listdir(mypath)]
    for file in filelist:
        os.remove(os.path.join(mypath, file))

    # st.subheader('Picking your purrfect picture...')
    # my_slot1 = st.empty()
    # my_slot1.subheader('Picking your purrfect picture...')
    my_slot_result.subheader('Picking your purrfect picture...')

    video_name = f.name
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(f.read())
    cap = cv.VideoCapture(tfile.name)

    # total number of frames
    frames = cap.get(cv.CAP_PROP_FRAME_COUNT)
    print('Total num of frames:',frames)
    # frame per second
    fps = cap.get(cv.CAP_PROP_FPS)
    print('FPS is:', fps)
    fine = fine
    skip = int(fps/10)*fine
    print('SKIP is',skip)

    # calculate dusration of the video
    duration = int(frames / fps)
    print('Duration of the video is', duration)

    my_bar = my_slot_result_progress.progress(0)
    
    # for percent_complete in range(100):
    #     time.sleep(0.1)
    #     my_bar.progress(percent_complete+1)




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
    cat_count=0
    while True:
        my_bar.progress(int(i*100/frames))
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if ret:
            frame_filename = video_name+'_'+str(i)+'.jpg'
            print(frame_filename)
            frame_display = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

            detections = cascade_classifier.detectMultiScale(frame, 1.3, 5)
            if (len(detections)>0):
                cat_count+=1
                (x,y,w,h) = detections[0]
                # frame = cv.rectangle(frame,(x,y),(x+w,y+h),(151, 191, 157),2) # Comment out on 3/13/2021 Save frame without bounding box
                # i+=1
                if cat_count%skip ==0:
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
                    cat_x = x/img.shape[1]
                    cat_y = y/img.shape[0]
                    # to_ctr = math.dist(img_ctr,cat_ctr)
                    # to_ctr = math.dist((cat_x,cat_y),(0.5,0.5))

                    
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


                    rows.append([frame_filename,lp_cat,lp_all,lp_ratio,lp_cat_canny,lp_all_canny,lp_ratio_canny,blur,cat_x,cat_y,cat_size,size_ratio,eyes,ears,nose,img_ctr])
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

    try:
        # Feature engineering of YOLO detected facial features
        import ast
        # Convert to actual list and count number of eyes/ears/nose detected
        test['eyes'] = test['eyes'].apply(lambda x: ast.literal_eval(x))
        test['num_eye'] = test['eyes'].apply(lambda x: len(x))
        test['ears'] = test['ears'].apply(lambda x: ast.literal_eval(x))
        test['num_ear'] = test['ears'].apply(lambda x: len(x))
        test['nose'] = test['nose'].apply(lambda x: ast.literal_eval(x))
        test['num_nose'] = test['nose'].apply(lambda x: len(x))
        test['img_ctr'] = test['img_ctr'].apply(lambda x: ast.literal_eval(x))

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


        # Correct ridiculous values
        # add video column
        test['video'] = test['frame_name'].apply(lambda x: x.split('.M')[0])

        # Examine cat face detection
        test['avg_face'] = test.groupby(['video'])['face_size'].transform(np.mean)
        test['nose_to_ctr']=test.apply(lambda x: np.sqrt((x['nose_x']-x['img_ctr'][0])**2 + \
                                                    (x['nose_y']-x['img_ctr'][1])**2),axis=1)
        test['avg_to_ctr'] = test.groupby(['video'])['nose_to_ctr'].transform(np.mean)
        # Correct face_size
        correct_face = test.apply(lambda x: x.face_size if x.face_size>10000 else x.avg_face, axis=1)
        test['face_size'] = correct_face
        # Correct to_ctr
        correct_to_crt = test.apply(lambda x: x.nose_to_ctr if x.face_size>10000 else x.avg_to_ctr, axis=1)

        
        


        X_test_df = test.copy()
        # Treat features

        # Continuous variables
        features = ['lp_cat', 'lp_all', 'lp_ratio', 'lp_cat_canny', 'lp_all_canny', 'lp_ratio_canny',\
                    'face_size', 'size_ratio',\
                    'num_eye','num_ear', 'num_nose', 
                    'eye_w1', 'eye_w2', 'eye_w', 'eye_h1', 'eye_h2', 'eye_h',
                    'eye_size', 'eye_ratio',
                    'ear_w1', 'ear_w2', 'ear_w', 'ear_h1', 'ear_h2', 'ear_h',
                    'nose_to_ctr',
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
        model = pickle.load(open('model.pkl','rb'))
        # Prediction
        test['prob'] = model.predict_proba(X_test)[:, 1]
        # Save the DataFrame
        df = test.sort_values('prob',ascending=False)
        df.to_csv('./data/df_predict.csv')


        # Display best frame
        # my_slot1.subheader('Your purrfect picture!')
        my_slot_result_progress.empty()
        st.balloons()
        my_slot_result.subheader('Your purrfect picture!')
        best_frame = df.frame_name.iloc[0]
        # stframe = st.empty()
        # stframe.image('data/' + best_frame)
        my_slot_result_image.image('data/' + best_frame)
        # my_slot_result_like
        col1, col2,col3 = my_slot_result_like.beta_columns([1,0.26,0.2])
        # col1.markdown(":hearts:")
        col2.button('♥️...Love it!')
        col3.button('👿...Nope!')
        

    except:
        my_slot_result.subheader('Sorry...no cat face detected!  😿')
        my_slot_result_progress.empty()

 



    print('Done')








if choice == "Your purrfect pic":
    # Check if data folder has images already
    import os

    mypath='data/'
    filelist = [f for f in os.listdir(mypath)]

    my_slot_headline = st.empty()
    my_slot_upload= st.empty()
    my_slot_slider = st.empty()
    my_slot_file = st.empty()
    my_slot_pick_buton = st.empty()
    my_slot_result = st.empty()
    my_slot_result_progress =st.empty()
    my_slot_result_image = st.empty()
    my_slot_result_like = st.empty()
    my_slot_previous_result_head = st.empty()
    my_slot_previous_result_image = st.empty()



    

    # st.subheader('Upload a new video')
    # f = st.file_uploader('Upload your video here:',type=['MOV','MP4'],key='another video')

    # how_fine = st.select_slider(
    #     'How do you want us to go through the frames?',
    #     options=['course (I want it fast)', 'medium', 'fine (I can wait)'],)
    # st.write('You selected:', how_fine)

    # fine_convert = {'course (I want it fast)':2, 'medium':1, 'fine (I can wait)':0.5}
    # fine = fine_convert[how_fine]


    # if st.button('Start picking'):
    #     if f is not None:
    #         new_video_pick(f,fine)

    #     else:
    #         st.write('Please upload a video first!')


    
    # st.subheader("Your Purrfect Picture")

    if 'df_predict.csv' in filelist:
        try:
            my_slot_previous_result_head.subheader("Previous Purrfect Picture:")
            st.write("Don't like the photo we picked? No worries!\
                    Use the customized criteria selector on the left!")
            df = pd.read_csv('./data/df_predict.csv')
            # Display best frame
            best_frame = df.frame_name.iloc[0]
            # stframe = st.empty()
            # stframe.image('data/' + best_frame)
            my_slot_previous_result_image.image('data/' + best_frame)
            # # Display the DataFrame
            # show_df = st.checkbox('Show Frame Data')
            # if show_df:
            #     st.write(df.iloc[:,1:])
            
        except:
            my_slot_result.subheader("Please upload a new video!")
            # f = st.file_uploader('Upload your video here:',type=['MOV','MP4'],key='new video')
            # if f is not None:
            #     new_video_pick(f)
            #     video_name = f.name
            #     # # Display the DataFrame
            #     # show_df = st.checkbox('Show Frame Data',key='new video')
            #     # if show_df:
            #     #     st.write(df.iloc[:,1:])


    else:
        my_slot_headline.subheader("Let's tyr it out!")        
        # # Upload video for the first time
        # f = st.file_uploader('Upload your video here:',type=['MOV','MP4'],key='new video')
        # if f is not None:
        #     new_video_pick(f)
        #     video_name = f.name
        #     # # Display the DataFrame
        #     # show_df = st.checkbox('Show Frame Data',key='new video')
        #     # if show_df:
        #     #     st.write(df.iloc[:,1:])



    my_slot_headline.subheader('Upload a new video')
    f = my_slot_upload.file_uploader('Upload your video here:',type=['MOV','MP4'],key='another video test')
    if f is not None:
        my_slot_file.write(f.name)

    how_fine = my_slot_slider.select_slider(
        'How do you want the frames to be sifted?',
        options=['course (I want it fast)', 'medium', 'fine (I can wait)'],key='another video test')
    # st.write('You selected:', how_fine)

    fine_convert = {'course (I want it fast)':2, 'medium':1, 'fine (I can wait)':0.5}
    fine = fine_convert[how_fine]
    
    if my_slot_pick_buton.button('Start picking  🐾 ',key='empty test'):
        if f is not None:
            new_video_pick(f,fine)

        else:
            st.write('Please upload a video first!')

    


    # if st.button('Start picking'):
    #     if f is not None:
    #         new_video_pick(f,fine)

    #     else:
    #         st.write('Please upload a video first!')



    

   

    


    # try:
    #     # Display best frame
    #     best_frame = df.frame_name.iloc[0]
    #     stframe = st.empty()
    #     stframe.image('data/' + best_frame)
    # except:
    #     f = st.file_uploader('Upload your video here:',type=['MOV','MP4'],key='new video')
    #     if f is not None:
    #         new_video_pick(f)
    #         video_name = f.name
    #         # Display the DataFrame
    #         show_df = st.checkbox('Show Frame Data',key='new video')
    #         if show_df:
    #             st.write(df.iloc[:,1:])

    
if choice == "Customize picker":

    st.subheader('Customized picker')
    st.write("Don't like the photo we picked? No worries!\
        Use the customized criteria selector on the left!")

    
    my_slot_summary = st.empty()
    my_slot_summary2 = st.empty()

    # Create empty slots
    my_slot_image = st.empty()
    my_slot_caption_like = st.empty()
    

    my_slot_caption = st.empty()
    my_slot_dataframe = st.empty()
    

    try:
        # Default
        df = pd.read_csv('./data/df_predict.csv').sort_values(['prob'],ascending=[False])
        df['face_frame_ctr'] = abs((df['eye_x1']+df['eye_x2'])/2 - 0.5) +\
                           abs(((df['eye_y1']+df['eye_y2'])/2+df['nose_y'])/2 - 0.5)
        df['en_x_delta'] = abs((df['eye_x1']+df['eye_x2'])/2 - df['nose_x'])
        df['eye_h_sum'] = (df['eye_h1']+df['eye_h2'])*df['eye_ratio']
        df['eye_shape'] = (df['eye_h']/df['eye_w'])
        df['eye_shape_1'] = (df['eye_h1']/df['eye_w1'])
        df['eye_shape_2'] = (df['eye_h2']/df['eye_w2'])
        df['eye_shape_diff'] = abs(df['eye_shape_1']-df['eye_shape_2'])
        df['eye_height_diff'] = abs(df['eye_h1']-df['eye_h2'])
        # /df['en_x_delta']
        df['new_eye_size'] = df['eye_h']/df['size_ratio']
        best_frame = df.frame_name.iloc[0]
        # st.write(df)


        # Narrow down the options
        thres = int(len(df)*.3)+1
        # print('Length of df is',thres)
        df_filtered = df.sort_values('lp_cat',ascending=False).head(thres)
        


        # Sidebar for filter
        st.sidebar.subheader('Criteria Selector')
        criterion = st.sidebar.radio(
            "What are your criterion for a good picture?",
            ('Original','Sharp','Big eyes', 'Face at center', 'So close to the camera!', 'Give me something funny!'))

        if criterion == 'Original':
            df = pd.read_csv('./data/df_predict.csv').sort_values(['prob'],ascending=[False])
            my_slot_summary = st.empty()
            # my_slot_dataframe.write(df.iloc[:,1:])
            my_slot_image.image('data/' + best_frame)
            my_slot_caption.write('Picture purrfect!')
            
        else:
            if criterion == 'Sharp':
                df_filtered = df.sort_values('lp_cat_canny',ascending=False)
                # my_slot_caption.write('Am I sharp enough?')
                text = 'Am I sharp enough?'

            elif criterion == 'Face at center':
                # Generate features to address face angle
                df_filtered = df_filtered[df_filtered.num_eye == 2].sort_values('en_x_delta') 
                df_filtered = df_filtered.sort_values(['lp_cat_canny','face_frame_ctr'],ascending=[False,True]) .iloc[1:]
                # my_slot_caption.write('When the cat face is at the center of the photo.')  
                text = 'I am the center of the universe!'   

            elif criterion == 'So close to the camera!':
                df_filtered = df_filtered.sort_values('size_ratio',ascending=False)
                # my_slot_caption.write('This is when the cat face is the closest to your camera...')    
                text='Am I too close to the camera?'

            elif criterion == 'Big eyes':
                df_filtered = df_filtered.sort_values('new_eye_size',ascending=False).iloc[1:]
                df_filtered = df_filtered.sort_values(['eye_shape'],ascending=[False]).head(5)
                df_filtered = df_filtered.sort_values(['lp_cat_canny'],ascending=[False])
                # my_slot_caption.write('This is when the cat opened its eye.') 
                text='This is when I have my eyes wide open!'

            elif criterion == 'Give me something funny!':
                # df_filtered = df_filtered.sort_values('eye_shape',ascending=True).iloc[[thres//2]]
                df_filtered = df_filtered.sort_values('eye_h_sum',ascending=True).head(3)
                # df_filtered = df_filtered.sort_values(['lp_cat_canny'],ascending=[False])
                # my_slot_caption.write('Hi there! Am I cute?') 
                text='Hi there! Am I cute?'
                


            # Summary
            my_slot_summary2.write('Selected criteria: ' + criterion)
            # if df_filtered.prob.iloc[0] < .5:
            if len(df_filtered) == 0:
                df_filtered = df.sort_values('lp_cat',ascending=False).head(thres)
            else:
                df_filtered = df_filtered
            
            if df_filtered.lp_ratio_canny.iloc[0] < .5:
                my_slot_summary.subheader('Sorry...this is the best we could find:')
            else:
                my_slot_summary.subheader('How about this?')
            # Best frame
            best_frame = df_filtered.frame_name.iloc[0]
            my_slot_image.image('data/' + best_frame)
            # my_slot_dataframe.write(df_filtered.iloc[:,1:])


            
            # my_slot_result_like
            col1, col2,col3 = my_slot_caption_like.beta_columns([1,0.28,0.2])
            # col1.markdown(":hearts:")
            col2.button('♥️...Love it!')
            col3.button('👿...Nope!')
            col1.write(text)



    except:
        my_slot_summary.subheader("Please upload a video first in Your Purrfect Pic!")


if choice == "Home":

    st.subheader('Capture the best momemt with your furry friend!')

    from PIL import Image
    image1 = Image.open('./img/chiffon.jpg')
    image2 = Image.open('./img/cheddar.jpg')
    image3 = Image.open('./img/chiffon2.jpg')
    image4 = Image.open('./img/cheddar2.jpg')


    cat_imgs=[image1,image2,image3,image4]
    st.image(cat_imgs,width=174)

    st.subheader('Ever struggled to take a good picture for your cat?')
    st.write("We tend to take a bunch of photos in fear of missing out.\
        Especially for cute animals since it's even more difficult to capture a good one when\
        you cannot tell them to hold still. Afterwards, it's extremely time-consuming to go through all the photos and manually select the best one to keep.\
        ")
    st.write("Now you can let Picture Purrfect do the hard job for you! Just take a short video of your cat and the best moment will be picked automatically!")


if choice == "Analysis":
    my_slot_header_all = st.write('In this section, you can examine the statistics of cat detection in your video.')    


    result = pd.read_csv('./data/df_predict.csv').iloc[:,1:].sort_values('frame_name').reset_index(drop=True)
    result['EYE_count']=[str(x) for x in result.num_eye.tolist()]

    result['new_eye_h'] = result['eye_h']/result['face_size']
    result['new_eye_w'] = result['eye_w']/result['face_size']


    # First plot
    result_sort = result[['frame_name','new_eye_w','new_eye_h','lp_cat_canny','EYE_count']].sort_values('new_eye_h').iloc[20:-5]
    result_sort.columns = ['Frame Name','Eye Width','Eye Height','Face Sharpness','Number of eyes detected']

    c = alt.Chart(result_sort).mark_circle(opacity=.8).encode(
        x=alt.X('Eye Height',axis=alt.Axis(gridColor='black',gridWidth=.1,labels=False),
                scale=Scale(domain=[np.mean(result['new_eye_h'])/2, 1.5*np.mean(result['new_eye_h'])])
            ),
        y=alt.Y('Eye Width',axis=alt.Axis(labels=False)),
        size='Face Sharpness',
    #     color='Number of eyes detected',
    #     color = alt.Color('Number of eyes detected',
    #                       scale=alt.Scale(scheme = 'set2')
        color = alt.Color('Number of eyes detected',
                        scale=alt.Scale(range=['#C7648E','#3A5AB1',])
                        ),
        tooltip=['Frame Name',]).properties(width=700, height=250
    )

    st.subheader('Eye Shape Analysis')
    st.write(c)


    # Second plot
    MAX = max(result.face_size)
    MIN = min(result.face_size)
    section = (max(result.face_size)-min(result.face_size))//3

    FACE_SIZE = []
    for x in result.face_size.tolist():
        if x<MIN+section:
            FACE_SIZE.append('small')
        elif MIN+section < x < MIN + 2*section:
            FACE_SIZE.append('medium')
        else:
            FACE_SIZE.append('large')

    result['Face size'] = FACE_SIZE

    result_middle_all = result.sort_values('new_eye_h').iloc[20:-10]
    result_middle = result_middle_all[['lp_cat_canny','lp_ratio','Face size']]
    result_middle.columns = ['Cat face sharpness','Sharpness ratio (cat:whole frame)','Face size']


    points = alt.Chart(result_middle).mark_circle(size=200).encode(
        alt.X('Cat face sharpness'),
        alt.Y('Sharpness ratio (cat:whole frame)'),
    #     color='FACE_SIZE',
        color = alt.Color('Face size',
                        scale=alt.Scale(range=['#D690AC','#6F88CD','#64C7B1',])),
    )

    top_hist = alt.Chart(result_middle).mark_area(
        opacity=.5, interpolate='step'
    ).encode(
        alt.X('Cat face sharpness:Q', 
            bin=alt.Bin(maxbins=20), 
            stack=None, 
            
            ),
        alt.Y('count(*):Q', 
            stack=None, 
            ),
        alt.Color('Face size:N'),
    ).properties(height=60)

    right_hist = alt.Chart(result_middle).mark_area(
        opacity=.5, interpolate='step'
    ).encode(
        alt.Y('Sharpness ratio (cat:whole frame):Q', 
            bin=alt.Bin(maxbins=20), 
            stack=None,
            ),
        alt.X('count(*):Q', 
            stack=None, 
            ),
        alt.Color('Face size:N'),
    ).properties(width=60)

    chart = top_hist & (points | right_hist)

    st.subheader('Blur Detection Analysis')
    st.write(chart)








def draw_all(
    key,
    plot=False,
):
    st.write(
        """
        # Example Widgets
        
        These widgets don't do anything. But look at all the new colors they got 👀 
    
        ```python
        # First some code.
        streamlit = "cool"
        theming = "fantastic"
        both = "💥"
        ```
        """
    )

    st.checkbox("Is this cool or what?", key=key)
    st.radio(
        "How many balloons?",
        ["1 balloon 🎈", "2 balloons 🎈🎈", "3 balloons 🎈🎈🎈"],
        key=key,
    )
    st.button("🤡 Click me", key=key)

    # if plot:
    #     st.write("Oh look, a plot:")
    #     x1 = np.random.randn(200) - 2
    #     x2 = np.random.randn(200)
    #     x3 = np.random.randn(200) + 2

    #     hist_data = [x1, x2, x3]
    #     group_labels = ["Group 1", "Group 2", "Group 3"]

    #     fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

    #     st.plotly_chart(fig, use_container_width=True)

    st.file_uploader("You can now upload with style", key=key)
    st.slider(
        "From 10 to 11, how cool are themes?", min_value=10, max_value=11, key=key
    )
    # st.select_slider("Pick a number", [1, 2, 3], key=key)
    st.number_input("So many numbers", key=key)
    st.text_area("A little writing space for you :)", key=key)
    st.selectbox(
        "My favorite thing in the world is...",
        ["Streamlit", "Theming", "Baloooons 🎈 "],
        key=key,
    )
    # st.multiselect("Pick a number", [1, 2, 3], key=key)
    # st.color_picker("Colors, colors, colors", key=key)
    with st.beta_expander("Expand me!"):
        st.write("Hey there! Nothing to see here 👀 ")
    st.write("")
    # st.write("That's our progress on theming:")
    # st.progress(0.99)
    if plot:
        st.write("And here's some data and plots")
        st.json({"data": [1, 2, 3, 4]})
        st.dataframe({"data": [1, 2, 3, 4]})
        st.table({"data": [1, 2, 3, 4]})
        st.line_chart({"data": [1, 2, 3, 4]})
        # st.help(st.write)
    st.write("This is the end. Have fun building themes!")





    # if plot:
    #     st.write("Oh look, a plot:")
    #     x1 = np.random.randn(200) - 2
    #     x2 = np.random.randn(200)
    #     x3 = np.random.randn(200) + 2

    #     hist_data = [x1, x2, x3]
    #     group_labels = ["Group 1", "Group 2", "Group 3"]

    #     fig = ff.create_distplot(hist_data, group_labels, bin_size=[0.1, 0.25, 0.5])

    #     st.plotly_chart(fig, use_container_width=True)

    st.file_uploader("You can now upload with style", key=key)
    st.slider(
        "From 10 to 11, how cool are themes?", min_value=10, max_value=11, key=key
    )
    # st.select_slider("Pick a number", [1, 2, 3], key=key)
    st.number_input("So many numbers", key=key)
    st.text_area("A little writing space for you :)", key=key)
    st.selectbox(
        "My favorite thing in the world is...",
        ["Streamlit", "Theming", "Baloooons 🎈 "],
        key=key,
    )
    # st.multiselect("Pick a number", [1, 2, 3], key=key)
    # st.color_picker("Colors, colors, colors", key=key)
    with st.beta_expander("Expand me!"):
        st.write("Hey there! Nothing to see here 👀 ")
    st.write("")
    # st.write("That's our progress on theming:")
    # st.progress(0.99)
    if plot:
        st.write("And here's some data and plots")
        st.json({"data": [1, 2, 3, 4]})
        st.dataframe({"data": [1, 2, 3, 4]})
        st.table({"data": [1, 2, 3, 4]})
        st.line_chart({"data": [1, 2, 3, 4]})
        # st.help(st.write)
    st.write("This is the end. Have fun building themes!")

    

if choice == "Test":

    
    result = pd.read_csv('./data/df_predict.csv').iloc[:,1:].sort_values('frame_name').reset_index(drop=True)
    result['EYE_count']=[str(x) for x in result.num_eye.tolist()]

    result['new_eye_h'] = result['eye_h']/result['face_size']
    result['new_eye_w'] = result['eye_w']/result['face_size']

    


    import pandas as pd
    from IPython.core.display import display,HTML

    df = pd.DataFrame([['A231', 'Book', 5, 3, 150], 
                    ['M441', 'Magic Staff', 10, 7, 200]],
                    columns = ['Code', 'Name', 'Price', 'Net', 'Sales'])

    # your images
    images = ['https://vignette.wikia.nocookie.net/2007scape/images/7/7a/Mage%27s_book_detail.png/revision/latest?cb=20180310083825',
            'https://i.pinimg.com/originals/d9/5c/9b/d95c9ba809aa9dd4cb519a225af40f2b.png'] 
    images = ['data/'+ x for x in result.frame_name.tolist()][:2]


    df['image'] = images

    # convert your links to html tags 
    def path_to_image_html(path):
        return '<img src="'+ path + '" width="60" >'

    pd.set_option('display.max_colwidth', None)
    import streamlit.components.v1 as components
    components.html(df.to_html(escape=False ,formatters=dict(image=path_to_image_html)))

    st.write(df)


    np.random.seed(0)

    n_objects = 20
    n_times = 50

    # Create one (x, y) pair of metadata per object
    locations = pd.DataFrame({
        'id': range(n_objects),
        'x': np.random.randn(n_objects),
        'y': np.random.randn(n_objects)
    })

    # Create a 50-element time-series for each object
    timeseries = pd.DataFrame(np.random.randn(n_times, n_objects).cumsum(0),
                            columns=locations['id'],
                            index=pd.RangeIndex(0, n_times, name='time'))

    # Melt the wide-form timeseries into a long-form view
    timeseries = timeseries.reset_index().melt('time')

    # Merge the (x, y) metadata into the long-form view
    timeseries['id'] = timeseries['id'].astype(int)  # make merge not complain
    data = pd.merge(timeseries, locations, on='id')

    # Data is prepared, now make a chart

    selector = alt.selection_single(empty='all', fields=['id'])

    base = alt.Chart(data).properties(
        width=250,
        height=250
    ).add_selection(selector)

    points = base.mark_point(filled=True, size=200).encode(
        x='mean(x)',
        y='mean(y)',
        color=alt.condition(selector, 'id:O', alt.value('lightgray'), legend=None),
    )

    timeseries = base.mark_line().encode(
        x='time',
        y=alt.Y('value', scale=alt.Scale(domain=(-15, 15))),
        color=alt.Color('id:O', legend=None)
    ).transform_filter(
        selector
    )

    points | timeseries

    from bokeh.plotting import figure, output_file, show, ColumnDataSource
    from bokeh.models import HoverTool

    output_file("toolbar.html")
    

    source = ColumnDataSource(
            data=dict(
                x=[1, 2, 3, 4, 5],
                y=[2, 5, 8, 2, 7],
                desc=['A', 'b', 'C', 'd', 'E'],
                imgs = ['data/'+ x for x in result.frame_name.tolist()]
            )
        )

    hover = HoverTool(
            tooltips="""
            <div>
                <div>
                    <img
                        src="@imgs"  alt="@imgs" 
                        style="float: left; margin: 0px 15px 15px 0px;"
                        border="2"
                    ></img>
                </div>
                <div>
                    <span style="font-size: 17px; font-weight: bold;">@desc</span>
                    <span style="font-size: 15px; color: #966;">[$index]</span>
                </div>
                <div>
                    <span style="font-size: 15px;">Location</span>
                    <span style="font-size: 10px; color: #696;">($x, $y)</span>
                </div>
            </div>
            """
        )

    p = figure(plot_width=400, plot_height=400, tools=[hover],
            title="Mouse over the dots")

    p.circle('x', 'y', size=20, source=source)

    st.bokeh_chart(p)

    # st_title = st.empty()
    # st_progress_bar = st.empty()

    # class tqdm:
    #     def __init__(self, iterable, title=None):
    #         if title:
    #             st_title.write(title)
    #         self.prog_bar = st_progress_bar.progress(0)
    #         self.iterable = iterable
    #         self.length = len(iterable)
    #         self.i = 0

    #     def __iter__(self):
    #         for obj in self.iterable:
    #             yield obj
    #             self.i += 1
    #             current_prog = self.i / self.length
    #             self.prog_bar.progress(current_prog)



    # for i in tqdm(range(200), title='tqdm style progress bar'):
    #     time.sleep(0.05)

    # st_title.empty()
    # st_progress_bar.empty()

    st.balloons()


    col1, col2 = st.beta_columns([0.2,1])
    # col1.markdown(":hearts:")
    col1.button('♥️...Love it!')
    col2.button('👿...Nope!')

    # Draw some dummy content in main page and sidebar.
    draw_all("main", plot=True)
    with st.sidebar:
        draw_all("sidebar")

    



    

    def paginator(label, items, items_per_page=10, on_sidebar=True):
        """Lets the user paginate a set of items.
        Parameters
        ----------
        label : str
            The label to display over the pagination widget.
        items : Iterator[Any]
            The items to display in the paginator.
        items_per_page: int
            The number of items to display per page.
        on_sidebar: bool
            Whether to display the paginator widget on the sidebar.
            
        Returns
        -------
        Iterator[Tuple[int, Any]]
            An iterator over *only the items on that page*, including
            the item's index.
        Example
        -------
        This shows how to display a few pages of fruit.
        >>> fruit_list = [
        ...     'Kiwifruit', 'Honeydew', 'Cherry', 'Honeyberry', 'Pear',
        ...     'Apple', 'Nectarine', 'Soursop', 'Pineapple', 'Satsuma',
        ...     'Fig', 'Huckleberry', 'Coconut', 'Plantain', 'Jujube',
        ...     'Guava', 'Clementine', 'Grape', 'Tayberry', 'Salak',
        ...     'Raspberry', 'Loquat', 'Nance', 'Peach', 'Akee'
        ... ]
        ...
        ... for i, fruit in paginator("Select a fruit page", fruit_list):
        ...     st.write('%s. **%s**' % (i, fruit))
        """

        # Figure out where to display the paginator
        if on_sidebar:
            location = st.sidebar.empty()
        else:
            location = st.empty()

        # Display a pagination selectbox in the specified location.
        items = list(items)
        n_pages = len(items)
        n_pages = (len(items) - 1) // items_per_page + 1
        page_format_func = lambda i: "Page %s" % i
        page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

        # Iterate over the items in the page to let the user display them.
        min_index = page_number * items_per_page
        max_index = min_index + items_per_page
        import itertools
        return itertools.islice(enumerate(items), min_index, max_index)
        
    sunset_imgs = [
    'https://unsplash.com/photos/-IMlv9Jlb24/download?force=true',
    'https://unsplash.com/photos/ESEnXckWlLY/download?force=true',
    'https://unsplash.com/photos/mOcdke2ZQoE/download?force=true',
    'https://unsplash.com/photos/GPPAjJicemU/download?force=true',
    'https://unsplash.com/photos/JFeOy62yjXk/download?force=true',
    'https://unsplash.com/photos/kEgJVDkQkbU/download?force=true',
    'https://unsplash.com/photos/i9Q9bc-WgfE/download?force=true',
    'https://unsplash.com/photos/tIL1v1jSoaY/download?force=true',
    'https://unsplash.com/photos/-G3rw6Y02D0/download?force=true',
    'https://unsplash.com/photos/xP_AGmeEa6s/download?force=true',
    'https://unsplash.com/photos/4iTVoGYY7bM/download?force=true',
    'https://unsplash.com/photos/mBQIfKlvowM/download?force=true',
    'https://unsplash.com/photos/A-11N8ItHZo/download?force=true',
    'https://unsplash.com/photos/kOqBCFsGTs8/download?force=true',
    'https://unsplash.com/photos/8DMuvdp-vso/download?force=true'
    ]

    sunset_imgs = [
    'https://unsplash.com/photos/-IMlv9Jlb24/download?force=true',
    'https://unsplash.com/photos/ESEnXckWlLY/download?force=true',
    'https://unsplash.com/photos/mOcdke2ZQoE/download?force=true',
    'https://unsplash.com/photos/GPPAjJicemU/download?force=true',
    'https://unsplash.com/photos/JFeOy62yjXk/download?force=true',
    'https://unsplash.com/photos/kEgJVDkQkbU/download?force=true',
    'https://unsplash.com/photos/i9Q9bc-WgfE/download?force=true',
    'https://unsplash.com/photos/tIL1v1jSoaY/download?force=true',
    'https://unsplash.com/photos/-G3rw6Y02D0/download?force=true',
    'https://unsplash.com/photos/xP_AGmeEa6s/download?force=true',
    'https://unsplash.com/photos/4iTVoGYY7bM/download?force=true',
    'https://unsplash.com/photos/mBQIfKlvowM/download?force=true',
    'https://unsplash.com/photos/A-11N8ItHZo/download?force=true',
    'https://unsplash.com/photos/kOqBCFsGTs8/download?force=true',
    'https://unsplash.com/photos/8DMuvdp-vso/download?force=true'
    ]

    image_iterator = paginator("Select a sunset page", sunset_imgs)
    indices_on_page, images_on_page = map(list, zip(*image_iterator))
    st.image(images_on_page, width=100, caption=indices_on_page)
    

    page_bg_img = '''
    <style>
    body {
    background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
    background-size: cover;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)


    image_address = "https://i.pinimg.com/originals/80/38/5d/80385d6fa0e6cafef67442020b096b85.jpg"
    page_bg_img = '''
    <style>
    body {
    background-image: url("https://i.pinimg.com/originals/80/38/5d/80385d6fa0e6cafef67442020b096b85.jpg");
    
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

    from PIL import Image
    image = Image.open('chiffon.jpg')
    st.image(image, caption='Find the best picture for you furry friend!')

    image1 = Image.open('chiffon.jpg')
    image2 = Image.open('cheddar.jpg')
    image3 = Image.open('chiffon2.jpg')
    image4 = Image.open('cheddar2.jpg')


    cat_imgs=[image1,image2,image3,image4]
    image_iterator = paginator("Select a sunset page", cat_imgs)
    indices_on_page, images_on_page = map(list, zip(*image_iterator))
    st.image(images_on_page, width=174)





    st.subheader('This is a testing section')

    values = st.select_slider(
        'Select a range of values',
        options=['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet'])
    st.write('Values:', values)

    values = st.select_slider(
        'How do you want us to go through the frames?',
        options=['course (I want it fast)', 'medium', 'fine (I can wait)'],)
    st.write('You selected:', values)



    genre = st.radio(
        "What's your favorite movie genre",
        ('Comedy', 'Drama', 'Documentary'))
        
    if genre == 'Comedy':
        st.write('You selected comedy.')
    else:
        st.write("You didn't select comedy.")



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



st.sidebar.header("")
st.sidebar.header("")
# st.sidebar.header("")
# st.sidebar.header("")

st.sidebar.markdown('###### Made by Katie Huang 2021')