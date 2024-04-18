from flask import Flask, render_template, Response, request, redirect, send_from_directory, url_for
import cv2
from fer import FER
import os
import datetime, time
import scipy
import torch
from collections import Counter
import numpy as np
import random
from diffusers import AudioLDM2Pipeline

global capture, rec_frame, grey, switch, neg, face, rec, out, emotion_str, mp3_link
capture=0
emotion_str=""
app = Flask(__name__,template_folder='templates')

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes
faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

color_range = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}

color_moods = {
    'brightred': ['excitement', 'speed', 'blood', 'rage', 'violence'],
    'dullred': ['love', 'romance', 'passion'],
    'brightgreen': ['balance', 'harmony', 'serenity', 'joy', 'hope'],
    'dullgreen': ['disgust', 'greed', 'jealousy'],
    'brightblue': ['calm', 'tranquillity', 'peace'],
    'dullblue': ['sadness', 'aloofness', 'melancholy']
}

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(00-02)', '(04-06)', '(08-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']
emotion_detector = FER(mtcnn=True)
faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
str_prompt="Hello. This is your prompt"
mp3_link="static/audio/generated.wav"

def find_dominant_color(image, k=3):
    # Load the image
    # image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

    # Flatten the image
    pixels = image.reshape(-1, 3)

    # Perform k-means clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), k, None, 
                                    criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Count the occurrences of each label
    label_counts = Counter(labels.flatten())

    # Find the label with the highest count
    dominant_label = max(label_counts, key=label_counts.get)

    # Return the dominant color
    dominant_color = centers[dominant_label].astype(int)
    return dominant_color

def is_bright(image, threshold=100):
    # Load the image in grayscale
    # image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 

    # Calculate the average pixel intensity
    average_intensity = image.mean()
    
    # Determine if the image is overall bright or dull based on the threshold
    if average_intensity > threshold:
        return True  # Bright
    else:
        return False  # Dull

def assign_label(dominant_color, color_range):
    min_distance = float('inf')
    assigned_label = None
    
    for label, color in color_range.items():
        distance = np.linalg.norm(np.array(color) - np.array(dominant_color))
        if distance < min_distance:
            min_distance = distance
            assigned_label = label
    
    return assigned_label

def assign_prompt(color, is_bright, gender, mood, age):
    text_adults=["from early 2000s","from late 90s","retro classic from early 90s"]
    text_kids=["EDM music","for babies","for kids"]
    text_oldies=["classic rock from early 80s","retro classic from early 70s"]
    if age>=0 and age<=2:
        text = random.choice(text_kids)
    elif age>=3 and age<=7:
        text = random.choice(text_kids)
    elif age>=8 and age<=12:
        text = text_kids[0]
    elif age>=13 and age<=20:
        text = random.choice(text_adults)
    elif age>=21 and age<=32:
        text = random.choice(text_adults)
    elif age>=33 and age<=45:
        text = random.choice(text_adults)
    elif age>=46 and age<=55:
        text = random.choice(text_oldies)
    elif age>=56 and age<=90:
        text = random.choice(text_oldies)
    else:
        text = 'for all age group'

    if color == 'red':
        if is_bright:
            text2 = random.choice(color_moods['brightred'])
        else:
            text2 = random.choice(color_moods['dullred'])

    elif color == 'green':
        if is_bright:
            text2 = random.choice(color_moods['brightgreen'])
        else:
            text2 = random.choice(color_moods['dullgreen'])
    
    else:
        if is_bright:
            text2 = random.choice(color_moods['brightblue'])
        else:
            text2 = random.choice(color_moods['dullblue'])

    prompt = text + " music for " + gender + " about " + text2 + ' expressing ' + mood

    return prompt

def generate_prompt(frame, gender, mood, age):
    dominant_color = find_dominant_color(frame)
    assigned_label = assign_label(dominant_color, color_range)
    bright = is_bright(frame)
    return assign_prompt(assigned_label, bright, gender, mood, age)

def gen_frames():  
  """
  This function generates frames from the webcam and performs basic processing.
  """
  global capture, emotion_str
  cap = cv2.VideoCapture(0)
  padding=20
  while True:  ## Capturing 1st 120 frames
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
      print("Failed to grab frame")
      break
    #if(cnt==150):
    #   break
    # Simple processing (replace with your OpenCV logic)
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]
        if(face is not None):
         blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]
        
        #results = emotion_detector.detect_emotions(resultImg)
        emotion, score = emotion_detector.top_emotion(resultImg)
         #emotion=max(results[0]['emotions'], key = results[0]['emotions'].get)
        cv2.putText(resultImg, f"{emotion}: {score}", (faceBox[0], faceBox[1]+50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        #if(gender is not None and emotion is not None and age is not None):
        emotion_str=generate_prompt(frame,gender,emotion,int(age[1:3]))

    # Encode frame as JPEG for web streaming
    ret, buffer = cv2.imencode('.jpg', resultImg)
    if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['shots', "shot_{}.png".format(str(now).replace(":",''))])
                #cv2.imwrite(p, frame)
                #emotion_str=generate_prompt(frame,gender,emotion,int(age[1:3]))
    #if(emotion_str!=[]):
    # generate_prompt(resultImg,random.choice(emotion_str)[0],random.choice(emotion_str)[1],random.choice(emotion_str)[2])
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
         
  cap.release()
  cv2.destroyAllWindows()
  render_template('index.html',audio=mp3_link,prompt=str_prompt)

@app.route('/')
def index():
  """
  Render the HTML template for the video feed.
  """
  global emotion_str
  if(emotion_str is not None):
   text=emotion_str
   gen_frames()
  else:
   text=""
   gen_frames()
  return render_template('index.html',audio=mp3_link,prompt=text)

@app.route('/video')
def video():
  """
  Video streaming route that generates frames using gen_frames function.
  """
  return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate', methods=['POST'])
def generate_audio():
  global mp3_link
  if request.method == 'POST':
    text_prompt = request.form['text_prompt']  # Access form data using 'name' attribute
    # Process the form data (e.g., store in database, send email)
    mp3_link = get_mp3_file(text_prompt)
    return render_template('index.html', audio=mp3_link,prompt="")
  else:
    return "Something went wrong!"  # Handle non-POST requests (optional)
    
@app.route('/',methods=['POST'])
def tasks():
    global switch,camera, capture, emotion_str, mp3_link
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            # global capture
            capture=1
            gen_frames()
            return render_template('index.html',audio=mp3_link,prompt=emotion_str)

@app.route('/audio/<path:filename>')
def serve_audio(filename):
    return send_from_directory('static/audio', filename)

def get_mp3_file(prompt):
  repo_id = "cvssp/audioldm2"
  pipe = AudioLDM2Pipeline.from_pretrained(repo_id)
  if(torch.cuda.is_available()):
    device="cuda" 
  else:
    device="cpu"
  pipe = pipe.to(device)

# define the prompts
  #prompt = "A 90's hard meta rock song expressing extreme anger and empowering women"
  negative_prompt = "Low quality."

# set the seed
  generator=torch.Generator(device).manual_seed(0)

# run the generation
  audio = pipe(
    prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=200,
    audio_length_in_s=5.0,
    num_waveforms_per_prompt=3,
  ).audios

# save the best audio sample (index 0) as a .wav file
  audio_file_path = os.path.join(os.path.dirname(__file__), 'static', 'audio', '3.wav')
  os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
  scipy.io.wavfile.write(audio_file_path, rate=16000, data=audio[0])
  return audio_file_path



if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)  # Set host to 0.0.0.0 for external access
