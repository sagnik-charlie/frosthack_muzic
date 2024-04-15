from flask import Flask, render_template, Response, request
import cv2
import os
import scipy
import torch
from diffusers import AudioLDM2Pipeline

# Initialize Flask app
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

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList=['Male','Female']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)

def gen_frames():  
  """
  This function generates frames from the webcam and performs basic processing.
  """
  cap = cv2.VideoCapture(0)  # 0 for default webcam
  padding=20
  while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
      print("Failed to grab frame")
      break

    # Simple processing (replace with your OpenCV logic)
    resultImg,faceBoxes=highlightFace(faceNet,frame)
    for faceBox in faceBoxes:
        face=frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds=genderNet.forward()
        gender=genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds=ageNet.forward()
        age=ageList[agePreds[0].argmax()]

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)

    # Encode frame as JPEG for web streaming
    ret, buffer = cv2.imencode('.jpg', resultImg)
    frame = buffer.tobytes()
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

  cap.release()
  cv2.destroyAllWindows()

@app.route('/')
def index():
  """
  Render the HTML template for the video feed.
  """
  return render_template('index.html')

@app.route('/video')
def video():
  """
  Video streaming route that generates frames using gen_frames function.
  """
  return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/generate', methods=['POST'])
def generate_audio():
  if request.method == 'POST':
    text_prompt = request.form['text_prompt']  # Access form data using 'name' attribute
    
    # Process the form data (e.g., store in database, send email)
    mp3_link = get_mp3_file(text_prompt)
    return render_template('index.html', audio=mp3_link)
  else:
    return "Something went wrong!"  # Handle non-POST requests (optional)

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
  audio_file_path = os.path.join(os.path.dirname(__file__), 'static', 'audio', 'generated.wav')
  os.makedirs(os.path.dirname(audio_file_path), exist_ok=True)
  scipy.io.wavfile.write(audio_file_path, rate=16000, data=audio[0])
  return 'static/audio/generated.wav'



if __name__ == '__main__':
  app.run(host='0.0.0.0', debug=True)  # Set host to 0.0.0.0 for external access
