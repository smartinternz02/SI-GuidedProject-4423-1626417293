import numpy as np
import cv2
import os
from tensorflow.keras.models import load_model
from flask import Flask, render_template, Response
from gtts import gTTS
from tensorflow.keras.preprocessing import image


model = load_model('savedmodel.h5')

app = Flask(__name__)

print("streaming the video........")
camera = cv2.VideoCapture(0) 
vals = ['A', 'B','C','D','E','F','G','H','I']

def detect(frame):
    img = image.load_img(r"file:///C:/Users/Ayyappa/OneDrive/Documents/ganesh/mnist/Dataset/test_set/B/1.png",target_size=(64,64))
    
    x=image.img_to_array(img) 
    index=['A', 'B','C','D','E','F','G','H','I']
    x=np.expand_dims(x,axis=0)  
    a=np.argmax(model.predict(x),axis=1)
    result = str(index[a[0]])
    return result
    
    
@app.route('/')
def index():
    return render_template('index.html')



def gen_frames():  
    while True:
        (grabbed, frame) = camera.read()
            
           
        if not grabbed:
            break
        
        data = detect(frame)
           
        text = "It indicates "+data
        
        cv2.putText(frame, text, (10, frame.shape[0] - 25),cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 255), 4)
        
        speech = gTTS(text = data, lang = 'en', slow = False)
        speech.save("text1.mp3")
        os.system("start text1.mp3")
            
        key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        w = cv2.VideoWriter(r"output.avi", fourcc, 25,(frame.shape[1], frame.shape[0]), True)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        fre = buffer.tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + fre + b'\r\n')
        
   
        
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
