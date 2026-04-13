from flask import Flask, render_template, request, send_file
import requests
import cv2
import numpy as np
import io
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    file = request.files.read()
    key = request.form.get('key')
    
    # --- CPU強力補正 (OpenCV) ---
    nparr = np.frombuffer(file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # シャープネス処理（輪郭をパキパキにする）
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    img = cv2.filter2D(img, -1, kernel)

    # --- iLoveIMG API連携 (キーがある場合) ---
    if key and len(key) > 20:
        try:
            start = requests.get("https://iloveapi.com", 
                                 headers={"Authorization": f"Bearer {key}"}, timeout=10).json()
            server, task = start['server'], start['task']
            
            requests.post(f"https://{server}/v1/upload", 
                          data={'task': task}, 
                          files={'file': ('image.jpg', file)})
            
            requests.post(f"https://{server}/v1/process", 
                          json={'task': task, 'tool': 'upscaleimg', 'upscale_factor': 2})
            
            res = requests.get(f"https://{server}/v1/download/{task}")
            img = cv2.imdecode(np.frombuffer(res.content, np.uint8), cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"API Error: {e}")

    # 結果をJPEGとして返す
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return send_file(io.BytesIO(buffer), mimetype='image/jpeg')

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
