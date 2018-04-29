# USAGE
# Start the server:
#       python run_keras_server.py
# Submit a request via cURL:
#       curl -X POST -F image=@jemma.png 'http://localhost:5000/predict'
# Submita a request via Python:
#       python simple_request.py 


# import the necessary packages
import keras.models
import skimage
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import imagenet_utils
from threading import Thread
from PIL import Image
import numpy as np
import base64
import flask
import redis
import uuid
import time
import json
import sys
import io

# constants for image dimensions and datatype
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
IMAGE_CHANS = 3
IMAGE_DTYPE = 'int8'

# constants for server queueing
IMAGE_QUEUE = "image_queue"
BATCH_SIZE = 32
SERVER_SLEEP = 0.25
CLIENT_SLEEP = 0.25

# start Flask Server
app = flask.Flask(__name__)
# connect to Redis
db = redis.StrictRedis(host='localhost', port=6379, db=0)
# initialize the model variable
model = None

"""10 type (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)"""
def f(x):
    return {
        0: 'airplane',
        1: 'bird',
        2: 'automobile',
        3: 'cat',
        4: 'deer',
        5: 'truck',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'dog'
    }.get(x, '')

def base64_encode_image(a):
    """serialize binary image a to base64 string."""
    return base64.b64encode(a).decode("utf-8")

def base64_decode_image(a, dtype, shape):
    """de-serialize base64 string to numpy array."""
    if sys.version_info.major == 3:
        a = bytes(a, encoding='utf-8')

    a = np.frombuffer(base64.decodestring(a), dtype=dtype)
    a = a.reshape(shape)
    
    return a

def prepare_image(image, target):
    """pre-process the image."""
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = imagenet_utils.preprocess_input(image)

    return image

def classify_process():
    """Keras Model Server"""
    
    print("* loading model...")
    # load the pretrained-keras model
    # you can replace this with your own model
    model = keras.models.load_model('cifar10.h5')
    print("* model loaded...")

    # pool for new images to classify
    while True:
        
        # poll next batch of images from the Redis queue
        queue = db.lrange(IMAGE_QUEUE, 0, BATCH_SIZE - 1)
        imageIDs = []
        batch = None
        for q in queue:
            q = json.loads(q.decode("utf-8"))
                        
            image = base64_decode_image(q["image"], IMAGE_DTYPE, 
                    (4, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANS))
            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])
            imageIDs.append(q["id"])

        # predict all images in the batch
        if len(imageIDs) > 0:
            print("* batch size: {}".format(batch.shape))
            preds = model.predict(batch)
            
            # construct results records 
            for (imageID, resultSet) in zip(imageIDs, preds):
                output = []
                for idx,pred in enumerate(resultSet):
                    if pred:
                        label = f(idx)
                        probability = 'Yes'
                    else:
                        label = f(idx)
                        probability = 'No'
                    r = {'label': label, 'probability': probability}
                    output.append(r)
                    
                # stores the results in the Redis queue
                db.set(imageID, json.dumps(output))
            
            # delete processed images from the queue
            db.ltrim(IMAGE_QUEUE, len(imageIDs), -1)
            
        # take a nap....
        time.sleep(SERVER_SLEEP)

        
@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            # read and prepare the image
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (IMAGE_WIDTH, IMAGE_HEIGHT))

            # ensure that numpy array is C-contiguous as well,
            # otherwise we won't be able to serialize it
            image = image.copy(order="C")
            
            # generate a unique ID
            k = str(uuid.uuid4())
            # create a new image record in the queue
            d = {"id": k, "image": base64_encode_image(image)}
            db.rpush(IMAGE_QUEUE, json.dumps(d))
            
            # Repeated poll the results from the queue
            # If the prediction has already finished, 
            # read the results and delete the record from the queue
            while True:
                output = db.get(k)
                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                # take a nap...
                time.sleep(CLIENT_SLEEP)
            data["success"] = True

    return flask.jsonify(data)

if __name__ == "__main__":
    print("* starting model service...")
    t = Thread(target=classify_process, args=())
    t.daemon = True
    t.start()

    print("* starting web service ...")
    app.run()
