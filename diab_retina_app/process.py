import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import os


def process_img(img):
    
    np.set_printoptions(suppress=True)

    # Load the model
    model = tensorflow.keras.models.load_model(os.path.dirname(__file__) + '/keras_model.h5')

    
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    
    image = Image.open(os.path.dirname(__file__) + '/test/' + img)

    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)

    image_array = np.asarray(image)

    
    # image.show()

    
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    
    data[0] = normalized_image_array

    
    prediction = model.predict(data)
    # print(prediction)

    
    pred_new = prediction[0]
    pred = max(pred_new)

    print(pred_new)
    index = pred_new.tolist().index(pred)

    #plot the graph
    import matplotlib.pyplot as plt

    
    left = [1, 2, 3, 4, 5]

    
    height = pred_new.tolist()
    new_height = []
    for i in height:
        new_height.append(round(i, 2) * 100)

    print(height)

    print(new_height)
    tick_label = ['No DR', 'Mild', 'Moderate', 'Sever', 'Proliferative']

    
    plt.bar(left, new_height, tick_label=tick_label,
            width=0.8, color=['red', 'green'])

    
    plt.xlabel('x - axis')
    
    plt.ylabel('y - axis')
    
    plt.title('Diabetic Retinopathy')

    
    plt.savefig(os.path.dirname(__file__) + '/output/graph.png')
    plt.show()
    result = []

    if index == 0:
        result.append("No DR")
    elif index == 1:
        result.append("Mild")
    elif index == 2:
        result.append("Moderate")
    elif index == 3:
        result.append("Sever")
    elif index == 4:
        result.append("Proliferative")

    accuracy = round(pred, 2)
    result.append("-")
    result.append(accuracy * 100)

    return result
