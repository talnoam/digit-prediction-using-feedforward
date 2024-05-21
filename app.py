from flask import Flask, render_template, request
from scipy.stats import truncnorm
import numpy as np
import tqdm
import base64
import re
import codecs, json 

import imageio
from PIL import Image

def imread(filename, mode=None):
    if mode == 'L':
        img = Image.open(filename).convert('L')
        return np.array(img)
    else:
        return imageio.imread(filename)

# Replace imsave with imageio.imwrite
def imsave(filename, arr):
    imageio.imwrite(filename, arr)

# Replace imresize with PIL's resize method
def imresize(arr, size):
    img = Image.fromarray(arr)
    return img.resize(size, Image.ANTIALIAS)

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.e ** -x)

def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, 
                     (upp - mean) / sd, 
                     loc=mean, 
                     scale=sd)

activation_function = sigmoid

class NeuralNetwork:
    
    def __init__(self, 
                 no_of_in_nodes, 
                 no_of_out_nodes, 
                 no_of_hidden_nodes,
                 learning_rate):
        self.no_of_in_nodes = no_of_in_nodes
        self.no_of_out_nodes = no_of_out_nodes
        self.no_of_hidden_nodes = no_of_hidden_nodes
        self.learning_rate = learning_rate 
        # self.create_weight_matrices()

        # Load weights
        wih_txt = codecs.open('./weights/wih.json', 'r', encoding='utf-8').read()
        wih_json = json.loads(wih_txt)
        self.wih = np.array(wih_json)

        who_txt = codecs.open('./weights/who.json', 'r', encoding='utf-8').read()
        who_json = json.loads(who_txt)
        self.who = np.array(who_json)
        
        
    def create_weight_matrices(self):
        """ A method to initialize the weight matrices of the neural network"""
        rad = 1 / np.sqrt(self.no_of_in_nodes)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.wih = X.rvs((self.no_of_hidden_nodes, 
                                       self.no_of_in_nodes))
        rad = 1 / np.sqrt(self.no_of_hidden_nodes)
        X = truncated_normal(mean=0, 
                             sd=1, 
                             low=-rad, 
                             upp=rad)
        self.who = X.rvs((self.no_of_out_nodes, 
                                        self.no_of_hidden_nodes))
        
        
    
    def train_single(self, input_vector, target_vector):
        """
        input_vector and target_vector can be tuple, 
        list or ndarray
        """
        
        output_vectors = []
        input_vector = np.array(input_vector, ndmin=2).T
        target_vector = np.array(target_vector, ndmin=2).T
        
        output_vector1 = np.dot(self.wih, 
                                input_vector)
        output_hidden = activation_function(output_vector1)
        
        output_vector2 = np.dot(self.who, 
                                output_hidden)
        output_network = activation_function(output_vector2)
        
        output_errors = target_vector - output_network
        self.output_errors = output_errors
        # update the weights:
        tmp = output_errors * output_network * \
              (1.0 - output_network)     
        tmp = self.learning_rate  * np.dot(tmp, 
                                           output_hidden.T)
        self.who += tmp


        # calculate hidden errors:
        hidden_errors = np.dot(self.who.T, 
                               output_errors)
        # update the weights:
        tmp = hidden_errors * output_hidden * (1.0 - output_hidden)
        self.wih += self.learning_rate * np.dot(tmp, input_vector.T)
        

    def train(self, data_array, 
              labels_one_hot_array,
              epochs=1,
              intermediate_results=False):
        intermediate_weights = []
        errors = []
        for epoch in range(epochs): 
            sum_error = 0 
            print("*")  
            for i in range(len(data_array)):
                self.train_single(data_array[i], 
                                  labels_one_hot_array[i])
                sum_error += sum([self.output_errors**2 for i in range(len(self.output_errors))])
            d['epoch'].append(epoch)
            d['error'].append(sum_error)
              
            error = 0
            for i in range(len(self.output_errors)):
              error += self.output_errors[i]

            errors.append(error[0])

            if intermediate_results:
                intermediate_weights.append((self.wih.copy(), 
                                             self.who.copy()))
            
        return (intermediate_weights, errors)        
            
    def confusion_matrix(self, data_array, labels):
        cm = {}
        for i in range(len(data_array)):
            res = self.run(data_array[i])
            res_max = res.argmax()
            target = labels[i][0]
            if (target, res_max) in cm:
                cm[(target, res_max)] += 1
            else:
                cm[(target, res_max)] = 1
        return cm
        
    
    def run(self, input_vector):
        """ input_vector can be tuple, list or ndarray """
        
        input_vector = np.array(input_vector, ndmin=2).T

        output_vector = np.dot(self.wih, input_vector)
        output_vector = activation_function(output_vector)
        
        output_vector = np.dot(self.who, output_vector)
        output_vector = activation_function(output_vector)
    
        return output_vector
    
    def evaluate(self, data, labels):
        corrects, wrongs = 0, 0
        for i in range(len(data)):
            res = self.run(data[i])
            res_max = res.argmax()
            if res_max == labels[i]:
                corrects += 1
            else:
                wrongs += 1
        return corrects, wrongs
    
    def predict_digit(self, data):
        res = self.run(data)
        return str(res.argmax())

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict/', methods=['GET','POST'])
def predict():
    # get data from drawing canvas and save as image
    parseImage(request.get_data())

    # read parsed image back in 8-bit, black and white mode (L)
    x = imread('output.png', mode='L')
        
    x = np.invert(x)
    image_size = 28 # width and length

    # Resize the image
    x_resized = imresize(x, (image_size, image_size))

    # Convert the resized image to a NumPy array if not already
    if not isinstance(x_resized, np.ndarray):
        x_resized = np.array(x_resized)

    # reshape image data for use in neural network
    image_pixels = image_size * image_size
    image_matrix = x_resized.reshape(image_pixels)
    image_matrix = image_matrix.tolist()

    # Normalize data
    norm = [float(i)/max(image_matrix) * 0.99 + 0.01 for i in image_matrix]

    ANN = NeuralNetwork(no_of_in_nodes = image_pixels, 
                               no_of_out_nodes = 10, 
                               no_of_hidden_nodes = 100,
                               learning_rate = 0.15)


    return ANN.predict_digit(norm)

def parseImage(imgData):
    # parse canvas bytes and save as output.png
    imgstr = re.search(b'base64,(.*)', imgData).group(1)
    with open('output.png','wb') as output:
        output.write(base64.standard_b64decode(imgstr))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)