import numpy as np
import pandas as pd
import cv2 #for preprocess custom images

############## activation functions ############################
def ReLU(x):
    #if X<0 then return 0
    #if x>0 return x
    return np.maximum(0,x)

#returns value btw 0 and 1
def softmax(x) :
    exp = np.exp(x - np.max(x))
    return exp / exp.sum(axis=0)
    #return np.exp(x) / sum(np.exp(x), axis = 0)

#derivatives of activation functions
def ReLU_derivative(x):
    return x > 0 



############ the neural network ##################################
layers = 3

#number of nuerons in each layer
inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 784, 10, 10


# initialise weights and biases ##########
#hidden layer
hidden_weights = np.random.rand(hiddenLayerNeurons,inputLayerNeurons)  - 0.5
hidden_bias = np.random.rand(hiddenLayerNeurons,1) - 0.5
#output layer
output_weights = np.random.rand(outputLayerNeurons,hiddenLayerNeurons) - 0.5 
output_bias = np.random.rand(outputLayerNeurons,1)  - 0.5



# training the ANN on the train dataset
def train(X,Y, epochs, lr) :
    #declaring global variables
    global hidden_weights, hidden_bias, output_weights, output_bias

    #X = the features based on which y is predicted = value of pixels
    #Y = value to be predicted = label 

    input_activations = X
    expected_output = one_hot(Y)

    #training episodes
    for _ in range(epochs) :
        #### forward propagation ##########################
        #hidden_layer
        hidden_z = np.dot(hidden_weights, input_activations) + hidden_bias
        hidden_activations = ReLU(hidden_z)

        #output layer
        output_z = np.dot(output_weights , hidden_activations) + output_bias
        output_activations = softmax(output_z)


        #### back propagation #################################
        ### the loss function and the loss ########
        #cost function: MSE
            #J = (predicted_output - Y)^2   #represented as E

        #error E
        dZ_output = 2*(output_activations - expected_output)  #dE/dZ_output

        #calculate the gradient
        #output layer
        dW_output = np.dot(dZ_output , hidden_activations.T) /m
        dB_output = np.sum(dZ_output, 1)/m

        #hidden_layer
        #hidden layer error
        dZ_hidden = np.dot(output_weights.T , dZ_output)*ReLU_derivative(hidden_z) #dE/dZ_hidden

        dW_hidden = np.dot(dZ_hidden , X.T) /m
        dB_hidden = np.sum(dZ_hidden, 1)/m


        ######## updating weights and biases a/c to gradient #######
        # formula: w = w - lr*dW
        #hidden layer
        hidden_weights = hidden_weights - lr*(dW_hidden)
        hidden_bias = hidden_bias - (lr)*np.reshape(dB_hidden , (hiddenLayerNeurons, 1))

        #output layer
        output_weights = output_weights - (lr)*dW_output
        output_bias = output_bias - (lr)*np.reshape(dB_output, (outputLayerNeurons,1))
    #training completed

    #return the final values of the weights and biases
    return hidden_weights, hidden_bias , output_weights , output_bias



#one hot encoding
def one_hot(Y) :
    one_hot_Y = np.zeros((Y.max() + 1, Y.size))
    one_hot_Y[Y, np.arange(Y.size)] = 1
    #one_hot_Y = one_hot_Y.T
    return one_hot_Y


#testing
def test(X, Y) :
    #declaring global variables
    global hidden_weights, hidden_bias, output_weights, output_bias

    #X, Y of the test set
    # X = input
    expected_output = Y

    ######forward propagation only ########
    #hidden layer
    hidden_activations = ReLU(np.dot(hidden_weights, X) + hidden_bias)

    #output layer
    output_activations = softmax(np.dot(output_weights , hidden_activations) + output_bias)  

    #### calculate accuracy #######
    predicted_output = np.argmax(output_activations, 0)

    #print predicted , expected output
    print("Expected output: ", expected_output)
    print("predicted output: ",predicted_output)

    accuracy = np.sum(predicted_output == expected_output)/ expected_output.size

    return accuracy


#classify an image
def classify(hw, hb, ow, ob, img) :
    #input = pixel values = img = 1d array with 784 columns

    #hidden layer
    hidden_activations = ReLU(np.dot(hidden_weights, img) + hidden_bias)

    #output layer
    output_activations = softmax(np.dot(output_weights , hidden_activations) + output_bias)  

    #classify
    predicted_output = np.argmax(output_activations, 0)

    return predicted_output


################## custom images ############
def preprocess_image(img) :
    #display original img
    cv2.imshow("original",img)
    cv2.waitKey(0)

    #add padding to image
    #img = cv2.copyMakeBorder(img, 90, 90, 0, 0, cv2.BORDER_CONSTANT, None, value = 210)
 
    #resize image
    new_dim = (28, 28)  #28x28 pixels
    img = cv2.resize(img, new_dim, interpolation = cv2.INTER_AREA)

    #normalise image
    img = img/255.

    #invert image
    img = cv2.bitwise_not(img)
    #the given img is dark digit on light background
    #we want light digit on dark background

    #threshholding
    img[img<0.1] = 0
    #img[img>0.85] = 1
    #img = cv2.threshold(img, 0.1, 1, cv2.THRESH_BINARY_INV)[1] 

    #reshape image 
    img = img.reshape((784, 1))

    #display processed image
    #cv2.imshow('processed image', img[1])
    #cv2.waitKey(0)

    return img


########## driver code ################################################
#######################################################################

##### dataset ############################################
data_df = pd.read_csv("")

data = np.array(data_df) #converting df to numpy array
    #60,000 rows  #785 columns
    # 1 row = 1 image
    #col 0 = label = what digit is this?
    #col 1-784 = grayscale values of each pixel

#X = the features based on which y is predicted = value of pixels
#Y = value to be predicted = label 

#m = number of training images
m = 1500

#dividing data into train set and test set
np.random.shuffle(data) #shuffle data before splitting

#training set
train_set = data[0:m].T #only m images of the data set will be used for training
X_train = train_set[1:785]  #row 1-784 are the features  #shape = (784, m)
Y_train = train_set[0]  #label #value to be predicted # row 0
X_train = X_train / 255.  #scaling

#test data set
test_set = data[m: m+200 ].T #200 test images
X_test = test_set[1:785]
Y_test = test_set[0]
X_test = X_test / 255.


################### training and testing process ################
#hyperparameters
epochs = 500
lr = 0.15 #learning rate

hw, hb, ow, ob = train(X_train, Y_train, epochs , lr)
accuracy = test(X_test, Y_test)
print("\nAccuracy: ", accuracy)

########## classifying custom image #####################
#open image
#img = cv2.imread('C:\\Users\\Samruddhi\\Desktop\\Neural Networks\\MNIST Handwritten digit classification\\custom data\\two.jpeg', 0)
#img[r][c] = grayscale value of pixel at that index

#pre-process the img
#img = preprocess_image(img)
#print(classify(hw, hb, ow, ob, img))
