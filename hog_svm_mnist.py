import cv2
from sklearn import svm
from sklearn.metrics import classification_report
from keras.datasets import mnist
import numpy as np

def deskew(img, imgSize):
    # calculate image moments
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        # no deskewing needed
        return img.copy()

    # calculate skew based on central moments
    skew = m['mu11'] / m['mu02']

    # calculate affine transformation to correct skewness
    M = np.float32([[1, skew, -0.5*imgSize*skew], [0, 1, 0]])

    # apply affine transformation
    img = cv2.warpAffine(img, M, (imgSize, imgSize), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)

    return img

# Load the mnist dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# train and test on 5000 samples (remove the below 4 lines to train and test on whole dataset)
trainX = trainX[:5000] 
trainY = trainY[:5000]
testX = testX[:5000]
testY = testY[:5000]

imsize = 28 # size of image (28x28)

# HOG parameters:
winSize = (imsize, imsize) # 28, 28
blockSize = (imsize//2, imsize//2) # 14, 14    
cellSize = (imsize//2, imsize//2) #14, 14
blockStride = (imsize//4, imsize//4) # 7, 7
nbins = 9
signedGradients = True
derivAperture = 1
winSigma = -1.0
histogramNormType = 0
L2HysThreshold = 0.2
gammaCorrection = 1
nlevels = 64

# define the HOG descriptor
hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, 
                        histogramNormType, L2HysThreshold, gammaCorrection, nlevels, signedGradients)

# compute HOG descriptors
train_descriptors = []
for i in range(trainX.shape[0]):
    trainX[i] = deskew(trainX[i], 28) # deskew the current image
    descriptor = hog.compute(trainX[i]) # compute the HOG features
    train_descriptors.append(descriptor) # append it to the train decriptors list

test_descriptors = []
for i in range(testX.shape[0]):
    testX[i] = deskew(testX[i], 28) # deskew the current image
    descriptor = hog.compute(testX[i]) # compute the HOG features
    test_descriptors.append(descriptor) # append it to the test descriptors list

#train_descriptors = np.array(train_descriptors)
train_descriptors = np.resize(train_descriptors, (trainX.shape[0], 81))

#test_descriptors = np.array(test_descriptors)
test_descriptors = np.resize(test_descriptors, (testX.shape[0], 81))

# classifier
clf = svm.SVC(C=1.0, kernel='rbf')
clf.fit(train_descriptors, trainY)

# print the classification report
print(classification_report(testY, clf.predict(test_descriptors)))

# visualize the predictions
for i in range(testX.shape[0]):
    # resize the image to be 10x bigger
    img = cv2.resize(testX[i], None, fx=10, fy=10)
    # make prediction on the current image
    prediction = clf.predict(test_descriptors[i:i+1])
    # write the predicted number on the image
    cv2.putText(img, 'prediction:' + str(prediction[0]), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 255)
    # display the image
    cv2.imshow('img', img)
    
    # get the pressed key
    key = cv2.waitKey(0)
    # if the pressed key is q, destroy the window and break out of the loop
    if key == ord('q'):
        cv2.destroyAllWindows()
        break
        
