"""
File: hopfield.py
Author: Trevor Stalnaker
Assignment 3
Date: 10 October 2018
"""

import numpy as np

"""
Implementation of a Hopefield Network
"""
class Hopfield():

    def __init__(self, n):
        self.t = np.zeros((n,n))

    def learn(self, array):
        for row in array:
            self.t += np.outer((2*row)-1,(2*row)-1)
            np.fill_diagonal(self.t, 0)
        
    def test(self, u, i=5):
        for x in range(i):
            u = (np.dot(u, self.t) >= 0).astype('int')
        return u
            

"""
The main method of this assignment
"""
def main():
    
    #Part One: Generate Some Training Data
    matrix = np.random.randint(0,2,(5,30))

    #Part Two: Display a Confusion Matrix
    print("\nPart 2: Vector-cosine confusion matrix of an array with itself ----------------------\n")
    show_confusion(matrix, matrix)

    #Part Three: Noise It Up!
    print("\nPart 3: Confusion matrix with 25 percent noise ------------------------------------\n")
    show_confusion(noisy_copy(matrix, 0.25), matrix)

    #Part Four: Code Up Your Hopfield Net
    print("\nPart 4: Recovering small patterns with a Hopfield net -----------------------------\n")
    index = np.random.randint(0,5)
    vector = matrix[index]
    hop = Hopfield(30)
    hop.learn(matrix)
    output = hop.test(vector)
    print("Recover pattern, no noise:")
    print("Input:  " + str(vector))
    print("Output: " + str(output))
    print("Vector Cosine: " + str(getVectorCosine(vector, output)))

    noisy = noisy_copy(matrix, .25)[index]
    noisy_output = hop.test(noisy)
    print("\nRecover pattern, 25% noise:")
    print("Input:    " + str(noisy))
    print("Output:   " + str(noisy_output))
    print("Original: " + str(vector))
    print("Vector Cosine: " + str(getVectorCosine(vector, noisy_output)))

    #Part Five: Improving the Capacity
    print("\nPart 5: Recovering big patterns ----------------------------------------------------\n")
    print("Confusion matrix for 1000-element vectors with 25 percent noise:\n")
    hugeMatrix = np.random.randint(0,2,(10,1000))
    noisyMatrix = noisy_copy(hugeMatrix,.25)
    bigHop = Hopfield(1000)
    bigHop.learn(hugeMatrix)
    show_confusion(noisyMatrix, hugeMatrix)
    print("\nRecovering patterns with 25 percent noise:\n")
    for index in range(len(hugeMatrix)):
        bigOut = bigHop.test(noisyMatrix[index])
        print("Vector cosine on pattern " + str(index) + " = " + str(getVectorCosine(hugeMatrix[index],bigOut)))
        
    #Extra Credit
    print("\nExtra Credit: Recovering ASCII Art\n")
    print("Original Images:\n")
    print("Logo:\n")
    print(asciiArt)
    print("\nStar:\n")
    print(asciiStar)
    asciiVector = imageToArray(asciiArt)
    asciiStarVector = imageToArray(asciiStar)
    artMatrix = np.stack((asciiVector, asciiStarVector))
    noisyArt = noisy_copy(artMatrix, .25)[0]
    noisyStar = noisy_copy(artMatrix, .25)[1]
    print("\nNoisy Inputs:\n")
    print("Logo:")
    print("\n" + arrayToImage(noisyArt, 32) + "\n")
    print("Star:")
    print("\n" + arrayToImage(noisyStar, 32) + "\n")
    extraHop = Hopfield(1024)
    extraHop.learn(artMatrix)
    artOut = extraHop.test(noisyArt)
    starOut = extraHop.test(noisyStar)
    print("\nRestored Output:\n")
    print("Logo:")
    print("\n" + arrayToImage(artOut, 32) + "\n")
    print("Star:")
    print("\n" + arrayToImage(starOut, 32) + "\n")
##    random = np.random.randint(0,2,1024)
##    print("Randomly Generated Pattern:\n")
##    print(arrayToImage(random, 32))
##    print("\n" + arrayToImage(extraHop.test(random), 32) + "\n")
    
    
    
"""
Returns the vector cosine of two vectors
"""
def getVectorCosine(a, b):
    return round(np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b)), 2)

"""
Prints a confusion matrix of two given matrices
"""
def show_confusion(a, b):
    rowCount = 0
    for row in a:
        rowCount += 1
        lyst = []
        columnCount = 0
        for column in b:
            columnCount += 1
            lyst.append(getVectorCosine(row, column))
            if rowCount == columnCount:
                break      
        print(' '.join(["%.2f"]*len(lyst)) % tuple(lyst))

"""
Creates a noisy copy of a matrix
"""
def noisy_copy(a, prob):
    copy = np.copy(a)
    for row in range(len(copy)):
        for column in range(len(copy[row])):
            if prob > np.random.random_sample():
                copy[row, column] -= (2*copy[row, column] - 1) #flips the bit
    return copy

"""
Converts and ASCII art image into a numpy array
"""
def imageToArray(image):
    return np.asarray([int(x) for x in image.replace("\n","").replace("*","1").replace("  ","0")])

"""
Converts a numpy array into an ASCII art image
"""
def arrayToImage(array, width):
    string = ""
    count = 1
    for element in array:
        string += str(element)
        if count%width == 0 and count != len(array):
            string += "\n"
        count += 1
    return string.replace("0"," ").replace("1","*")

asciiArt = """********************************
********************************
********************************
*****                      *****
*****                      *****
**************    **************
**************    **************
**************    **************
**************    **************
**************    **************
**************    **************
**************    **************
**************    **************
**************    **************
********************************
********************************
*****    ***************    ****
*****    ***************    ****
*****    ***************    ****
*****    ***************    ****
*****    ***************    ****
*****                       ****
************************    ****
************************    ****
************************    ****
************************    ****
************************    ****
************************    ****
************************    ****
********************************
********************************
********************************"""

asciiStar = """********************************
********************************
********************************
********************************
********************************
********************************
********************************
********************************
***************  ***************
***************  ***************
**************    **************
*************      *************
*                              *
*****                      *****
*******                 ********
**********            **********
**********            **********
*********     ****     *********
********   **********   ********
*******  **************  *******
********************************
********************************
********************************
********************************
********************************
********************************
********************************
********************************
********************************
********************************
********************************
********************************"""
            
    

if __name__ == '__main__':
    main()
