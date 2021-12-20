
import numpy as np
import matplotlib.pyplot as plt
import logging
import scipy.spatial.distance as scipy
from scipy.sparse import random


def generate_input_vector(vector_length,density):
        rand = (random(1,vector_length,density= density).A)[0]
        ones = [int(np.rint(x)) for x in rand]
        return ones

class Sensory_Layer:
    def __init__(self,number_of_neurons=100,omega=1):
        self.Neurons = []
        self.number_of_neurons = number_of_neurons
        self.normalized_matrix = None
        self.omega = omega
        self.initialize_neurons(omega=omega)

    ## Nueron class models a single neuron in the sensory layer. Note that neural responses dont really make
    ## sense outside of the context of the sensory layer, so keep that in mind if you decide to 
    ## instantiate a neuron without a sensory layer
    class Neuron:
            
            ## Neuron positions will be determined by their index in the Neuron list.
            def __init__(self,omega=1,index=0,number_of_neurons_in_layer=100):
                self.omega=omega
                self.index = index
                self.number_of_neurons_in_layer = number_of_neurons_in_layer
                self.phi = (self.index*2*np.pi-np.pi)/self.number_of_neurons_in_layer

            def neural_response(self,input):
                ## I am going to assume that phi will be determined by a neurons index with respect to the total
                ## number of neurons in a layer. 
                output_vec = np.zeros(self.number_of_neurons_in_layer)
                for input_index in range(0,self.number_of_neurons_in_layer):
                    ## Theta will be determined by the index of the input vector divided by the total number of 
                    ## neurons in a given layer times 2pi. Observe that for omega = 1, f_ij() = 1 (maximal) when phi==theta
                    theta = (input_index*2*np.pi-np.pi)/self.number_of_neurons_in_layer
                    alpha = input[input_index]
                    output_vec[input_index] = input[input_index]*self.f_ij(self.phi,theta)
                return output_vec

            def f_ij(self,phi,theta): # Eq 1 in Bays. This gives the response of a neuron to input theta, 
                                      # for any value of theta. Each neuron has a preferred orientation phi.
                return np.exp((1/self.omega)*np.cos(phi-theta)-1)

    def initialize_neurons(self,omega=1):
        ## we will make it so that the index in the list corresponds to 
        ## its relative location to the other neurons
        if self.number_of_neurons <=0:
            self.error_list.append("Quantity of neurons in sensory layer must be an integer greater that 0")
            return False
        for i in range(0,int(self.number_of_neurons)):
            self.Neurons.append(self.Neuron(index=i,number_of_neurons_in_layer=self.number_of_neurons,omega=omega))

    def population_response(self,response_matrix = None):
        transposed = np.transpose(response_matrix)
        population_response = np.sum(transposed, axis = 0)
        return population_response
       

    def generate_output(self,input_vec,plot = False,set_normalized_matrix = False,normalize = True,gain_gamma = 1):
        ## input vector must be of the length of the number of neurons
        ## input can be zeros and ones, but in bays, equation 2, there is a scaling factor alpha associated
        ## with each input. You can provide a vector of any real numbers and the output will be scaled accordingly
        if(self.number_of_neurons!=len(input_vec)):
            print("Input vector must be the same as number of neurons in layer")
            print("input vec len: " + str(len(input_vec)))
            print("number of neurons: "+ str(self.number_of_neurons))
            return 0

        f_ij_matrix = []
        for i in range(0,self.number_of_neurons):
            f_ij_matrix.append(self.Neurons[i].neural_response(input_vec))
    
        ## This is an option to return and plot before normalization
        ## set normalize flag to False if you choose this option
        if not normalize:
            if plot:
                c = plt.imshow(f_ij_matrix)
                plt.colorbar(c)
                plt.title("Not normalized output matrix")
                plt.show()
            return f_ij_matrix
            
        ## Normalize == True here 
        denominator = np.sum(f_ij_matrix)
        if denominator==0:
            print("Input non-existent")
            return 0
        ## We assume that a firing rate r_ij is an integer greater or equal to 0
        normalized_matrix = ((gain_gamma/denominator)*np.array(f_ij_matrix))
            
        
        ## This is a flag to set the normalized matrix of a layer 
        ## Basically, it sets the output of a layer in response to a certain input.
        ## To change this, just run new input through this function and set the set_normalized_matrix to true
        if set_normalized_matrix:
            self.normalized_matrix = normalized_matrix


        if plot:
            c = plt.imshow(normalized_matrix)
            plt.title("Normalized output matrix")
            plt.colorbar(c)
            plt.show()
        return np.array(normalized_matrix)

    

    def return_spiking_distribution(self,n_ij_matrix,T=100,plot=False):
        #n_ij_matrix = n_ij_matrix.astype(int)

        spiking_probabilities = np.ones((self.number_of_neurons,self.number_of_neurons))
        for i in range(0,self.number_of_neurons):
            for j in range(0,self.number_of_neurons):
                n_ij = np.rint(n_ij_matrix[i][j])
                r_ij = self.normalized_matrix[i][j]
                if r_ij ==0:
                    spiking_probabilities[i][j]=0
                    continue
                spiking_probabilities[i][j] = (((T*r_ij)**n_ij)/np.math.factorial(n_ij))*np.exp(-1*r_ij*T)
        if plot:
            c = plt.imshow(spiking_probabilities)
            plt.colorbar(c)
            plt.title("Probablity of firing at rate n")
            plt.show()
        return spiking_probabilities

    def maximum_liklihood(self,N,F):
        N = np.transpose(N)
        F = np.transpose(F)
        temp = np.zeros(len(N))
        for p in range(0,len(N)):
            sum = 0
            for i in range (0,len(N)):
                sum+= N[i][p]*np.log(F[i][p]+.00001)
            temp[p]=sum
        return np.argmax(temp)
    def maximum_liklihood_cos(self,N):
        N = np.transpose(N)
        temp = np.zeros(len(N))
        for p in range(0,len(N)):
            sum = 0
            for i in range (0,len(N)):
                theta = (p*2*np.pi-np.pi)/len(N)
                sum+= N[i][p]*np.cos(self.Neurons[i].phi-theta)
            temp[p]=sum
        return np.argmax(temp)

    def percent_similarity(self,original_input,probability_matrix):
        argmax_indeces = list(set(np.argmax(np.transpose(probability_matrix),axis=0)))
        predicted_input = np.zeros(len(original_input))
        for i in argmax_indeces:
            predicted_input[i]=1
        truth_vec = (original_input==predicted_input)
        count = 0
        for i in truth_vec:
            if not i:
                count+=1
        print(str((1-count/len(original_input))*100)+" percent of the original input recalled")


