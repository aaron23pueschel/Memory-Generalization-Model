
import numpy as np
import matplotlib.pyplot as plt
import logging



class Sensory_Layer:
    def __init__(self,number_of_neurons=100,omega = 1):
        self.Neurons = []
        self.number_of_neurons = number_of_neurons
        self.alphas = []
        self.normalized_matrix = None
        self.initialize_neurons()
        
    class Neuron:
        ## A neuron will take in a vector whose elements are either 0 or 1 and output a vector based on
        ## equations 1-4 in bays

        ## Neuron position will be determined by its index in the Neuron list.
        def __init__(self,omega=1,index=0,number_of_neurons_in_layer=100):
            self.omega = omega
            self.index = index
            self.number_of_neurons_in_layer = number_of_neurons_in_layer

        def tuning_curve(self,input):
            ## I am going to assume that phi will be determined by a neurons index with respect to the total
            ## number of neurons in a layer. 
            phi = (self.index*2*np.pi)/self.number_of_neurons_in_layer

            ## Theta will be determined by the index of the input vector divided by the total number of 
            ## neurons in a given layer time 2pi. Observe that for omega = 1, f_ij() = 1 (maximal) when phi==theta

            output_vec = np.zeros(self.number_of_neurons_in_layer)
            for input_index in range(0,self.number_of_neurons_in_layer):
                theta = (input_index*2*np.pi)/self.number_of_neurons_in_layer
                alpha = input[input_index]
                output_vec[input_index] = input[input_index]*self.f_ij(phi,theta)
            return output_vec


        def f_ij(self,phi,theta):
            return np.exp((1/self.omega)*np.cos(phi-theta)-1)

    def generate_output(self,input_vec,plot = False,set_normalized_matrix = False,normalize = True):
        ## input vector must be of the length of the number of neurons
        ## input can be zeros and ones, but in bays, equation 2, there is a scaling factor alpha associated
        ## with each input. You can provide a vector of any integers and the output will be scaled accordingly
        if(self.number_of_neurons!=len(input_vec)):
            print("Input vector must be the same as number of neurons in layer")
            print("input vec len: " + str(len(input_vec)))
            print("number of neurons: "+ str(self.number_of_neurons))
            return 0
        f_ij_matrix = []
        for i in range(0,self.number_of_neurons):
            f_ij_matrix.append(self.Neurons[i].tuning_curve(input_vec))
        if not normalize:
            return f_ij_matrix
        normalized_matrix = []
        for rows in range(0,len(f_ij_matrix)):
            normalized_matrix.append(f_ij_matrix[rows]/np.sum(f_ij_matrix[rows]))
        if plot:
            plt.imshow(normalized_matrix)
            plt.title("Normalized output matrix")
            plt.show()
        if set_normalized_matrix:
            self.normalized_matrix = normalized_matrix
        return normalized_matrix
    def return_spiking_distribution(self,n_ij_matrix,T=100,plot=False):

        ## Im pretty sure that n_ij has to be supplied by the user. Since n_ij is obviously an element 
        ## from a matrix (the firing predicted rate of neuron i in response to input j) it follows that the user
        ## must supply an INTEGER matrix of predicted firing rates. 

        spiking_probabilities = np.ones((self.number_of_neurons,self.number_of_neurons))
        for i in range(0,self.number_of_neurons):
            for j in range(0,self.number_of_neurons):
                n_ij = n_ij_matrix[i][j]
                r_ij = self.normalized_matrix[i][j]
                spiking_probabilities[i][j] = ((T*r_ij)**n_ij)/np.math.factorial(n_ij)*np.exp(-1*r_ij*T)
        if plot:
            c = plt.imshow(spiking_probabilities)
            plt.colorbar(c)
            plt.title("Probablity of firing given matrix of firing rates n")
            plt.show()
        return spiking_probabilities

    def initialize_neurons(self):
        ## we will make it so that the index in the list corresponds to 
        ## its relative location to the other neurons
        if self.number_of_neurons <=0:
            self.error_list.append("Quantity of neurons in sensory layer must be an integer greater that 0")
            return False
        for i in range(0,int(self.number_of_neurons)):
            self.Neurons.append(self.Neuron(index=i,number_of_neurons_in_layer=self.number_of_neurons))

    


    