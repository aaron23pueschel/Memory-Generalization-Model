
import numpy as np
import matplotlib.pyplot as plt
import pemguin as pm

## If we decide to use Pemguin for the sensory layers this is where we call it. 

class Sensory_Layer:
    def __init__(self,pemguin=False, number_of_neurons=100,pemguin_encoder=None,pemguin_decoder = None):
        ## I am going to initialize a parameter that allows us to create a sensory layer
        ## with or without pemguin. In the space below we will initialize any constants
        ## that we will use throughout the model. I have listed sample constants below:
        self.PI = 3.1415925
        self.TAU = 1.23456
        self.Neurons = []
        self.number_of_neurons = number_of_neurons
        self.pemguin_encoder = None
        self.weights_matrix = None
        self.error_list = []

        if not pemguin:
            self.initialize_neurons()
            self.initialize_lateral_connections()
        else:
            self.pemguin_encoder = pemguin_encoder
            self.pemguin_decoder = pemguin_decoder
            
        
        ## Here we will call the functions to initialize the neurons based off of the parameters listed above



    class Neuron:
        def __init__(self):
            # I am not sure what this will look like yet but I assume that a neuron will have a tuning 
            # function or else some kind of constants:

            self.TAU = 1.2345
            self.PI = 3.1415926
            self.additional_parameters = None
            self.generate_tuning_function()

        def generate_tuning_function(self):
            return 0
        def change_parameters(self):
            return 0
        
    def initialize_neurons(self):
        ## we will make it so that the index in the list corresponds to 
        ## its relative location to the other neurons
        if self.number_of_neurons <=0:
            self.error_list.append("Quantity of neurons in sensory layer must be an integer greater that 0")
            return False
        for i in range(0,int(self.number_of_neurons)):
            self.Neurons.append(self.Neuron())
        return True

    def initialize_lateral_connections(self):
        ## We have to consider how a neuron with a static connection will be connected to other neurons
        ## for now, we can assume an all to all network. We can represent this as an adjacency matrix whose
        ## elements are the weights of the synapses
        ## I will create an all to all matrix (except for self connected) whose weights range from 0-2pi and drop off 
        ## as in a gaussian distribution. The exact formula for this dropoff will be determined later.
        number_of_neurons = int(self.number_of_neurons)
        weights_matrix = np.zeros((number_of_neurons,number_of_neurons))
        for i in range(0,self.number_of_neurons):
            for j in range(0,self.number_of_neurons):
                if i==j:
                    weights_matrix[i][j]=0
                    continue
                ## I just came up with this but it kind of resembles the relationship between 
                ## neurons close to eachother. feel free to plot
                weights_matrix[i][j] = np.exp(np.cos(2*np.pi*(i+j)/self.number_of_neurons))
        self.weights_matrix = weights_matrix

    def get_weights_matrix(self):
        return self.weights_matrix
                
    


    