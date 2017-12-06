package neuralnet;

import java.util.ArrayList;
import java.util.Arrays;

import java.lang.Cloneable;

public class NeuralNetwork implements Cloneable {
	@Override
	public NeuralNetwork clone() {
		NeuralNetwork copy = new NeuralNetwork(this.neuronCounts, 0);
		for(int i = 0; i < neuronCounts.length - 1; i ++) {
			for(int j = 0; j < neuronCounts[i + 1]; j ++) {
				copy.biases[i][j] = this.biases[i][j];
				for(int k = 0; k < neuronCounts[i]; k ++) {
					copy.weights[i][j] = this.weights[i][j];
				}
			}
		}
		return copy;
	}
	
	//the number of neurons in each layer
	public int[] neuronCounts;
	//most number of neurons in a layer
	public final int neuronMax;
	//bias of each neuron (hidden and output layers only), from layer and number
	//e.g. biases[0][1] is the bias of the 2nd neuron in the 1st hidden layer
	public Double[][] biases;
	//weight of each connection (hidden and output layers only), from layer, number and number
	//e.g. weights[0][1][2] is the weight of the connection between
	//the 2nd neuron in the 1st hidden layer and the 3rd neuron in the layer before it
	public Double[][][] weights;
	
	static int getMax(int[] arr) {
		int max = 0;
		for(int i : arr) {
			if(i > max) {
				max = i;
			}
		}
		return max;
	}
	static double dotProduct(Double[] a, Double[] b) {
		double result = 0;
		for(int i = 0; i < a.length; i ++) {
			result += a[i] * b[i];
		}
		return result;
	}
	static Double sigmoid(Double z) {
		return 1.0 / (1.0 + Math.exp(-z));
	}
	
	public NeuralNetwork(int[] neuronCounts) {
		this.neuronCounts = neuronCounts;
		neuronMax = getMax(neuronCounts);
		//The first (input) layer has no biases
		this.biases = new Double[neuronCounts.length - 1][neuronMax];
		this.weights = new Double[neuronCounts.length - 1][neuronMax][neuronMax];
		//sets each weight and bias to a random real number between
		//0 and 1
		for(int i = 0; i < neuronCounts.length - 1; i ++) {
			for(int j = 0; j < neuronCounts[i + 1]; j ++) {
				biases[i][j] = Math.random() * 2.0 - 1;
				for(int k = 0; k < neuronCounts[i]; k ++) {
					//System.out.println("i: " + i);
					//System.out.println("j: " + j);
					//System.out.println("k: " + k);
					weights[i][j][k] = Math.random() * 2.0 - 1;
				}
			}
		}
	}
	//the int parameter is not used
	//it was included so that another version of the constructor that does not init
	//the network with random values can be created
	public final static int NO_INIT = 0;
	public NeuralNetwork(int[] neuronCounts, int dummy) {
		this.neuronCounts = neuronCounts;
		neuronMax = getMax(neuronCounts);
		//The first (input) layer has no biases
		this.biases = new Double[neuronCounts.length - 1][neuronMax];
		this.weights = new Double[neuronCounts.length - 1][neuronMax][neuronMax];
	}
	
	//returns output of the network if input is input
	public Double[] feedForward(Double[] input) {
		ArrayList<Double> lastLayer = new ArrayList<>(Arrays.asList(input));
		for(int layer = 1; layer < neuronCounts.length; layer ++) {
			ArrayList<Double> currentLayer = new ArrayList<Double>(neuronMax);
			for(int neuron = 0; neuron < neuronCounts[layer]; neuron ++) {
				currentLayer.add(
						sigmoid(
								dotProduct(lastLayer.toArray(new Double[lastLayer.size()]), 
										weights[layer - 1][neuron])
								+ biases[layer - 1][neuron]
								)
						);
			}
			lastLayer = currentLayer;
			
		}
		return lastLayer.toArray(new Double[lastLayer.size()]);
	}
}
