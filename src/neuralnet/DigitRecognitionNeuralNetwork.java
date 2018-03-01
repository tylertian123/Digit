package neuralnet;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import mnist.MNISTImage;

public class DigitRecognitionNeuralNetwork {
	
	static class SigmoidActivation implements ActivationFunction {
		@Override
		public double activation(double z) {
			return 1.0 / (1.0 + Math.exp(-z));
		}

		@Override
		public double activationDerivative(double z) {
			return activation(z) * (1 - activation(z));
		}
	}
	static class QuadraticCost implements CostFunction {

		@Override
		public double costDerivative(double y, double a) {
			return (a - y);
		}
		
	}

	public static final ActivationFunction SIGMOID_ACTIVATION = new SigmoidActivation();
	public static final CostFunction QUADRATIC_COST = new QuadraticCost();
	
	/*
	 * Although the input layer does not have weights and biases, space is still allocated for them
	 * so the indices are less confusing
	 */
	public final int layers;
	//the number of neurons in each layer
	public int[] neuronCounts;
	//most number of neurons in a layer
	public final int neuronMax;
	//bias of each neuron, from layer and number
	//e.g. biases[1][1] is the bias of the 2nd neuron in the 1st hidden layer
	public double[][] biases;
	//weight of each connection, from layer, number and number
	//e.g. weights[1][1][2] is the weight of the connection between
	//the 2nd neuron in the 1st hidden layer and the 3rd neuron in the layer before it
	public double[][][] weights;
	//The activation and cost functions
	protected final ActivationFunction activationFunction;
	protected final CostFunction costFunction;
	
	static int getMax(int[] arr) {
		int max = 0;
		for(int i : arr) {
			if(i > max) {
				max = i;
			}
		}
		return max;
	}
	static double dotProduct(double[] a, double[] b) {
		double result = 0;
		for(int i = 0; i < a.length; i ++) {
			result += a[i] * b[i];
		}
		return result;
	}
	static double dotProduct(final double[] a, final double[] b, final int len) {
		double result = 0;
		for(int i = 0; i < len; i ++) {
			result += a[i] * b[i];
		}
		return result;
	}
	
	public DigitRecognitionNeuralNetwork(int[] neuronCounts, ActivationFunction activation, CostFunction cost) {
		activationFunction = activation;
		costFunction = cost;
		this.layers = neuronCounts.length;
		this.neuronCounts = neuronCounts;
		neuronMax = getMax(neuronCounts);
		//The first (input) layer has no biases
		this.biases = new double[layers][neuronMax];
		this.weights = new double[layers][neuronMax][neuronMax];
		//sets each weight and bias to a random real number between
		//0 and 1
		//No need to initialize the first layer's weights and biases
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				biases[i][j] = Math.random() * 2.0 - 1;
				for(int k = 0; k < neuronCounts[i - 1]; k ++)
					weights[i][j][k] = Math.random() * 2.0 - 1;
			}
		}
	}
	
	public int classify(MNISTImage img) {
		double[] lastActivations = new double[neuronMax];
		double[] input = img.asNeuralNetworkInput();
		for(int i = 0; i < input.length; i ++) {
			lastActivations[i] = input[i];
		}
		double[] activations = new double[neuronMax];
		
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				activations[j] = activationFunction.activation(
						dotProduct(lastActivations, weights[i][j], neuronCounts[i - 1])
						+ biases[i][j]);
				//System.out.println("Layer " + i + " neuron " + j + ": " + dotProduct(lastActivations, weights[i][j], neuronCounts[i - 1]));
				//System.out.println(neuronCounts[i]);
			}
			lastActivations = activations.clone();
		}
		int maxIndex = 0;
		for(int i = 0; i < 10; i ++) {
			if(lastActivations[i] > lastActivations[maxIndex]) {
				maxIndex = i;
			}
		}
		return maxIndex;
	}
	
	public void SGD(MNISTImage[] trainingData, int batchSize, double learningRate, int epochs) {
		SGD(trainingData, batchSize, learningRate, epochs, null);
	}
	public void SGD(MNISTImage[] trainingData, int batchSize, double learningRate, int epochs, MNISTImage[] evalData) {
		if(evalData != null) {
			System.out.println("No Training:\nEvaluating...");
			int correct = 0;
			for(int i = 0; i < evalData.length; i ++) {
				//System.out.println(evalData[i].getAvg());
				if(classify(evalData[i]) == evalData[i].classification) {
					correct ++;
				}
			}
			double percentage = ((double) correct) / evalData.length * 100;
			System.out.println(percentage + "% correctly classified.");
		}
		for(int epoch = 1; epoch <= epochs; epoch ++) {
			List<MNISTImage> l = Arrays.asList(trainingData);
			Collections.shuffle(l);
			
			if(evalData != null) {
				System.out.println("Epoch #" + epoch);
				System.out.println("Learning...");
			}
			
			for(int i = 0; i < trainingData.length; i += batchSize) {
				MNISTImage[] miniBatch = new MNISTImage[batchSize];
				for(int j = 0; j < batchSize && i + j < trainingData.length; j ++) {
					miniBatch[j] = trainingData[i + j];
				}
				learnFromMiniBatch(miniBatch, learningRate);
			}
			
			if(evalData != null) {
				System.out.println("Evaluating...");
				int correct = 0;
				for(int i = 0; i < evalData.length; i ++) {
					if(classify(evalData[i]) == evalData[i].classification) {
						correct ++;
					}
				}
				double percentage = ((double) correct) / evalData.length * 100;
				System.out.println(percentage + "% correctly classified.");
			}
		}
	}
	
	public void learnFromMiniBatch(MNISTImage[] miniBatch, double learningRate) {
		int batchSize = 0;
		double[][] biasDerivativesTotal = new double[layers][neuronMax];
		double[][][] weightDerivativesTotal = new double[layers][neuronMax][neuronMax];
		
		for(MNISTImage trainingSample : miniBatch) {
			if(trainingSample != null) {
				batchSize ++;
				//Expected output
				double[] y = trainingSample.generateExpectedOutput();
				//Activations
				double[][] a = new double[layers][neuronMax];
				//Weighted sums
				double[][] z = new double[layers][neuronMax];
				//Errors
				double[][] e = new double[layers][neuronMax];
				
				//Feedforward
				a[0] = trainingSample.asNeuralNetworkInput();
				for(int i = 1; i < layers; i ++) {
					for(int j = 0; j < neuronCounts[i]; j ++) {
						z[i][j] = dotProduct(a[i - 1], weights[i][j], neuronCounts[i - 1]) + biases[i][j];
						a[i][j] = activationFunction.activation(z[i][j]);
					}
				}
				//Calculate error for output layer
				for(int j = 0; j < neuronCounts[layers - 1]; j ++) {
					e[layers - 1][j] = activationFunction.activationDerivative(z[layers - 1][j]) 
							* costFunction.costDerivative(y[j], a[layers - 1][j]);
					//System.out.println(e[layers - 1][j]);
				}
				//Backpropagate
				for(int i = layers - 2; i >= 0; i --) {
					for(int j = 0; j < neuronCounts[i]; j ++) {
						//Perform the sigma
						double err = 0.0;
						for(int k = 0; k < neuronCounts[i + 1]; k ++) {
							err += e[i + 1][k] * weights[i + 1][k][j];
						}
						e[i][j] = err;
					}
				}
				//Calculate the weight and bias derivatives and add to total
				//Skip input layer
				for(int i = 1; i < layers; i ++) {
					for(int j = 0; j < neuronCounts[i]; j ++) {
						biasDerivativesTotal[i][j] += e[i][j];
						for(int k = 0; k < neuronCounts[i - 1]; k ++) {
							weightDerivativesTotal[i][j][k] = e[i][j] * a[i - 1][k];
						}
					}
				}
			}
		}
		
		//Divide to take the average
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				//System.out.println(biasDerivativesTotal[i][j]);
				biasDerivativesTotal[i][j] /= batchSize;
				for(int k = 0; k < neuronCounts[i - 1]; k ++) {
					//System.out.println(weightDerivativesTotal[i][j][k]);
					weightDerivativesTotal[i][j][k] /= batchSize;
				}
			}
		}
		//Update the new weights and biases
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				biases[i][j] = biases[i][j] - learningRate * biasDerivativesTotal[i][j];
				for(int k = 0; k < neuronCounts[i - 1]; k ++) {
					weights[i][j][k] = weights[i][j][k] - learningRate * weightDerivativesTotal[i][j][k];
				}
			}
		}
	}
}
