package neuralnet;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import mnist.MNISTImage;

/*
 * A neural network that classifies MNIST database digits
 */
public class DigitRecognitionNeuralNetwork {
	//Pre-defined activation and cost functions
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
	static class CrossEntropySigmoidCost implements CostFunction {
		@Override
		public double costDerivative(double y, double a) {
			return (1 - y) / (1 - a) - y / a;
		}
	}

	public static final ActivationFunction SIGMOID_ACTIVATION = new SigmoidActivation();
	public static final CostFunction QUADRATIC_COST = new QuadraticCost();
	public static final CostFunction CROSSENTROPY_SIGMOID_COST = new CrossEntropySigmoidCost();
	
	/*
	 * Although the input layer does not have weights and biases, space is still allocated for them
	 * so the indices are less confusing
	 */
	protected final int layers;
	//the number of neurons in each layer
	protected int[] neuronCounts;
	//most number of neurons in a layer
	protected final int neuronMax;
	//bias of each neuron, from layer and number
	//e.g. biases[1][1] is the bias of the 2nd neuron in the 1st hidden layer
	protected double[][] biases;
	//weight of each connection, from layer, number and number
	//e.g. weights[1][1][2] is the weight of the connection between
	//the 2nd neuron in the 1st hidden layer and the 3rd neuron in the layer before it
	protected double[][][] weights;
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
	
	//Creates a non-rectangular array
	static double[][] createJaggedArray(int[] lengths) {
		double[][] arr = new double[lengths.length][];
		for(int i = 0; i < arr.length; i ++)
			arr[i] = new double[lengths[i]];
		return arr;
	}
	//Used to create a jagged array of rectangular arrays
	static double[][][] createJaggedArray3d(int[] widths, int[] heights) {
		if(widths.length != heights.length)
			throw new IllegalArgumentException("Dimension arrays provided are not of the same length");
		double[][][] arr = new double[widths.length][][];
		for(int i = 0; i < arr.length; i ++)
			arr[i] = new double[widths[i]][heights[i]];
		return arr;
	}
	//Creates a 3-dimensional array in the shape of the weights to store them in
	double[][][] createWeightsArray() {
		int[] sizes2 = new int[layers];
		for(int i = 0; i < layers - 1; i ++)
			sizes2[i + 1] = neuronCounts[i];
		//First layer contains no weights, so each neuron is connected to 0 others
		sizes2[0] = 0;
		return createJaggedArray3d(neuronCounts, sizes2);
	}
	//Creates a 2-dimensional array in the shape of the biases to store them in
	double[][] createBiasesArray() {
		return createJaggedArray(neuronCounts);
	}
	
	//Creates the network and initializes each weight and bias with a normal distribution random number
	//with mean of 0 and standard deviation 1
	//The activation and cost are also custom specified
	public DigitRecognitionNeuralNetwork(int[] neuronCounts, ActivationFunction activation, CostFunction cost) {
		Random r = new Random();
		activationFunction = activation;
		costFunction = cost;
		this.layers = neuronCounts.length;
		this.neuronCounts = neuronCounts;
		neuronMax = getMax(neuronCounts);
		//The first (input) layer has no biases, but memory is still allocated to keep indices simple
		this.biases = createBiasesArray();
		this.weights = createWeightsArray();
		//sets each weight and bias to a random real number between
		//0 and 1
		//No need to initialize the first layer's weights and biases
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				biases[i][j] = r.nextGaussian();
				for(int k = 0; k < neuronCounts[i - 1]; k ++)
					weights[i][j][k] = r.nextGaussian();
			}
		}
	}
	
	//Feedforwards and outputs the 'classification' of the digit
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
	
	//Stochastic Gradient Descent
	public void SGD(MNISTImage[] trainingData, int batchSize, double learningRate, int epochs) {
		SGD(trainingData, batchSize, learningRate, epochs, null);
	}
	public void SGD(MNISTImage[] trainingData, int batchSize, double learningRate, int epochs, MNISTImage[] evalData) {
		double maxPercentage = 0.0;
		int maxEpoch = -1;
		if(evalData != null) {
			System.out.println("No Training:\nEvaluating...");
			int correct = 0;
			for(int i = 0; i < evalData.length; i ++) {
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
			
			//Separate the shuffled training samples into mini-batches and train with each mini-batch
			for(int i = 0; i < trainingData.length; i += batchSize) {
				List<MNISTImage> miniBatchList = l.subList(i, Math.min(i + batchSize, l.size()));
				MNISTImage[] miniBatch = new MNISTImage[miniBatchList.size()];
				miniBatchList.toArray(miniBatch);
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
				if(percentage > maxPercentage) {
					maxPercentage = percentage;
					maxEpoch = epoch;
				}
			}
		}
		if(evalData != null)
			System.out.printf("Max classification rate: %f%%, reached at Epoch #%d", maxPercentage, maxEpoch);
	}
	//Uses gradient descent and backpropagation to learn from a mini-batch
	public void learnFromMiniBatch(MNISTImage[] miniBatch, double learningRate) {
		//The size of the batch
		//Only incremented for values that are non-null
		int batchSize = 0;
		//Summed dC/db and dC/dw
		double[][] biasDerivativesTotal = createBiasesArray();
		double[][][] weightDerivativesTotal = createWeightsArray();
		
		for(MNISTImage trainingSample : miniBatch) {
			if(trainingSample != null) {
				batchSize ++;
				//Expected output
				double[] y = trainingSample.generateExpectedOutput();
				//Activations
				double[][] a = createBiasesArray();
				//Weighted sums
				double[][] z = createBiasesArray();
				//Errors
				double[][] e = createBiasesArray();
				
				//Feedforward
				a[0] = trainingSample.asNeuralNetworkInput();
				for(int i = 1; i < layers; i ++) {
					for(int j = 0; j < neuronCounts[i]; j ++) {
						//Dot product of last layer's activations with this layer's weights added to the bias 
						z[i][j] = dotProduct(a[i - 1], weights[i][j]) + biases[i][j];
						//Put through the activation function
						a[i][j] = activationFunction.activation(z[i][j]);
					}
				}
				//Calculate error for output layer
				for(int j = 0; j < neuronCounts[layers - 1]; j ++) {
					//The error for a neuron in the output layer =
					//activation'(z) * dC/da
					e[layers - 1][j] = activationFunction.activationDerivative(z[layers - 1][j]) 
							* costFunction.costDerivative(y[j], a[layers - 1][j]);
				}
				//Backpropagate
				for(int i = layers - 2; i >= 0; i --) {
					for(int j = 0; j < neuronCounts[i]; j ++) {
						//Perform the sigma
						double err = 0.0;
						for(int k = 0; k < neuronCounts[i + 1]; k ++) {
							//The error of a neuron in the next layer * the weight connecting them
							err += e[i + 1][k] * weights[i + 1][k][j];
						}
						//dC/da * da/dz = dC/dz
						err *= activationFunction.activationDerivative(z[i][j]);
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
