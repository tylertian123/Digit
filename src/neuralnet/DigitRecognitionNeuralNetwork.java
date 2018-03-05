package neuralnet;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Random;

import javax.imageio.ImageIO;

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
		@Override
		public byte getCode() {
			return 0;
		}
	}
	static class QuadraticCost implements CostFunction {
		@Override
		public double costDerivative(double y, double a) {
			return (a - y);
		}
		@Override
		public byte getCode() {
			return 0;
		}
	}
	static class CrossEntropySigmoidCost implements CostFunction {
		@Override
		public double costDerivative(double y, double a) {
			return (1 - y) / (1 - a) - y / a;
		}
		@Override
		public byte getCode() {
			return 1;
		}
	}

	public static final ActivationFunction SIGMOID_ACTIVATION = new SigmoidActivation();
	public static final CostFunction QUADRATIC_COST = new QuadraticCost();
	public static final CostFunction CROSSENTROPY_SIGMOID_COST = new CrossEntropySigmoidCost();
	
	public static final byte SAVE_FORMAT_VER = 0x01;
	protected static final ActivationFunction[] ACTIVATION_LIST = new ActivationFunction[] {
			SIGMOID_ACTIVATION
	};
	protected static final CostFunction[] COST_LIST	= new CostFunction[] {
			QUADRATIC_COST,
			CROSSENTROPY_SIGMOID_COST
	};
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
	protected ActivationFunction activationFunction;
	protected CostFunction costFunction;
	
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
		//No need to initialize the first layer's weights and biases
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				biases[i][j] = r.nextGaussian();
				for(int k = 0; k < neuronCounts[i - 1]; k ++)
					//Initialize each weight to a gaussian with mean 0 and standard deviation 1/sqrt(Nin)
					//Where Nin the number of weights the neuron has
					weights[i][j][k] = r.nextGaussian() / Math.sqrt(neuronCounts[i - 1]);
			}
		}
	}
	//Loads a neural network from file
	//For the specific format see saveDataAs
	public DigitRecognitionNeuralNetwork(File f) throws IOException, NeuralNetworkException {
		DataInputStream in = new DataInputStream(new FileInputStream(f));
		byte version = in.readByte();
		switch(version) {
		case 0x01:
		{
			ArrayList<Integer> countsList = new ArrayList<Integer>();
			int count;
			while((count = in.readInt()) != 0) {
				countsList.add(count);
			}
			this.neuronCounts = new int[countsList.size()];
			for(int i = 0; i < neuronCounts.length; i ++)
				neuronCounts[i] = countsList.get(i);
			this.neuronMax = getMax(neuronCounts);
			this.layers = neuronCounts.length;
			byte activationType = in.readByte();
			byte costType = in.readByte();
			
			boolean found = false;
			for(int i = 0; i < ACTIVATION_LIST.length; i ++) {
				if(ACTIVATION_LIST[i].getCode() == activationType) {
					this.activationFunction = ACTIVATION_LIST[i];
					found = true;
					break;
				}
			}
			if(!found) {
				in.close();
				throw new NeuralNetworkException("Unsupported activation function type");
			}
			found = false;
			for(int i = 0; i < COST_LIST.length; i ++) {
				if(COST_LIST[i].getCode() == costType) {
					this.costFunction = COST_LIST[i];
					found = true;
					break;
				}
			}
			if(!found) {
				in.close();
				throw new NeuralNetworkException("Unsupported cost function type");
			}
			
			weights = createWeightsArray();
			biases = createBiasesArray();
			for(int i = 1; i < layers; i ++) {
				for(int j = 0; j < neuronCounts[i]; j ++) {
					for(int k = 0; k < neuronCounts[i - 1]; k ++) {
						weights[i][j][k] = in.readDouble();
					}
				}
			}
			for(int i = 1; i < layers; i ++) {
				for(int j = 0; j < neuronCounts[i]; j ++) {
					biases[i][j] = in.readDouble();
				}
			}
		}
			break;
		default: in.close(); throw new NeuralNetworkException("Unsupported format");
		}
		in.close();
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
	//Returns how many images were correctly classified
	public int evaluate(MNISTImage[] data) {
		int total = 0;
		for(MNISTImage img : data)
			if(this.classify(img) == img.classification)
				total ++;
		return total;
	}
	
	//Stochastic Gradient Descent
	public void SGD(MNISTImage[] trainingData, int batchSize, double learningRate, int epochs, double regularizationConstant) {
		SGD(trainingData, batchSize, learningRate, epochs, regularizationConstant, null);
	}
	public void SGD(MNISTImage[] trainingData, int batchSize, double learningRate, int epochs, double regularizationConstant, MNISTImage[] evalData) {
		SGD(trainingData, batchSize, learningRate, epochs, regularizationConstant, evalData, false);
	}
	public void SGD(MNISTImage[] trainingData, int batchSize, double learningRate, int epochs, double regularizationConstant, MNISTImage[] evalData, boolean generateGraph) {
		double maxPercentage = 0.0;
		int maxEpoch = -1;
		double[] percentages = null;
		if(evalData != null) {
			percentages = new double[epochs];
			System.out.println("No Training:\nEvaluating...");
			double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
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
				learnFromMiniBatch(miniBatch, learningRate, regularizationConstant, trainingData.length);
			}
			
			if(evalData != null) {
				System.out.println("Evaluating...");
				double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
				System.out.println(percentage + "% correctly classified.");
				if(percentage > maxPercentage) {
					maxPercentage = percentage;
					maxEpoch = epoch;
				}
				percentages[epoch - 1] = percentage;
			}
		}
		if(evalData != null)
			System.out.printf("Max classification rate: %f%%, reached at Epoch #%d", maxPercentage, maxEpoch);
		if(generateGraph) {
			BufferedImage graph = new BufferedImage(percentages.length * 10, 500, BufferedImage.TYPE_INT_RGB);
			Graphics2D g = (Graphics2D) graph.getGraphics();
			g.setPaint(Color.WHITE);
			g.fillRect(0, 0, percentages.length * 10, 500);
			g.setPaint(Color.RED);
			for(int i = 0; i < percentages.length - 1; i ++) {
				g.drawLine(i * 10, (int) (500 - (percentages[i] * 5)), (i + 1) * 10, (int) (500 - (percentages[i + 1] * 5)));
			}
			try {
				ImageIO.write(graph, "png", new File("classification_rate_progression.png"));
			} 
			catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	public void SGDAndSave(MNISTImage[] trainingData, int batchSize, double learningRate, int epochs, double regularizationConstant, MNISTImage[] evalData, File outFile) throws IOException {
		double maxPercentage = 0.0;
		int maxEpoch = -1;
		
		File[] netData = new File[epochs];
		
		System.out.println("No Training:\nEvaluating...");
		double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
		System.out.println(percentage + "% correctly classified.");
		
		for(int epoch = 1; epoch <= epochs; epoch ++) {
			List<MNISTImage> l = Arrays.asList(trainingData);
			Collections.shuffle(l);

			System.out.println("Epoch #" + epoch);
			System.out.println("Learning...");
			
			//Separate the shuffled training samples into mini-batches and train with each mini-batch
			for(int i = 0; i < trainingData.length; i += batchSize) {
				List<MNISTImage> miniBatchList = l.subList(i, Math.min(i + batchSize, l.size()));
				MNISTImage[] miniBatch = new MNISTImage[miniBatchList.size()];
				miniBatchList.toArray(miniBatch);
				learnFromMiniBatch(miniBatch, learningRate, regularizationConstant, trainingData.length);
			}
			
			System.out.println("Evaluating...");
			percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
			System.out.println(percentage + "% correctly classified.");
			if(percentage > maxPercentage) {
				maxPercentage = percentage;
				maxEpoch = epoch;
			}
			File f = File.createTempFile("tmpnet", null);
			saveData(f);
			netData[epoch - 1] = f;
			f.deleteOnExit();
		}
		System.out.printf("Max classification rate: %f%%, reached at Epoch #%d", maxPercentage, maxEpoch);
		if(outFile.exists())
			outFile.delete();
		Files.copy(netData[maxEpoch - 1].toPath(), outFile.toPath());
	}
	/*
	 * SGD Using A Scheduled Learning Rate
	 * trainingData - Training data
	 * batchSize - Size of each mini-batch
	 * initLearningRate - Initial learning rate (eta)
	 * regularizationConstant - L2 regularization constant (lambda)
	 * evalData - Performance evaluation data
	 * schedule - The number of epochs with no performance increase before moving to the next cycle
	 * newRateFactor - The number to multiply the current learning rate by to get the next learning rate
	 * cycles - The max number of cycles to continue for
	 */
	public void SGDScheduledEta(MNISTImage[] trainingData, int batchSize, double initLearningRate, double regularizationConstant, MNISTImage[] evalData, int schedule, double newRateFactor, int cycles) {
		int epoch = 1;
		double eta = initLearningRate;
		int lastMaxEpoch = 1;
		double lastMaxRate = 0.0;
		
		double allTimeBest = 0.0;
		int bestCycle = -1;
		int bestEpoch = -1;
		
		for(int cycle = 1; cycle <= cycles; cycle ++) {
			while(true) {
				List<MNISTImage> l = Arrays.asList(trainingData);
				Collections.shuffle(l);
				System.out.printf("Cycle #%d, Epoch #%d:\nLearning...\n", cycle, epoch);
				
				for(int i = 0; i < trainingData.length; i += batchSize) {
					List<MNISTImage> miniBatchList = l.subList(i, Math.min(i + batchSize, l.size()));
					MNISTImage[] miniBatch = new MNISTImage[miniBatchList.size()];
					miniBatchList.toArray(miniBatch);
					learnFromMiniBatch(miniBatch, eta, regularizationConstant, trainingData.length);
				}
				
				double percentage = ((double) this.evaluate(evalData)) / evalData.length * 100;
				System.out.printf("%f%% correctly classified.", percentage);
				if(percentage > allTimeBest) {
					allTimeBest = percentage;
					bestCycle = cycle;
					bestEpoch = epoch;
				}
				if(percentage > lastMaxRate) {
					lastMaxRate = percentage;
					lastMaxEpoch = epoch;
				}
				else {
					if(epoch - lastMaxEpoch > schedule) {
						lastMaxEpoch = 1;
						lastMaxRate = 0.0;
						epoch = 1;
						break;
					}
				}
				epoch ++;
			}
			
			eta *= newRateFactor;
		}
		System.out.printf("Training finished.\nAll-time best was %f%% at Cycle #%d, Epoch #%d.\n", allTimeBest, bestCycle, bestEpoch);
	}
	
	//Uses gradient descent and backpropagation to learn from a mini-batch
	public void learnFromMiniBatch(MNISTImage[] miniBatch, double learningRate, double regularizationConstant, int dataSize) {
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
					weights[i][j][k] = weights[i][j][k] * (1 - learningRate * regularizationConstant / dataSize)
							- learningRate * weightDerivativesTotal[i][j][k];
				}
			}
		}
	}
	/*
	 * Format for version 0x01:
	 * Version code - 1 byte
	 * Input layer neuron count - 4 bytes
	 * Hidden layer 1 neuron count - 4 bytes
	 * Hidden layer 2 neuron count - 4 bytes
	 * ...
	 * Output layer neuron count - 4 bytes
	 * 0 - 4 bytes
	 * Activation type code - 1 byte
	 * Cost type code - 1 byte
	 * Weight[1][0][0] - 8 bytes
	 * Weight[1][0][1] - 8 bytes
	 * ...
	 * Bias[1][0] - 8 bytes
	 * Bias[1][1] - 8 bytes
	 * ...
	 */
	public void saveData(File f) throws IOException {
		if(!f.exists())
			f.createNewFile();
		DataOutputStream out = new DataOutputStream(new FileOutputStream(f));
		out.writeByte(SAVE_FORMAT_VER);
		
		for(int i = 0; i < layers; i ++)
			out.writeInt(neuronCounts[i]);
		out.writeInt(0);
		
		out.writeByte(activationFunction.getCode());
		out.writeByte(costFunction.getCode());
		
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				for(int k = 0; k < neuronCounts[i - 1]; k ++) {
					out.writeDouble(weights[i][j][k]);
				}
			}
		}
		for(int i = 1; i < layers; i ++) {
			for(int j = 0; j < neuronCounts[i]; j ++) {
				out.writeDouble(biases[i][j]);
			}
		}
		
		out.close();
	}
}
