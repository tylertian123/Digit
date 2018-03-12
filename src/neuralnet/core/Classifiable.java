package neuralnet.core;

/**
 * Interface representing "classifiable" things, for use with the ClassificationNeuralNetwork.
 */
public interface Classifiable {
	/**
	 * The "classification" of this classifiable object.
	 * @return The "classification" of this object
	 */
	public Object getClassification();
	/**
	 * The neural network input form of this classifiable object.
	 * @return A double[], the input to a neural network this object represents
	 */
	public double[] asNeuralNetworkInput();
	/**
	 * The expected output of a neural network if this object was given as input.
	 * @return A double[], the expected output of a neural network if this object was given as input.
	 */
	public double[] generateExpectedOutput();
	/**
	 * Maps the output from a neural network to a "classification".
	 * For example, an activation of [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] could represent a "2".
	 * The "classification" can be any type, as long as it matches with getClassification().
	 * @param networkOutput - A double[], the output of a neural network
	 * @return The "classification" that output represents
	 */
	public Object toClassification(double[] networkOutput);
}
