package neuralnet.core;

/**
 * Interface for an activation function of a neural network. <br>
 * Implement this and pass into the constructor of a neural network to use.
 */
public interface ActivationFunction {
	/**
	 * The activation function itself. E.g. The sigmoid function
	 * @param z - The weighted sum of a neuron
	 * @return The activation of a neuron
	 */
	public double activation(double z);
	/**
	 * The derivative of the activation function.
	 * @param z - The weighted sum of a neuron
	 * @return The derivative of this activation function at z
	 */
	public double activationDerivative(double z);
	/**
	 * The code of this activation function.<br>
	 * Each activation function has a unique code for it to be identified.
	 * This value is used when saving and loading neural networks; it is saved in the file.
	 * The ClassificationNeuralNetwork class keeps a list of these functions to look up the correct one when loading.
	 * For custom activation functions not included in the list, loading a network causes the activation function
	 * of that particular network to be set to <em>null</em>, and thus it <b>must</b> be set later <b>manually</b>
	 * through the setActivationFunction() method.
	 * @return The code of this activation function
	 */
	public byte getCode();
}
