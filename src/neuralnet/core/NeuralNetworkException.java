package neuralnet.core;

/**
 *	An exception thrown by a neural network to indicate something went wrong.
 */
public class NeuralNetworkException extends Exception {
	/**
	 * 
	 */
	private static final long serialVersionUID = -3333626948737630370L;
	public NeuralNetworkException() {
		super();
	}
	public NeuralNetworkException(String message) {
		super(message);
	}
}
