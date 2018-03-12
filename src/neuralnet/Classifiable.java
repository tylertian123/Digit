package neuralnet;

/*
 * Interface representing "classifiable" things, for use with the ClassificationNeuralNetwork
 */
public interface Classifiable {
	//Returns the "classification" of this object
	//The classification can be any type
	public Object getClassification();
	//Returns this object, in the form of a input to a neural network
	public double[] asNeuralNetworkInput();
	//Returns the expected output of a neural network with this object as input
	public double[] generateExpectedOutput();
	//Maps the output of a neural network to a "classification"
	//The "classification" must be the same type as returned by getClassification()
	public Object toClassification(double[] networkOutput);
}
