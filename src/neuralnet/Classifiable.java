package neuralnet;

public interface Classifiable {
	public int getClassification();
	public double[] asNeuralNetworkInput();
	public double[] generateExpectedOutput();
}
