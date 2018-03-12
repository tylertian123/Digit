package mnist;

import neuralnet.Classifiable;

/*
 * A training/evaluation image
 */
public class MNISTImage implements Classifiable {
	public static final int SIZE = 28;
	public static final int PIXEL_COUNT = SIZE * SIZE;
	//Data is stored in a byte array
	public byte[] data;
	int classification;
	
	public MNISTImage(byte[] dat, int classification) {
		if(dat.length != PIXEL_COUNT)
			throw new IllegalArgumentException("Data array provided is of wrong size");
		data = dat;
		this.classification = classification;
	}
	public MNISTImage(int classification) {
		data = new byte[PIXEL_COUNT];
		this.classification = classification;
	}
	
	//Returns a double[] that has the same length as the number of pixels in an image.
	//Each value (previously 0 ~ 255) is scaled down to between 0 and 1
	public double[] asNeuralNetworkInput() {
		double[] output = new double[PIXEL_COUNT];
		for(int i = 0; i < PIXEL_COUNT; i ++) {
			output[i] = ((double) (((int) data[i]) & 0xFF)) / 255;
			//System.out.println(output[i]);
		}
		return output;
	}
	//Returns a double[10], the expected output of the neural network for this training image
	//All values are 0.0 except one
	//The index of the 1.0 output is the classification
	public double[] generateExpectedOutput() {
		double[] output = new double[10];
		output[classification] = 1.0;
		return output;
	}
	
	//Prints out ASCII art to show the digit. For debugging only.
	public void draw() {
		for(int i = 0; i < PIXEL_COUNT; i ++) {
			if(i % SIZE == 0) System.out.println();
			
			if(((int) data[i] & 0xFF) > 127)
				System.out.print(1);
			else 
				System.out.print(0);
			
		}
	}
	
	public byte get(int x, int y) {
		return data[y * SIZE + x];
	}
	public void set(int x, int y, byte val) {
		data[y * SIZE + x] = val;
	}
	
	@Override
	public Object getClassification() {
		return this.classification;
	}
	@Override
	public Object toClassification(double[] networkOutput) {
		int maxIndex = 0;
		for(int i = 0; i < 10; i ++) {
			if(networkOutput[i] > networkOutput[maxIndex]) {
				maxIndex = i;
			}
		}
		return maxIndex;
	}
}
