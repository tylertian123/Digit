package mnist;

public class MNISTImage {
	public static final int SIZE = 28;
	public static final int PIXEL_COUNT = SIZE * SIZE;
	
	public byte[] data;
	public byte classification;
	
	public MNISTImage(byte[] dat, byte classification) {
		if(dat.length != PIXEL_COUNT)
			throw new IllegalArgumentException("Data array provided is of wrong size");
		data = dat;
		this.classification = classification;
	}
	public MNISTImage() {
		data = new byte[PIXEL_COUNT];
	}
	
	public double[] asNeuralNetworkInput() {
		double[] output = new double[PIXEL_COUNT];
		for(int i = 0; i < PIXEL_COUNT; i ++) {
			output[i] = (double) (((int) data[i]) & 0xFF) / 255;
			//System.out.println(output[i]);
		}
		return output;
	}
	public double[] generateExpectedOutput() {
		double[] output = new double[10];
		output[classification] = 1.0;
		return output;
	}
	
	public void draw() {
		for(int i = 0; i < PIXEL_COUNT; i ++) {
			if(i % SIZE == 0) System.out.println();
			
			if(((int) data[i] & 0xFF) > 127)
				System.out.print(1);
			else 
				System.out.print(0);
			
		}
	}
	
	public double getAvg() {
		int avg = 0;
		for(byte b : data)
			avg += b;
		return avg / (double) PIXEL_COUNT;
	}
}
