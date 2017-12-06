package mnist;

public class MNISTImage {
	public static final int SIZE = 28;
	public static final int PIXEL_COUNT = SIZE * SIZE;
	
	public byte[] data = new byte[PIXEL_COUNT];
	public byte classification;
	
	public MNISTImage(byte[] dat, byte classification) {
		data = dat.clone();
		this.classification = classification;
	}
}
