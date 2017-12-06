package mnist;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class MINSTLoader {
	public static final int TRAIN_IMG_COUNT = 60000;
	public static final int EVAL_IMG_COUNT = 10000;
	
	public static MNISTImage[] loadTrainingImages() throws IOException, FileNotFoundException {
		MNISTImage[] mnistImages = new MNISTImage[TRAIN_IMG_COUNT];
		
		File fileLabels = new File("data\\t10k-labels-idx1-ubyte");
		File fileImages = new File("t10k-images-idx3-ubyte");
		FileInputStream labels = new FileInputStream(fileLabels);
		FileInputStream images = new FileInputStream(fileImages);
		//Discard the first 8 bytes from both files since they contain data we already have
		byte[] buf = new byte[8];
		labels.read(buf);
		images.read(buf);
		
		for(int i = 0; i < TRAIN_IMG_COUNT; i ++) {
			mnistImages[i].classification = (byte) labels.read();
			//"Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black)."
			//(From MNIST website)
			for(int j = 0; j < MNISTImage.PIXEL_COUNT; j ++) {
				mnistImages[i].data[j] = (byte) images.read();
			}
		}
		
		labels.close();
		images.close();
		
		return mnistImages;
	}
}
