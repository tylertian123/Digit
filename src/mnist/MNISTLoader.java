package mnist;

import java.io.File;
import java.io.FileInputStream;
import java.io.BufferedInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

/*
 * Loads the MNIST images 
 * Generates arrays of MNISTImages from files
 */
public class MNISTLoader {
	public static final int TRAINING_IMG_COUNT = 50000;
	public static final int TESTING_IMG_COUNT = 10000;
	public static final int VALIDATION_IMG_COUNT = 10000;
	
	public static MNISTImage[] loadImagesBasic(int imgCount, File imagesFile, File labelsFile, int offset) throws FileNotFoundException, IOException {
		MNISTImage[] mnistImages = new MNISTImage[imgCount];
		BufferedInputStream images = new BufferedInputStream(new FileInputStream(imagesFile));
		BufferedInputStream labels = new BufferedInputStream(new FileInputStream(labelsFile));
		//Discard first bytes
		//According to MNIST website the first 8 bytes of labels and first 16 bytes of images are not actual data
		byte[] buf = new byte[8];
		labels.read(buf);
		images.read(buf);
		images.read(buf);
		//Data buffer
		byte[] data = new byte[MNISTImage.PIXEL_COUNT];
		
		for(int i = 0; i < offset; i ++) {
			images.read(data);
			labels.read();
		}
		for(int i = 0; i < imgCount; i ++) {
			//"Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black)."
			//(From MNIST website)
			images.read(data);
			
			//clone the data because otherwise keeping the same copy would make every loaded image the same
			mnistImages[i] = new MNISTImage(data.clone(), (byte) labels.read());
		}
		
		labels.close();
		images.close();
		return mnistImages;
	}
	
	public static MNISTImage[] loadTrainingImages() throws IOException, FileNotFoundException {
		return loadImagesBasic(TRAINING_IMG_COUNT, new File("data\\training_images"), new File("data\\training_labels"), 0);
	}
	public static MNISTImage[] loadTestingImages() throws IOException, FileNotFoundException {
		return loadImagesBasic(TESTING_IMG_COUNT, new File("data\\testing_images"), new File("data\\testing_labels"), 0);
	}
	public static MNISTImage[] loadValidationImages() throws IOException, FileNotFoundException {
		return loadImagesBasic(VALIDATION_IMG_COUNT, new File("data\\training_images"), new File("data\\training_labels"), TRAINING_IMG_COUNT);
	}
}
