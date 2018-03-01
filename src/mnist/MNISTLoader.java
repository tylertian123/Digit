package mnist;

import java.io.File;
import java.io.FileInputStream;
import java.io.BufferedInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;

public class MNISTLoader {
	public static final int TRAINING_IMG_COUNT = 50000;
	public static final int TESTING_IMG_COUNT = 10000;
	public static final int VALIDATION_IMG_COUNT = 10000;
	
	public static MNISTImage[] loadImagesBasic(int imgCount, File imagesFile, File labelsFile) throws FileNotFoundException, IOException {
		MNISTImage[] mnistImages = new MNISTImage[imgCount];
		BufferedInputStream images = new BufferedInputStream(new FileInputStream(imagesFile));
		BufferedInputStream labels = new BufferedInputStream(new FileInputStream(labelsFile));
		//Discard first bytes
		byte[] buf = new byte[8];
		labels.read(buf);
		images.read(buf);
		images.read(buf);
		byte[] data = new byte[MNISTImage.PIXEL_COUNT];
		
		for(int i = 0; i < imgCount; i ++) {
			//"Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black)."
			//(From MNIST website)
			images.read(data);
			mnistImages[i] = new MNISTImage(data.clone(), (byte) labels.read());
		}
		
		labels.close();
		images.close();
		return mnistImages;
	}
	
	public static MNISTImage[] loadTrainingImages() throws IOException, FileNotFoundException {
		return loadImagesBasic(TRAINING_IMG_COUNT, new File("data\\training_images"), new File("data\\training_labels"));
	}
	public static MNISTImage[] loadTestingImages() throws IOException, FileNotFoundException {
		return loadImagesBasic(TESTING_IMG_COUNT, new File("data\\testing_images"), new File("data\\testing_labels"));
	}
}
