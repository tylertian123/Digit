package main;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

import mnist.DatabaseExpander;
import mnist.MNISTImage;
import mnist.MNISTLoader;

public class DatabaseExpansion {

	public static void expand12() throws FileNotFoundException, IOException {
		MNISTImage[] original = MNISTLoader.loadTrainingImages();
		MNISTImage[] artificial = DatabaseExpander.expandByTranslation(original, 2);
		int pt1Len = artificial.length / 2;
		MNISTImage[] pt1 = new MNISTImage[pt1Len];
		MNISTImage[] pt2 = new MNISTImage[artificial.length - pt1Len];
		System.arraycopy(artificial, 0, pt1, 0, pt1Len);
		System.arraycopy(artificial, pt1Len, pt2, 0, pt2.length);
		MNISTLoader.saveImages(pt1, new File("data\\expanded_training_images_1"), new File("data\\expanded_training_labels_1"));
		MNISTLoader.saveImages(pt2, new File("data\\expanded_training_images_2"), new File("data\\expanded_training_labels_2"));
	}
	public static void expand3() throws FileNotFoundException, IOException {
		MNISTImage[] original = MNISTLoader.loadTrainingImages();
		MNISTImage[] artificial = DatabaseExpander.expandByTranslation2(original, 2);
		MNISTLoader.saveImages(artificial, new File("data\\expanded_training_images_3"), new File("data\\expanded_training_labels_3"));
	}
	public static void main(String[] args) throws FileNotFoundException, IOException {
		expand12();
		expand3();
	}

}
