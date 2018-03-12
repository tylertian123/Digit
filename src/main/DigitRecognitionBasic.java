package main;

import mnist.MNISTLoader;

import java.io.File;

//import mnist.DatabaseExpander;
import mnist.MNISTImage;
import neuralnet.classification.CompositeClassifier;
import neuralnet.core.ClassificationNeuralNetwork;

public class DigitRecognitionBasic {

	public static void main(String[] args) {
		try {
			MNISTImage[] trainingImages = MNISTLoader.loadTrainingImages();
			MNISTImage[] testingImages = MNISTLoader.loadTestingImages();
			MNISTImage[] expanded1 = MNISTLoader.loadImagesFromFile(new File("data\\expanded_training_images_1"), new File("data\\expanded_training_labels_1"));
			MNISTImage[] expanded2 = MNISTLoader.loadImagesFromFile(new File("data\\expanded_training_images_2"), new File("data\\expanded_training_labels_2"));
			
			MNISTImage[] expandedImages = new MNISTImage[trainingImages.length + expanded1.length + expanded2.length];
			System.arraycopy(trainingImages, 0, expandedImages, 0, trainingImages.length);
			System.arraycopy(expanded1, 0, expandedImages, trainingImages.length, expanded1.length);
			System.arraycopy(expanded2, 0, expandedImages, trainingImages.length + expanded1.length, expanded2.length);
			
			ClassificationNeuralNetwork<MNISTImage> net = new ClassificationNeuralNetwork<MNISTImage>(
					new int[] { MNISTImage.PIXEL_COUNT, 50, 10 },
					ClassificationNeuralNetwork.TANH_ACTIVATION,
					ClassificationNeuralNetwork.QUADRATIC_COST);
			net.scheduledSGD(trainingImages, 5, 0.5, 5.0, 0.3, testingImages, 3, 0.5, 4);
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
