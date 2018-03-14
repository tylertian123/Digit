package main;

import mnist.MNISTLoader;

import java.io.File;

import mnist.DatabaseExpander;
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
			MNISTImage[] expanded3 = MNISTLoader.loadImagesFromFile(new File("data\\expanded_training_images_3"), new File("data\\expanded_training_labels_3"));
			
			MNISTImage[] expandedImagesLarge = DatabaseExpander.concatArrays(trainingImages, expanded1, expanded2);
			MNISTImage[] expandedImagesSmall = DatabaseExpander.concatArrays(trainingImages, expanded3);
			
			MNISTImage[] smallDataset = new MNISTImage[1000];
			System.arraycopy(trainingImages, 0, smallDataset, 0, 1000);
			ClassificationNeuralNetwork<MNISTImage> net = new ClassificationNeuralNetwork<MNISTImage>(
					new int[] { MNISTImage.PIXEL_COUNT, 50, 10 },
					ClassificationNeuralNetwork.SIGMOID_ACTIVATION,
					ClassificationNeuralNetwork.CROSSENTROPY_SIGMOID_COST);
			//net.scheduledSGD(trainingImages, 3, 0.20, 0.5, 0.6, testingImages, 3, 0.5, 4);
			//net.dropoutSGD(smallDataset, 2, 0.10, 0.5, 0.4, 30, testingImages);
			
			//net.scheduledDropoutSGD(trainingImages, 1, 0.050, 0.5, 0.6, testingImages, 4, 0.25, 4);
			//net.saveData(new File("dropout.ann"));
			
			net.scheduledSGD(expandedImagesSmall, 1, 0.05, 5.0, 0.6, testingImages, 4, 0.25, 4);
			net.saveData(new File("Momentum_expanded.ann"));
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
