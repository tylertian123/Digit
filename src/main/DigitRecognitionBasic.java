package main;

import mnist.MNISTLoader;

import java.io.File;

import mnist.DatabaseExpander;
import mnist.MNISTImage;
import neuralnet.ClassificationNeuralNetwork;
import neuralnet.CompositeClassifier;

public class DigitRecognitionBasic {

	public static void main(String[] args) {
		try {
			MNISTImage[] trainingData = MNISTLoader.loadTrainingImages();
			//MNISTImage[] expandedTrainingData = MNISTLoader.loadImagesFromFile(new File("data\\expanded_training_images"), new File("data\\expanded_training_labels"));
			MNISTImage[] evalData = MNISTLoader.loadTestingImages();
			MNISTImage[] validationData = MNISTLoader.loadValidationImages();
			
			/*MNISTImage[] expandedData = new MNISTImage[trainingData.length + expandedTrainingData.length];
			System.arraycopy(trainingData, 0, expandedData, 0, trainingData.length);
			System.arraycopy(expandedTrainingData, 0, expandedData, trainingData.length, expandedTrainingData.length);
			
			ClassificationNeuralNetwork<MNISTImage> net = 
					new ClassificationNeuralNetwork<MNISTImage>(new int[] {MNISTImage.PIXEL_COUNT, 100, 10}, 
							ClassificationNeuralNetwork.SIGMOID_ACTIVATION,
							ClassificationNeuralNetwork.CROSSENTROPY_SIGMOID_COST);
			net.SGDScheduledEta(expandedData, 1, 0.06, 5.0, evalData, 4, 0.5, 5);
			net.saveData(new File("expandedresult.ann"));*/
			MNISTImage[] expandedTrainingData = DatabaseExpander.expandByTranslation(trainingData, 2);
			MNISTImage[] pt1 = new MNISTImage[expandedTrainingData.length / 2];
			MNISTImage[] pt2 = new MNISTImage[expandedTrainingData.length - pt1.length];
			System.arraycopy(expandedTrainingData, 0, pt1, 0, pt1.length);
			System.arraycopy(expandedTrainingData, expandedTrainingData.length / 2, pt2, 0, pt2.length);
			MNISTLoader.saveImages(pt1, new File("data\\expanded_training_images_1"), new File("data\\expanded_training_labels_1"));
			MNISTLoader.saveImages(pt2, new File("data\\expanded_training_images_2"), new File("data\\expanded_training_labels_2"));
			
		} 
		catch (Exception e) {
			e.printStackTrace();
		} 
		
	}

}
