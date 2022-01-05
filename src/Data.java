import java.util.ArrayList;

import org.ejml.simple.SimpleMatrix;

import java.io.*;
import java.nio.file.Files;

public class Data {

    public ArrayList<ArrayList<SimpleMatrix>> trainingData = new ArrayList<ArrayList<SimpleMatrix>>(60000);
    public ArrayList<ArrayList<SimpleMatrix>> testData = new ArrayList<ArrayList<SimpleMatrix>>(10000);

    public void setTrainingImageData(String filename) throws IOException {
    	File file = new File(filename);
    	byte[] fileContent = Files.readAllBytes(file.toPath());

    	double[] imageInputsList = new double[784];
    	int pixel = 0;
    	// image data starts at byte 16
    	for (int i = 16; i < fileContent.length; i++) {
    		// since java is cool and doesn't support unsigned bytes, convert to int first
    		int value = Byte.toUnsignedInt(fileContent[i]);
    		// convert to value between 0 - 1
    		double pixelValue = value / 256.0;
    		imageInputsList[pixel] = pixelValue;
    		pixel++;
    		// each image is 28 x 28 = 784 pixels
    		if (pixel >= 784) {
    			SimpleMatrix imageInputs = new SimpleMatrix(784, 1, true, imageInputsList);
    			ArrayList<SimpleMatrix> tuple = new ArrayList<SimpleMatrix>();
    			tuple.add(imageInputs);
    			trainingData.add(tuple);
    			imageInputsList = new double[784];
    			pixel = 0;
    		}

    	}
    }

    public void setTestImageData(String filename) throws IOException {
    	File file = new File(filename);
    	byte[] fileContent = Files.readAllBytes(file.toPath());

    	double[] imageInputsList = new double[784];
    	int pixel = 0;
    	// image data starts at byte 16
    	for (int i = 16; i < fileContent.length; i++) {
    		// since java is cool and doesn't support unsigned bytes, convert to int first
    		int value = Byte.toUnsignedInt(fileContent[i]);
    		// convert to value between 0 - 1
    		double pixelValue = value / 256.0;
    		imageInputsList[pixel] = pixelValue;
    		pixel++;
    		// each image is 28 x 28 = 784 pixels
    		if (pixel >= 784) {
    			SimpleMatrix imageInputs = new SimpleMatrix(784, 1, true, imageInputsList);
    			ArrayList<SimpleMatrix> tuple = new ArrayList<SimpleMatrix>();
    			tuple.add(imageInputs);
    			testData.add(tuple);
    			imageInputsList = new double[784];
    			pixel = 0;
    		}

    	}
    }

    public void setTrainingLabelData(String filename) throws IOException {
         File file = new File(filename);
         byte[] fileContent = Files.readAllBytes(file.toPath());

         // label data starts at byte 8 of the file
         for (int i = 8; i < fileContent.length; i++) {
              SimpleMatrix outputs = new SimpleMatrix(10, 1);
              int index = fileContent[i];
              outputs.set(index, 1.0);
              trainingData.get(i - 8).add(outputs);
         }

    }

    public void setTestLabelData(String filename) throws IOException {
    	 File file = new File(filename);
         byte[] fileContent = Files.readAllBytes(file.toPath());

         // label data starts at byte 8 of the file
         for (int i = 8; i < fileContent.length; i++) {
              SimpleMatrix outputs = new SimpleMatrix(10, 1);
              int index = fileContent[i];
              outputs.set(index, 1.0);
              testData.get(i - 8).add(outputs);
         }
    }

}
