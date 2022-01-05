import java.io.IOException;
import java.util.Scanner;

public class Driver {

	public static void main(String[] args) throws IOException {
		
//		Scanner input = new Scanner(System.in);
//		while (true) {
//			Data data = new Data();
//			data.setTrainingImageData("./data/train-images.idx3-ubyte");
//			data.setTrainingLabelData("./data/train-labels.idx1-ubyte");
//			data.setTestImageData("./data/t10k-images.idx3-ubyte");
//			data.setTestLabelData("./data/t10k-labels.idx1-ubyte");
//			
//			String ans = "";
//			do {
//				System.out.println("1. create new network");
//				System.out.println("2. load network");
//				ans = input.nextLine();
//			} while (!( ans.equals("1") || ans.equals("2")));
//			
//			if (ans.equals("1")) {
//				System.out.println("enter network dimensions (layer1, layer2, ..., outputLayer");
//				Network net = new Network
//			}
//			
//			else {
//				
//			}
//		}
		
		Data data = new Data();
		data.setTrainingImageData("./data/train-images.idx3-ubyte");
		data.setTrainingLabelData("./data/train-labels.idx1-ubyte");
		data.setTestImageData("./data/t10k-images.idx3-ubyte");
		data.setTestLabelData("./data/t10k-labels.idx1-ubyte");
		System.out.println("set data");
		
		int[] sizes = {784, 30, 10};
		Network net = new Network(sizes);
		net.gradientDescent(data.trainingData, 30, 10, 3.0, data.testData);
		net.saveNetwork("best network");

		// not as accurate due to saving rounding errors i think
		System.out.println("loading best network");
		Network net2 = new Network("best network");
		System.out.println(String.format("%d / %d", net2.selfEvaluate(data.testData), data.testData.size()));
		System.out.println(String.format("%d / %d", net2.selfEvaluate(data.testData), data.testData.size()));

		int[] results = net2.indetify(data.testData);
		System.out.println("expected: "+results[1]);
		System.out.println("network output: "+results[0]);
		
	}

}
