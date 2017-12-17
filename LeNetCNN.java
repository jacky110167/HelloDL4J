package maven_dl4jBeginner;

import java.io.IOException;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class LeNetCNN{

	public static void main(String[] args) throws IOException {
		int channels = 1;
		int outputNum = 10;
		int batchSize = 32;
		int epochs = 1;
		int iterations = 1;
		int seed = 123;
		
		DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
		DataSetIterator mnistTest  = new MnistDataSetIterator(batchSize, false, seed);
		
		//Construct the neural network
		MultiLayerConfiguration mlconf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(iterations)
				.regularization(true).l2(0.0005)
				.learningRate(0.01)
				.weightInit(WeightInit.XAVIER)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.updater(Updater.NESTEROVS)
				.list()
				.layer(0, new ConvolutionLayer.Builder(5, 5)
						.nIn(channels)
						.stride(1, 1)
						.nOut(20)
						.activation(Activation.IDENTITY)
						.build()
						)
				.layer(1, new SubsamplingLayer
						.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(2, new ConvolutionLayer.Builder(5, 5)
						.stride(1, 1)
						.nOut(50)
						.activation(Activation.IDENTITY)
						.build()
						)
				.layer(3, new SubsamplingLayer
						.Builder(SubsamplingLayer.PoolingType.MAX)
						.kernelSize(2, 2)
						.stride(2, 2)
						.build())
				.layer(4, new DenseLayer
						.Builder()
						.activation(Activation.RELU)
						.nOut(500)
						.build())
				.layer(5, new OutputLayer
						.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
						.nOut(outputNum)
						.activation(Activation.SOFTMAX)
						.build())
				.setInputType(InputType.convolutionalFlat(28, 28, 1))
				.backprop(true).pretrain(false).build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(mlconf);
		model.init();
		model.setListeners(new ScoreIterationListener(1));
		
		for(int i = 0; i < epochs; i++) {
			model.fit(mnistTrain);
			
			Evaluation eva = new Evaluation(outputNum);
			
			while(mnistTest.hasNext()) {
				DataSet ds = mnistTest.next();
				INDArray label = ds.getLabels();
				INDArray predict = model.output(ds.getFeatureMatrix(), false);
				eva.eval(label, predict);
			}
			
			System.out.println(eva.stats());
		}
	}
}
