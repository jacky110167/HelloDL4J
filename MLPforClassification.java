package maven_dl4jBeginner;

import java.io.File;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

public class MLPforClassification {

	public static void main(String[] args) throws Exception {
		int seed = 123;
		double learningRate = 0.01;
		int batchSize = 50;
		int nEpochs = 30;
		int numInputs = 2;
		int numOutputs = 2;
		int numHiddenNodes = 20;
		
		//Load the training data
		RecordReader rr = new CSVRecordReader();
		File csvfilepath = new File("src/main/resources/linear_data_train.csv");
		rr.initialize(new FileSplit(csvfilepath));
		DataSetIterator trainIter = new RecordReaderDataSetIterator(rr, batchSize, 0, 2);
		
		//Load the evaluating data
		File testfilepath = new File("src/main/resources/linear_data_train.csv");
		RecordReader testrr = new CSVRecordReader();
		testrr.initialize(new FileSplit(testfilepath));
		DataSetIterator testIter = new RecordReaderDataSetIterator(testrr, batchSize, 0, 2);
			
	 	MultiLayerConfiguration MLConf = new NeuralNetConfiguration.Builder()
				.seed(seed)
				.iterations(1)
				.optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
				.learningRate(learningRate)
				.updater(Updater.NESTEROVS)
				.list()
				.layer(0, new DenseLayer.Builder()
						.nIn(numInputs)
						.nOut(numHiddenNodes)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.RELU)
						.build()
						)
				.layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
						.weightInit(WeightInit.XAVIER)
						.activation(Activation.SOFTMAX)
						.nIn(numHiddenNodes)
						.nOut(numOutputs)
						.build()
						)
				.pretrain(false).backprop(true).build();
		
		MultiLayerNetwork model = new MultiLayerNetwork(MLConf);
		model.init();
		model.setListeners(new ScoreIterationListener(10));
		
		for(int n = 0; n < nEpochs; n++) {
			model.fit(trainIter);
		}
		
		System.out.println("Establishing the model......");
		
		Evaluation eva = new Evaluation(numOutputs);
		
		while(testIter.hasNext()) {
			DataSet ds = testIter.next();
			INDArray features = ds.getFeatureMatrix();
			INDArray label = ds.getLabels();
			INDArray predict = model.output(features, false);
			
			eva.eval(label, predict);
		}
		
		System.out.println(eva.stats());
		
		/***********************Plotting the data***********************/
		
		double xMin = -15;
		double xMax = 15;
		double yMin = -15;
		double yMax = 15;
		
		int PointsPerAxis = 100;
		double[][] evaPoints = new double[PointsPerAxis*PointsPerAxis][2];
		int count = 0;
		
		for(int i = 0; i < PointsPerAxis; i++) {
			for(int j = 0; j < PointsPerAxis; j++) {
				double x = i * (xMax - xMin)/ (PointsPerAxis-1) + xMin;
				double y = i * (yMax - yMin)/ (PointsPerAxis-1) + yMin;
				
				evaPoints[count][0] = x;
				evaPoints[count][1] = y;
				
				count++;
			}
		}
		
		INDArray allXYPoints = Nd4j.create(evaPoints);
		INDArray predictionsAtXYPoints = model.output(allXYPoints);
		
		rr.initialize(new FileSplit(csvfilepath));
		rr.reset();
		int trainPoints = 500;
		trainIter = new RecordReaderDataSetIterator(rr, trainPoints, 0, 2);
		DataSet ds = trainIter.next();
		PlotUtil.plotTrainingData(ds.getFeatures(), 
								  ds.getLabels(), 
								  allXYPoints, 
								  predictionsAtXYPoints,
								  PointsPerAxis
								  );
		
		testrr.initialize(new FileSplit(testfilepath));
		testrr.reset();
		int testPoints = 100;
		testIter = new RecordReaderDataSetIterator(testrr, testPoints, 0, 2);
		ds = testIter.next();
		INDArray testPredict = model.output(ds.getFeatures());
		PlotUtil.plotTestData(ds.getFeatures(), 
								 ds.getLabels(),
								 testPredict,
								 allXYPoints, 
								 predictionsAtXYPoints,
								 PointsPerAxis
								 );
	}
}
