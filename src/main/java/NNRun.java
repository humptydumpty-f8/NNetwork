import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.nnet.Perceptron;
import org.neuroph.nnet.learning.BackPropagation;
import org.neuroph.util.TransferFunctionType;

import java.util.Arrays;

import static org.neuroph.util.TransferFunctionType.SIGMOID;

public class NNRun {
    static final int inputSize = 2;
    static final int outputSize = 1;

    public static void main(String[] args) {
        //todo function
        NeuralNetwork ann = new Perceptron(inputSize, outputSize, TransferFunctionType.STEP);

        DataSet ds = new DataSet(inputSize, outputSize);
        DataSetRow rOne = new DataSetRow(new double[]{0, 1}, new double[]{1});
        ds.addRow(rOne);
        DataSetRow rTwo = new DataSetRow(new double[]{1, 1}, new double[]{0});
        ds.addRow(rTwo);
        DataSetRow rThree = new DataSetRow(new double[]{0, 0}, new double[]{0});
        ds.addRow(rThree);
        DataSetRow rFour = new DataSetRow(new double[]{1, 0}, new double[]{1});
        ds.addRow(rFour);

        BackPropagation backPropagation = new BackPropagation();
        backPropagation.setMaxIterations(1000);
        ann.learn(ds, backPropagation);

        ann.setInput(1,1);
        ann.calculate();
        double[]networkOutputOne = ann.getOutput();
        System.out.println(Arrays.toString(networkOutputOne));
    }
}
