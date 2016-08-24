package org.mvlabs.weka.test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Random;
import java.util.ResourceBundle;

import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.core.Instances;

import weka.core.converters.ConverterUtils.DataSource;

public class NBTest1 {

	public static void main(String[] args) throws Exception {
		
		// create a source folder name src/env/java and an .gitIgnore -> /env/ at src 
		//put the correct values for your environment
		
		ResourceBundle bundle = ResourceBundle.getBundle("org/mvlabs/properties/props");
		
		BufferedReader breader = new BufferedReader(new FileReader(bundle.getString("file.email")));
		Instances train = new Instances(breader);
		train.setClassIndex(train.numAttributes() - 1); // index of the attribute used to classify
		breader.close();
		
		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);
		Evaluation eval = new Evaluation(train);
		eval.crossValidateModel(nb, train, 10, new Random(1));
		System.out.println(eval.toSummaryString("\nresults\n=======\n", true));
		System.out.println(eval.fMeasure(1) + " " + eval.precision(1) + " " + eval.recall(1));
		
		DataSource source2 = new DataSource(bundle.getString("file.emailtest"));
        Instances test = source2.getDataSet();
        // setting class attribute if the data format does not provide this information
        // For example, the XRFF format saves the class attribute information as well
        if (test.classIndex() == -1)
            test.setClassIndex(test.numAttributes() - 1);
        
        double label = nb.classifyInstance(test.instance(0));
        System.out.println("label (0):" + label);
        test.instance(0).setClassValue(label);
        System.out.println(train.classAttribute().value((int) label));
        
        label = nb.classifyInstance(test.instance(1));
        System.out.println("label (1):" + label);
        test.instance(1).setClassValue(label);
        System.out.println(train.classAttribute().value((int) label));
        
        label = nb.classifyInstance(test.instance(2));
        System.out.println("label (2):" + label);
        test.instance(2).setClassValue(label);
        System.out.println(train.classAttribute().value((int) label));
        
		
	}

}
