/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import static java.lang.Math.floor;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;
import static jdk.nashorn.internal.parser.TokenType.EOF;

/**
 *
 * @author demorgan
 */
public class NeuralNetwork {

    /**
     * @param args the command line arguments
     */
    static final boolean Debug_Enable = false;

    static int n = 15;     
    static int m = Math.floorDiv(n, 2);
    static double l_rate = 0.7;
    static double momentum = 0.3;
    static int number_of_epochs = 1000;
    static double bias = 0.9;
    static Double []bias_weight=new Double[m];
    public static void main(String[] args) throws FileNotFoundException, IOException {

        BufferedReader in = new BufferedReader(new FileReader("part1-forestfires.txt"));
        in.mark(100000000);
        NumberFormat formatter = new DecimalFormat("#0.00000");
        Random rand = new Random();     //will be used to choose random output
        
        Double Ave_RMSE=0.0;

        Double in_weights[][] = new Double[n][m];   //input layer weights
        Double h_weights[] = new Double[m];         //hidden layer weigths
        Double[] inputs = new Double[n];            //input values
        Double h_outputs[] = new Double[m];         //hidden layer outputs
        for (int i = 0; i < m; i++) {
            h_outputs[i] = 0.0;
        }

        Double chosen_output = -1.0;                //tobe used later
        Double real_output = -1.0;

        Double h_chosen_outputs[] = new Double[m];      //chosen outputs hidden layer outputs
        Double i_chosen_outputs[] = new Double[m];      //chosen outputs inputs

        String raw;
        String[] split;

        Double[] Min = new Double[10];                  //minimum values
        Double[] Max = new Double[10];                  //maximum values

        init_MinMax(Min, Max);                          //initilize
        if (Debug_Enable) {
            PrintMinMax(Min, Max);
        }

        int rastgele;
        int trstart1 = 0, trstart2 = 0;

        Double[][] best_input_weights;
        Double[] best_hidden_weights;

        Double smallest_error[] = new Double[5];
        Double RMSE = 0.0;
        int ss = 0;
        for (int f = 0; f < 5; f++) {

            for (int aa = 0; aa < 5; aa++) {
                smallest_error[aa] = 1.0;
            }
            in_weights = initilize_i_weights(in_weights);   //initilize the inputlayer weights
            h_weights = initilize_h_weights(h_weights);     //initilize hidden layer weights
            best_hidden_weights = h_weights.clone();
            best_input_weights = in_weights.clone();
            ss = 0;
            switch (f) {
                case 0:
                    trstart1 = 352;
                    trstart2 = 440;
                    break;
                case 1:
                    trstart1 = 0;
                    trstart2 = 88;
                    break;
                case 2:
                    trstart1 = 88;
                    trstart2 = 176;
                    break;
                case 3:
                    trstart1 = 176;
                    trstart2 = 264;
                    break;
                case 4:
                    trstart1 = 264;
                    trstart2 = 352;
                    break;
            }
            int tt;
            for (int a = 0; a < number_of_epochs; a++) {                  //number of epochs
                in.reset();
                tt = 0;
                int i = 0;
                rastgele = (rand.nextInt() % 440 + 440) % 440;     // to evaluate negative values
                while(rastgele >= trstart1 && rastgele < trstart2 )
                    rastgele = (rand.nextInt() % 440 + 440) % 440;     // to evaluate negative values
                while (i < 440) {                                 //For each line of input. 352 lines total
                    raw = in.readLine();
                    if (!(i >= trstart1 && i < trstart2)) { //trainset
                        tt++;
                        if (Debug_Enable) {
                            System.out.println("\n" + i + ". line started:");
                        }

                        split = raw.split(",");
                        for (int j = 1; j < 9; j++) {
                            inputs[j - 1] = Double.valueOf(split[j]); //First 8 input node filled with numerical inputs.
                            inputs[j - 1] = MinMax_Value(inputs[j - 1], j - 1, Min, Max); //MinMax Normalization
                        }
                        if (Debug_Enable) {
                            System.out.println("First 8 input node filled with numerical inputs.");
                        }
                        Double[] encoded = Encode(split[0]);

                        for (int j = 0; j < 7; j++) {
                            inputs[j + 8] = encoded[j];       //Encoded categorical data is filled to last 7 nodes
                        }
                        Double output = go_forward(inputs, in_weights, h_weights, h_outputs);

                        if (Debug_Enable) {
                            System.out.println("Encoded categorical data is filled to last 7 nodes");
                            System.out.println("\nFull input data is: ");
                            for (int j = 0; j < n; j++) {
                                System.out.print(inputs[j] + " ");
                            }
                            System.out.println("\nStarting the forward propagation");
                            System.out.println("The " + i + "th output is : " + output);
                        }

                        if (i == rastgele) {
                            chosen_output = output;        //choosing one random output.
                            real_output = MinMax_Value(Double.valueOf(split[8]), 8, Min, Max);
                            i_chosen_outputs = inputs;
                            h_chosen_outputs = h_outputs;
                        }
                    }
                    i++;
                }       //end of 1 epoch/train

                Double out_error = output_error(chosen_output, real_output);
                if (Debug_Enable) {
                    System.out.println("NUmber of train" + tt);
                    System.out.println("\n\t\t\t\tEPOCH NUMBER: " + a);
                    System.out.println("real output is " + rastgele + "th:" + real_output);
                    System.out.println("chosen output is: " + chosen_output);
                    System.out.println("output error rate is: " + out_error);
                }
                
                if (out_error < smallest_error[f]) {
                    smallest_error[f] = out_error;
                    best_hidden_weights = h_weights;
                    best_input_weights = in_weights;
                }
                
                //Weight updates with out_error
                Update_Weights(out_error, in_weights, h_weights, i_chosen_outputs, h_chosen_outputs);
                
                
                if (Debug_Enable) {
                    printweights(in_weights, h_weights);
                    System.out.println("Update!");
                    printweights(in_weights, h_weights);//updated
                }
            }
            //end of 1 train epochs

            System.out.println("Smallest error of fold " + f + "  is :" + smallest_error[f]);
            in.reset();
            if(Debug_Enable)System.out.println("starting the TEST");
            int x = 0;
            while (x < 440) {                                 //For each line of input. 440 lines total
                raw = in.readLine();
                if (x >= trstart1 && x < trstart2) { //test set
                    if (Debug_Enable) {
                        System.out.println("\n" + x + ". line started:");
                    }

                    split = raw.split(",");
                    for (int j = 1; j < 9; j++) {
                        inputs[j - 1] = Double.valueOf(split[j]); //First 8 input node filled with numerical inputs.
                        inputs[j - 1] = MinMax_Value(inputs[j-1], j - 1, Min, Max); //MinMax Normalization
                    }
                    if (Debug_Enable) {
                        System.out.println("First 8 input node filled with numerical inputs.");
                    }

                    Double[] encoded = Encode(split[0]);
                    Double output = go_forward(inputs, best_input_weights, best_hidden_weights, h_outputs);
                    
                    Double actual = MinMax_Value(Double.valueOf(split[9]), 9, Min, Max);
                    
                    RMSE += Math.pow(actual - output, 2);   //once minmax RMSE 
                    ss++;
                    
                    for (int j = 0; j < 7; j++) {
                        inputs[j + 8] = encoded[j];       //Encoded categorical data is filled to last 7 nodes
                    }

                    if (Debug_Enable) {
                        System.out.println("Encoded categorical data is filled to last 7 nodes");
                        System.out.println("\nFull input data is: ");
                        for (int j = 0; j < n; j++) {
                            System.out.print(inputs[j] + " ");
                        }
                        System.out.println("\nStarting the forward propagation for TEST");
                    
                        System.out.println("The " + x + "th output is : " + output);
                    }
                }
                x++;
            } //end of 1 fold
            if(Debug_Enable) System.out.println("number of RMSE addition: " + ss);
            ss = 0;
            RMSE /= 88;
            //RMSE = Math.sqrt(RMSE);
            NumberFormat formatt = new DecimalFormat("#0.0000");
            RMSE = Double.valueOf(formatt.format(RMSE));
            System.out.println("ROOT MEAN SQUARED ERROR OF FOLD " + f + " is: " + RMSE);
            Ave_RMSE+=RMSE;
            RMSE = 0.0;
            System.out.println("");
        } //end of all folds
        
        System.out.println("Average RMSE is: " + Ave_RMSE/5);
        in.close();
    }

    static void Update_Weights(Double out_error, Double[][] in_weights, Double[] h_weights,
            Double[] inputs, Double h_chosen_outputs[]) {

        Double h_errors[] = new Double[h_weights.length];

        for (int i = 0; i < h_weights.length; i++) {
            h_weights[i] += l_rate * momentum * out_error * h_chosen_outputs[i];
            h_errors[i] = h_chosen_outputs[i] * (1.0 - h_chosen_outputs[i]) * out_error * h_weights[i];
            //System.out.println("error of hidden node:"+i+"is:" + h_errors[i]);
            bias_weight[i]+=l_rate * momentum * out_error * h_chosen_outputs[i];
        }
        if (Debug_Enable) System.out.println("");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                in_weights[j][i] += l_rate * momentum * h_errors[i] * inputs[j];
                if (Debug_Enable) {
                    System.out.print("Einput" + j + "-" + i + ":" + in_weights[j][i] + "\t");
                }
            }
            if (Debug_Enable) {
                System.out.println("");
            }
        }

    }

    static void printweights(Double[][] in_weights, Double[] h_weights) {

        NumberFormat formatter = new DecimalFormat("#0.00000");

        System.out.println("\nInput layer weights");
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                in_weights[j][i] = Double.valueOf(formatter.format(in_weights[j][i]));
                System.out.print(" \t" + j + "-" + i + ":" + in_weights[j][i]);
            }
            System.out.println();
        }
        System.out.println("Hidden layer weights");
        for (int i = 0; i < m; i++) {
            h_weights[i] = Double.valueOf(formatter.format(h_weights[i]));
            System.out.print(" \t" + i + ":" + h_weights[i]);
        }
        System.out.println("");
    }

    static Double output_error(Double chosen_output, Double real_output) {
        NumberFormat formatter = new DecimalFormat("#0.00000");
        Double error = chosen_output * (1 - chosen_output) * (real_output - chosen_output);
        error = Double.valueOf(formatter.format(error));
        return error;
    }

    static Double[] initilize_h_weights(Double[] weigts) {
        Random a = new Random();
        NumberFormat formatter = new DecimalFormat("#0.0000");
        for (int i = 0; i < m; i++) {
            weigts[i] = Double.valueOf(formatter.format(a.nextDouble() - 0.5));
            bias_weight[i] = Double.valueOf(formatter.format(a.nextDouble() - 0.5));
            if(Debug_Enable)System.out.print("H_W " + i + ":" + weigts[i].toString() + "\t");
        }
        return weigts;
    }

    static Double[][] initilize_i_weights(Double[][] weigts) {
        Random a = new Random();
        NumberFormat formatter = new DecimalFormat("#0.0000");
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                weigts[i][j] = Double.valueOf(formatter.format(a.nextDouble() - 0.5));
                if(Debug_Enable)System.out.print(" W " + i + "-" + j + ":" + weigts[i][j] + "\t");
            }
            if(Debug_Enable) System.out.println("");
        }

        return weigts;
    }

    static Double[] Encode(String input) {
        Double[] output = new Double[7];

        for (int i = 0; i < 7; i++) {
            output[i] = 0.0;
        }

        switch (input) {
            case "mon":
                output[0] = 1.0;
                break;
            case "tue":
                output[1] = 1.0;
                break;
            case "wed":
                output[2] = 1.0;
                break;
            case "thu":
                output[3] = 1.0;
                break;
            case "fri":
                output[4] = 1.0;
                break;
            case "sat":
                output[5] = 1.0;
                break;
            case "sun":
                output[6] = 1.0;
                break;
            default:
                System.out.println("Error on encoding.");
                System.exit(0);
        }
        return output;
    }

    static Double go_forward(Double real_inputs[], Double in_weights[][], Double h_weights[], Double h_outputs[]) {

        //Initialize: input data, initial weights. 
        Double h_inputs[] = new Double[m];
        for (int i = 0; i < m; i++) {
            h_inputs[i] = 0.0;
            h_outputs[i] = 0.0;
        }

        Double o_in = 0.0;
        Double o_out;

        for (int j = 0; j < m; j++) {       //inputs * weights
            for (int i = 0; i < n; i++) {
                h_inputs[j] += in_weights[i][j] * real_inputs[i] ;
            }
            h_inputs[j] += bias*bias_weight[j];
            h_outputs[j] = sigmoid(h_inputs[j]);
            o_in += h_outputs[j] * h_weights[j];
        }

        o_out = sigmoid(o_in);

        NumberFormat formatter = new DecimalFormat("#0.00000");

        o_out = Double.valueOf(formatter.format(o_out));

        return o_out;
    }

    static double sigmoid(double x) {

        double result = 1.0 / (1.0 + Math.pow(Math.E, -x));
        // NumberFormat formatter = new DecimalFormat("#0.00000");     //yavaşlatmaya gerek yok şimdilik
        // result = Double.valueOf(formatter.format(result));
        return result;
    }

    static void init_MinMax(Double[] min, Double[] max) throws FileNotFoundException, IOException {

        BufferedReader in = new BufferedReader(new FileReader("part1-forestfires.txt"));

        String raw = in.readLine();                     //First line is categories and stuff...

        String[] split;
        raw = in.readLine();
        split = raw.split(",");

        for (int j = 1; j < 10; j++) {           //initilize the data with first input data
            min[j - 1] = Double.valueOf(split[j]);
            max[j - 1] = Double.valueOf(split[j]);
        }
        min[9] = Double.valueOf(split[9]);
        max[9] = Double.valueOf(split[9]);

        while ((raw = in.readLine()) != null) {     //compare the rest of the lines and update min max

            split = raw.split(",");
            //WTF is going on
            for (int j = 1; j < 10; j++) {
                Double value = Double.valueOf(split[j]);
                if (value < min[j - 1]) {
                    min[j - 1] = value;
                }
                if (value > max[j - 1]) {
                    max[j - 1] = value;
                }
            }
            Double asd = Double.valueOf(split[9]);
            if (asd < min[9]) {
                min[9] = asd;
            }
            if (asd > max[9]) {
                max[9] = asd;
            }

        }

        in.close();
        // 0   1     2       3    4     5     6   7    8   9 (out)
        //fri, 92.4, 117.9, 668, 12.2, 19.6, 33, 5.4,  0,  0
    }

    static double MinMax_Value(Double value, int index, Double[] min, Double[] max) {

        value = (value - min[index]) / (max[index] - min[index]);

        NumberFormat formatter = new DecimalFormat("#0.00000");

        value = Double.valueOf(formatter.format(value));

        return value;
    }

    static void PrintMinMax(Double[] Min, Double[] Max) {
        System.out.println("\nMinimum Values:");
        for (int i = 0; i < 9; i++) {
            System.out.print(Min[i] + " ");
        }

        System.out.println("\nMaximum Values");
        for (int i = 0; i < 8; i++) {
            System.out.print(Max[i] + " ");
        }
        System.out.println("");
    }
}
