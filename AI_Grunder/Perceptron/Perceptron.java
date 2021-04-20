class Perceptron {
    float[] weights = new float[2];

    //Constructor
    Perceptron() {
        //Initialize the weights randomly
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random(-1,1);
        }
    }

    int guess(float[] inputs) {
        float sum = 0;
        for (int i = 0; i < weights.length; i++) {
            sum += inputs[i]*weights[i];
        }
    }
}