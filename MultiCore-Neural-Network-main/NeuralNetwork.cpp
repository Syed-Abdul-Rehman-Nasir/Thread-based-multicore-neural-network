#include <iostream>
#include <windows.h>
#include <ctime>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include <pthread.h>
#include <fcntl.h>
#include <queue>
#include <algorithm>
#include <process.h>

using namespace std;


// Process structure for layer management
struct Process {
    DWORD pid;
    int burst_time;
    int arrival_time;
    int priority;
    int remaining_time;
    HANDLE process_handle;
    
    Process(int b, int a, int pr) : 
        burst_time(b), arrival_time(a), priority(pr), remaining_time(b) {
        process_handle = NULL;
        pid = 0;
    }
};

// Scheduler types
enum SchedulerType {
    FCFS,
    SJN,
    RR
};

// Windows-specific sleep function (in milliseconds)
void win_sleep(int milliseconds) {
    Sleep(milliseconds);
}

// Process Manager for Layer scheduling
class ProcessManager {
private:
    vector<Process> processes;
    SchedulerType scheduler_type;
    int time_quantum;

public:
    ProcessManager(SchedulerType type = FCFS, int quantum = 2) 
        : scheduler_type(type), time_quantum(quantum) {}

    void addProcess(const Process& p) {
        processes.push_back(p);
    }

    void schedule() {
        switch(scheduler_type) {
            case FCFS:
                scheduleFCFS();
                break;
            case SJN:
                scheduleSJN();
                break;
            case RR:
                scheduleRR();
                break;
        }
    }

private:
    void scheduleFCFS() {
        sort(processes.begin(), processes.end(), 
             [](const Process& a, const Process& b) {
                 return a.arrival_time < b.arrival_time;
             });
        
        for (Process& p : processes) {
            win_sleep(p.burst_time * 100);
        }
    }

    void scheduleSJN() {
        sort(processes.begin(), processes.end(), 
             [](const Process& a, const Process& b) {
                 return a.burst_time < b.burst_time;
             });
        
        for (Process& p : processes) {
            win_sleep(p.burst_time * 100);
        }
    }

    void scheduleRR() {
        queue<Process> process_queue;
        for (Process& p : processes) {
            process_queue.push(p);
        }

        while (!process_queue.empty()) {
            Process current = process_queue.front();
            process_queue.pop();

            int execution_time = min(time_quantum, current.remaining_time);
            win_sleep(execution_time * 100);
            current.remaining_time -= execution_time;

            if (current.remaining_time > 0) {
                process_queue.push(current);
            }
        }
    }
};

// Matrix multiplication function
double** Mul(double** x, int r1, int c1, double** y, int r2, int c2) {
    double** result = new double*[r1];
    for (int i = 0; i < r1; ++i) {
        result[i] = new double[c2];
        for (int j = 0; j < c2; ++j) {
            result[i][j] = 0.0;
        }
    }

    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            for (int k = 0; k < c1; k++) {
                result[i][j] += x[i][k] * y[k][j];
            }
        }
    }
    return result;
}

// Activation functions and their derivatives
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x) {
    double sig = sigmoid(x);
    return sig * (1.0 - sig);
}

// Calculate f(x)
double* calculateFx(double x) {
    double* result = new double[2];
    result[0] = (x * x + x + 1) / 2.0;
    result[1] = (x * x - x) / 2.0;
    return result;
}

// Enhanced Neuron structure with backpropagation capabilities
struct Neuron {
    int nWeights;
    double** weights;
    double* bias;
    double output;
    double delta;
    vector<double> inputs;

    Neuron() : bias(new double(0.0)), output(0.0), delta(0.0) {}

    double* activationFtn(double** input) {
        inputs.clear();
        for (int i = 0; i < nWeights; i++) {
            inputs.push_back(input[0][i]);
        }
        
        double** x = Mul(input, 1, nWeights, weights, nWeights, 1);
        x[0][0] += *bias;
        output = sigmoid(x[0][0]);
        double* result = new double(output);
        
        for (int i = 0; i < 1; ++i) {
            delete[] x[i];
        }
        delete[] x;
        return result;
    }

    void updateWeights(double learning_rate) {
        for (int i = 0; i < nWeights; i++) {
            weights[i][0] += learning_rate * delta * inputs[i];
        }
        *bias += learning_rate * delta;
    }
};

// Thread arguments for processing
struct threadArgs {
    double** data;
    Neuron* neu;
    pthread_mutex_t* mutex;
    double* outputArr;
    int index;
};

// Thread function
void* funct(void* args) {
    threadArgs* arg = (threadArgs*)args;
    pthread_mutex_lock(arg->mutex);
    double* output = arg->neu->activationFtn(arg->data);
    pthread_mutex_unlock(arg->mutex);
    arg->outputArr[arg->index] = *output;
    delete output;
    return NULL;  // Add this return statement
}

// Enhanced Layer structure with process-based implementation
struct Layer {
    double** onData;
    Neuron* neuron;
    int nNodes = 0;
    pthread_mutex_t nodeMutex;
    ProcessManager process_manager;
    vector<double*> outputs;
    vector<double> deltas;

    Layer(SchedulerType scheduler_type = FCFS) : 
        process_manager(scheduler_type) {
        pthread_mutex_init(&nodeMutex, NULL);
    }

    double** forward() {
        double** outputArr = new double*[1];
        outputArr[0] = new double[nNodes];
        outputs.clear();

        for (int i = 0; i < nNodes; i++) {
            Process p(1, 0, 1);  // burst_time, arrival_time, priority
            process_manager.addProcess(p);
            
            pthread_mutex_lock(&nodeMutex);
            double* output = neuron[i].activationFtn(onData);
            pthread_mutex_unlock(&nodeMutex);
            
            outputArr[0][i] = *output;
            outputs.push_back(output);
        }

        process_manager.schedule();
        return outputArr;
    }

    void backward(const vector<double>& next_layer_deltas, const vector<vector<double>>& next_layer_weights, double learning_rate) {
        deltas.resize(nNodes);
        
        for (int i = 0; i < nNodes; i++) {
            double error = 0.0;
            if (next_layer_deltas.empty()) {  // Output layer
                error = next_layer_weights[i][0] - *outputs[i];
            } else {  // Hidden layer
                for (size_t j = 0; j < next_layer_deltas.size(); j++) {
                    error += next_layer_deltas[j] * next_layer_weights[j][i];
                }
            }
            deltas[i] = error * sigmoid_derivative(*outputs[i]);
            neuron[i].delta = deltas[i];
            neuron[i].updateWeights(learning_rate);
        }
    }
};

// Enhanced Neural Network structure with backpropagation
struct NeuralNetwork {
    Layer* layers;
    int nLayers;
    int* nNodes;
    double learning_rate;

    NeuralNetwork(int numLayers, int* numNodes, int* weights, vector<double> values, 
                  double lr = 0.01, SchedulerType scheduler_type = FCFS) 
        : nLayers(numLayers), nNodes(numNodes), learning_rate(lr) {
        
        layers = new Layer[nLayers];
        int counter = 0;

        for (int i = 0; i < nLayers; i++) {
            layers[i] = Layer(scheduler_type);
            layers[i].nNodes = numNodes[i];
            layers[i].neuron = new Neuron[numNodes[i]];
            
            for (int j = 0; j < numNodes[i]; j++) {
                layers[i].neuron[j].weights = new double*[weights[i]];
                layers[i].neuron[j].nWeights = weights[i];
                for (int k = 0; k < weights[i]; k++) {
                    layers[i].neuron[j].weights[k] = new double[1];
                    layers[i].neuron[j].weights[k][0] = values[counter++];
                }
            }
        }
    }

    double* train(vector<vector<double>>& input, vector<double>& target) {
    // Forward propagation
    double* output = propogation(input);
    
    // Backward propagation
    for (int i = nLayers - 1; i >= 0; i--) {
        vector<double> next_deltas;
        vector<vector<double>> next_weights;
        
        if (i == nLayers - 1) {  // Output layer
            next_weights.push_back(target);
        } else {  // Hidden layers
            for (int j = 0; j < layers[i + 1].nNodes; j++) {
                vector<double> weights;
                for (int k = 0; k < layers[i + 1].neuron[j].nWeights; k++) {
                    weights.push_back(layers[i + 1].neuron[j].weights[k][0]);
                }
                next_weights.push_back(weights);
            }
            next_deltas = layers[i + 1].deltas;
        }
        
        layers[i].backward(next_deltas, next_weights, learning_rate);
    }
    
    return output;
}

    double* propogation(vector<vector<double>>& input) {
        double* outputFx = nullptr;
        double** currentInput = new double*[1];
        currentInput[0] = new double[input[0].size()];

        for (size_t i = 0; i < input[0].size(); i++) {
            currentInput[0][i] = input[0][i];
        }

        for (int i = 0; i < nLayers; i++) {
            layers[i].onData = currentInput;
            double** output = layers[i].forward();
            delete[] currentInput[0];
            delete[] currentInput;
            currentInput = output;
        }

        outputFx = calculateFx(currentInput[0][0]);
        delete[] currentInput[0];
        delete[] currentInput;
        return outputFx;
    }
};

// Read weights from file function
vector<double> read_file_to_1d_vector(const string& filename) {
    vector<double> result;
    ifstream input_file(filename);

    if (input_file.is_open()) {
        string line;
        while (getline(input_file, line)) {
            double value;
            istringstream line_stream(line);
            string token;
            while (getline(line_stream, token, ',')) {
                istringstream token_stream(token);
                token_stream >> value;
                result.push_back(value);
            }
        }
        input_file.close();
    } else {
        cout << "Unable to open file." << endl;
        exit(0);
    }
    return result;
}

// Main function
int main() {
    cout << "******** Neural Network with Backpropagation ********" << endl;
    cout << "Program started..." << endl;

    cout << "Initializing training data..." << endl;
    // Training data
    vector<vector<double>> training_inputs = {
        {0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}
    };
    vector<vector<double>> training_targets = {
        {0.0}, {1.0}, {1.0}, {0.0}
    };

    cout << "Setting up network parameters..." << endl;
    int numLayers = 3;
    int nodes[] = {4, 4, 1};
    int weights[] = {2, 4, 4};

    cout << "Reading weights from config file..." << endl;
    string filename = "config.txt";
    vector<double> assigning_weights;
    try {
        assigning_weights = read_file_to_1d_vector(filename);
        cout << "Successfully read " << assigning_weights.size() << " weights" << endl;
    } catch (const exception& e) {
        cout << "Error reading weights: " << e.what() << endl;
        return 1;
    }

    cout << "Creating neural network..." << endl;
    NeuralNetwork nn(numLayers, nodes, weights, assigning_weights, 0.01, RR);
    cout << "Neural network created successfully" << endl;

    // Training loop with timeout and progress tracking
    cout << "Beginning training phase..." << endl;
    int epochs = 1000;
    int max_time_seconds = 60; // Maximum training time of 60 seconds
    time_t start_time = time(nullptr);
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        if (difftime(time(nullptr), start_time) > max_time_seconds) {
            cout << "Training timeout after " << epoch << " epochs." << endl;
            break;
        }

        cout << "\nStarting epoch " << epoch << "..." << endl;
        double total_error = 0.0;
        
        for (size_t i = 0; i < training_inputs.size(); i++) {
            cout << "Processing training sample " << i + 1 << "/" << training_inputs.size() << "\r";
            vector<vector<double>> input = {training_inputs[i]};
            
            cout << "  Forward propagation..." << endl;
            double* output = nn.train(input, training_targets[i]);
            
            cout << "  Calculating error..." << endl;
            total_error += pow(training_targets[i][0] - output[0], 2);
            delete[] output;
        }
        
        double avg_error = total_error / training_inputs.size();
        cout << "\nEpoch " << epoch << ", Error: " << avg_error << endl;
        
        // Add early stopping if error is small enough
        if (avg_error < 0.01) {
            cout << "Reached target error. Stopping training." << endl;
            break;
        }

        // Add a small delay to prevent CPU overload
        Sleep(10);
    }

    cout << "\nTraining completed. Starting testing phase..." << endl;
    
    // Test the trained network
    cout << "\nTesting the trained network:" << endl;
    vector<vector<double>> input_value(1, vector<double>(2));
    cout << "Enter values of input: " << endl;
    cin >> input_value[0][0] >> input_value[0][1];

    cout << "Processing test input..." << endl;
    double* output = nn.propogation(input_value);
    cout << "\nForward Propagation Output: " << output[0] << endl;
    cout << "fx1: " << output[0] << ", fx2: " << output[1] << endl;

    delete[] output;
    return 0;
}