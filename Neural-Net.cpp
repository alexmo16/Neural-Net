// Authors: alexmo16, Dave Miller

#include "pch.h"
#include <vector>
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <fstream>
#include <sstream>

struct Connection
{
	double weight;
	double deltaWeight;
};

class TrainingData
{
public:
	TrainingData(const std::string filename);
	bool isEof(void) { return m_trainingDataFile.eof(); }
	void getTopology(std::vector<unsigned> &topology);

	// Returns the number of input values read from the file:
	unsigned getNextInputs(std::vector<double> &inputVals);
	unsigned getTargetOutputs(std::vector<double> &targetOutputVals);

private:
	std::ifstream m_trainingDataFile;
};

void TrainingData::getTopology(std::vector<unsigned> &topology)
{
	std::string line;
	std::string label;

	getline(m_trainingDataFile, line);
	std::stringstream ss(line);
	ss >> label;
	if (this->isEof() || label.compare("topology:") != 0) {
		abort();
	}

	while (!ss.eof()) {
		unsigned n;
		ss >> n;
		topology.push_back(n);
	}

	return;
}

TrainingData::TrainingData(const std::string filename)
{
	m_trainingDataFile.open(filename.c_str());
}

unsigned TrainingData::getNextInputs(std::vector<double> &inputVals)
{
	inputVals.clear();

	std::string line;
	std::getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("in:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			inputVals.push_back(oneValue);
		}
	}

	return inputVals.size();
}

unsigned TrainingData::getTargetOutputs(std::vector<double> &targetOutputVals)
{
	targetOutputVals.clear();

	std::string line;
	getline(m_trainingDataFile, line);
	std::stringstream ss(line);

	std::string label;
	ss >> label;
	if (label.compare("out:") == 0) {
		double oneValue;
		while (ss >> oneValue) {
			targetOutputVals.push_back(oneValue);
		}
	}

	return targetOutputVals.size();
}

// *********************** class Neuron ****************************

class Neuron;

typedef std::vector< Neuron > Layer;

class Neuron 
{
public:
	Neuron( unsigned numOutputs, unsigned neuronIndex );
	
	void setOutputValue( double value ) { this->m_outputValue = value; };
	double getOutputValue() const { return this->m_outputValue; };
	void feedForward( const Layer &prevLayer );
	void calculateOutputGradients( double targetValue );
	void calculateHiddenGradients( const Layer &nextLayer );
	void updateInputWeights( Layer &prevLayer );

private:
	static double eta;   // [0.0...1.0] net traning rate
	static double alpha; // [0.0...1.0] multiplier of last weight change (momentum)
	static double transferFunction( double x );
	static double transferFunctionDerivative( double x );
	static double randomWeight( void ) { return std::rand() / double( RAND_MAX ); }
	double sumDOW( const Layer &nextLayer ) const;
	double m_outputValue;
	std::vector< Connection > m_outputWeights;
	unsigned m_neuronIndex;
	double m_gradient;
};

double Neuron::eta = 0.15;
double Neuron::alpha = 0.5;

Neuron::Neuron( unsigned numOutputs, unsigned neuronIndex )
{
	this->m_neuronIndex = neuronIndex;

	for ( unsigned connections = 0; connections < numOutputs; ++connections )
	{
		m_outputWeights.push_back( Connection() );
		m_outputWeights.back().weight = randomWeight();
	}
}

void Neuron::feedForward( const Layer &prevLayer )
{
	double sum = 0.0;

	// Sum the previous layer's outputs (which are our inputs)
	// Include the bias node from the previous layer.
	for ( unsigned neuronNumber = 0; neuronNumber < prevLayer.size(); ++neuronNumber )
	{
		sum += prevLayer[ neuronNumber ].getOutputValue() * 
			prevLayer[ neuronNumber ].m_outputWeights[ m_neuronIndex ].weight;
	}

	m_outputValue = Neuron::transferFunction( sum );
}

double Neuron::transferFunction( double x )
{
	// tanh - output range [-1.0...1.0]
	return tanh( x );
}

double Neuron::transferFunctionDerivative( double x )
{
	// tanh - derivative
	return 1 - x * x;
}

void Neuron::calculateOutputGradients( double targetValue )
{
	double delta = targetValue - m_outputValue;
	m_gradient = delta * Neuron::transferFunctionDerivative( m_outputValue );
}

void Neuron::calculateHiddenGradients( const Layer &nextLayer )
{
	double dow = sumDOW( nextLayer );
	m_gradient = dow * Neuron::transferFunctionDerivative( m_outputValue );
}

double Neuron::sumDOW( const Layer &nextLayer ) const
{
	double sum = 0.0;
	for ( unsigned neuronNumber = 0; neuronNumber < nextLayer.size() - 1; ++neuronNumber )
	{
		sum += m_outputWeights[ neuronNumber ].weight * nextLayer[ neuronNumber ].m_gradient;
	}

	return sum;
}

void Neuron::updateInputWeights( Layer &prevLayer )
{
	// The weights to be updated are in the Connectoin container
	// in the neurons in the preceding layer
	for ( unsigned neuronNumber = 0; neuronNumber < prevLayer.size(); ++neuronNumber )
	{
		Neuron &neuron = prevLayer[ neuronNumber ];

		double oldDeltaWeight = neuron.m_outputWeights[ m_neuronIndex ].deltaWeight;
		double newDeltaWeight =
			// Individual input, magnified by the gradient and train rate:
			// eta = overall net learning rate alpha = momentum
			eta
			* neuron.getOutputValue()
			* m_gradient
			// Also add momentum = a fraction of the previus delta weight
			+ alpha
			* oldDeltaWeight;

		neuron.m_outputWeights[ m_neuronIndex ].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[ m_neuronIndex ].weight += newDeltaWeight;
	}
}

// *********************** class Network ***************************
class Network
{
public:
	Network( const std::vector< unsigned > &topology );

	void feedForward( const std::vector< double > &inputValues );
	void backPropagation( const std::vector< double > &targetValues );
	void getResults( std::vector< double > &resultValues ) const;
	double getRecentAverageError(void) const { return this->m_recentAverageSmoothingFactor; }

private:
	std::vector< Layer > m_layers; // m_layers[ layerNum ][ neuronNum ]
	double m_error;
	double m_recentAverageError;
	static double m_recentAverageSmoothingFactor;
};

double Network::m_recentAverageSmoothingFactor = 100.0; // Number of training samples to average over

Network::Network( const std::vector< unsigned > &topology )
{
	unsigned numLayers = topology.size();
	for ( unsigned layerNumber = 0; layerNumber < numLayers; ++layerNumber ) 
	{
		m_layers.push_back( Layer() );
		std::cout << "Made a Layer!!" << std::endl;

		// if we're at the last layer, there's no outputs,
		// else the number of ouputs is the number of Neuron in the next layer
		unsigned numOutputs = layerNumber == topology.size() - 1 ? 0 : topology[ layerNumber + 1 ];

		// we have made a mew Layer, now fill it with neurons, and
		// add a bias neuron to the layer:
		for (unsigned neuronNumber = 0; neuronNumber <= topology[ layerNumber ]; ++neuronNumber )
		{
			// back, gives the last element in a container
			m_layers.back().push_back( Neuron( numOutputs, neuronNumber ) );
			std::cout << "Made a Neuron!!" << std::endl;
		}

		// Force the bias node's output value to 1.0. It's the last neuron created above
		m_layers.back().back().setOutputValue( 1.0 );
	}
}

void Network::feedForward( const std::vector< double > &inputValues )
{
	assert( inputValues.size() == m_layers[0].size() - 1 );

	// Assign (latch) the input values into the input neurons
	for ( unsigned index = 0; index < inputValues.size(); ++index )
	{
		m_layers[ 0 ][ index ].setOutputValue( inputValues[ index ] );
	}

	// forward propagate
	for ( unsigned layerNumber = 1; layerNumber < m_layers.size() - 1; ++layerNumber )
	{
		// instead of copying the value in a new memory space,
		// it's like a pointer to the vector's index.
		Layer &previousLayer = m_layers[ layerNumber - 1 ];
		for ( unsigned neuronNumber = 0; neuronNumber < m_layers[ layerNumber ].size() - 1; ++neuronNumber )
		{
			m_layers[ layerNumber ][ neuronNumber ].feedForward( previousLayer );
		}
	}
}

void Network::backPropagation( const std::vector< double > &targetValues )
{
	// Calculate overall network error RMS
	Layer &outputLayer = m_layers.back();
	m_error = 0.0;

	for ( unsigned neuronNumber = 0; neuronNumber < outputLayer.size() - 1; ++neuronNumber )
	{
		double delta = targetValues[ neuronNumber ] - outputLayer[ neuronNumber ].getOutputValue();
		m_error += delta * delta;
	}
	m_error /= outputLayer.size() - 1; // get average error squared
	m_error = sqrt( m_error ); // RMS

	// Implement a recent average measurement:
	m_recentAverageError =
		( m_recentAverageError * m_recentAverageSmoothingFactor + m_error )
		/ ( m_recentAverageSmoothingFactor + 1.0 );

	// Calculate output layer gradients
	for ( unsigned neuronNumber = 0; outputLayer.size() - 1; ++neuronNumber )
	{
		outputLayer[ neuronNumber ].calculateOutputGradients( targetValues[ neuronNumber ] );
	}

	// Calculate gradients on hidden layers
	for ( unsigned layerNumber = m_layers.size() - 2; layerNumber > 0; --layerNumber )
	{
		Layer &hiddenLayer = m_layers[ layerNumber ];
		Layer &nextLayer = m_layers[ layerNumber + 1 ];

		for ( unsigned neuronNumber = 0; neuronNumber < hiddenLayer.size(); ++neuronNumber )
		{
			hiddenLayer[neuronNumber].calculateHiddenGradients(nextLayer);
		}
	}

	// For all layers from outputs to first hidden layer,
	// update connection weights
	for ( unsigned layerNumber = m_layers.size() - 1; layerNumber > 0; --layerNumber )
	{
		Layer &layer = m_layers[ layerNumber ];
		Layer &prevLayer = m_layers[ layerNumber - 1 ];

		for ( unsigned neuronNumber = 0; neuronNumber < layer.size() - 1; ++neuronNumber )
		{
			layer[ neuronNumber ].updateInputWeights( prevLayer );
		}
	}
}

void Network::getResults( std::vector< double > &resultValues ) const
{
	resultValues.clear();

	for ( unsigned neuronNumber = 0; neuronNumber < m_layers.back().size() - 1; ++neuronNumber )
	{
		resultValues.push_back( m_layers.back()[ neuronNumber ].getOutputValue() );
	}
}

// *************************** main ****************************

void showVectorValues( std::string label, std::vector< double > &v )
{
	std::cout << label << " ";

}

int main()
{
	TrainingData trainData( "trainingData.txt" );

	std::vector< unsigned > topology;
	trainData.getTopology( topology );
	Network myNet( topology );

	std::vector< double > inputValues, targetValues, resultValues;
	int trainingPass = 0;

	while ( !trainData.isEof() )
	{
		++trainingPass;
		std::cout << std::endl << "Pass " << trainingPass;

		// get new input data and feed it forward:
		if ( trainData.getNextInputs( inputValues ) != topology[ 0 ] )
		{
			break;
		}
		showVectorValues( ": Inputs:", inputValues );
		myNet.feedForward( inputValues );

		// collect the net's actual results:
		myNet.getResults( resultValues );
		showVectorValues( "Outputs:", resultValues );

		// train the net what the outputs should have been:
		trainData.getTargetOutputs( targetValues );
		showVectorValues( "Targets:", targetValues );
		assert( targetValues.size() == topology.back() );

		myNet.backPropagation( targetValues );

		// report how well the training is working, averaged
		std::cout << "Net recent average error: " << myNet.getRecentAverageError() << std::endl;
	}

	std::cout << std::endl << "Done" << std::endl;
}
