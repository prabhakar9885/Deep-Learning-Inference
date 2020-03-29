#include <vector>

#ifndef MathOps
#define MathOps

using namespace std;

namespace utils
{
	void axpb_c(vector<float> a, vector<float>& x, float b)
	{
		for (size_t i = 0; i < x.size(); i++)
		{
			x[i] = a[i] * x[i] + b;
		}
	}

	void computeActivation(vector<float>& x, Activation activation)
	{
		switch (activation)
		{
		case Activation::SIGMOID:
			break;
		case Activation::ReLU:
			break;
		case Activation::SOFTMAX:
			break;
		default:
			throw "Unidentified Activation type";
		}
	}
}
#endif // MathOps
