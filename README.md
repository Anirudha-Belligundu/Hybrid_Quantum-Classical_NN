# Hybrid Quantum-Classical Neural Network for MNIST Classification

## Project Description

This project implements a hybrid quantum-classical neural network for MNIST digit classification using the TorchQuantum framework. The goal is to demonstrate the effectiveness of quantum-enhanced models for visual pattern recognition while optimizing model complexity.

## Files Included

- `hybrid_mnist.py`: Main Python script that defines and trains the hybrid model
- `models/`: Contains the quantum circuit definitions and model wrappers
- `results/`: Directory storing training logs and confusion matrix
- `utils/`: Helper scripts for visualization and evaluation

## Requirements

- Python 3.8+
- torch>=1.9.0
- torchvision
- torchquantum
- matplotlib
- scikit-learn


## Results

- **Test Accuracy**: 95.47%
- **Train Accuracy**: 98.97%
- **Loss (final epoch)**: 0.0982
- **Quantum Backend**: TorchQuantum CPU simulator
- **Qubits Used**: 4
- **Entanglement**: Full via Controlled-Z gates
- **Confusion Matrix**: Provided in `results/`

Key Insight: Hybrid QNNs achieve comparable results to classical models while consuming fewer trainable parameters, which is highly beneficial for quantum resource-constrained systems.


## Reference Material

1. M. Schuld and F. Petruccione, *Machine Learning with Quantum Computers*, Springer, 2021.
2. V. Havlíček et al., “Supervised learning with quantum-enhanced feature spaces,” *Nature*, vol. 567, no. 7747, pp. 209–212, 2019.
3. TorchQuantum Documentation: https://github.com/mit-han-lab/torchquantum
4. MNIST Dataset: http://yann.lecun.com/exdb/mnist/
5. Qiskit Textbook (for conceptual clarity): https://qiskit.org/textbook

## Conclusion

This project demonstrates how hybrid quantum-classical neural networks can be used effectively for image classification tasks like MNIST. By utilizing TorchQuantum, we constructed circuits with only 4 qubits and introduced entanglement using Controlled-Z gates, yet achieved a test accuracy of over 95%. This validates the feasibility of quantum neural networks for practical problems even under simulator constraints. As quantum hardware advances, such hybrid models are expected to play a significant role in edge devices and AI accelerators. The experiment also reflects how quantum models can learn with fewer parameters—showing promise for resource-efficient AI in the quantum era.
