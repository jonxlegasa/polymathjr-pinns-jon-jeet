import json
from visualizer import PowerSeriesVisualizer

# Example usage
if __name__ == "__main__":
  # Example: Create sample JSON file for demonstration
  # You will replace this with your actual JSON file
  
  # Generate example data
  example_data = {
    "10": [1.0, 2.1, 0.9, 0.15],  # Coefficients for 10 neurons
    "20": [1.0, 2.05, 0.95, 0.18],  # Coefficients for 20 neurons
    "30": [1.0, 2.02, 0.98, 0.165],  # Coefficients for 30 neurons
    "40": [1.0, 2.01, 0.99, 0.168],  # Coefficients for 40 neurons
    "50": [1.0, 2.0, 1.0, 0.167],  # Coefficients for 50 neurons
  }
  
  # Save example JSON
  with open('example_coefficients.json', 'w') as f:
    json.dump(example_data, f, indent=2)
  
  # True power series coefficients
  # For example: y = 1 + 2x + x^2 + (1/6)x^3
  true_coeffs = [1.0, 2.0, 1.0, 1.0/6.0]
  
  # Create and show the visualizer
  visualizer = PowerSeriesVisualizer(
    json_file_path='example_coefficients.json',
    true_coefficients=true_coeffs,
    x_range=(-2, 2),
    num_points=1000,
    neuron_range=(10, 50),
    initial_neurons=30,
    title="Neural Network Power Series Approximation"
  )
  
  visualizer.show()
