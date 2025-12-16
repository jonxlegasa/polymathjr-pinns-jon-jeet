import json
import numpy as np
from visualizer import PowerSeriesVisualizer, GeneralizedVisualizer, PlotConfig, SliderConfig

# Example 2: Power Series Visualizer with Loss Data
def example_with_loss():
    """Usage including loss data from training."""

# Create example predicted coefficients JSON
    example_data = {
    "10": [1.0, 2.1, 0.9, 0.15, 0.02],
    "20": [1.0, 2.05, 0.95, 0.18, 0.025],
    "30": [1.0, 2.02, 0.98, 0.165, 0.0167],
    "40": [1.0, 2.01, 0.99, 0.168, 0.0169],
    "50": [1.0, 2.0, 1.0, 0.167, 0.0166],
    }

    with open('example_coefficients.json', 'w') as f:
        json.dump(example_data, f, indent=2)
        # True coefficients
        true_coeffs = [1.0, 2.0, 1.0, 1.0/6.0, 1.0/60.0]
# Create example loss data for each neuron count
        loss_data = {}
   
    for n_neurons in [10, 20, 30, 40, 50]:
        num_iterations = 1000
        iterations = np.arange(num_iterations)
# Simulate decreasing loss (you'll replace this with your actual data)
        total_loss = 1.0 * np.exp(-iterations / 200) + 0.01 * np.random.randn(num_iterations)
        bc_loss = 0.3 * np.exp(-iterations / 150) + 0.005 * np.random.randn(num_iterations)
        pde_loss = 0.5 * np.exp(-iterations / 250) + 0.005 * np.random.randn(num_iterations)
        supervised_loss = 0.2 * np.exp(-iterations / 180) + 0.003 * np.random.randn(num_iterations)
        loss_data[n_neurons] = {
            'iterations': iterations.tolist(),
            'total_loss': np.maximum(total_loss, 1e-6).tolist(),  # Avoid negative for log scale
            'bc_loss': np.maximum(bc_loss, 1e-6).tolist(),
            'pde_loss': np.maximum(pde_loss, 1e-6).tolist(),
            'supervised_loss': np.maximum(supervised_loss, 1e-6).tolist()
        }
    # Create visualizer with loss data
    visualizer = PowerSeriesVisualizer(
    json_file_path='example_coefficients.json',
    true_coefficients=true_coeffs,
    loss_data=loss_data,
    x_range=(0, 1),
    num_points=1000,
    neuron_range=(10, 50),
    initial_neurons=30
    )

    visualizer.show()

# Example 4: loading data from real files
def example_with_real_data():
    """
    Example showing how to load and use your actual training data.
    """
    
    # Load your predicted coefficients
    with open('path/to/your/coefficients.json', 'r') as f:
        predicted_coeffs = json.load(f)
    
    # Load your loss data (assuming you saved it as JSON or CSV)
    # Option A: From JSON
    with open('path/to/your/loss_data.json', 'r') as f:
        loss_data = json.load(f)
    
    # Option B: From CSV (if you have separate CSV files per neuron count)
    # import pandas as pd
    # loss_data = {}
    # for n_neurons in [10, 20, 30, 40, 50]:
    #   df = pd.read_csv(f'path/to/loss_data_{n_neurons}_neurons.csv')
    #   loss_data[n_neurons] = {
    #     'iterations': df['iteration'].tolist(),
    #     'total_loss': df['total_loss'].tolist(),
    #     'bc_loss': df['bc_loss'].tolist(),
    #     'pde_loss': df['pde_loss'].tolist(),
    #     'supervised_loss': df['supervised_loss'].tolist()
    #   }
    
    # Your true coefficients from analytical solution
    true_coeffs = [1.0, 2.0, 1.0, 1.0/6.0, 1.0/60.0]  # Replace with your actual values
    
    # Create visualizer
    visualizer = PowerSeriesVisualizer(
        json_file_path='path/to/your/coefficients.json',
        true_coefficients=true_coeffs,
        loss_data=loss_data,
        x_range=(0, 1),
        num_points=1000,
        neuron_range=(10, 50),
        initial_neurons=30
    )
    visualizer.show()

if __name__ == "__main__":
    example_with_loss()
