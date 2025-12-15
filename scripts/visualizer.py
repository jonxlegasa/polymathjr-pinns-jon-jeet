import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json
import sys

from matplotlib.widgets import Button, Slider


def setup_backend():
  """
  Try different backends until one works.
  """
  backends = ['Qt5Agg', 'QtAgg', 'GTK3Agg', 'GTK4Agg', 'WXAgg', 'TkAgg']
  
  for backend in backends:
    try:
      matplotlib.use(backend, force=True)
      # Test if backend works by creating a test figure
      fig = plt.figure()
      plt.close(fig)
      print(f"Successfully using backend: {backend}")
      return True
    except (ImportError, ModuleNotFoundError, ValueError) as e:
      continue
  
  print("ERROR: No interactive backend available.")
  print("Please install one of the following:")
  print("  - PyQt5: pip install PyQt5  (RECOMMENDED)")
  print("  - PyQt6: pip install PyQt6")
  print("  - Tkinter: Use system Python with python3-tk")
  print("  - GTK3: pip install PyGObject")
  print("  - WX: pip install wxPython")
  return False


# Setup backend before any plotting
if not setup_backend():
  sys.exit(1)


class PowerSeriesVisualizer:
  """
  A visualization tool for comparing neural network power series predictions
  against true power series solutions.
  
  This class creates an interactive plot where you can:
  - Adjust the neuron count using a slider to see different predictions
  - Compare predicted vs true power series
  - Reset to initial values using a button
  """
  
  def __init__(self, json_file_path, true_coefficients, x_range=(-1, 1),
         num_points=1000, neuron_range=None, initial_neurons=None,
         title="Power Series Comparison"):
    """
    Initialize the visualizer.
    
    Parameters:
    -----------
    json_file_path : str
      Path to JSON file containing predicted coefficients
      Format: {"10": [c0, c1, c2, ...], "20": [c0, c1, c2, ...], ...}
    true_coefficients : list or array-like
      Coefficients of the true power series [c0, c1, c2, ...]
    x_range : tuple
      (x_min, x_max) for evaluation range
    num_points : int
      Number of points to evaluate the series
    neuron_range : tuple or None
      (min_neurons, max_neurons) for slider. If None, inferred from data
    initial_neurons : int or None
      Starting neuron count. If None, uses minimum available
    title : str
      Plot title
    """
    # Load predicted coefficients from JSON
    with open(json_file_path, 'r') as f:
      data = json.load(f)
    
    # Convert string keys to integers and store coefficients
    self.predicted_coeffs_dict = {int(k): np.array(v) for k, v in data.items()}
    self.neuron_counts = sorted(self.predicted_coeffs_dict.keys())
    
    # Store true coefficients
    self.true_coefficients = np.array(true_coefficients)
    
    # Set up x-range
    self.x_data = np.linspace(x_range[0], x_range[1], num_points)
    
    # Determine neuron range and initial value
    if neuron_range is None:
      self.neuron_range = (min(self.neuron_counts), max(self.neuron_counts))
    else:
      self.neuron_range = neuron_range
    
    if initial_neurons is None:
      self.initial_neurons = min(self.neuron_counts)
    else:
      self.initial_neurons = initial_neurons
    
    self.title = title
    
    # Evaluate true series (fixed)
    self.y_true = self._evaluate_power_series(self.true_coefficients, self.x_data)
    
    # Create the figure and initial plots
    self.fig, self.ax = plt.subplots(figsize=(10, 6))
    
    # Plot true series (will not change)
    self.line_true, = self.ax.plot(self.x_data, self.y_true, 
                     lw=2, label='True Series', 
                     color='blue', linestyle='--')
    
    # Plot predicted series (will change with slider)
    y_predicted = self._get_predicted_series(self.initial_neurons)
    self.line_predicted, = self.ax.plot(self.x_data, y_predicted,
                        lw=2, label=f'Predicted (Neurons: {self.initial_neurons})',
                        color='red')
    
    self.ax.set_xlabel('x')
    self.ax.set_ylabel('y')
    self.ax.set_title(self.title)
    self.ax.legend()
    self.ax.grid(True, alpha=0.3)
    
    # Adjust layout to make room for slider and button
    self.fig.subplots_adjust(left=0.15, bottom=0.25)
    
    # Create slider for neuron count
    ax_neuron = self.fig.add_axes([0.25, 0.1, 0.65, 0.03])
    self.neuron_slider = Slider(
      ax=ax_neuron,
      label='Neuron Count',
      valmin=self.neuron_range[0],
      valmax=self.neuron_range[1],
      valinit=self.initial_neurons,
      valstep=1
    )
    
    # Create reset button
    reset_ax = self.fig.add_axes([0.8, 0.025, 0.1, 0.04])
    self.reset_button = Button(reset_ax, 'Reset', hovercolor='0.975')
    
    # Register callbacks
    self.neuron_slider.on_changed(self.update)
    self.reset_button.on_clicked(self.reset)
  
  def _evaluate_power_series(self, coefficients, x):
    """
    Evaluate a power series at given x values.
    
    Series: y = c0 + c1*x + c2*x^2 + c3*x^3 + ...
    
    Parameters:
    -----------
    coefficients : array-like
      Coefficients [c0, c1, c2, ...] where ci is coefficient of x^i
    x : array-like
      Points at which to evaluate the series
      
    Returns:
    --------
    array : Evaluated series values
    """
    result = np.zeros_like(x)
    for i, coeff in enumerate(coefficients):
      result += coeff * (x ** i)
    return result
  
  def _get_predicted_series(self, neuron_count):
    """
    Get predicted series for a given neuron count.
    
    Parameters:
    -----------
    neuron_count : int or float
      The desired neuron count
      
    Returns:
    --------
    array : The evaluated predicted series
    """
    neuron_count = int(round(neuron_count))
    
    # If exact match exists, use it
    if neuron_count in self.predicted_coeffs_dict:
      coeffs = self.predicted_coeffs_dict[neuron_count]
    else:
      # Otherwise, find nearest available neuron count
      nearest = min(self.neuron_counts, key=lambda x: abs(x - neuron_count))
      coeffs = self.predicted_coeffs_dict[nearest]
    
    return self._evaluate_power_series(coeffs, self.x_data)
  
  def update(self, val):
    """
    Update the predicted series plot when slider value changes.
    
    Parameters:
    -----------
    val : float
      The new slider value (neuron count)
    """
    neuron_count = int(round(val))
    y_predicted = self._get_predicted_series(neuron_count)
    self.line_predicted.set_ydata(y_predicted)
    self.line_predicted.set_label(f'Predicted (Neurons: {neuron_count})')
    self.ax.legend()
    self.fig.canvas.draw_idle()
  
  def reset(self, event):
    """Reset slider to initial value."""
    self.neuron_slider.reset()
  
  def show(self):
    """Display the interactive plot."""
    plt.show()


