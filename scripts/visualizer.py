import matplotlib
matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt
import numpy as np
import json
import sys

from matplotlib.widgets import Button, Slider
from typing import Dict, List, Callable, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class PlotConfig:
  """Configuration for a single plot/subplot."""
  data_key: str  # Key to access data in the data dictionary
  title: str
  xlabel: str
  ylabel: str
  plot_type: str = 'line'  # 'line', 'scatter', 'semilogy', etc.
  colors: Optional[List[str]] = None
  linestyles: Optional[List[str]] = None
  labels: Optional[List[str]] = None
  grid: bool = True


@dataclass
class SliderConfig:
  """Configuration for a slider widget."""
  name: str
  label: str
  valmin: float
  valmax: float
  valinit: float
  valstep: Optional[float] = 1
  position: Tuple[float, float, float, float] = (0.25, 0.1, 0.65, 0.03)


class GeneralizedVisualizer:
  """
  A generalized visualization framework for interactive multi-plot analysis.
  
  This class supports:
  - Multiple subplots with different configurations
  - Multiple interactive sliders
  - Automatic data updates when sliders change
  - Flexible plot types (line, scatter, log scale, etc.)
  """
  
  def __init__(self, 
         data_dict: Dict[str, Any],
         plot_configs: List[PlotConfig],
         slider_configs: List[SliderConfig],
         layout: Tuple[int, int] = (2, 2),
         figsize: Tuple[int, int] = (14, 10),
         main_title: str = "Analysis Dashboard"):
    """
    Initialize the generalized visualizer.
    
    Parameters:
    -----------
    data_dict : dict
      Dictionary containing all data needed for plots
      Example: {
        'neuron_counts': {10: {...}, 20: {...}, ...},
        'static_data': {'x': [...], 'y': [...]},
        ...
      }
    plot_configs : list of PlotConfig
      Configuration for each subplot
    slider_configs : list of SliderConfig
      Configuration for each slider
    layout : tuple
      (rows, cols) for subplot layout
    figsize : tuple
      Figure size (width, height)
    main_title : str
      Main title for the figure
    """
    self.data_dict = data_dict
    self.plot_configs = plot_configs
    self.slider_configs = slider_configs
    self.layout = layout
    self.figsize = figsize
    self.main_title = main_title
    
    # Store current slider values
    self.slider_values = {config.name: config.valinit for config in slider_configs}
    
    # Create figure and subplots
    self.fig = plt.figure(figsize=self.figsize)
    self.fig.suptitle(self.main_title, fontsize=16, fontweight='bold')
    
    # Adjust layout for sliders and buttons
    num_sliders = len(slider_configs)
    bottom_margin = 0.15 + (num_sliders * 0.04)
    self.fig.subplots_adjust(left=0.1, right=0.95, top=0.93, 
                   bottom=bottom_margin, hspace=0.3, wspace=0.3)
    
    # Create subplots
    self.axes = []
    self.lines = {}  # Store line objects for updating
    
    for idx, config in enumerate(plot_configs):
      ax = self.fig.add_subplot(layout[0], layout[1], idx + 1)
      self.axes.append(ax)
      self._setup_plot(ax, config, idx)
    
    # Create sliders
    self.sliders = {}
    self._create_sliders()
    
    # Create reset button
    self._create_reset_button()
  
  def _setup_plot(self, ax, config: PlotConfig, plot_idx: int):
    """
    Set up a single plot based on its configuration.
    
    Parameters:
    -----------
    ax : matplotlib axis
      The axis to plot on
    config : PlotConfig
      Configuration for this plot
    plot_idx : int
      Index of this plot
    """
    ax.set_title(config.title, fontweight='bold')
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    if config.grid:
      ax.grid(True, alpha=0.3)
    
    # Initialize empty lines that will be updated
    self.lines[plot_idx] = []
    
    # Initial plot (will be updated by _update_plot)
    self._update_plot(ax, config, plot_idx)
  
  def _update_plot(self, ax, config: PlotConfig, plot_idx: int):
    """
    Update a plot with current slider values.
    
    Parameters:
    -----------
    ax : matplotlib axis
      The axis to update
    config : PlotConfig
      Configuration for this plot
    plot_idx : int
      Index of this plot
    """
    # Clear previous lines
    for line in self.lines[plot_idx]:
      line.remove()
    self.lines[plot_idx] = []
    
    # Get data based on current slider values
    data = self._get_plot_data(config.data_key)
    
    if data is None:
      return
    
    # Handle different plot types
    if config.plot_type == 'line':
      self._plot_lines(ax, data, config, plot_idx)
    elif config.plot_type == 'semilogy':
      self._plot_semilogy(ax, data, config, plot_idx)
    elif config.plot_type == 'scatter':
      self._plot_scatter(ax, data, config, plot_idx)
    
    ax.legend()
  
  def _plot_lines(self, ax, data, config: PlotConfig, plot_idx: int):
    """Plot standard line plots."""
    if isinstance(data, dict):
      # Multiple lines
      for idx, (key, values) in enumerate(data.items()):
        # Handle both dict format and direct array format
        if isinstance(values, dict):
          x_data = values.get('x', np.arange(len(values['y'])))
          y_data = values['y']
        else:
          # Direct array
          x_data = np.arange(len(values))
          y_data = values
        
        color = config.colors[idx] if config.colors and idx < len(config.colors) else None
        linestyle = config.linestyles[idx] if config.linestyles and idx < len(config.linestyles) else '-'
        label = config.labels[idx] if config.labels and idx < len(config.labels) else str(key)
        
        line, = ax.plot(x_data, y_data, color=color, linestyle=linestyle, 
                label=label, lw=2)
        self.lines[plot_idx].append(line)
    else:
      # Single line
      if isinstance(data, dict):
        x_data = data.get('x', np.arange(len(data['y'])))
        y_data = data['y']
      else:
        x_data = np.arange(len(data))
        y_data = data
      
      label = config.labels[0] if config.labels else 'Data'
      
      line, = ax.plot(x_data, y_data, label=label, lw=2)
      self.lines[plot_idx].append(line)
  
  def _plot_semilogy(self, ax, data, config: PlotConfig, plot_idx: int):
    """Plot with logarithmic y-axis."""
    if isinstance(data, dict):
      for idx, (key, values) in enumerate(data.items()):
        # Handle both dict format and direct array format
        if isinstance(values, dict):
          x_data = values.get('x', np.arange(len(values['y'])))
          y_data = values['y']
        else:
          # Direct array
          x_data = np.arange(len(values))
          y_data = values
        
        color = config.colors[idx] if config.colors and idx < len(config.colors) else None
        linestyle = config.linestyles[idx] if config.linestyles and idx < len(config.linestyles) else '-'
        label = config.labels[idx] if config.labels and idx < len(config.labels) else str(key)
        
        # Ensure positive values for log scale
        y_data = np.maximum(y_data, 1e-10)
        
        line, = ax.semilogy(x_data, y_data, color=color, linestyle=linestyle,
                  label=label, lw=2)
        self.lines[plot_idx].append(line)
    else:
      if isinstance(data, dict):
        x_data = data.get('x', np.arange(len(data['y'])))
        y_data = data['y']
      else:
        x_data = np.arange(len(data))
        y_data = data
      
      label = config.labels[0] if config.labels else 'Data'
      
      # Ensure positive values for log scale
      y_data = np.maximum(y_data, 1e-10)
      
      line, = ax.semilogy(x_data, y_data, label=label, lw=2)
      self.lines[plot_idx].append(line)
  
  def _plot_scatter(self, ax, data, config: PlotConfig, plot_idx: int):
    """Plot scatter plots."""
    if isinstance(data, dict):
      for idx, (key, values) in enumerate(data.items()):
        if isinstance(values, dict):
          x_data = values.get('x', np.arange(len(values['y'])))
          y_data = values['y']
        else:
          x_data = np.arange(len(values))
          y_data = values
        
        color = config.colors[idx] if config.colors and idx < len(config.colors) else None
        label = config.labels[idx] if config.labels and idx < len(config.labels) else str(key)
        
        scatter = ax.scatter(x_data, y_data, color=color, label=label, s=20)
        self.lines[plot_idx].append(scatter)
    else:
      if isinstance(data, dict):
        x_data = data.get('x', np.arange(len(data['y'])))
        y_data = data['y']
      else:
        x_data = np.arange(len(data))
        y_data = data
      
      label = config.labels[0] if config.labels else 'Data'
      
      scatter = ax.scatter(x_data, y_data, label=label, s=20)
      self.lines[plot_idx].append(scatter)
  
  def _get_plot_data(self, data_key: str) -> Optional[Any]:
    """
    Retrieve data for plotting based on current slider values.
    
    Parameters:
    -----------
    data_key : str
      Key to access data in self.data_dict
      
    Returns:
    --------
    Data for plotting (format depends on plot type)
    """
    # This is a flexible method that can be overridden for custom data retrieval
    # For now, it supports basic key lookup and slider-dependent data
    
    if '.' in data_key:
      # Support nested keys like 'neuron_data.predictions'
      keys = data_key.split('.')
      data = self.data_dict
      for key in keys:
        if isinstance(data, dict) and key in data:
          data = data[key]
        else:
          return None
      
      # If data is slider-dependent (dict with numeric keys)
      if isinstance(data, dict) and all(isinstance(k, (int, float)) for k in data.keys()):
        # Get data for current slider value
        slider_val = list(self.slider_values.values())[0]  # Use first slider
        if slider_val in data:
          return data[slider_val]
        else:
          # Find nearest key
          nearest = min(data.keys(), key=lambda x: abs(x - slider_val))
          return data[nearest]
      
      return data
    else:
      # Simple key lookup
      return self.data_dict.get(data_key)
  
  def _create_sliders(self):
    """Create all slider widgets."""
    for idx, config in enumerate(self.slider_configs):
      # Calculate slider position (stack them vertically)
      pos = list(config.position)
      pos[1] = 0.08 - (idx * 0.04)  # Stack sliders from bottom up
      
      ax_slider = self.fig.add_axes(pos)
      slider = Slider(
        ax=ax_slider,
        label=config.label,
        valmin=config.valmin,
        valmax=config.valmax,
        valinit=config.valinit,
        valstep=config.valstep
      )
      
      # Register callback
      slider.on_changed(lambda val, name=config.name: self._on_slider_change(name, val))
      self.sliders[config.name] = slider
  
  def _on_slider_change(self, slider_name: str, value: float):
    """
    Callback when any slider changes.
    
    Parameters:
    -----------
    slider_name : str
      Name of the slider that changed
    value : float
      New value of the slider
    """
    # Update stored slider value
    self.slider_values[slider_name] = value
    
    # Update all plots
    for idx, (ax, config) in enumerate(zip(self.axes, self.plot_configs)):
      self._update_plot(ax, config, idx)
    
    self.fig.canvas.draw_idle()
  
  def _create_reset_button(self):
    """Create reset button to restore initial slider values."""
    button_ax = self.fig.add_axes([0.85, 0.02, 0.1, 0.03])
    self.reset_button = Button(button_ax, 'Reset', hovercolor='0.975')
    self.reset_button.on_clicked(self._on_reset)
  
  def _on_reset(self, event):
    """Reset all sliders to initial values."""
    for config in self.slider_configs:
      self.sliders[config.name].reset()
  
  def show(self):
    """Display the interactive visualization."""
    plt.show()


class PowerSeriesVisualizer(GeneralizedVisualizer):
  """
  Specialized visualizer for power series analysis with multiple plots.
  """
  
  def __init__(self,
         json_file_path: str,
         true_coefficients: List[float],
         loss_data: Optional[Dict] = None,
         x_range: Tuple[float, float] = (0, 1),
         num_points: int = 1000,
         neuron_range: Optional[Tuple[int, int]] = None,
         initial_neurons: Optional[int] = None):
    """
    Initialize the power series visualizer.
    
    Parameters:
    -----------
    json_file_path : str
      Path to JSON file with predicted coefficients
    true_coefficients : list
      True power series coefficients
    loss_data : dict, optional
      Dictionary with loss data per neuron count
      Format: {
        10: {'iterations': [...], 'total_loss': [...], 'bc_loss': [...], ...},
        20: {...},
        ...
      }
    x_range : tuple
      Range for x-axis
    num_points : int
      Number of evaluation points
    neuron_range : tuple, optional
      Min and max neurons for slider
    initial_neurons : int, optional
      Initial neuron count
    """
    # Load predicted coefficients
    with open(json_file_path, 'r') as f:
      predicted_coeffs = json.load(f)
    
    predicted_coeffs = {int(k): np.array(v) for k, v in predicted_coeffs.items()}
    neuron_counts = sorted(predicted_coeffs.keys())
    
    if neuron_range is None:
      neuron_range = (min(neuron_counts), max(neuron_counts))
    
    if initial_neurons is None:
      initial_neurons = min(neuron_counts)
    
    true_coefficients = np.array(true_coefficients)
    x_data = np.linspace(x_range[0], x_range[1], num_points)
    
    # Prepare data dictionary
    data_dict = self._prepare_data(
      predicted_coeffs, true_coefficients, x_data, loss_data, neuron_counts
    )
    
    # Define plot configurations
    plot_configs = self._create_plot_configs(loss_data is not None)
    
    # Define slider configuration
    slider_configs = [
      SliderConfig(
        name='neuron_count',
        label='Neuron Count',
        valmin=neuron_range[0],
        valmax=neuron_range[1],
        valinit=initial_neurons,
        valstep=1
      )
    ]
    
    # Determine layout based on number of plots
    num_plots = len(plot_configs)
    if num_plots <= 2:
      layout = (1, 2)
    elif num_plots <= 4:
      layout = (2, 2)
    elif num_plots <= 6:
      layout = (2, 3)
    else:
      layout = (3, 3)

    super().__init__(
      data_dict=data_dict,
      plot_configs=plot_configs,
      slider_configs=slider_configs,
      layout=layout,
      figsize=(16, 10),
      main_title="PINNs Analysis"
    )
  
  def _prepare_data(self, predicted_coeffs, true_coeffs, x_data, loss_data, neuron_counts):
    """Prepare all data needed for plots."""
    data = {}
    # Compute solutions for each neuron count
    true_solution = self._evaluate_power_series(true_coeffs, x_data)
    solutions = {}
    coeff_comparisons = {}
    coeff_errors = {}
    solution_errors = {}
    for n_neurons in neuron_counts:
      pred_coeffs = predicted_coeffs[n_neurons]
      pred_solution = self._evaluate_power_series(pred_coeffs, x_data)
      solutions[n_neurons] = {
        'x': x_data,
        'y_true': true_solution,
        'y_pred': pred_solution
      }
      # Coefficient comparison
      coeff_idx = np.arange(len(pred_coeffs))
      coeff_comparisons[n_neurons] = {
        'benchmark': {'x': coeff_idx, 'y': true_coeffs[:len(pred_coeffs)]},
        'pinn': {'x': coeff_idx, 'y': pred_coeffs}
      }
      # Coefficient errors
      min_len = min(len(true_coeffs), len(pred_coeffs))
      coeff_errors[n_neurons] = {
        'x': np.arange(min_len),
        'y': np.abs(true_coeffs[:min_len] - pred_coeffs[:min_len])
      }
      # Solution errors
      solution_errors[n_neurons] = {
        'x': x_data,
        'y': np.abs(true_solution - pred_solution)
      }
    data['solutions'] = solutions
    data['coeff_comparisons'] = coeff_comparisons
    data['coeff_errors'] = coeff_errors
    data['solution_errors'] = solution_errors
    # Add loss data if provided
    if loss_data is not None:
      data['loss_data'] = loss_data
    return data
  
  def _create_plot_configs(self, include_loss: bool) -> List[PlotConfig]:
    """Create plot configurations."""
    configs = [
      # ODE Solution Comparison
      PlotConfig(
        data_key='solutions',
        title='ODE Solution Comparison',
        xlabel='x',
        ylabel='u(x)',
        plot_type='line',
        colors=['blue', 'red'],
        linestyles=['--', '-'],
        labels=['Analytic Solution', 'PINN Power Series']
      ),
      # Coefficient Comparison
      PlotConfig(
        data_key='coeff_comparisons',
        title='Coefficient Comparison',
        xlabel='Coefficient Index',
        ylabel='Coefficient Value',
        plot_type='line',
        colors=['blue', 'red'],
        linestyles=['-', '-'],
        labels=['Benchmark', 'PINN']
      ),
      # Coefficient Error
      PlotConfig(
        data_key='coeff_errors',
        title='Absolute Error of Coefficients',
        xlabel='Coefficient Index',
        ylabel='Absolute Error',
        plot_type='semilogy',
        colors=['blue'],
        labels=['|Benchmark - PINN|']
      ),
      # Solution Error
      PlotConfig(
        data_key='solution_errors',
        title='Absolute Error of Solution',
        xlabel='x',
        ylabel='Error',
        plot_type='semilogy',
        colors=['blue'],
        labels=['|Analytic - Predicted|']
      )
    ]
    # Add loss plots if loss data is provided
    if include_loss:
      loss_config = PlotConfig(
        data_key='loss_data',
        title='Loss vs Iteration',
        xlabel='Iteration',
        ylabel='Loss',
        plot_type='semilogy',
        colors=['black', 'red', 'blue', 'green', 'orange'],
        labels=['Total Loss', 'BC Loss', 'PDE Loss', 'Supervised Loss', 'Other']
      )
      configs.append(loss_config)
    return configs

  def _evaluate_power_series(self, coefficients, x):
    """Evaluate power series."""
    result = np.zeros_like(x)
    for i, coeff in enumerate(coefficients):
      result += coeff * (x ** i)
    return result

  def _get_plot_data(self, data_key: str):
    """Override to handle power series specific data retrieval."""
    neuron_count = int(round(self.slider_values['neuron_count']))
    # Find nearest available neuron count if exact match doesn't exist
    available_counts = list(self.data_dict['solutions'].keys())
    if neuron_count not in available_counts:
      neuron_count = min(available_counts, key=lambda x: abs(x - neuron_count))

    if data_key == 'solutions':
      sol_data = self.data_dict['solutions'][neuron_count]
      return {
        'true': {'x': sol_data['x'], 'y': sol_data['y_true']},
        'pred': {'x': sol_data['x'], 'y': sol_data['y_pred']}
      }

    elif data_key == 'coeff_comparisons':
      return self.data_dict['coeff_comparisons'][neuron_count]

    elif data_key == 'coeff_errors':
      error_data = self.data_dict['coeff_errors'][neuron_count]
      # Return single series for error plot
      return {'error': {'x': error_data['x'], 'y': error_data['y']}}
    
    elif data_key == 'solution_errors':
      error_data = self.data_dict['solution_errors'][neuron_count]
      # Return single series for error plot
      return {'error': {'x': error_data['x'], 'y': error_data['y']}}
    
    elif data_key == 'loss_data':
      if 'loss_data' not in self.data_dict:
        return None
      
      loss_info = self.data_dict['loss_data'][neuron_count]
      # Return all loss components
      result = {}
      if 'total_loss' in loss_info:
        result['Total'] = {'x': loss_info['iterations'], 'y': loss_info['total_loss']}
      if 'bc_loss' in loss_info:
        result['BC'] = {'x': loss_info['iterations'], 'y': loss_info['bc_loss']}
      if 'pde_loss' in loss_info:
        result['PDE'] = {'x': loss_info['iterations'], 'y': loss_info['pde_loss']}
      if 'supervised_loss' in loss_info:
        result['Supervised'] = {'x': loss_info['iterations'], 'y': loss_info['supervised_loss']}
      return result
    
    return None
