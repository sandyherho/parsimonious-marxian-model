#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stable Marxian Figures Generator
--------------------------------
A Python implementation of a Marxian economic model with improved numerical stability
and visualization capabilities for academic publications.

License: WTFPL (http://www.wtfpl.net/)
Author: Sandy Herho <sandy.herho@email.ucr.edu>
Date: April 6, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import os
from typing import Tuple, Optional, List, Dict

class MarxianModel:
    """
    A class representing a Marxian economic model with improved numerical stability.
    
    This model implements a minimal 2-ODE system describing worker's share and capital dynamics
    in a Marxian economic framework.
    
    Attributes:
        alpha (float): Capital productivity parameter
        beta (float): Labor exploitation rate
        gamma (float): Class struggle intensity
        delta (float): Capital depreciation rate
        epsilon (float): Wage pressure parameter
        rho (float): Profit rate threshold for investment
    """
    
    def __init__(self, alpha: float = 0.3, beta: float = 0.6, gamma: float = 0.4, 
                 delta: float = 0.05, epsilon: float = 0.2, rho: float = 0.1):
        """
        Initialize the Marxian economic model with given parameters.
        
        Args:
            alpha: Capital productivity (default: 0.3)
            beta: Labor exploitation rate (default: 0.6)
            gamma: Class struggle intensity (default: 0.4)
            delta: Capital depreciation rate (default: 0.05)
            epsilon: Wage pressure parameter (default: 0.2)
            rho: Profit rate threshold for investment (default: 0.1)
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.rho = rho
        
        # Professional color palette for visualizations
        self.colors = {
            'worker_share': '#1f77b4',    # Blue
            'capital': '#2ca02c',          # Green
            'profit': '#d62728',           # Red
            'accent1': '#ff7f0e',          # Orange
            'accent2': '#9467bd',          # Purple
            'text': '#333333'             # Dark gray
        }
        
        # Set up plotting style
        self._setup_plot_style()

    def _setup_plot_style(self) -> None:
        """Configure matplotlib plotting style for publication-quality figures."""
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 12,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.figsize': (10, 8),
            'axes.grid': True,
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.labelpad': 10,
        })

    def profit_rate(self, w: float, k: float) -> float:
        """
        Calculate the profit rate based on worker's share and capital.
        
        Args:
            w: Worker's share (0 < w < 1)
            k: Capital stock (k > 0)
            
        Returns:
            The profit rate as a float
        """
        k_safe = max(k, 1e-6)  # Ensure positive capital
        return self.alpha * (1 - w) * k_safe**(self.alpha - 1) - self.delta

    def unemployment(self, k: float) -> float:
        """
        Calculate unemployment level based on capital stock.
        
        Args:
            k: Capital stock (k > 0)
            
        Returns:
            The unemployment rate as a float
        """
        return 1 / (1 + max(k, 1e-6))  # Ensure positive denominator

    def system(self, t: float, state: List[float]) -> List[float]:
        """
        The ODE system describing Marxian dynamics.
        
        Args:
            t: Time (not used directly, required by solve_ivp)
            state: Current state [worker's share, capital]
            
        Returns:
            List of derivatives [dw_dt, dk_dt]
        """
        w, k = state
        
        # Ensure state variables are in valid ranges
        w = np.clip(w, 0.01, 0.99)
        k = max(k, 1e-6)
        
        # Calculate intermediate values
        profit = self.profit_rate(w, k)
        u = self.unemployment(k)
        
        # Worker's share dynamics
        dw_dt = self.gamma * w * (1 - w) - self.epsilon * u * w
        
        # Capital accumulation
        if profit > self.rho:
            dk_dt = k * (profit - self.rho) - self.delta * k
        else:
            dk_dt = -self.delta * k
            
        return [dw_dt, dk_dt]

    def simulate(self, w0: float = 0.5, k0: float = 1.0, time_span: int = 100, 
                 num_points: int = 1000, save_csv: Optional[str] = None) -> Tuple[np.ndarray, ...]:
        """
        Run a simulation of the model with given initial conditions.
        
        Args:
            w0: Initial worker's share (default: 0.5)
            k0: Initial capital stock (default: 1.0)
            time_span: Simulation time span (default: 100)
            num_points: Number of time points (default: 1000)
            save_csv: Path to save CSV output (optional)
            
        Returns:
            Tuple of (time points, state variables, profit rates)
        """
        t_span = (0, time_span)
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        # Solve the ODE system
        solution = solve_ivp(
            self.system, 
            t_span, 
            [w0, k0], 
            method='RK45',
            t_eval=t_eval,
            rtol=1e-6,
            atol=1e-9
        )
        
        # Extract solution
        t = solution.t
        w = solution.y[0]
        k = solution.y[1]
        profit_rates = np.array([self.profit_rate(w[i], k[i]) for i in range(len(t))])
        
        # Save to CSV if requested
        if save_csv:
            self._save_to_csv(t, w, k, profit_rates, save_csv)
        
        return t, solution.y, profit_rates

    def _save_to_csv(self, t: np.ndarray, w: np.ndarray, k: np.ndarray, 
                    profit_rates: np.ndarray, filepath: str) -> None:
        """
        Save simulation results to a CSV file.
        
        Args:
            t: Time points
            w: Worker's share values
            k: Capital values
            profit_rates: Profit rate values
            filepath: Path to save the CSV file
        """
        try:
            import pandas as pd
            
            # Create directory if needed
            csv_dir = os.path.dirname(filepath)
            if csv_dir and not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            
            # Create and save DataFrame
            df = pd.DataFrame({
                'time': t,
                'workers_share': w,
                'capital': k,
                'profit_rate': profit_rates
            })
            df.to_csv(filepath, index=False)
            print(f"Simulation data saved to {filepath}")
        except ImportError:
            print("Could not save CSV as pandas is not installed.")

    def create_phase_portrait(self, w0: float = 0.6, k0: float = 1.0, 
                             save_path: str = "./figs/figure1.png") -> plt.Figure:
        """
        Create a phase portrait plot of the system dynamics.
        
        Args:
            w0: Initial worker's share (default: 0.6)
            k0: Initial capital stock (default: 1.0)
            save_path: Path to save the figure (default: "./figs/figure1.png")
            
        Returns:
            The matplotlib Figure object
        """
        # Run simulation
        t, y, _ = self.simulate(w0=w0, k0=k0)
        w, k = y[0], y[1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot trajectory
        ax.plot(w, k, '-', color=self.colors['accent1'], alpha=0.9, linewidth=2, label='System path')
        ax.plot(w[0], k[0], 'o', color=self.colors['accent2'], markersize=8, label='Starting point')
        
        # Create vector field
        w_grid, k_grid = np.meshgrid(np.linspace(0.1, 0.9, 20), np.linspace(0.1, 1.2, 20))
        dw = np.zeros_like(w_grid)
        dk = np.zeros_like(k_grid)
        
        for i in range(w_grid.shape[0]):
            for j in range(w_grid.shape[1]):
                dw[i, j], dk[i, j] = self.system(0, [w_grid[i, j], k_grid[i, j]])
        
        # Normalize vectors
        magnitude = np.sqrt(dw**2 + dk**2)
        mask = magnitude > 0
        dw[mask] /= magnitude[mask]
        dk[mask] /= magnitude[mask]
        
        # Add vector field
        ax.quiver(w_grid, k_grid, dw, dk, alpha=0.2, color='gray', 
                 angles='xy', scale_units='xy', scale=None, width=0.002)
        
        # Labels and formatting
        ax.set_xlabel("Worker's Share ($w$)")
        ax.set_ylabel('Capital ($k$)')
        ax.set_xlim(0.05, 0.95)
        ax.set_ylim(0.05, 1.05)
        ax.legend(loc='upper right', framealpha=0.9, bbox_to_anchor=(0.98, 0.98))
        plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
        
        # Save figure
        self._save_figure(fig, save_path)
        return fig

    def create_time_series(self, w0: float = 0.6, k0: float = 1.0, 
                          save_path: str = "./figs/figure2.png") -> plt.Figure:
        """
        Create a three-panel time series plot of system variables.
        
        Args:
            w0: Initial worker's share (default: 0.6)
            k0: Initial capital stock (default: 1.0)
            save_path: Path to save the figure (default: "./figs/figure2.png")
            
        Returns:
            The matplotlib Figure object
        """
        # Run simulation
        t, y, profit_rates = self.simulate(w0=w0, k0=k0)
        w, k = y[0], y[1]
        
        # Create figure with three panels
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 9))
        
        # Panel 1: Worker's share
        ax1.plot(t, w, '-', color=self.colors['worker_share'], linewidth=2.5)
        ax1.set_ylabel("Worker's Share ($w$)")
        
        # Panel 2: Capital accumulation
        ax2.plot(t, k, '-', color=self.colors['capital'], linewidth=2.5)
        ax2.set_ylabel('Capital ($k$)')
        
        # Panel 3: Profit rate with threshold
        ax3.plot(t, profit_rates, '-', color=self.colors['profit'], linewidth=2.5)
        ax3.axhline(y=self.rho, color='gray', linestyle='--', alpha=0.7, 
                   label=f'Investment Threshold ($\\rho={self.rho}$)')
        ax3.set_xlabel('Time ($t$)')
        ax3.set_ylabel('Profit Rate ($r(w,k)$)')
        ax3.legend(loc='lower right')
        
        # Adjust layout
        plt.subplots_adjust(hspace=0.4, left=0.15, right=0.95, top=0.95, bottom=0.1)
        
        # Save figure
        self._save_figure(fig, save_path)
        return fig

    def create_parameter_variation(self, gamma_values: List[float] = [0.2, 0.4, 0.6, 0.8],
                                  save_path: str = "./figs/figure3.png") -> plt.Figure:
        """
        Create a four-panel plot showing system behavior for different parameter values.
        
        Args:
            gamma_values: List of gamma (class struggle) values to plot
            save_path: Path to save the figure (default: "./figs/figure3.png")
            
        Returns:
            The matplotlib Figure object
        """
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        axes = axes.flatten()
        
        # For each gamma value, create a phase portrait
        for i, gamma in enumerate(gamma_values):
            # Create temporary model with this gamma
            temp_model = MarxianModel(gamma=gamma)
            t, y, _ = temp_model.simulate(w0=0.5, k0=1.0)
            w, k = y[0], y[1]
            
            # Plot trajectory
            axes[i].plot(w, k, '-', color=self.colors['accent1'], linewidth=2)
            axes[i].plot(w[0], k[0], 'o', color=self.colors['accent2'], markersize=6)
            
            # Add vector field
            w_grid, k_grid = np.meshgrid(np.linspace(0.4, 0.9, 10), np.linspace(0.2, 1.05, 10))
            dw = np.zeros_like(w_grid)
            dk = np.zeros_like(k_grid)
            
            for j in range(w_grid.shape[0]):
                for l in range(w_grid.shape[1]):
                    dw[j, l], dk[j, l] = temp_model.system(0, [w_grid[j, l], k_grid[j, l]])
            
            # Normalize vectors
            magnitude = np.sqrt(dw**2 + dk**2)
            mask = magnitude > 0
            dw[mask] /= magnitude[mask]
            dk[mask] /= magnitude[mask]
            
            # Add vector field
            axes[i].quiver(w_grid, k_grid, dw, dk, alpha=0.15, color='gray', 
                         angles='xy', scale_units='xy', scale=None, width=0.002)
            
            # Add title and labels
            axes[i].set_title(f'$\\gamma = {gamma}$')
            axes[i].set_xlim(0.4, 0.9)
            axes[i].set_ylim(0.2, 1.05)
            axes[i].set_xlabel("Worker's Share ($w$)")
            axes[i].set_ylabel("Capital ($k$)")
        
        # Adjust layout
        plt.subplots_adjust(wspace=0.3, hspace=0.3, left=0.1, right=0.95, top=0.95, bottom=0.1)
        
        # Save figure
        self._save_figure(fig, save_path)
        return fig

    def create_crisis_comparison(self, save_path: str = "./figs/figure4.png") -> plt.Figure:
        """
        Create a comparison plot of two crisis scenarios.
        
        Args:
            save_path: Path to save the figure (default: "./figs/figure4.png")
            
        Returns:
            The matplotlib Figure object
        """
        # Crisis Scenario 1: High exploitation, low class struggle
        model_expl = MarxianModel(beta=0.8, gamma=0.2)
        t1, y1, _ = model_expl.simulate(w0=0.5, k0=1.0)
        w1, k1 = y1[0], y1[1]
        
        # Crisis Scenario 2: Strong class struggle, high depreciation
        model_class = MarxianModel(gamma=0.8, delta=0.15)
        t2, y2, _ = model_class.simulate(w0=0.5, k0=1.0)
        w2, k2 = y2[0], y2[1]
        
        # Create figure with custom layout
        fig = plt.figure(figsize=(12, 10))
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.08], height_ratios=[1, 1],
                            left=0.1, right=0.9, wspace=0.15, hspace=0.15)
        
        # Create subplots
        ax1 = fig.add_subplot(gs[0, 0])  # Top left - Exploitation time series
        ax2 = fig.add_subplot(gs[0, 1])  # Top right - Exploitation phase portrait
        ax3 = fig.add_subplot(gs[1, 0])  # Bottom left - Class struggle time series
        ax4 = fig.add_subplot(gs[1, 1])  # Bottom right - Class struggle phase portrait
        cax = fig.add_subplot(gs[:, 2])  # Colorbar
        
        # Top left: High Exploitation Time Series
        ax1.plot(t1, w1, '-', color=self.colors['worker_share'], linewidth=2, label="Worker's Share")
        ax1.plot(t1, k1/np.max(k1), '--', color=self.colors['capital'], linewidth=2, label="Capital (Scaled)")
        ax1.set_xlabel("Time ($t$)")
        ax1.set_ylabel("Value ($w$, $k/k_{max}$)")
        ax1.legend(loc='best')
        
        # Top right: High Exploitation Phase Portrait
        scatter1 = ax2.scatter(w1, k1, c=t1, cmap='plasma', s=25, alpha=0.8)
        ax2.plot(w1, k1, '-', color='gray', alpha=0.3, linewidth=1)
        ax2.set_xlabel("Worker's Share ($w$)")
        ax2.set_ylabel("Capital ($k$)")
        ax2.set_xlim(0.43, 0.51)
        
        # Bottom left: Class Struggle Time Series
        ax3.plot(t2, w2, '-', color=self.colors['worker_share'], linewidth=2, label="Worker's Share")
        ax3.plot(t2, k2/np.max(k2), '--', color=self.colors['capital'], linewidth=2, label="Capital (Scaled)")
        ax3.set_xlabel("Time ($t$)")
        ax3.set_ylabel("Value ($w$, $k/k_{max}$)")
        ax3.legend(loc='best')
        
        # Bottom right: Class Struggle Phase Portrait
        scatter2 = ax4.scatter(w2, k2, c=t2, cmap='plasma', s=25, alpha=0.8)
        ax4.plot(w2, k2, '-', color='gray', alpha=0.3, linewidth=1)
        ax4.set_xlabel("Worker's Share ($w$)")
        ax4.set_ylabel("Capital ($k$)")
        ax4.set_xlim(0.48, 0.83)
        
        # Add colorbar
        cbar = plt.colorbar(scatter1, cax=cax)
        cbar.set_label('Time ($t$)', labelpad=15)
        
        # Save figure
        self._save_figure(fig, save_path)
        return fig

    def _save_figure(self, fig: plt.Figure, filepath: str) -> None:
        """
        Save a figure to file, creating directories if needed.
        
        Args:
            fig: Matplotlib figure to save
            filepath: Path to save the figure
        """
        # Create directory if needed
        fig_dir = os.path.dirname(filepath)
        if fig_dir and not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        
        # Save figure
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Figure saved to {filepath}")

def main():
    """Main function to generate all figures and data."""
    # Create output directories
    for directory in ['./figs', './data']:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
    
    # Initialize model with default parameters
    model = MarxianModel(gamma=0.6)
    
    # Generate all figures
    model.create_phase_portrait()
    model.create_time_series()
    model.create_parameter_variation()
    model.create_crisis_comparison()
    
    # Save base case simulation data
    model.simulate(w0=0.6, k0=1.0, save_csv='./data/base_case.csv')
    
    print("All figures and data generated successfully!")

if __name__ == "__main__":
    main()
