import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings

sns.set_theme('paper')

class BinaryClassificationFeatureExplorer:
    def __init__(self, dataframe, target):
        self.df = dataframe
        self.target = target
        self.numerical_features = [col for col in self.df.columns if self.df[col].dtype in ['int64', 'float64'] and col != self.target]
        self.transformations = {
            'None': lambda x: x,
            'Log': lambda x: np.log(x.clip(lower=0.01)),  
            'Sqrt': np.sqrt,
            'Exp': np.exp
        }
        self.display_data = self.df 

    def update_display_data(self, feature, transformation, outlier_percent):
        try:
            transformed_feature = self.transformations[transformation](self.df[feature])
            self.df['transformed'] = transformed_feature  

            
            lower_bound = transformed_feature.quantile(outlier_percent)
            upper_bound = transformed_feature.quantile(1 - outlier_percent)
            
            self.display_data = self.df[(transformed_feature >= lower_bound) & (transformed_feature <= upper_bound)]
        except Exception as e:
            warnings.warn(str(e))
            self.display_data = self.df 

    def plot_features(self):
        if 'transformed' not in self.display_data:
            warnings.warn("No transformed data available for plotting.")
            return
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))

        # Plot histogram
        sns.histplot(data=self.display_data, x='transformed', hue=self.target, element="step", stat="density", common_norm=False, ax=axes[0])
        axes[0].set_title('Histogram of Transformed Feature')
        axes[0].set_xlabel('Transformed Value')
        axes[0].set_ylabel('Density')

        # Plot boxplot
        sns.boxplot(x=self.target, y='transformed', data=self.display_data, ax=axes[1], hue = self.target)
        axes[1].set_title('Box Plot of Transformed Feature')
        axes[1].set_xlabel('Target')
        axes[1].set_ylabel('Transformed Value')

        plt.tight_layout()
        plt.show()

    def explore_numerical_variables(self):
        feature_dropdown = widgets.Dropdown(options=self.numerical_features, description='Feature:', index=0)
        transformation_dropdown = widgets.Dropdown(options=list(self.transformations.keys()), description='Transformation:', index=0)
        outlier_slider = widgets.FloatSlider(value=0.00, min=0, max=0.25, step=0.05, description='Outlier Percent:')
        output = widgets.Output()

        def update_plot(*args):
            self.update_display_data(feature_dropdown.value, transformation_dropdown.value, outlier_slider.value)
            with output:
                clear_output(wait=True)
                self.plot_features()

        feature_dropdown.observe(update_plot, names='value')
        transformation_dropdown.observe(update_plot, names='value')
        outlier_slider.observe(update_plot, names='value')
        display(widgets.VBox([feature_dropdown, transformation_dropdown, outlier_slider]), output)
        update_plot() 