import pandas as pd
import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
import warnings

sns.set_theme('paper')

class FeatureExplorer:
    def __init__(self, dataframe, target, goal, outlier_method='quantile'):
        self.df = dataframe
        self.target = target
        self.numerical_features = [col for col in dataframe.columns if dataframe[col].dtype in ['int64', 'float64'] and col != target]
        self.categorical_features = [col for col in dataframe.columns if dataframe[col].dtype == 'object']
        self.features = self.numerical_features + self.categorical_features
        self.transformations = {'None': lambda x: x, 'Log': lambda x: np.log(x.clip(lower=0.01)), 'Sqrt': np.sqrt, 'Exp': np.exp}
        self.goal = goal
        self.outlier_method = outlier_method
        self.final_features = {feature: {'final': False, 'transformation': 'None'} for feature in self.features}

    def update_display_data(self, feature, transformation, outlier_percent):
        transformed_feature = self.transformations[transformation](self.df[feature])
        self.df['transformed'] = transformed_feature

        if self.outlier_method == 'quantile':
            lower_bound = transformed_feature.quantile(outlier_percent)
            upper_bound = transformed_feature.quantile(1 - outlier_percent)
            self.display_data = self.df[(transformed_feature >= lower_bound) & (transformed_feature <= upper_bound)]
        elif self.outlier_method == 'z':
            mean = transformed_feature.mean()
            std_dev = transformed_feature.std()
            z_scores = (transformed_feature - mean) / std_dev
            self.display_data = self.df[(z_scores > -outlier_percent) & (z_scores < outlier_percent)]
        else:
            self.display_data = self.df


    def explore_features(self):
        feature_dropdown = widgets.Dropdown(options=self.features, description='Feature:')
        transformation_dropdown = widgets.Dropdown(options=list(self.transformations.keys()), description='Transform:')
        outlier_slider = widgets.FloatSlider(value=0, min=0, max=0.25, step=0.01, description='Outlier Threshold:')
        feature_checkbox = widgets.Checkbox(value=False, description='Select Feature')
        output = widgets.Output()

        def update_feature_display(*args):
            feature = feature_dropdown.value
            transformation = transformation_dropdown.value
            outlier_threshold = outlier_slider.value
            feature_checkbox.value = self.final_features[feature]['final']
            transformation_dropdown.value = self.final_features[feature]['transformation']

            with output:
                clear_output(wait=True)
                self.final_features[feature]['transformation'] = transformation
                self.final_features[feature]['final'] = feature_checkbox.value
                self.update_display_data(feature, transformation, outlier_threshold)

                if feature in self.numerical_features:
                    self.plot_features()
                elif feature in self.categorical_features:
                    self.plot_categorical_features(feature)

        feature_dropdown.observe(update_feature_display, names='value')
        transformation_dropdown.observe(update_feature_display, names='value')
        outlier_slider.observe(update_feature_display, names='value')
        feature_checkbox.observe(lambda change: self.update_final_status(feature_dropdown.value, change.new), names='value')

        display(widgets.VBox([feature_dropdown, transformation_dropdown, outlier_slider, feature_checkbox, output]))
        update_feature_display()  # Initial call to display data

    def update_final_status(self, feature, is_final):
        self.final_features[feature]['final'] = is_final

    def get_final_feature_details(self):
        """
        Returns a copy of the final_features dictionary to prevent direct manipulation.
        """
        return self.final_features.copy()

    def finalize_features(self):
        """
        Applies the selected transformations to the features marked as final and compiles them into a new DataFrame.
        This method does not consider the outlier threshold, ensuring all data is transformed as specified.
        """
        final_df = pd.DataFrame(index=self.df.index)
        for feature, details in self.final_features.items():
            if details['final']:
                transformed_data = self.transformations[details['transformation']](self.df[feature])
                final_df[feature] = transformed_data
        return final_df
    
    def plot_features(self):
        if 'transformed' not in self.display_data.columns:
            warnings.warn("No transformed data available for plotting.")
            return

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        sns.histplot(data=self.display_data, x='transformed', hue=self.target, element="step", ax=axes[0])
        sns.violinplot(x=self.target, y='transformed', data=self.display_data, ax=axes[1], hue=self.target)
        plt.tight_layout()
        plt.show()
        self.display_correlation_scorecard('transformed')


    def plot_categorical_features(self, feature):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # countplot
        sns.countplot(x=feature, data=self.df, ax=axes[0])
        axes[0].set_title(f'Count Plot of {feature}')
        axes[0].set_ylabel('Counts')
        axes[0].set_xlabel(feature)
        axes[0].tick_params(axis='x', rotation=45)

        # barplot
        sns.barplot(x=feature, y=self.target, data=self.df, ax=axes[1], estimator=np.mean, ci=None)
        axes[1].set_title(f'Mean of {self.target} per {feature}')
        axes[1].set_ylabel(f'Mean of {self.target}')
        axes[1].set_xlabel(feature)
        axes[1].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.show()
        # cramers v heatmap
        self.display_cramers_v_scorecard(feature)


    def display_correlation_scorecard(self, feature):
        correlation_value = self.df[[feature, self.target]].corr().iloc[0, 1]
        correlation_matrix = pd.DataFrame([[correlation_value]], columns=[self.target], index=[feature])
        plt.figure(figsize=(5, 1))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap=sns.color_palette("Reds", as_cmap=True), vmin=0, vmax=1)
        plt.title(f'Correlation between {feature} and {self.target}')
        plt.show()

    @staticmethod
    def z_score_threshold(value):
        return 3 - (value - 0.01) * (1.5 / 0.24)

    def cramers_v(self, x, y):
        confusion_matrix = pd.crosstab(x, y)
        chi2 = ss.chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.sum().sum()
        phi2 = chi2/n
        r, k = confusion_matrix.shape
        phi2_corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
        r_corr = r - ((r-1)**2)/(n-1)
        k_corr = k - ((k-1)**2)/(n-1)
        return np.sqrt(phi2_corr / min((k_corr-1), (r_corr-1)))

    def display_cramers_v_scorecard(self, feature):
        cramers_v_value = self.cramers_v(self.df[feature], self.df[self.target])
        cramers_v_matrix = pd.DataFrame([[cramers_v_value]], columns=[self.target], index=[feature])
        plt.figure(figsize=(5, 1))
        sns.heatmap(cramers_v_matrix, annot=True, fmt=".2f", cmap=sns.color_palette("Reds", as_cmap=True), vmin=0, vmax=1)
        plt.title(f"Cramer's V between {feature} and {self.target}")
        plt.show()
