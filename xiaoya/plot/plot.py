from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class Plot:

    def __init__(self) -> None:
        pass

    def plot_vis_dataset(
            self,
            data: List,
            save_path: str,
        ) -> None:
        """
        Plot the distribution of the dataset.

        Args:
            data: List.
                List of patients' data.
            save_path: str.
                Path to save the plot.
        """

        for feature in data:
            plt.hist(feature['value'], bins=20, edgecolor='black')
            plt.title(f'{feature["name"]}')
            plt.xlabel('Value')
            plt.ylabel('Count')
            plt.savefig(f'{save_path}/{feature["name"]}_hist.png')

    def plot_feature_importance(
            self,
            data: List,
            save_path: str,
            feature_num: int=10,
        ) -> None:
        """
        Plot the feature importance as a bar chart.

        Args:
            data: List.
                List of patients' data.
            save_path: str.
                Path to save the plot.
            feature_num: int.
                Number of features to plot, default 10.
        """

        importance = dict(zip(data['name'], data['value']))
        importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        names = [item[0] for item in importance[:feature_num]]
        values = [item[1] for item in importance[:feature_num]]

        plt.figure(figsize=(12, 6))
        plt.barh(names, values, color='blue', alpha=0.75)
        plt.xlabel('Importance Index')
        plt.title('Feature Importance')
        plt.xlim(min(values) - 0.001, 1)  # Adjust the x-axis range for better visualization
        plt.gca().invert_yaxis()  # Invert y-axis for top-down display
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance.png')
        

    def plot_risk_curve(
            self,
            data: List,
            save_path: str,
        ) -> None:

        plt.figure(figsize=(6, 6))
        


    def plot_patient_embedding(
            self,
            data: List,
            save_path: str,
            dimension: int = 2,    
        ) -> None:
        """
        Plot patients' embeddings in 2D or 3D space.

        Args:
            data: List.
                List of patients' embeddings.
            save_path: str.
                Path to save the plot.
            dimension: int.
                Dimension of the plot. Must be 2 or 3.
        """

        assert dimension in [2, 3], "dimension must be 2 or 3"

        plt.figure(figsize=(6, 6))
        for patient in data:   
            if dimension == 2: 
                df_subset = pd.DataFrame(data=patient[0]['value'], columns=['2d-one', '2d-two', 'target'])
                sns.scatterplot(
                    x="2d-one",
                    y="2d-two",
                    hue="target",
                    palette=sns.color_palette("coolwarm", as_cmap=True),
                    data=df_subset,
                    legend=False,
                    alpha=0.3,
                )
            elif dimension == 3:
                df_subset = pd.DataFrame(data=patient[0]['value'], columns=['3d-one', '3d-two', '3d-three', 'target'])
                sns.scatterplot(
                    x="3d-one",
                    y="3d-two",
                    hue="target",
                    palette=sns.color_palette("coolwarm", as_cmap=True),
                    data=df_subset,
                    legend=False,
                    alpha=0.3,
                )
        plt.savefig(save_path)
