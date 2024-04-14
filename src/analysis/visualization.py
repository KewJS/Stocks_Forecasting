import numpy as np
import pandas as pd
from typing import Any
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objs as go
from plotly.offline import iplot
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf


class Visualization:
    def __init__(self):
        pass
    
    
    def line_plot(self, df: pd.DataFrame, x: str, y: str, title: Any=None) -> plt.figure:
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(12, 3))
        sns.lineplot(data=df, x=x, y=y)
        ax.set_title(title, fontsize=12)
        
        return fig
    
    
    def missing_cum_plot(self, df: pd.DataFrame, col: str) -> plt.figure:
        fig, axs = plt.subplots(2, 1, figsize=(18,5))
        axs[0].plot(df[col], "-", lw=1)
        axs[0].set_title(col)
        axs[1].plot(np.cumsum(df[col].isna()), "-")
        axs[1].set_title(f"Cumulative count of missing {col} values")
        plt.tight_layout()
        
        return fig
    
    
    def ts_components_plot(self, daily_comp, weekly_comp, yearly_comp, city=None):
        f, axes = plt.subplots(6, 1, figsize=(15,16))

        # # setting figure title and adjusting title position and size
        plt.suptitle("Summary of Seasonal Decomposition of {}".format(city), y=0.92, fontsize=15, fontweight="bold")
        plt.subplots_adjust(wspace=0.5, hspace=0.4)
        
        # # plotting trend component
        axes[0].plot(daily_comp.observed)
        axes[0].set_title("Observed component", fontdict={"fontsize": 12})

        # # plotting trend component
        axes[1].plot(daily_comp.trend)
        axes[1].set_title("Trend component", fontdict={"fontsize": 12})

        # # plotting daily seasonal component
        axes[2].plot(daily_comp.seasonal)
        axes[2].set_title("Daily seasonal component", fontdict={"fontsize": 12})

        # # plotting weekly seasonal component
        axes[3].plot(weekly_comp.seasonal)
        axes[3].set_title("Weekly seasonal component", fontdict={"fontsize": 12})

        # # plotting yearly seasonality
        axes[4].plot(yearly_comp.seasonal)
        axes[4].set_title("Yearly seasonal component", fontdict={"fontsize": 12})

        # # plotting residual of decomposition
        axes[5].plot(daily_comp.resid)
        axes[5].set_title("Residual component", fontdict={"fontsize": 12})

        for a in axes:
            a.set_ylabel("MW")
            
        return fig
    

    def plotly_line_plot(self, df, groupby_col, col):
        fig = go.Figure()

        # Iterate over unique groups
        for group in df[groupby_col].unique():
            group_df = df[df[groupby_col] == group]
            fig.add_trace(go.Scatter(
                x=group_df.index,
                y=group_df[col],
                name=f"{group}",
                mode="lines"
            ))

        # Add dropdown menu
        buttons = []
        for group in df[groupby_col].unique():
            visible = [True if g == group else False for g in df[groupby_col].unique()]
            buttons.append(dict(
                label=f"{group}",
                method="update",
                args=[{"visible": visible}, {"title": f"Group {group}"}]
            ))

        fig.update_layout(
            updatemenus=[go.layout.Updatemenu(
                buttons=buttons,
                active=0,
                direction="down",
                x=0.2,
                y=1.2,
            )],
            width=1000,
            height=400
        )

        fig.update_layout(
            title="<b>Line Plot of '{}'</b>".format(col),
            title_x=0.5, title_y=0.9,
            xaxis_title="datetime",
            yaxis_title="value"
        )

        # Show the plot
        return fig.show()
    
    
    def single_lineplot(self, df, title):
        data = []
        
        value = go.Scatter(
            x=df.index,
            y=df.netsolar,
            mode="lines",
            name="netsolar",
            marker=dict(),
            text=df.index,
            line=dict(color="rgba(0,0,0, 0.3)"),
        )
        data.append(value)

        layout = dict(
            title=title,
            xaxis=dict(title="Date", ticklen=5, zeroline=False),
            yaxis=dict(title="Netsolar", ticklen=5, zeroline=False),
        )

        fig = dict(data=data, layout=layout)
        iplot(fig)
    
    
    def roll_mean_plot(self, train_dict, roll_mean_d):    
        fig = go.Figure()
        
        for city in train_dict["city"].unique():
            fig.add_trace(go.Scatter(x=train_dict[train_dict["city"]==city].index,
                                     y=roll_mean_d[city]["netsolar"],
                                     name=city),
                        )

        fig.update_layout(
            title="<b>10 Days Rolling Average of Net Solar Radiation</b>",
            title_x=0.5, title_y=0.9,
            )
                    
        return fig.show()
    
    
    def pacf_plot(self, x_var, lags):
        fig, ax = plt.subplots(figsize=(13,3))
        plot_pacf(x_var, lags=lags, ax=ax)
        
        return plt.show()


    def acf_plot(self, x_var, lags):
        fig, ax = plt.subplots(figsize=(13,3))
        plot_acf(x_var, lags=lags, ax=ax)
        
        return plt.show()