import fire
import seaborn as sns
import matplotlib.pyplot as plt

from src.config import Config
from src.analysis.scraper import Scraper


def stock_line_plot(x_var, y_var, data):
    fig, ax = plt.subplots(figsize=(20, 5))
    
    num_data_points = len(data)
    display_date_range = "from {} to {}".format(data[x_var][0].strftime("%Y-%m-%d"), 
                                                data[x_var][num_data_points-1].strftime("%Y-%m-%d")
                                                )
    
    sns.lineplot(x=x_var, y=y_var, data=data, ax=ax)
    ax.set_title("Daily close price for " + Config.SCRAPER_CONFIG["STOCK_TICKER"] + ", " + display_date_range)
    plt.grid(which='major', axis='y', linestyle='--')
    
    return plt.show()


if __name__== "__main__":
    scraper = Scraper()
    fire.Fire(scraper.get_stock_data(Config.SCRAPER_CONFIG["STOCK_TICKER"], 
                                     Config.SCRAPER_CONFIG["ALPHA_VANTAGE_API"]
                                     )
              )