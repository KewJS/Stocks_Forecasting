import fire

from src.config import Config
from src.analysis.scraper import Scraper

if __name__== "__main__":
    scraper = Scraper()
    fire.Fire(scraper.get_stock_data(Config.SCRAPER_CONFIG["STOCK_TICKER"], 
                                     Config.SCRAPER_CONFIG["ALPHA_VANTAGE_API"]
                                     )
              )