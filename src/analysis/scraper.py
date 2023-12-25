import os
import fire
import requests
import numpy as np
import pandas as pd
from ftplib import FTP
from pathlib import Path
from datetime import datetime, timedelta
from pvlib.iotools import read_surfrad
from typing import Optional

from src.config import Config
from src.base.logger import get_console_logger

logger = get_console_logger()

class Logger(object):
    info     = print
    warning  = print
    error    = print
    critical = print 
    

class Scraper(Config):
    def __init__(self, logger=Logger()):
        self.logger = logger
        
        
    def download_ohlc_data_from_coinbase(
        self,
        product_id: Optional[str] = Config.SCRAPER_CONFIG["CRYPTO_PRODUCT_ID"],
        from_day: Optional[str] = Config.SCRAPER_CONFIG["CRYPTO_FROM_DAY"],
        to_day: Optional[str] = Config.SCRAPER_CONFIG["CRYPTO_TO_DAY"],
        ):
        """Downloads historical candles from Coinbase API and saves data to disk
        Reference: https://docs.cloud.coinbase.com/exchange/reference/exchangerestapi_getproductcandles

        Args:
            product_id (Optional[str], optional): _description_. Defaults to Config.SCRAPER_CONFIG["CRYPTO_PRODUCT_ID"].
            from_day (Optional[str], optional): _description_. Defaults to Config.SCRAPER_CONFIG["CRYPTO_FROM_DAY"].
            to_day (Optional[str], optional): _description_. Defaults to Config.SCRAPER_CONFIG["CRYPTO_TO_DAY"].
        """
        days = pd.date_range(start=from_day, end=to_day, freq="1D")
        days = [day.strftime("%Y-%m-%d") for day in days]

        data = pd.DataFrame()

        if not Path(os.path.join(Config.FILES["RAW_DATA"], "crypto")).exists():
            self.logger.info("Create directory for 'crypto'")
            Path(os.path.join(Config.FILES["RAW_DATA"], "crypto")).mkdir(parents=True)
    
        for day in days:
            filename = Path(os.path.join(self.FILES["RAW_DATA"], "crypto", f"{day}.parquet"))
            if filename.exists():
                self.logger.info(f"File {filename} already exists, skipping")
                data_one_day = pd.read_parquet(filename)
            else:
                self.logger.info(f"Downloading data for {day}")
                data_one_day = self.download_data_for_one_day(product_id, day)
                data_one_day.to_parquet(filename, index=False)
                
            data = pd.concat([data, data_one_day])
            
        data.to_parquet(os.path.join(self.FILES["PREPROCESS_DATA"], "crypto", f"{self.FILES['PREPROCESS_CRYPTO_FILE']}.parquet"), index=False)
        
        return os.path.join(self.FILES["PREPROCESS_DATA"], "crypto", f"{self.FILES['PREPROCESS_CRYPTO_FILE']}.parquet")


    def download_data_for_one_day(self, product_id: str, day: str) -> pd.DataFrame:
        """Download one day of data and returns pandas dataframe

        Args:
            product_id (str): product_id to download
            day (str): day of data to download in format YYYY-MM-DD

        Returns:
            pd.DataFrame: _description_
        """
        start = f"{day}T00:00:00"
        end = (datetime.strptime(day, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")
        end = f"{end}T00:00:00"
        
        URL = f"https://api.exchange.coinbase.com/products/{product_id}/candles?start={start}&end={end}&granularity=3600"
        r = requests.get(URL)
        data = r.json()
        
        return pd.DataFrame(data, columns=["time", "low", "high", "open", "close", "volume"])
        
        
    def noaa_scraper(self):
        """Downloads data from NOAA FTP server and saves data to data/preprocess_data
        """
        for year in self.ANALYSIS_CONFIG["YEAR_LIST"]:
            for location in self.ANALYSIS_CONFIG["LOCATION_LIST"]:
                source = f"data/radiation/surfrad/{location}/{year}"
                destination = os.path.join(self.FILES["RAW_DATA"], "radiation", location, year)
                if not os.path.exists(destination):
                    os.makedirs(destination)

                start = datetime.now()
                ftp = FTP("ftp.gml.noaa.gov")
                ftp.login()
                ftp.cwd(source)
                # ftp.retrlines("LIST") Enable to view what is inside the server

                ftp.nlst()
                files = ftp.nlst()
                self.logger.info("Downloading datasets from {} at {}...".format(location, year))
                for file in files:
                    self.logger.info("Downloading..." + file)
                    ftp.retrbinary(f"RETR {file}", open(os.path.join(destination, file), "wb").write)

                ftp.close()
                self.logger.info("\nftp connection is now closed")

                end = datetime.now()
                diff = end - start
                self.logger.info("\nAll files download for " + str(diff.seconds) + "s")
                
    
    def get_stock_data(self,
                       ticker: str=Config.SCRAPER_CONFIG["STOCK_TICKER"], 
                       api_token: str=Config.SCRAPER_CONFIG["ALPHA_VANTAGE_API"]) -> pd.DataFrame:
        """
        Get historical stock data from www.alphavantage.com
        
        Documentation: https://www.alphavantage.co/documentation/#symbolsearch
        """
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={api_token}'
        r = requests.get(url)
        data = r.json()
        
        df = pd.DataFrame(data["Time Series (Daily)"]).T
        
        if not Path(os.path.join(Config.FILES["RAW_DATA"], "stocks")).exists():
            self.logger.info("Create directory for 'stocks'")
            Path(os.path.join(Config.FILES["RAW_DATA"], "stocks")).mkdir(parents=True)
        
        df.to_parquet(os.path.join(Config.FILES["RAW_DATA"], "stocks", f"{Config.FILES['PREPROCESS_STOCK_FILE']}.parquet"), index=False)
        
        try:
            df_preprocess = (
                df.rename(columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume"
                    })
                .reset_index()
                .rename(columns={"index": "time"})
                )
        except :
            raise self.logger.error("Error: {}".format(data["Information"]))
            
        if not Path(os.path.join(Config.FILES["PREPROCESS_DATA"], "stocks")).exists():
            self.logger.info("Create directory for 'stocks'")
            Path(os.path.join(Config.FILES["PREPROCESS_DATA"], "stocks")).mkdir(parents=True)

        df_preprocess.to_parquet(os.path.join(Config.FILES["PREPROCESS_DATA"], "stocks", f"{Config.FILES['PREPROCESS_STOCK_FILE']}.parquet"), index=False)
        
        self.logger.info("Successfully retrieved stock data for {} - {}".format(ticker, df.shape[0]))
        
        return df_preprocess


if __name__== "__main__":
    scraper = Scraper(product_id = Config.SCRAPER_CONFIG["CRYPTO_PRODUCT_ID"],
                      from_day = Config.SCRAPER_CONFIG["CRYPTO_FROM_DAY"],
                      to_day = Config.SCRAPER_CONFIG["CRYPTO_TO_DAY"],
                      )
    fire.Fire(scraper.download_ohlc_data_from_coinbase())