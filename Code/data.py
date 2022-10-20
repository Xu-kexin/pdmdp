import pandas as pd
import datetime
from config import *

import torch
from torch.utils.data import Dataset

from utils import *

class SalesDataset(Dataset):
    def __init__(self, csv_file_path, preprocessed):
        sep_print("Read Step-LSTM Dataset")
        if preprocessed:
            # Read processed dataset
            print("Loading processed dataset......")
            self.dataframe = pd.read_csv(csv_file_path)
        else:    
            # Preprocess raw dataset
            print("Loading unprocessed dataset......")
            self.dataframe = self.preprocess(pd.read_csv(csv_file_path))
            print("Dataset processed.")
        print("Step-LSTM Dataset Loaded")
    
    def preprocess(self, df):
        df.dropna(axis=0, how='any', inplace=True)  # Drop NaN

        # Filtration
        l = set()
        for index, row in df.iterrows():
            # Extract specific shipping line
            if (row["POL_CDE"] != POL_CDE) or (row["POD_CDE"] != POD_CDE): l.add(index)
            # Keep valide transactions between 08:00 and 18:00
            hour = int(df["WBL_AUD_DTDL"][index].split()[1].split(":")[0])
            if (hour > 18) or (hour < 8): l.add(index)

        df.drop(index=df.index[list(l)], inplace=True)
        df.reset_index(inplace=True)

        # Create a new dataframe to complete the missing values with 0s
        df_c = pd.DataFrame({"DATE":[], "HOUR":[], "NUM":[], "PRICE":[], "EPISODE":[]}) # completed dataframe
        pointer = 0

        # Automatically choose the begin and end date for each shipping route
        begin_date_str, end_date_str = df["WBL_AUD_DTDL"][0].split()[0], df["WBL_AUD_DTDL"][len(df)-1].split()[0]
        yy, mm, dd = begin_date_str.split('/')
        begin_date = datetime.date(int(yy), int(mm), int(dd))
        yy, mm, dd = end_date_str.split('/')
        end_date = datetime.date(int(yy), int(mm), int(dd))

        # Data completion - complete the missing data (missing time, missing price, missing epoch)
        counter = 0
        # Complete zero price to the lastest no-zero price in the past
        current_price = df[df["PRICE"] != 0].reset_index()["PRICE"][0]
        date = begin_date
        delta_date = datetime.timedelta(days=1)
        while (date <= end_date):
            date_str = date.strftime("%Y-%m-%d")
            date += delta_date
            # Data selection (transaction hour from 08:00 to 18:00, interval 2 hours)
            for hour_str in ['8:00', '10:00', '12:00', '14:00', '16:00', '18:00']:
                if (pointer < len(df)):
                    date_ptr, hour_ptr = df["WBL_AUD_DTDL"][pointer].split()
                    yy, mm, dd = date_ptr.split('/')
                    date_ptr = (datetime.date(int(yy), int(mm), int(dd))).strftime("%Y-%m-%d")
                if (date_str != date_ptr) or (hour_str != hour_ptr) or (pointer >= len(df)):
                    temp_row = {"DATE":date_str, "HOUR":hour_str, "NUM":0, "PRICE":current_price, "EPISODE": counter // EPIS_LENGTH}
                    counter += 1
                    df_c = pd.concat([df_c, pd.DataFrame(temp_row,index=[0])], ignore_index=True)
                else:
                    temp_profit, temp_num = 0, 0
                    # Add the sales data (for different voyages) of the same step together
                    while (date_str == date_ptr) and (hour_str == hour_ptr):
                        # Calculate the weighted average price in a step
                        temp_profit += df["PRICE"][pointer] * df["NUM"][pointer]
                        # Calculate the cumulated sold volume in a step
                        temp_num += df["NUM"][pointer]
                        pointer += 1
                        if (pointer >= len(df)): break
                        date_ptr, hour_ptr = df["WBL_AUD_DTDL"][pointer].split()
                        yy, mm, dd = date_ptr.split('/')
                        date_ptr = (datetime.date(int(yy), int(mm), int(dd))).strftime("%Y-%m-%d")
                    # If no product has been sold in this step, current price does not change
                    if temp_profit != 0:
                        current_price = temp_profit / temp_num
                    temp_row = {"DATE":date_str, "HOUR":hour_str, "NUM":temp_num, "PRICE":current_price, "EPISODE": counter // EPIS_LENGTH}
                    counter += 1
                    df_c = pd.concat([df_c,pd.DataFrame(temp_row,index=[0])],ignore_index=True)
        df_c.to_csv(UNNORMALIZED_DATA_PATH, index=False)
        df_c = self.normalize_dataframe(df_c)
        df_c.to_csv(PROCESSED_OUTPUT_PATH, index=False)
        return df_c
        
    def __len__(self):
        return len(self.dataframe) - T
    
    def __getitem__(self, index):
        # Return the sample of certain index
        if torch.is_tensor(index):
            index = index.tolist()
        
        np_data = self.dataframe[["NUM", "PRICE", "EPISODE"]].iloc[range(index,index+T+1)].values
        # Mask the last vol, since we mean to predict it with step-wise LSTM
        np_data[-1, 0] = -1
        np_label = self.dataframe[["NUM"]].iloc[[index+T]].values[0]
        torch_data = torch.from_numpy(np_data).float()
        torch_label = torch.from_numpy(np_label).float()
        
        return torch_data, torch_label
    
    def normalize_dataframe(self, df):
        # Normalize the price and volume dataset 
        price_mean = df["PRICE"].mean()
        price_std = df["PRICE"].std()
        df["PRICE"] = (df["PRICE"] - price_mean) / price_std
        
        vol_mean = df["NUM"].mean()
        vol_std = df["NUM"].std()
        df["NUM"] = (df["NUM"] - vol_mean) / vol_std
        
        # Save price & volume statistics to csv file
        df_stats = pd.DataFrame({"price_mean":[price_mean], "price_std":[price_std], "vol_mean":[vol_mean], "vol_std":[vol_std]})
        df_stats.to_csv(STATS_PATH, index=False)
        return df
      
class RepDataset(Dataset):
    def __init__(self, representation):
        self.repmatrix = representation
        sep_print("Episode-LSTM Dataset Loaded")
        
    def __len__(self):
        return self.repmatrix.shape[0] - L
    
    def __getitem__(self, index):
        # Return the sample of certain index
        return self.repmatrix[index:index+L, :], self.repmatrix[index+L]
    
class EpsSaleDataset(Dataset):
    def __init__(self, sales_data):
        # [Vol, Price]
        # sep_print("Episode Sales Dataset Loaded")
        
        # Normalize sales dataset
        stats = pd.read_csv(STATS_PATH)
        sales_data[:, 0] = (sales_data[:, 0] - stats['vol_mean'].to_numpy())/stats['vol_std'].to_numpy()
        sales_data[:, 1] = (sales_data[:, 1] - stats['price_mean'].to_numpy())/stats['price_std'].to_numpy()
        self.sales_data = sales_data
        
    def __len__(self):
        return self.sales_data.shape[0] - T - 1
    
    def __getitem__(self, index):
        np_data = self.sales_data[index:index+T+1]
        np_label = np.expand_dims(np.array(self.sales_data[index+T][0]),0)
        return torch.FloatTensor(np_data), torch.FloatTensor(np_label)
