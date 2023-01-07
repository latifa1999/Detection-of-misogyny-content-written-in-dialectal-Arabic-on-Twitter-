import streamlit as st
import pandas as pd
import plotly.express as px
from plotly import graph_objects as go
import numpy as np
import pickle
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import plotly.express as px
import pyarabic.araby as araby
import re
import torch
import pytorch_lightning as pl
from transformers import AutoTokenizer, AutoModel
from torch import nn,optim
from torch.utils.data import Dataset, DataLoader
from nltk.stem.isri import ISRIStemmer
import re
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
import folium
from streamlit_folium import folium_static



def geolocate(country):
    try:
        # Geolocate the center of the country
        loc = geolocator.geocode(country)
        # And return latitude and longitude
        return (loc.latitude, loc.longitude)
    except:
        # Return missing value
        return np.nan

geolocator = Nominatim(user_agent="achrafs758@gmail.com")

class ArabicDataset(Dataset):
    def __init__(self,data,max_len,model_type="Mini"):
        super().__init__()
        self.labels = data["label"].values
        self.texts = data["text"].values
        self.max_len = max_len
        model = {"Mini": "asafaya/bert-mini-arabic",
                "Medium": "asafaya/bert-medium-arabic",
                "Base": "asafaya/bert-base-arabic",
                "Large": "asafaya/bert-large-arabic"}
        self.tokenizer = AutoTokenizer.from_pretrained(model[model_type])
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        text = " ".join(self.texts[idx].split())
        label = self.labels[idx]
        inputs = self.tokenizer(text,padding='max_length',
                                max_length=self.max_len,truncation=True,return_tensors="pt")
        #input_ids,token_type_ids,attention_mask
        return {
            "inputs":{"input_ids":inputs["input_ids"][0],
                      "token_type_ids":inputs["token_type_ids"][0],
                      "attention_mask":inputs["attention_mask"][0],
                     },
            "labels": torch.tensor(label,dtype=torch.long) 
        }
        

        

        
class ArabicDataModule(pl.LightningDataModule):
    def __init__(self,train_path,val_path,batch_size=12,max_len=100,model_type="Mini"):
        super().__init__()
        self.train_path,self.val_path= train_path,val_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.model_type = model_type
    
    def setup(self,stage=None):
        train = pd.read_csv(self.train_path)
        val = pd.read_csv(self.val_path)
        self.train_dataset = ArabicDataset(data=train,max_len=self.max_len,model_type=self.model_type)
        self.val_dataset = ArabicDataset(data=val,max_len=self.max_len,model_type=self.model_type)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset,batch_size=self.batch_size,shuffle=False,num_workers=4)

class ArabicBertModel(pl.LightningModule):
    def __init__(self,model_type="Mini"):
        super().__init__()
        model = {"Mini": ("asafaya/bert-mini-arabic",256),
                "Medium": ("asafaya/bert-medium-arabic",512),
                "Base": ("asafaya/bert-base-arabic",768),
                "Large": ("asafaya/bert-large-arabic",1024)}
        self.bert_model = AutoModel.from_pretrained(model[model_type][0])
        self.fc = nn.Linear(model[model_type][1],18)
    
    def forward(self,inputs):
        out = self.bert_model(**inputs)#inputs["input_ids"],inputs["token_type_ids"],inputs["attention_mask"])
        pooler = out[1]
        out = self.fc(pooler)
        return out
    
    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=0.0001)
    
    def criterion(self,output,target):
        return nn.CrossEntropyLoss()(output,target)
    
    #TODO: adding metrics
    def training_step(self,batch,batch_idx):
        x,y = batch["inputs"],batch["labels"]
        out = self(x)
        loss = self.criterion(out,y)
        return loss
    
    def validation_step(self,batch,batch_idx):
        x,y = batch["inputs"],batch["labels"]
        out = self(x)
        loss = self.criterion(out,y)
        return loss

from tqdm.auto import tqdm
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load('/kaggle/input/my-model/model.pth', map_location=device)
model.to(device)

stemmer = ISRIStemmer()

def processDocument(doc, stemmer): 

    #Replace @username with empty string
    doc = re.sub(r'@[^\s]+', ' ', doc)
    doc = re.sub(r'_', ' ', doc)
    doc = re.sub(r'\n', ' ', doc)
    doc = re.sub(r'\r', ' ', doc)
    doc = re.sub(r'Ù…Ø³ØªØ®Ø¯Ù…@', ' ', doc)
    doc = re.sub(r'[a-z,A-Z]', '', doc)
    doc = re.sub(r'\d', '', doc)
    #Convert www.* or https?://* to " "
    doc = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',doc)
    #Replace #word with word
    doc = re.sub(r'#([^\s]+)', r'\1', doc)
    # remove punctuations
    # normalize the tweet
#     doc= normalize_arabic(doc)
    # remove repeated letters
#     doc=remove_repeating_char(doc)
    #stemming
#     doc = stemmer.stem(doc)
   
    return doc

def remove_hashtag(df, col = 'text'):
    for letter in r'#.][!XR':
        df[col] = df[col].astype(str).str.replace(letter,'', regex=True)

def normalize_arabic(text):
    text = re.sub("[Ã˜Â¥Ã˜Â£Ã˜Â¢Ã˜Â§]", "Ã˜Â§", text)
    text = re.sub("Ã™â€°", "Ã™Å ", text)
    text = re.sub("Ã˜Â©", "Ã™â€¡", text)
    text = re.sub("ÃšÂ¯", "Ã™Æ’", text)
    text = re.sub(r'@[^\s]+', ' ', text)
    text = re.sub(r'\r', ' ', text)
    text = re.sub(r'[a-z,A-Z]', '', text)
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))',' ',text)
    text = re.sub(r'\d', '', text)
    return text

def removeWeirdChars(text):
    weirdPatterns = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u'\U00010000-\U0010ffff'
                               u"\u200d"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\u3030"
                               u"\ufe0f"
                               u"\u2069"
                               u"\u2066"
                               u"\u200c"
                               u"\u2068"
                               u"\u2067"
                               "]+", flags=re.UNICODE)
    return weirdPatterns.sub(r'', text)
def remove_repeating_char(text):
    return re.sub(r'(.)\1+', r'\1', text)

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)



st.set_page_config(layout="wide")

with st.sidebar:
    st.title("Mysoginic tweets detection Web Application")
    st.write("1. Upload your dataset.")
    st.write("2. View a summary of your dataset.")
    st.write("3. Clean and preprocessing your data.")
    st.write("4. Prediction of mysoginic tweets.")
    st.write("5. Visualization of the results.")
    st.write("6. Take a look on results.")


df = None
if 'df' not in st.session_state:
    st.session_state.df = df
else:
    df = st.session_state.df

st.session_state.eval = False

with st.expander("Guidlines", expanded=True):
    st.write("1. Dataframe should be in csv format.")
    st.write("2. The dataframe should contain a column named 'text' which contains the tweets.")
    st.write("3. The dataframe should contain a column named 'Country' .")


with st.expander("Upload your data", expanded=True):
    file = st.file_uploader("Upload a dataset", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
        st.write(df)
        st.write("Dataset shape:", df.shape)

with st.expander("Data Summary"):
    if df is not None:
        st.write(df.describe())
        st.write("Dataset shape:", df.shape)
        st.write("Number of Nan values across columns:", df.isna().sum())
        st.write("Total number of Nan values:", df.isna().sum().sum())


with st.expander("Data Cleaning"):
    if df is not None:
        st.write("Select cleaning options")
        drop_na0 = st.checkbox("Drop all rows with Nan values")
        drop_duplicates = st.checkbox("Remove duplicates")
        drop_colmuns = st.checkbox("Drop specific columns")
        if drop_colmuns:
            columns = st.multiselect("Select columns to drop", df.columns)
        if st.button("Apply data cleaning"):
            if drop_na0:
                df = df.dropna(axis=0)
            if drop_duplicates:
                df = df.drop_duplicates()
            if drop_colmuns:
                df = df.drop(columns, axis=1)
            remove_hashtag(df)
            df["text"] = df["text"].apply(lambda x: remove_repeating_char(x))
            df["text"] = df['text'].apply(lambda x: araby.strip_diacritics(x))
            df["text"] = df['text'].apply(lambda x: normalize_arabic(x))
            df["text"] = df["text"].apply(lambda x: removeWeirdChars(x))
            df["text"] = df["text"].apply(lambda x: processDocument(x,stemmer))
                            
            st.session_state.df = df
            st.write(df)
            st.write("Dataset shape:", df.shape)
            df['label']=-1
            df.to_csv('Final_Data.csv',index=False)
            st.write("Total number of Nan values remaining:", df.isna().sum().sum())


with st.expander("Prediction"):
    if df is not None:
            if model:
                preds = []
                real_values = []

                load = ArabicDataModule(train_path="/kaggle/working/Final_Data.csv",val_path = "/kaggle/working/Final_Data.csv",batch_size=512,max_len=60)
                load.setup()
                test_dataloader = load.val_dataloader()

                progress_bar = tqdm(range(len(test_dataloader)))

                model.eval()
                for batch in test_dataloader:    
                    x,y = batch["inputs"],batch["labels"]
                    inp = {k: v.to(device) for k, v in x.items()}
                    
                    with torch.no_grad():
                        outputs = model(inp)

                    predictions = torch.argmax(outputs, dim=-1)
                    
                    preds.extend(predictions)
                    real_values.extend(y)

                    progress_bar.update()
 
                preds = torch.stack(preds).cpu()
                real_values = torch.stack(real_values).cpu()
                df=pd.read_csv('/kaggle/working/Final_Data.csv')
                df=df.drop(columns=['label'])
                df["misogyny"]=preds.tolist()

                df['misogyny']=df['misogyny'].apply(lambda x:'misogyny' if x==0 else 'none')
                st.session_state.df = df
                col5, col6 = st.columns((1,1))
                with col5:
                    st.subheader("Dataframe")
                    st.write(df)
                with col6:
                    st.subheader("Pie figure")
                    words = []
                    labels = df['misogyny'].value_counts().keys()
                    values = df['misogyny'].value_counts()
    # Plot pie chart 
                    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3, textinfo='label+percent', textfont_size=15)])
                    fig.layout.update(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig)

with st.expander("Visualization"):
    if df is not None:
            if geolocator:
                df2 = df.groupby('Country')['misogyny'].apply(lambda x: (x=='misogyny').sum()).reset_index(name='misogyny')
                df2['misogyny'] = (df2['misogyny']/df2['misogyny'].sum())*100
                a = df.groupby('Country')['misogyny'].apply(lambda x: (x=='none').sum()).reset_index(name='none')
                df2['none'] = (a['none']/a['none'].sum())*100
                df2['Latitude']=df2['Country'].apply(geolocate).apply(lambda x: x[0])
                df2['Longitude']=df2['Country'].apply(geolocate).apply(lambda x: x[1])

                # add marker one by one on the map
                col7, col8 = st.columns((1,1))
                with col7:
                    st.subheader("Map")
                    f = folium.Figure(width=900, height=1000)
            # Make an empty map
                    m = folium.Map(location=[27,48], tiles="OpenStreetMap", zoom_start=5).add_to(f)
                    for i in range(0,len(df2)):
                        folium.Circle(
                            location=[df2.iloc[i]['Latitude'], df2.iloc[i]['Longitude']],
                            popup=str(round(df2.iloc[i]['misogyny'],2))+"%",
                            radius=float(df2.iloc[i]['misogyny'])*4000,
                            color='crimson',
                            fill=True,
                            fill_color='crimson'
                        ).add_to(m)

                    folium_static(m)
                with col8:
                    st.subheader("Bar Plot: Percentage of misogynistic tweets per country")
                    fig = px.bar(df2, x="Country", y=["misogyny", "none"], barmode='group', height=600, width=600, color_discrete_sequence=['crimson', 'lightseagreen'])
                    fig.layout.update(margin=dict(l=0, r=0, t=0, b=0))
                    st.plotly_chart(fig)