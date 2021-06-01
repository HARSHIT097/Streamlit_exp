
############################import libraries#############

import streamlit as st
#import pyEX as p
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import urllib
import requests
from datetime import datetime
from keras.models import load_model
# import Image from pillow to open images
from PIL import Image
import pickle
import matplotlib.dates as mpdates
from mplfinance.original_flavor import candlestick_ohlc
#from mplfinance import candlestick_ohlc
import plotly.express as px
import datetime as dt
from streamlit import caching
import plotly.graph_objects as go
from plotly.subplots import make_subplots
caching.clear_cache()
#########################functions

def calcMovingAverage(data, size):
    df = data.copy()
    df['sma'] = df['open'].rolling(size).mean()
    df['ema'] = df['open'].ewm(span=size, min_periods=size).mean()
    df.dropna(inplace=True)
    return df

#############modelprediction

urllib.request.urlretrieve("https://github.com/HARSHIT097/Streamlit_exp/blob/main/stock_price_pred/scaler.pkl?raw=true",
                           "scaler.pkl")

urllib.request.urlretrieve(
    "https://github.com/HARSHIT097/Streamlit_exp/blob/main/stock_price_pred/modelmsft.h5?raw=true", "modelmsft.h5")

urllib.request.urlretrieve(
    "https://github.com/HARSHIT097/Streamlit_exp/blob/main/stock_price_pred/ftestMSFT.pkl?raw=true", "ftestMSFT.pkl")

urllib.request.urlretrieve(
    "https://github.com/HARSHIT097/Streamlit_exp/blob/main/stock_price_pred/df_msft.pkl?raw=true", "df_msft.pkl")
urllib.request.urlretrieve(
    "https://github.com/HARSHIT097/Streamlit_exp/blob/main/stock_price_pred/df_nifty.pkl?raw=true", "df_nifty.pkl")
urllib.request.urlretrieve(
    "https://github.com/HARSHIT097/Streamlit_exp/blob/main/stock_price_pred/scalerMSFT.pkl?raw=true", "scalerMSFT.pkl")
urllib.request.urlretrieve(
    "https://github.com/HARSHIT097/Streamlit_exp/blob/main/stock_price_pred/modelMSFT.pkl?raw=true", "modelMSFT.pkl")

def data_update_1():
    # sym = 'MSFT'
    timeframe = '1mm'
    df = c.chartDF(symbol=sym, timeframe=timeframe)[['open', 'low', 'close', 'volume']]
    df.reset_index(inplace=True)
    df.read_csv(r'https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/test.csv')
    # data_load_state.text("Done! (using st.cache)")


@st.cache
def load_data(nrows):
    data = pd.read_csv('https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/test.csv',
                       nrows=nrows)
    return data


@st.cache
def load_msft_data(nrows):
    data = pd.read_csv('https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/MSFT.csv',
                       nrows=nrows)
    # data = pd.read_csv('https://github.com//HARSHIT097//Streamlit_exp//blob//main//stock_price_pred//MSFT.csv',nrows=nrows)
    return data


@st.cache
def load_nfty_data(nrows):

    data = pd.read_csv('https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/DataFrame.csv',
                       nrows=nrows)
    # data = pd.read_csv('https://github.com//HARSHIT097//Streamlit_exp//blob//main//stock_price_pred//DataFrame.csv',nrows=nrows)
    return data
#nrows = len(data.index)
######################title#########and############sidebaar###########

st.markdown("<h1 style='text-align: left; color: blue;'>Welcome!\n</h1>"

            "<h1 style='text-align: centre; color: red;'>Predicting Stock Prices</h1>",
            unsafe_allow_html=True)
# st.title("""Title: **Predicting** *Stock Prices*""")
# title of the checkbox is 'Show/Hide'
if st.checkbox("Show Credits"):
    st.sidebar.markdown("<h1 style='text-align: left; color: green;'>Welcome!</h1>",
                        unsafe_allow_html=True)

    urllib.request.urlretrieve("https://technocolabs.tech//assets//img//logo1.png", "logo1.png")
    # img = Image.open("logo1.png")
    # img.show()
    img = Image.open("logo1.png")

    # st.text[website](https://technocolabs.tech/)
    # display image using streamlit
    # width is used to set the width of an image
    st.sidebar.image(img, width=200)

    st.sidebar.subheader("Credits")

    st.sidebar.subheader("Under Guidance of")
    # **Guidance @ CDAC-ACTS, Pune**\n
    st.sidebar.info(
        """
        1. Yasin Sir\n
        2. Team @ [Technocolab](https://www.linkedin.com/company/technocolabs/)\n
        """)
    st.sidebar.subheader("Contributors/Project Team")
    st.sidebar.info(
        "1. [Nayana](https://www.linkedin.com/in/nayana-s-49482b180/)\n"
        "2. [Harshit Singh](https://www.linkedin.com/in/harshit-singh-097/)\n"
        "3. [Yogendra](https://www.linkedin.com/in/yogendrasingh5015/)\n"
        "4. [Snehashish](https://www.linkedin.com/in/snehashish-chakraborty-884598121/)\n"
        "5. [Pranay](https://www.linkedin.com/in/harshit-singh-097/)"
    )
    st.sidebar.info("[contact us](https://www.technocolabs.tech/)\n")

st.write("""
Heloo! Welcome to the ** Project**
#@Technocolab
""")
#############################################dataset and raw data
my_expander6 = st.beta_expander("Get fundamentals by choosing Stocks(SNP 500)", expanded=False)
with my_expander6:
    # https://raw.githubusercontent.com/teobeeguan/market_profile/main/Datasets/SP500.csv

    snp500 = pd.read_csv("https://raw.githubusercontent.com/teobeeguan/market_profile/main/Datasets/SP500.csv")
    symbols = snp500['Symbol'].sort_values().tolist()

    ticker = st.selectbox(
        'Choose a S&P 500 Stock',
        symbols)
    stock = yf.Ticker(ticker)
    info = stock.info
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    infoType = st.radio(
        "Choose an info type",
        ('Company Profile','Market trend', 'Fundamental Info', 'Market Info', 'Business Summary')
    )
    if infoType == 'Company Profile':

        st.title('Company Profile')
        st.subheader(info['longName'])

        col01, col02, col03 = st.beta_columns(3)

        #sector = st.markdown('** Sector **: ' + info['sector'])
        sector = info['sector']
        col01.header("Sector")
        col01.write(sector)

        # grayscale = original.convert('LA')
        industry = info['industry']
        #industry = st.markdown('** Industry **: ' + info['industry'])
        col02.header("Industry")
        col02.write(industry)

        website = info['website']
        col03.header("website")
        col03.write(website)

        col04, col05, col06 = st.beta_columns(3)

        market = info['market']
        col04.header("Market")
        col04.write(market)

        # grayscale = original.convert('LA')
        quo = info['quoteType']
        col06.header("quote type")
        col06.write(quo)

        exch = info['exchange']
        col05.header("Exchange")
        col05.write(exch)

    elif infoType == 'Business Summary':
        st.markdown('** Business Summary **')
        st.info(info['longBusinessSummary'])

    elif infoType == 'Fundamental Info':
        fundInfo = {
            'Enterprise Value (USD)': info['enterpriseValue'],
            'Enterprise To Revenue Ratio': info['enterpriseToRevenue'],
            'Enterprise To Ebitda Ratio': info['enterpriseToEbitda'],
            'Net Income (USD)': info['netIncomeToCommon'],
            'Profit Margin Ratio': info['profitMargins'],
            'Forward PE Ratio': info['forwardPE'],
            'PEG Ratio': info['pegRatio'],
            'Price to Book Ratio': info['priceToBook'],
            'Forward EPS (USD)': info['forwardEps'],
            'Beta ': info['beta'],
            'Book Value (USD)': info['bookValue'],
            'Dividend Rate (%)': info['dividendRate'],
            'Dividend Yield (%)': info['dividendYield'],
            'Five year Avg Dividend Yield (%)': info['fiveYearAvgDividendYield'],
            'Payout Ratio': info['payoutRatio']
        }

        fundDF = pd.DataFrame.from_dict(fundInfo, orient='index')
        fundDF = fundDF.rename(columns={0: 'Value'})
        st.subheader('Fundamental Info')
        st.table(fundDF)
    elif infoType == 'Market Info':
        marketInfo = {
            "Volume": info['volume'],
            # "Average Volume": info['averageVolume'],
            "Market Cap": info["marketCap"],
            "Float Shares": info['floatShares'],
            'Bid Size': info['bidSize'],
            'Ask Size': info['askSize'],
            "Share Short": info['sharesShort'],
            "Regular Market Price (USD)": info['regularMarketPrice']
            # 'Short Ratio': info['shortRatio'],
            # 'Share Outstanding': info['sharesOutstanding']
        }

        marketDF = pd.DataFrame(data=marketInfo, index=[0])
        st.table(marketDF)
    else:
        start = dt.datetime.today() - dt.timedelta(2 * 365)
        end = dt.datetime.today()
        df = yf.download(ticker, start, end)
        df = df.reset_index()
        fig = go.Figure(
            data=go.Scatter(x=df['Date'], y=df['Adj Close'])
        )
        fig.update_layout(
            title={
                'text': "Stock Prices Over Past Two Years",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)



def show_info(df):

    #df = pd.read_csv("test_002.csv")
    col1, col2 = st.beta_columns(2)

    original = df.open[0]
    col1.header("Open Price")
    col1.write(original)

    # grayscale = original.convert('LA')
    grayscale = df.close[0]
    col2.header("Prev Close")
    col2.write(grayscale)

    #volume = df.volume[0]
    #col3.header("Volume")
    #col3.write(volume)

    st.write("last day return")
    x = 0
    col4, col5, col6, col7 = st.beta_columns(4)
    return1 = (df.open[x] - df.close[0]) / (df.open[x]) * 100
    fo = "{:.2f}".format(return1)
    col4.header("Return % last day")
    col4.write(fo)

    return2 = (df.open[x] - df.close[6]) / (df.open[x]) * 100
    fo2 = "{:.2f}".format(return2)
    col5.header("Return % last week")
    col5.write(fo2)

    return3 = (df.open[x] - df.close[30]) / (df.open[x]) * 100
    fo3 = "{:.2f}".format(return3)
    col6.header("Return % last month")
    col6.write(fo3)

    return4 = (df.open[x] - df.close[365]) / (df.open[x]) * 100
    fo4 = "{:.2f}".format(return4)
    col7.header("Return % last year")
    col7.write(fo4)
    
my_expander7 = st.beta_expander("Basic Info ", expanded=False)
with my_expander7:
    st.header('Select Dataset to view info')
    df = pd.DataFrame({
        'first column': ["MSFT", "NIFTY"]
    })
    option1 = st.selectbox(
        'Select prediction Interval',
        df['first column'])

    
    if option1 == "MSFT":
        'You selected:', option1
        df1 = pd.read_csv("https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/test_002.csv")
        df1.columns = [x.lower() for x in df1.columns]
        #df1.columns.lowercase()
        #data.columns = map(str.lower, data.columns)
        show_info(df1)
    if option1 == "NIFTY":
        'You selected:', option1
        df2 = pd.read_csv("https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/Datasets/DataFrame.csv")
        #df1.columns = [x.lower() for x in df1.columns]
        #df1.columns.lowercase()
        #data.columns = map(str.lower, data.columns)
        show_info(df2)
        
my_expander2 = st.beta_expander("Plotting Visualization", expanded=False)
with my_expander2:
    genre = st.radio(
        "Options:",
        ('comparative', 'interactive'))
    plt.style.use('dark_background')

    # extracting Data for plotting

    df = pd.read_csv('https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/Datasets/MSFT.csv')
    df.columns = map(str.lower, df.columns)
    df = df[['date', 'open', 'high',
             'low', 'close']]

    # convert into datetime object
    df['date'] = pd.to_datetime(df['date'])

    if genre == 'comparative':
        st.write('All data')
        # apply map function
        df['date'] = df['date'].map(mpdates.date2num)

        # creating Subplots
        fig, ax = plt.subplots()

        # plotting the data
        candlestick_ohlc(ax, df.values, width=0.6,
                         colorup='green', colordown='red',
                         alpha=0.8)

        # allow grid
        ax.grid(True)

        # Setting labels
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')

        # setting title
        plt.title('Prices For the Period 01-07-2020 to 15-07-2020')

        # Formatting Date
        date_format = mpdates.DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        fig.autofmt_xdate()

        fig.tight_layout()

        # show the plot
        # plt.show()
        st.write(fig)
        #st.write("Connect us through linkendin(link available in credit section)")
    elif genre == 'interactive':
        st.write("Customize by selecting the area")
        fig4 = go.Figure(data=[go.Candlestick(x=df['date'],
                                              open=df['open'], high=df['high'],
                                              low=df['low'], close=df['close'])
                               ])

        fig4.update_layout(xaxis_rangeslider_visible=False)
        # fig4.show()
        st.write(fig4)




my_expander3 = st.beta_expander("Indicators", expanded=False)
with my_expander3:

    df = pd.read_csv("https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/test_002.csv")

    df['MA12'] = df['open'].rolling(12).mean()
    df['VMA12'] = df['volume'].rolling(12).mean()
    genre = st.radio(
        "12 days Moving Average with",
        ('Open', 'Volume', 'Custom'))
    if genre == 'Open':
        st.write('Open with 12days moving average ')
        df_melt = df.melt(id_vars='date', value_vars=["MA12", "open"])
        fig1 = px.line(df_melt, x="date", y='value', template='plotly_dark', color="variable")
        st.write(fig1)

    elif genre == 'Volume':
        st.write("Volume with 12days moving Average ")
        df_melt = df.melt(id_vars='date', value_vars=["VMA12", "volume"])
        fig2 = px.line(df_melt, x="date", y='value', template='plotly_dark', color="variable")
        st.write(fig2)

    elif genre == 'Custom':
        st.write("Sorry! This feature is under construction, the result may vary.")
        coMA1, coMA2 = st.beta_columns(2)

        with coMA1:
            numYearMA = st.number_input('Insert period (Year): ', min_value=1, max_value=10, value=2, key=0)

        with coMA2:
            windowSizeMA = st.number_input('Window Size (Day): ', min_value=5, max_value=500, value=20, key=1)

        start = dt.datetime.today() - dt.timedelta(numYearMA * 365)
        end = dt.datetime.today()
        # dataMA = yf.download(ticker, start, end)
        df_ma = calcMovingAverage(df, windowSizeMA)
        df_ma = df_ma.reset_index()
        figMA = go.Figure()

        figMA.add_trace(
            go.Scatter(
                x=df_ma['date'],
                y=df_ma['open'],
                name="Prices Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma['date'],
                y=df_ma['sma'],
                name="SMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.add_trace(
            go.Scatter(
                x=df_ma['date'],
                y=df_ma['ema'],
                name="EMA" + str(windowSizeMA) + " Over Last " + str(numYearMA) + " Year(s)"
            )
        )

        figMA.update_layout(legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ))

        figMA.update_layout(legend_title_text='Trend')
        figMA.update_yaxes(tickprefix="$")

        st.plotly_chart(figMA, use_container_width=True)



my_expander4 = st.beta_expander("Visualization and Prediction(on specific dataset)", expanded=False)
with my_expander4:

    st.header('Users Input Parameters')
    df = pd.DataFrame({
        'first column': ["MSFT 1- Day", "NIFTY 1-Minute"]
    })
    option = st.selectbox(
        'Select prediction Interval',
        df['first column'])

    if option == "NIFTY 1-Minute":
        'You selected:', option
        df = pickle.load(open('df_nifty.pkl', 'rb'))
        #df = pd.read_csv("Datasets/MSFT.csv")
        scaler = pickle.load(open('scaler.pkl', 'rb'))
        model = load_model('model.h5')
        with open('ftest.pkl', 'rb') as f:
            f_test = pickle.load(f)

        f_test = np.array(f_test)
        f_test = np.reshape(f_test, (f_test.shape[0], f_test.shape[1], 1))

        st.header('User Input Parameters')


        def user_input_features():
            date = st.date_input('Select Date')
            date = date.day
            hour = st.slider('Hour of the day', 9, 17, 9)
            minute = st.slider('Minute of the day', 0, 59, 5)

            return date, hour, minute


        day, hour, minute = user_input_features()

        f_predict = []
        n_days = 1
        minutes = 0
        hours = 8  # [9--17]

        for i in range(n_days):
            while (hours > 0):
                while (minutes < 60):
                    res = model.predict(f_test)
                    f_predict.append(res[0][0])
                    f_test = np.delete(f_test, [0], None)
                    f_test = np.append(f_test, res[0][0], None)
                    f_test = f_test.reshape(1, 60, 1)
                    minutes = minutes + 1
                hours = hours - 1
                minutes = 0
            hours = 8
            minutes = 0

        hour = hour - 8

        st.header("Prediction:")
        res = scaler.inverse_transform([[f_predict[day * hour * minute]]])

        col1, col2, col3 = st.beta_columns(3)

        original = df.open[0]
        col1.subheader("Open Price")
        col1.write(res[0][0])

        grayscale = df.open[0]
        col2.subheader("Prev Day Open")
        a = scaler.inverse_transform([[f_predict[day - 2]]])[0][0]
        col2.write(a)

        return1 = (a - res[0][0]) / (a) * 100
        # fo = "{:.2f}".format(return1)
        col3.subheader("Return %")
        fo = "{:.2f}".format(return1)
        col3.write(fo)

        st.header("Visualisation:")
        idx = pd.date_range("2021-04-01", periods=len(f_predict), freq="D")
        ts = pd.Series(range(len(idx)), index=idx)
        r = scaler.inverse_transform([f_predict]).reshape(-1, 1)
        r = r.reshape(r.shape[0])
        fig, ax = plt.subplots()
        ax = sns.lineplot(x=df.DateAndTime, y=df['open'], color='r')
        ax = sns.lineplot(x=idx, y=r)
        st.pyplot(fig)

    # MSFFTT
    if option == "MSFT 1- Day":
        'You selected:', option


        df = pickle.load(open('df_msft.pkl', 'rb'))
        scaler = pickle.load(open('scalerMSFT.pkl', 'rb'))
        model = load_model('modelmsft.h5')
        # model = joblib.load('modelMSFT.pkl')

        with open('ftestMSFT.pkl', 'rb') as f:
            f_test = pickle.load(f)

        f_test = np.array(f_test)
        f_test = np.reshape(f_test, (f_test.shape[0], f_test.shape[1], 1))


        def user_input_features():
            date = st.date_input('Select Date')
            # st.write(date.day)
            return date.day


        day = user_input_features()

        f_predict = []
        n_days = day

        for i in range(n_days):
            res = model.predict(f_test)
            f_predict.append(res[0][0])
            f_test = np.delete(f_test, [0], None)
            f_test = np.append(f_test, res[0][0], None)
            f_test = f_test.reshape(1, 60, 1)

        st.header("Prediction:")
        res = scaler.inverse_transform([[f_predict[day - 1]]])
        col1, col2, col3 = st.beta_columns(3)

        #original = df.Open[0]
        col1.subheader("Open Price")
        col1.write(res[0][0])

        #grayscale = df.Open[0]
        col2.subheader("Prev Day Open")
        a = scaler.inverse_transform([[f_predict[day - 2]]])[0][0]
        col2.write(a)

        return1 = (a - res[0][0]) / (a) * 100
        # fo = "{:.2f}".format(return1)
        fo2 = "{:.2f}".format(return1)
        col3.subheader("Return %")
        col3.write(fo2)

        st.header("Visualisation:")
        idx = pd.date_range("2021-05-04", periods=len(f_predict), freq="D")
        ts = pd.Series(range(len(idx)), index=idx)
        r = scaler.inverse_transform([f_predict]).reshape(-1, 1)
        r = r.reshape(r.shape[0])
        fig8, ax1 = plt.subplots()
        ax1 = sns.lineplot(x=df.Date[df.Date.dt.year > 2019], y=df['Open'], color='r')
        ax1 = sns.lineplot(x=idx, y=r)
        st.pyplot(fig8)


  
################footer - finalized##########
genre = st.radio(
    "Do you Like our project?",
    ('Yes', 'No', 'Not Interested'))
if genre == 'Yes':
    st.write('Thanks! for showing Love.')
    st.write("Connect us through linkendin(link available in credit section)")
elif genre == 'No':
    st.write("Recommend changes! ")
    st.write("Connect us through linkendin(link available in credit section)")
elif genre == 'Not Interested':
    st.write("No worry")
    st.write("Connect us through linkendin(link available in credit section)")



ctn1 = st.beta_container()
ctn1.subheader("**---------------------------------Caution!---------------------------------------**")
ctn1.write("""
This Project is used for only learning and development process. We don't encourage anyone 
to invest in stock based on any data represented here.
""")
