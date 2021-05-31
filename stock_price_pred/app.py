import streamlit as st
import pyEX as p
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from keras.models import load_model
# import Image from pillow to open images
from PIL import Image
import pickle
#########################functions


def data_update_1():
    # sym = 'MSFT'
    timeframe = '1mm'
    df = c.chartDF(symbol=sym, timeframe=timeframe)[['open', 'low', 'close', 'volume']]
    df.reset_index(inplace=True)
    df.read_csv(r'https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/test.csv')
    #data_load_state.text("Done! (using st.cache)")

@st.cache
def load_data(nrows):
    data = pd.read_csv('https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/test.csv', nrows=nrows)
    return data
@st.cache
def load_msft_data(nrows):
    data = pd.read_csv('https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/test.csv', nrows=nrows)
    #data = pd.read_csv('https://github.com//HARSHIT097//Streamlit_exp//blob//main//stock_price_pred//MSFT.csv',nrows=nrows)
    return data
@st.cache
def load_nfty_data(nrows):
    data = pd.read_csv('https://raw.githubusercontent.com/HARSHIT097/Streamlit_exp/main/stock_price_pred/test.csv', nrows=nrows)
    #data = pd.read_csv('https://github.com//HARSHIT097//Streamlit_exp//blob//main//stock_price_pred//DataFrame.csv',nrows=nrows)
    return data

st.markdown("<h1 style='text-align: center; color: blue;'>Welcome!\n</h1>"

            "<h1 style='text-align: left; color: red;'>Predicting Stock Prices</h1>",
            unsafe_allow_html=True)
#st.title("""Title: **Predicting** *Stock Prices*""")
# title of the checkbox is 'Show/Hide'
if st.checkbox("Show Credits"):

    st.sidebar.markdown("<h1 style='text-align: left; color: green;'>Welcome!</h1>",
            unsafe_allow_html=True)
    img = Image.open("logo.png")

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
        "1. [Nayana](https://www.linkedin.com/in/)\n"
        "2. [Harshit Singh](https://www.linkedin.com/in/harshit-singh-097/)\n"
        "3. [Yogendra](https://www.linkedin.com/in/)\n"
        "4. [Snehashish](https://www.linkedin.com/in/)\n"
        "5. [Pranay](https://www.linkedin.com/in//)"
    )
    st.sidebar.info("[contact us](https://www.technocolabs.tech/)\n")




st.write("""
Heloo! Welcome to the **demo Project**
#@Technocolab
""")

st.title("Symbol Pre Selected/Default Dataset:\n** MSFT **\n*--> Working on Microsoft data*\n")


dataset = st.selectbox("Choose the dataset:" ,
                       ['day-wise dataset', 'one-min dataset', 'choose symbol'])

#weekly_data = load_data(1000)
#df = pd.read_csv(r"Datasets/test.csv")
df = load_data(100)

show_raw_data = st.beta_expander("Raw Data", expanded=False)
with show_raw_data:
    #clicked = my_widget("second")
    st.write("Raw data")
    #list_clm = df.columns
    #df = df[["open",'low', 'close', "volume"]].set_index(df['date'])
    #df = df["open"]
    #df
    if dataset == "day-wise dataset":
        #df = pd.read_csv(r"Datasets//test.csv")
        df = load_msft_data(500)
    elif dataset == "one-min dataset":
        df = load_nfty_data(500)
        #df = pd.read_csv(r"Datasets//DataFrame.csv")


    # print the selected hobby
    # st.write("Selected dataset is: ", dataset)
    elif dataset == "choose symbol":
        sym = 'MSFT'
        sym = st.text_input("Enter Your Symbol")
        st.write("The dataset is selected as", sym)
        token = 'pk_4f41c633a06f40e09c67979fd397a16a'
        c = p.Client(api_token=token, version='stable')

        if st.button("Update data!"):
            st.success("Data Updated")
            st.write("we will update the request due to limited no. of api's")
            # data_update_1()
            st.text("Data updated!!!")
            df = load_data(10)
            #df = pd.read_csv(r"Datasets//test.csv")

            #df
    df
    """
data_update = st.beta_expander("Update Data", expanded=False)

with data_update:
    if st.button("Update data"):
        st.success("Data Updated")
        st.write("we will update the request due to limited no. of api's")
        # data_update_1()
        st.text("Data updated!!!")
        df = load_data(10)
        #df = pd.read_csv(r"Datasets//test.csv")
        df

col1, col2, col3, col4 = st.beta_columns(4)

original = df.open[0]
col1.header("Open Price")
col1.write(original)

#grayscale = original.convert('LA')
grayscale = df.close[0]
col2.header("Prev Close")
col2.write(grayscale)

volume = df.volume[0]
col3.header("Volume")
col3.write(volume)

return1 = (df.open[0] - df.open[1])/(df.open[0])*100
fo = "{:.2f}".format(return1)
col4.header("Return %")
col4.write(fo)


my_expander2 = st.beta_expander("Plotting Vizualization", expanded=True)
with my_expander2:
    df = load_data(10)
    #od = pd.read_csv(r"Datasets//test.csv")
    #od
    od_test = od[["date", "open", "close"]]
    #od_test
    st.subheader("Plotting Visualization")
    st.bar_chart(od_test.rename(columns={"date":"index"}).set_index("index"))

#############modelprediction

my_expander3 = st.beta_expander("Predictions", expanded=True)
with my_expander3:
    scaler = pickle.load(open('scalerMSFT.pkl', 'rb'))
    model = load_model('modelMSFT.h5')
    # model = joblib.load('modelMSFT.pkl')

    with open('ftestMSFT.pkl', 'rb') as f:
        f_test = pickle.load(f)

    f_test = np.array(f_test)
    f_test = np.reshape(f_test, (f_test.shape[0], f_test.shape[1], 1))


    def user_input_features():
        day = st.slider('Predicting Day', 1, 90, 1)
        return day


    day = user_input_features()

    f_predict = []
    n_days = day

    # f_test.append(training_set_scaled[22745:22805, 0])

    # In[92]:

    for i in range(n_days):
        res = model.predict(f_test)
        f_predict.append(res[0][0])
        f_test = np.delete(f_test, [0], None)
        f_test = np.append(f_test, res[0][0], None)
        f_test = f_test.reshape(1, 60, 1)

    st.subheader("Predicted Stock Price")
    res = scaler.inverse_transform([[f_predict[day - 1]]])

    st.write(res[0][0])





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
    st.write("Connect us through linkend in(link available in credit section)")
"""
ctn1 = st.beta_container()
ctn1.subheader("**---------------------------------Caution!---------------------------------------**")
ctn1.write("""
This Project is used for only learning and development process. We don't encourage anyone 
to invest in stock based on any data represented here.
""")


"""
#histogram
df = pd.DataFrame(weekly_data[:200],
                  columns = ["num_orders",
                "checkout_price",
                "base_price"])
df.hist()
plt.show()
st.pyplot()

#Line Chart
st.line_chart(df)

chart_data = pd.DataFrame(weekly_data[:40], columns=["num_orders", "base_price"])
st.area_chart(chart_data)
"""

