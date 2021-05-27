import streamlit as st
import pyEX as p
import pandas as pd
import requests
from datetime import datetime
# import Image from pillow to open images
from PIL import Image


token = 'pk_53f6f55f51234bc594c5aaee57dbf2a3'
c = p.Client(api_token=token, version='stable')

st.markdown("<h1 style='text-align: center; color: blue;'>Welcome!\n</h1>"

            "<h1 style='text-align: left; color: red;'>Predicting Stock Prices</h1>",
            unsafe_allow_html=True)
#st.title("""Title: **Predicting** *Stock Prices*""")
# title of the checkbox is 'Show/Hide'
if st.checkbox("Show Credits"):
    # dispaly the text if the checkbox returns True value

    #st.sidebar.header("**Welcome**")
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


st.title("Symbol Pre Selected:\n"
         "** MSFT **\n"
         "*--> Working on Microsoft data*\n")


data_update = st.beta_expander("Update Data", expanded=False)
with data_update:
    #clicked = my_widget("second")
    @st.cache
    def data_update_1():
        sym = 'MSFT'
        timeframe = '1mm'
        df = c.chartDF(symbol=sym, timeframe=timeframe)[['open', 'low', 'close', 'volume']]
        df.reset_index(inplace=True)
        df.to_csv('test.csv')
        #df
        #df=pd.read_csv("test.csv")
        #list_clm = df.columns
        #df = df[['open', 'close', 'volume', 'low']].set_index(df['date'])
        #return df
        print(df)
        # Create a button, that when clicked, shows a text
    if st.button("Update data"):
        st.success("Data Updated")
        data_update_1()
        st.text("Data updated!!!")


my_expander1 = st.beta_expander("Raw Data", expanded=False)
with my_expander1:
    #clicked = my_widget("second")

    df = pd.read_csv("test.csv")
    #list_clm = df.columns
    df = df[["open",'low', 'close', "volume"]].set_index(df['date'])

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
    #clicked = my_widget("second")
    od = pd.read_csv("test.csv")
    #od
    od_test = od[["date", "open", "close"]]
    #od_test
    st.subheader("Plotting Visualization")
    st.line_chart(od_test.rename(columns={"date":"index"}).set_index("index"))


genre = st.radio(
  "What's your favorite movie genre",
  ('Comedy', 'Drama', 'Documentary'))
if genre == 'Comedy':
   st.write('You selected comedy.')
elif genre == 'Drama':
    st.write("You select drama.")
elif genre == 'Documentary':
    st.write("You didn't select comedy.")

ctn1 = st.beta_container()
ctn1.subheader("**---------------------------------Caution!---------------------------------------**")
ctn1.write("""
This Project is used for only learning and development process. We don't encourage anyone 
to invest in stock based on any data represented here.
""")

if __name__ == "__main__":
    print("hello")