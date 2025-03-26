import streamlit as st

from st_clickable_images import clickable_images # pip install st_clickable_images
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors

import warnings
warnings.filterwarnings("ignore")


@st.cache_data
def load_data():
    items = pd.read_csv("magazine_items.csv")
    ritems = pd.read_csv("magazine_recommend_items.csv")
    return items, ritems

items, ritems = load_data()

## Popularity Recommendations Pre-Work ###
pitems=items[["title","average_rating","rating_number"]]
pitems["pscore"]=round((pitems['average_rating']*pitems['rating_number']) / (pitems['rating_number'].mean()),2)

########### Content Recommendations Pre-Work ######
cdata=items[["categories","details","average_rating","rating_number"]]

# Cat Col Encoding
cdata.categories.replace({'magazine subscriptions':'1','books':'0'},inplace=True)

for indx in range(len(cdata)):
    #cdata['details'][indx] = items['details'][indx].get("Date First Available")
    if isinstance(items['details'][indx], dict):
            cdata['details'][indx] = items['details'][indx].get("Date First Available")

label_encoder = LabelEncoder() 
cdata["details"]= label_encoder.fit_transform(cdata.details)


cosinemodel = NearestNeighbors(n_neighbors=6, metric='cosine')   # 6 to include input item
cosinemodel.fit(cdata)

def recommend_items(itemtitle, n=5): # n is number of recommendations
       
    # Finding the item index
    idx = items[items['title'].str.contains(itemtitle, case=False, na=False)].index
    if len(idx) == 0:
        return "Item not found!"
    
    idx = idx[0]  # Take first match if multiple exist
       # Model input
    data = cdata.iloc[idx]
    
    # Find N nearest neighbors
    distances, indices = cosinemodel.kneighbors([data], n_neighbors=n+1)
    items_indices = indices.flatten()[1:]  # Skip input 
        
    return items_indices
########### Streamlit UI ##############
st.subheader(":blue[🏷️ 🏷️ 🏷️ Magazine Subcription Recommendations 🏷️ 🏷️ 🏷️]", divider=True)
st.subheader(":red[Popular Magazine..!]")

# Taking Top 10 based on Popularity Score
indxs = pitems.sort_values(by='pscore', ascending=False)[0:10][['title']].index

# Display images as clickable grid
image_paths = [eval(val)['large'] for val in items['images'].iloc[indxs]]
captions = [val.title() for val in items['title'].iloc[indxs]]
categories = [val.title() for val in items['categories'].iloc[indxs]]
#description = [val.title() for val in items['description'].iloc[indxs]]
description = [str(val).title() if isinstance(val, str) else "No Description" for val in items['description'].iloc[indxs]]
ratings = [val for val in items['average_rating'].iloc[indxs]]
details = [val for val in items['details'].iloc[indxs]]

selected_index = clickable_images(
    image_paths,
    titles=captions,
    div_style={"display": "flex", "flex-wrap": "wrap", "gap": "30px"},
    img_style={"height": "90px", "border-radius": "10px", "cursor": "pointer"}
)

# Store selection
if selected_index is not None:
    st.session_state["selected_section"] = selected_index + 1

st.divider()

if "selected_section" in st.session_state:
    st.subheader(":green[Selected..]")
    indx = st.session_state['selected_section']-1
    col1, col2, col3 = st.columns(3)
    with col2:
        st.image(image_paths[indx], caption=captions[indx])
    st.write(":green[categories:]")
    st.write(categories[indx])
    st.write(":green[description:]")
    st.write(description[indx])
    st.write(":green[Average_Rating:]", ratings[indx])
    st.write(":green[details:]", details[indx])

    if st.button("Recommend Similar Items:"):
        st.divider()
        st.subheader(":red[Similar Magazine..!]")
        title = captions[indx].lower()
        indxs = recommend_items(title)

        # Display images as clickable grid
        image_paths = [eval(val)['large'] for val in items['images'].iloc[indxs]]
        captions = [val.title() for val in items['title'].iloc[indxs]]
        categories = [val.title() for val in items['categories'].iloc[indxs]]
        #description = [val.title() for val in items['description'].iloc[indxs]]
        description = [str(val).title() if isinstance(val, str) else "No Description" for val in items['description'].iloc[indxs]]
        ratings = [val for val in items['average_rating'].iloc[indxs]]
        details = [val for val in items['details'].iloc[indxs]]

        selected_index2 = clickable_images(image_paths,titles=captions,
                                          div_style={"display": "flex", "flex-wrap": "wrap", "gap": "30px"},
                                          img_style={"height": "200px", "border-radius": "10px", "cursor": "pointer"}
                                          )


