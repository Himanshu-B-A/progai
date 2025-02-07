import pandas as pd
import plotly.express as px

df = pd.read_csv(r"C:\Users\himub\OneDrive\Desktop\ISRO internship project\trial\6-12-25\data_rel.txt", 
                 delim_whitespace=True, header=None, 
                 names=['Date_Time', 'Agency', 'Event','a','b','c','d','e']) 
print(df)

df.sort_values(by='Date_Time', inplace=True)

# ✅ Create a bar chart with proper date ordering
fig = px.bar(df, 
             x='Date_Time',  
             y='Event',
              )
fig.show()