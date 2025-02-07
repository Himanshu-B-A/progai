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
             color='Event',  
             title="Correctly Ordered Vertical Bar Graph",
             labels={'Event_Count': 'Number of Events', 'Date_Time': 'Timestamp'},
             
             category_orders={'Date_Time': df['Date_Time'].tolist()})  # Forces Date Order

# ✅ Rotate x-axis labels for better readability
fig.update_layout(xaxis=dict(tickangle=-45))

# Show the graph
fig.show()