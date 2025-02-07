import pandas as pd
import plotly.express as px

# Load data with updated separator for whitespace
df = pd.read_csv(r"C:\Users\himub\OneDrive\Desktop\ISRO internship project\trial\6-12-25\data_rel.txt", 
                 sep='\s+', header=None, 
                 names=['Date_Time', 'Agency', 'Event','a','b','c','d','e']) 

# ✅ Convert Date_Time to datetime with explicit format
df['Date_Time'] = pd.to_datetime(df['Date_Time'], format='%Y-%m-%dT%H:%M:%S', errors='coerce')

# ✅ Group by Date_Time and Event, then count occurrences
df_counts = df.groupby(['Date_Time', 'Event']).size().reset_index(name='Event_Count')

# ✅ Sort by Date_Time
df_counts.sort_values(by='Date_Time', inplace=True)

# ✅ Create a correctly ordered vertical bar chart
fig = px.bar(df_counts, 
             x='Date_Time',  
             y='Event_Count',  
             color='Event',  
             title="Event Frequency Over Time",
             labels={'Event_Count': 'Number of Events', 'Date_Time': 'Timestamp'},
             barmode='group',
             category_orders={'Date_Time': df_counts['Date_Time'].astype(str).tolist()})  # Forces Date Order

# ✅ Rotate x-axis labels for better readability
fig.update_layout(xaxis=dict(tickangle=-45))

# Show the graph
fig.show()
