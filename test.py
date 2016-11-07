




temp_df = pd.DataFrame()
if ((df_complete['type'] == 'NDD') & (df_complete['writer'] == 'Mary') & (df_complete['status'] != '7')):
    temp_df['col A'] = df_complete['col a']
    temp_df['col B'] = 'good'
    temp_df['col C'] = df_complete['col c']




