import pandas as pd
import streamlit as st
import preprocessor,helper
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import random

# set Title
st.set_page_config(
        page_title="Whatsapp Chat Analyzer",
)



# hide footer streamLit
hide_st_style = """
            <style>
                footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)




# sidebar title
st.sidebar.title("Whatsapp Chat Analyzer")



uploaded_file = st.sidebar.file_uploader("Choose a file")



if uploaded_file is not None:


    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)


    # fetch unique users
    user_list = df['user'].unique().tolist()
    user_list.sort()
    user_list.insert(0, "Overall")


    selected_user = st.sidebar.selectbox("Show analysis wrt", user_list)


    if st.sidebar.button("Show Analysis"):

        st.balloons()
        # Stats Area
        num_messages, words, num_media_messages, num_links = helper.fetch_stats(selected_user, df)
        st.title("Top Statistics")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Messages", num_messages)
        col2.metric("Total Words", words)
        col3.metric("Media Shared", num_media_messages)
        col4.metric("Links Shared", num_links)



        # Sentiment Analysis of Messages
        positive_messages, negative_messages, neutral_messages = helper.fetch_sentiment_stats(selected_user, df)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Positive Messages", positive_messages)
        col2.metric("Negative Messages", negative_messages)
        col3.metric("Neutral Messages", neutral_messages)

        st.header("Analysis of Messages")
        fig = px.bar(y=[positive_messages, negative_messages, neutral_messages],
                     x=["Positive Messages", "Negative Messages", "Neutral Messages"],
                     labels={"x": "Messages", "y": "Count"})
        st.plotly_chart(fig)


        # monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)

        fig = px.line(timeline, x = 'time', y = 'message', labels={'time':'Dates', 'message':'Messages'})
        fig.update_xaxes(nticks=20)
        st.plotly_chart(fig)


        # daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)

        fig = px.line(daily_timeline, x = 'only_date', y = 'message', labels={'only_date': 'Dates', 'message':'Messages'})
        fig.update_xaxes(nticks=20)
        st.plotly_chart(fig)

        # activity map
        st.title('Activity Map')

        st.header("Most busy day")
        weekday_grouped_msg = helper.weekday_msg(selected_user, df)
        fig = px.line_polar(weekday_grouped_msg, r='count', theta='day_name', line_close=True)
        fig.update_traces(fill='toself')
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )),
            showlegend=False
        )
        st.plotly_chart(fig)

        st.header("Most busy month")
        month_grouped_msg = helper.month_activity_map(selected_user, df)
        fig = px.line_polar(month_grouped_msg, r='count', theta='month', line_close=True)
        fig.update_traces(fill='toself')
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                )),
            showlegend=False
        )
        st.plotly_chart(fig)


        st.title("Weekly Activity Map")
        user_heatmap = helper.activity_heatmap(selected_user, df)

        fig, ax = plt.subplots()
        ax = sns.heatmap(user_heatmap)

        st.pyplot(fig)

        # finding the busiest users in the group(Group level)
        if selected_user == 'Overall':
            st.title('Most Busy Users')

            busy_df, new_df = helper.most_busy_users(df)

            fig = px.bar(busy_df.dropna(), x='name', y='count', labels={'name': 'Users Name', 'count': 'Messages'},
                         height=500, width=800)
            fig.update_traces(marker_color='#EDCC8B', marker_line_color='#D4A29C',
                              marker_line_width=1.5, opacity=0.6)
            st.plotly_chart(fig)

            fig = px.pie(new_df, values='percent', names='name')
            st.plotly_chart(fig)

        # WordCloud
        st.title("Wordcloud")
        df_wc = helper.create_wordcloud(selected_user, df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        # most common words
        st.title('Most Commmon Words')
        most_common_df = helper.most_common_words(selected_user, df)

        fig = px.bar(most_common_df.head(15).dropna(), x='words', y='count',
                     labels={'words': 'Common Words', 'count': 'Words'},
                     height=500, width=800)
        fig.update_traces(marker_color='#EDCC8B', marker_line_color='#D4A29C',
                          marker_line_width=1.5, opacity=0.6)

        st.plotly_chart(fig)

        # emoji analysis
        emoji_df = helper.emoji_helper(selected_user, df)
        st.title("Emoji Analysis")

        st.header('Emoji TreeMap')
        fig = px.treemap(emoji_df, path=['emoji'],
                         values=emoji_df['count'].tolist(),
                         )
        st.plotly_chart(fig)
        st.header('Most Used Emoji')
        fig = px.pie(emoji_df.head(10), values='count', names='emoji')
        st.plotly_chart(fig)







