# DATA 1030 Project Report #
Tongxin Wang   
12/03/2019   
Brown University Data Science Inisitive   
<font color = maroon>TODO:LINK</font> 


## I. Introduction ##   
### Motivation: ###  
YouTube is the world's largest video sharing platform and currently has 2 billion monthly active users, which accounts for 45% of the world's entire online population. As of 2018, there are more than 23 million YouTube channels, and Top-10 YOuTube channels earned $\$180.5$ billion between June 2017 and June 2018. PewDiePie alone has cleared at least $\$9$ million a year from advertising earning on his YouTube channel. Not only creating contents, these top YouTubers also have vast influences on their subscribers as well as the off-line world. YouTube is a website full of video contents along with tremendous market value. YouTube releases statistics of top 200 trending videos daily and the insights gained from these videos would be highly valuable for YouTubers as well as commercial users to improve their video performances to generate higher revenues.  

### Problem to solve: ###
In this project, we are aiming to predict the total trending days of certain videos based on their performances, such as views, comments, likes and dislikes. Thus, the target variable is the total trending days, which makes this machine learning project a regression problem.

### About the Data set:### 

Data set: [Trending YouTube Video Statistics](https://www.kaggle.com/datasnaek/youtube-new)

The data set contains daily record of the top 200 trending YouTube videos in the US, GB, DE, CA, and FR regions (USA, Great Britain, Germany, Canada, and France, respectively), and each country data is stored in a separate file. In this project we will focus on the US trending videos.    

Duration: 14 Nov 2017 - 14 June 2018 (205 days)   

This data set contains 40946 rows and 16 features, which are the following:    

- video_id(object): unique IDs for videos
- trending_date(object): the date of the trending list
- title(object): video title
- channel_title(object): channel title
- category_id(int64): IDs for categories
- publish_time(object): publish time for videos
- tags(object): the tags of videos
- views(int64): total number of views
- likes(int64): total number of likes
- dislikes(int64): total number of dislikes
- comment_count(int64): total number of comments
- thumbnail_link(object): link of the thumbnail
- comments_disabled(bool): whether the video disable comment or not
- ratings_disabled(bool): whether the video disable rating or not
- video_error_or_removed(bool):whether the video has an error or not
- description(object): descriptions for videos

In addition to provided features, we added several features for EDA and modeling, which are:   
- like_rate: $\frac{likes}{views}$
- dislike_rate: $\frac{dislikes}{views}$
- comment_log: $log(comment\_count)$
- view_log: $log(views)$
- interaction_rate: $\frac{likes + dislikes + 2 * comment\_count}{views} * 100$




### Previous research:###  

This dataset is retrieved from Kaggle and it contains data from several countries, such as UK, Canada, Mexico, Japan, South Korea, France. etc. People have been using these data in the following ways:
Sentiment analysis in a variety of forms
Categorising YouTube videos based on their comments and statistics.
Training ML algorithms like RNNs to generate their own YouTube comments.
Analysing what factors affect how popular a YouTube video will be.
Statistical analysis over time
There are more than 500 kennels about this dataset on the Kaggle website about the aforementioned topics, and these projects could be found [here](https://www.kaggle.com/datasnaek/youtube-new/kernels). 

<!-- #region -->
## II. EDA ##   

In this section, we will analyse individual features. 

First of all, it would be interesting for us to see the videos with top number of views.   
From Figure 3.1, we can see that 8 out of 10 top viewed videos belong to the music genre. The video gains the most views between 11/14/2017 and 6/14/2018 is *Childish Gambino - This Is America (Official Video)*, and the number of its view is 225,211,923. The only two videos in the list which are not music are *YouTube Rewind: The Shape of 2017* and *Marvel Studios' Avengers: Infinity War Official*, and both of these two videos are entertainment. 

*Fig.3.1 : Top views Videos*
![Fig 3.1 Top views Videos](figures/top_views.png)


Figure 3.1 appears more interesting if we compare it to the Figure 3.2, which is Video counts by Category Names. 

*Fig.3.2 : video counts by categories*
![Fig 3.2 : video counts by categories](figures/category_count.png)

As we can see from Figure 3.2, entertainment genre ranks Top 1 in the number of video counts by category names, and it takes 25.5%. The next four genres are Music, How to and Style, Comedy and News and Politics. The genres which have the least number of trending videos are NonProfits and Activism and Shows. 

Although entertainment genre has the most number of videos, music genre generates the most views. 

Besides number of views, interaction rate is also one of the most important features of trending videos. Interaction rate is related to views, comments, likes and dislikes. The detailed explanation of interaction rate will be discussed in the Method section.

In Figure 3.3, we can see the video with the highest interaction rate. 

*Fig.3.3 : videos with top interaction rates*
![Fig 3.3 : interaction_rate](figures/interaction_rate.png)

The top one is the music video Finesse (Remix) by Bruno Mars. The Top 10 videos with the highest interaction rates are from Music, Entertainment and How to and Style categories. Thus, we can see that although entertainment takes the most numbers of videos, it does not guarantee neither the most number of views nor the highest interaction rates. 

In Figure 3.4, we can see the number of views of videos by category names as well as the log(views) by category names. 

*Fig.3.4 : views distribution by categories*
![Fig 3.4 : view_category](figures/view_category.png)

As we can see from the figure, the average of log(views) are around 6 for most of the categories, and most of the categories have outliers. In the top figure, the outlier with the highest views is in the Music category, and the result matches Figure 3.1. 

Figure 3.5 is the histogram of total trending days for each videos. 

*Fig.3.5 : histogram of total trending days for each videos*
![Fig 3.5 : histogram](figures/y_hist.png)

There are 6351 unique videos in the data set and the range of the total trending days are from 1 to 30. Most of the total trending days are less than 15, and the mean is 6.45 days. It is reasonable to have less videos with large total trending days, but it is also surprising to see that there is a drop of number of videos with 26 total trending days. 

Figure 3.6: Correlation matrix of the data frame. 

*Fig.3.6 : Correlation matrix of the data frame*
![Fig 3.6 : corr](figures/corr_coeff.png)

After preprocessing, there are 29 feature columns in the data frame. We can see most of the feature are less correlated, but x1_0.0 and x1_1.0 are negatively correlated as x1 is a binary feature. X2 and x3 have the similar characteristics as x1. It also makes sense to see that views, likes, dislikes, like_rate, dislike_rate and interaction_rate are positively correlated. 

<!-- #endregion -->

## III. Methods ##




## IV. Results


## V. Outlook


## VI. References ##    
https://www.businessofapps.com/data/youtube-statistics/#1 
https://influencermarketinghub.com/pewdiepie-net-worth/

```python

```
