## Using Vanila Decision Tree model with Spotify Dataset

A Decision Tree is an algorithm used for supervised learning problems such as classification or regression. A decision tree or a classification tree is a tree in which each internal (non-leaf) node is labeled with an input feature. Each leaf of the tree is labeled with a class or a probability distribution over the classes.

A tree can be "learned" by splitting the source set into subsets based on an attribute value test. This process is repeated on each derived subset in a recursive manner called **recursive partitioning**. The recursion is completed when the subset at a node has all the same value of the target variable or when splitting no longer adds values to the predictions. This process of top-down induction of decision trees is an example of a greedy algorithm, and it is the most commmon strategy for learning decision trees.

In this notebook I will be building a Decision Tree Classifier to determine whether or not the user might like a song based on its attributes using a Spotify dataset from [Kaggle](https://www.kaggle.com/geomack/spotifyclassification?select=data.csv). 




```python
# The usual packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 
%matplotlib inline

# To make our sets
from sklearn.model_selection import train_test_split
# The classifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# Scalars
from sklearn.preprocessing import StandardScaler, MinMaxScaler 

```

### Step 1: EDA


```python
# Load the data
df = pd.read_csv('data/spotify_attributes.csv')

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
      <th>target</th>
      <th>song_title</th>
      <th>artist</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0102</td>
      <td>0.833</td>
      <td>204600</td>
      <td>0.434</td>
      <td>0.021900</td>
      <td>2</td>
      <td>0.1650</td>
      <td>-8.795</td>
      <td>1</td>
      <td>0.4310</td>
      <td>150.062</td>
      <td>4.0</td>
      <td>0.286</td>
      <td>1</td>
      <td>Mask Off</td>
      <td>Future</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.1990</td>
      <td>0.743</td>
      <td>326933</td>
      <td>0.359</td>
      <td>0.006110</td>
      <td>1</td>
      <td>0.1370</td>
      <td>-10.401</td>
      <td>1</td>
      <td>0.0794</td>
      <td>160.083</td>
      <td>4.0</td>
      <td>0.588</td>
      <td>1</td>
      <td>Redbone</td>
      <td>Childish Gambino</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.0344</td>
      <td>0.838</td>
      <td>185707</td>
      <td>0.412</td>
      <td>0.000234</td>
      <td>2</td>
      <td>0.1590</td>
      <td>-7.148</td>
      <td>1</td>
      <td>0.2890</td>
      <td>75.044</td>
      <td>4.0</td>
      <td>0.173</td>
      <td>1</td>
      <td>Xanny Family</td>
      <td>Future</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.6040</td>
      <td>0.494</td>
      <td>199413</td>
      <td>0.338</td>
      <td>0.510000</td>
      <td>5</td>
      <td>0.0922</td>
      <td>-15.236</td>
      <td>1</td>
      <td>0.0261</td>
      <td>86.468</td>
      <td>4.0</td>
      <td>0.230</td>
      <td>1</td>
      <td>Master Of None</td>
      <td>Beach House</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.1800</td>
      <td>0.678</td>
      <td>392893</td>
      <td>0.561</td>
      <td>0.512000</td>
      <td>5</td>
      <td>0.4390</td>
      <td>-11.648</td>
      <td>0</td>
      <td>0.0694</td>
      <td>174.004</td>
      <td>4.0</td>
      <td>0.904</td>
      <td>1</td>
      <td>Parallel Lines</td>
      <td>Junior Boys</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Take a look at the shape of dataset
print(f'There are {df.shape[0]} rows and {df.shape[1]} columns in the dataset.')
```

    There are 2017 rows and 16 columns in the dataset.



```python
# Lets drop some columns

df.drop(columns=['song_title', 'artist'], inplace= True)

```

**Variables:**   


|    <strong>Variables</strong>        |  Description                                        |
|-------------------|-----------------------------------------------------|
|<strong>Acousticness</strong>  | This value describes how acoustic a song is. A score of 1.0 means the song is most likely to be an acoustic one.|   
|<strong>Danceability</strong>         | Danceability describes how suitablea track is for dancing based on a combination of musical elements (tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least dancable nd 1.0 is most dancable.|   
|<strong>Durationms</strong>         | Duration of a song in seconds.|
|<strong>Energy</strong>   | Represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy.|
|<strong>Instrumentalness</strong>       | This value represents the amount of vocals in the song. The closer it is to 1.0, the more instrumental the song is.|
|<strong>Key</strong>    | A system of functionally related chords deriving from the major and minor scales, with a central note, called the tonic (or keynote).                             |
|<strong>Liveness</strong>       | This value describes the probability that the song was recorded with a live audience. According to the official documentation "a value above 0.8 provides strong likelihood that track is live". |
|<strong>Loudness</strong>    |The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Values typical range between -60 and 0 db.|
|<strong>Mode</strong>|Mode indicates the modality(major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0.|
|<strong>Speechiness</strong>|Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Value above 0.66 describe tracks that are probably made entirely of spoken words. Values betweeen 0.33 and 0.66 describe tracks that may contain both music and speech, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks.      |
|<strong>Tempo</strong>    |The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration.|
|<strong>Time_signature</strong>|An estimated overall time signature of a track. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure).|
|<strong>Valence</strong>|A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry).|


**Target:**  

|    <strong>Target</strong>        |  Description                                        |
|-------------------|-----------------------------------------------------|
|<strong>Target</strong>|Boolean values, 1 is like and 0 is dislike|



```python
# Quickly check the data types
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 2017 entries, 0 to 2016
    Data columns (total 14 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   acousticness      2017 non-null   float64
     1   danceability      2017 non-null   float64
     2   duration_ms       2017 non-null   int64  
     3   energy            2017 non-null   float64
     4   instrumentalness  2017 non-null   float64
     5   key               2017 non-null   int64  
     6   liveness          2017 non-null   float64
     7   loudness          2017 non-null   float64
     8   mode              2017 non-null   int64  
     9   speechiness       2017 non-null   float64
     10  tempo             2017 non-null   float64
     11  time_signature    2017 non-null   float64
     12  valence           2017 non-null   float64
     13  target            2017 non-null   int64  
    dtypes: float64(10), int64(4)
    memory usage: 220.7 KB



```python
# Check for nulls
df.isnull().sum().any()
```




    False




```python
# Check for duplicates:
# Columns 
print(f'Duplicated columns: {df.T.duplicated().any()}')
# Rows 
print(f'Duplicated rows: {df.duplicated().any()}')
```

    Duplicated columns: False
    Duplicated rows: True


_Ideally, we would want to take the necessary steps to ensure that the data set is clean. However, today we have the luxuries of working with a cleaned dataset._



```python
# Summary statistics 
df.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>acousticness</th>
      <td>2017.0</td>
      <td>0.187590</td>
      <td>0.259989</td>
      <td>0.000003</td>
      <td>0.00963</td>
      <td>0.063300</td>
      <td>0.265</td>
      <td>0.995</td>
    </tr>
    <tr>
      <th>danceability</th>
      <td>2017.0</td>
      <td>0.618422</td>
      <td>0.161029</td>
      <td>0.122000</td>
      <td>0.51400</td>
      <td>0.631000</td>
      <td>0.738</td>
      <td>0.984</td>
    </tr>
    <tr>
      <th>duration_ms</th>
      <td>2017.0</td>
      <td>246306.197323</td>
      <td>81981.814219</td>
      <td>16042.000000</td>
      <td>200015.00000</td>
      <td>229261.000000</td>
      <td>270333.000</td>
      <td>1004627.000</td>
    </tr>
    <tr>
      <th>energy</th>
      <td>2017.0</td>
      <td>0.681577</td>
      <td>0.210273</td>
      <td>0.014800</td>
      <td>0.56300</td>
      <td>0.715000</td>
      <td>0.846</td>
      <td>0.998</td>
    </tr>
    <tr>
      <th>instrumentalness</th>
      <td>2017.0</td>
      <td>0.133286</td>
      <td>0.273162</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>0.000076</td>
      <td>0.054</td>
      <td>0.976</td>
    </tr>
    <tr>
      <th>key</th>
      <td>2017.0</td>
      <td>5.342588</td>
      <td>3.648240</td>
      <td>0.000000</td>
      <td>2.00000</td>
      <td>6.000000</td>
      <td>9.000</td>
      <td>11.000</td>
    </tr>
    <tr>
      <th>liveness</th>
      <td>2017.0</td>
      <td>0.190844</td>
      <td>0.155453</td>
      <td>0.018800</td>
      <td>0.09230</td>
      <td>0.127000</td>
      <td>0.247</td>
      <td>0.969</td>
    </tr>
    <tr>
      <th>loudness</th>
      <td>2017.0</td>
      <td>-7.085624</td>
      <td>3.761684</td>
      <td>-33.097000</td>
      <td>-8.39400</td>
      <td>-6.248000</td>
      <td>-4.746</td>
      <td>-0.307</td>
    </tr>
    <tr>
      <th>mode</th>
      <td>2017.0</td>
      <td>0.612295</td>
      <td>0.487347</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
    <tr>
      <th>speechiness</th>
      <td>2017.0</td>
      <td>0.092664</td>
      <td>0.089931</td>
      <td>0.023100</td>
      <td>0.03750</td>
      <td>0.054900</td>
      <td>0.108</td>
      <td>0.816</td>
    </tr>
    <tr>
      <th>tempo</th>
      <td>2017.0</td>
      <td>121.603272</td>
      <td>26.685604</td>
      <td>47.859000</td>
      <td>100.18900</td>
      <td>121.427000</td>
      <td>137.849</td>
      <td>219.331</td>
    </tr>
    <tr>
      <th>time_signature</th>
      <td>2017.0</td>
      <td>3.968270</td>
      <td>0.255853</td>
      <td>1.000000</td>
      <td>4.00000</td>
      <td>4.000000</td>
      <td>4.000</td>
      <td>5.000</td>
    </tr>
    <tr>
      <th>valence</th>
      <td>2017.0</td>
      <td>0.496815</td>
      <td>0.247195</td>
      <td>0.034800</td>
      <td>0.29500</td>
      <td>0.492000</td>
      <td>0.691</td>
      <td>0.992</td>
    </tr>
    <tr>
      <th>target</th>
      <td>2017.0</td>
      <td>0.505702</td>
      <td>0.500091</td>
      <td>0.000000</td>
      <td>0.00000</td>
      <td>1.000000</td>
      <td>1.000</td>
      <td>1.000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Lets check out the distributions of the tempo column

# Specify the positive and negative tempo
pos_tempo = df[df['target'] == 1]['tempo']
neg_tempo = df[df['target'] == 0]['tempo']

plt.figure(figsize = (12, 8))
pos_tempo.hist(alpha = 0.6, bins = 30, label='positive')
neg_tempo.hist(alpha = 0.6, bins = 30, label='negative')
plt.title('Song Tempo Like / Dislike Distribution')
plt.legend(loc = "upper right")

```




    <matplotlib.legend.Legend at 0x7f8d5a0bdd30>




    
![output_13_1](https://user-images.githubusercontent.com/70767722/124399237-28cd5200-dce8-11eb-9844-18174e5b9456.png)    



```python
# Lets check out the columns' distribution.

# Custom Color Palette
red_blue = ['#19B5FE', '#EF4836']
palette = sns.color_palette(red_blue)
sns.set_palette(palette)
sns.set_style('white')

# Setting the plots layout
plt.subplots(3,4, figsize=(20,15))

# Plotting
for i, column in enumerate(df.columns, 1):
    plt.subplot(4,4,i)
    plt.hist(df[df['target']==1][column], alpha =0.6, bins=30, label= 'positive')
    plt.hist(df[df['target']==0][column], alpha =0.6, bins=30, label= 'negative')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.title(f'song {column} like / dislike distribution')
    plt.legend()
plt.tight_layout()
```


    
![output_14_0](https://user-images.githubusercontent.com/70767722/124399239-2cf96f80-dce8-11eb-89bb-08aa3b015eca.png)    


- For the `danceability` feature, we can see a slight bias toward disliking song with lower density index and a slight preferece for songs with higher danceability index.
- For the `key` feature, we have 12 different keys which can be called the standard pitch class notation. For example, 0 is the key of C seems to be the second highest observation; it seems like 3 which is D sharp has the lowest number of observations as well as the highest relative distribution of dislike.
- For the song `loudness` feature, there is a big spike for songs at the very extreme end of loudness.

### Step 2: Building Decision Tree Classifier

We will: 

- Assign our features and target variables to `X` and `y`, respectively, 

- Create our training ( `X_train` ) and test ( `X_test` ) sets using `train_test_split()`,

- Scale `X_train` and `X_test` using the `MinMaxScaler()`, and 

- Standardize `X_train` and `X_test` using the `StandardScaler()`. 


```python
# Assign our features to X 
X = df.drop(['target'], axis=1)

# Assign our target to y 
y = df['target']

# Check independent variable and dependent variable
display(X)

display(y)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>acousticness</th>
      <th>danceability</th>
      <th>duration_ms</th>
      <th>energy</th>
      <th>instrumentalness</th>
      <th>key</th>
      <th>liveness</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>tempo</th>
      <th>time_signature</th>
      <th>valence</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.01020</td>
      <td>0.833</td>
      <td>204600</td>
      <td>0.434</td>
      <td>0.021900</td>
      <td>2</td>
      <td>0.1650</td>
      <td>-8.795</td>
      <td>1</td>
      <td>0.4310</td>
      <td>150.062</td>
      <td>4.0</td>
      <td>0.286</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.19900</td>
      <td>0.743</td>
      <td>326933</td>
      <td>0.359</td>
      <td>0.006110</td>
      <td>1</td>
      <td>0.1370</td>
      <td>-10.401</td>
      <td>1</td>
      <td>0.0794</td>
      <td>160.083</td>
      <td>4.0</td>
      <td>0.588</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.03440</td>
      <td>0.838</td>
      <td>185707</td>
      <td>0.412</td>
      <td>0.000234</td>
      <td>2</td>
      <td>0.1590</td>
      <td>-7.148</td>
      <td>1</td>
      <td>0.2890</td>
      <td>75.044</td>
      <td>4.0</td>
      <td>0.173</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.60400</td>
      <td>0.494</td>
      <td>199413</td>
      <td>0.338</td>
      <td>0.510000</td>
      <td>5</td>
      <td>0.0922</td>
      <td>-15.236</td>
      <td>1</td>
      <td>0.0261</td>
      <td>86.468</td>
      <td>4.0</td>
      <td>0.230</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.18000</td>
      <td>0.678</td>
      <td>392893</td>
      <td>0.561</td>
      <td>0.512000</td>
      <td>5</td>
      <td>0.4390</td>
      <td>-11.648</td>
      <td>0</td>
      <td>0.0694</td>
      <td>174.004</td>
      <td>4.0</td>
      <td>0.904</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2012</th>
      <td>0.00106</td>
      <td>0.584</td>
      <td>274404</td>
      <td>0.932</td>
      <td>0.002690</td>
      <td>1</td>
      <td>0.1290</td>
      <td>-3.501</td>
      <td>1</td>
      <td>0.3330</td>
      <td>74.976</td>
      <td>4.0</td>
      <td>0.211</td>
    </tr>
    <tr>
      <th>2013</th>
      <td>0.08770</td>
      <td>0.894</td>
      <td>182182</td>
      <td>0.892</td>
      <td>0.001670</td>
      <td>1</td>
      <td>0.0528</td>
      <td>-2.663</td>
      <td>1</td>
      <td>0.1310</td>
      <td>110.041</td>
      <td>4.0</td>
      <td>0.867</td>
    </tr>
    <tr>
      <th>2014</th>
      <td>0.00857</td>
      <td>0.637</td>
      <td>207200</td>
      <td>0.935</td>
      <td>0.003990</td>
      <td>0</td>
      <td>0.2140</td>
      <td>-2.467</td>
      <td>1</td>
      <td>0.1070</td>
      <td>150.082</td>
      <td>4.0</td>
      <td>0.470</td>
    </tr>
    <tr>
      <th>2015</th>
      <td>0.00164</td>
      <td>0.557</td>
      <td>185600</td>
      <td>0.992</td>
      <td>0.677000</td>
      <td>1</td>
      <td>0.0913</td>
      <td>-2.735</td>
      <td>1</td>
      <td>0.1330</td>
      <td>150.011</td>
      <td>4.0</td>
      <td>0.623</td>
    </tr>
    <tr>
      <th>2016</th>
      <td>0.00281</td>
      <td>0.446</td>
      <td>204520</td>
      <td>0.915</td>
      <td>0.000039</td>
      <td>9</td>
      <td>0.2180</td>
      <td>-6.221</td>
      <td>1</td>
      <td>0.1410</td>
      <td>190.013</td>
      <td>4.0</td>
      <td>0.402</td>
    </tr>
  </tbody>
</table>
<p>2017 rows × 13 columns</p>
</div>



    0       1
    1       1
    2       1
    3       1
    4       1
           ..
    2012    0
    2013    0
    2014    0
    2015    0
    2016    0
    Name: target, Length: 2017, dtype: int64



```python
# Create our training and test sets, 20% test size, random state of 5 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5, stratify=y)

# Check shape 
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
```

    (1613, 13) (404, 13) (1613,) (404,)


Now that we have our `X_train` and `X_test` let's transform them with our two different scalers. 

Remember, we need to: 
1. Instantiate the scaler
2. Fit the scaler to the **training** data 
3. Transform both the training and test features

_While we didn't need to scale our data for our Decision Tree model performance, it's best practice to scale our data anyways._

**MinMaxScaler**


```python
# Instantiate the scaler
MM = MinMaxScaler()

# Fit the scaler
MM.fit(X_train)

# Transform the training and test sets 
X_train_mm = MM.transform(X_train)
X_test_mm = MM.transform(X_test)
```

**StandardScaler**


```python
# Instantiate the scaler
SS = StandardScaler()

# Fit the scaler
SS.fit(X_train)

# Transform the training and test sets 
X_train_ss = SS.transform(X_train)
X_test_ss = SS.transform(X_test)
```

The goal of the tree is to ultimately split observations into groups of homogenous target values (0 or 1), giving us a set of "paths" to follow to determine if this user liked or disliked a particular song.

Now we will fit sklearn's `DecisionTreeClassifier()` with different `max_depth`s for both sets of transformed data. 

The paramater `max_depth` is what the name suggests: The maximum depth that you allow the tree to grow to. In general, the deeper you allow your tree to grow, the more complex your model will become because you will have more splits and it captures more information about the data and this is one of the root causes of overfitting in decision trees because your model will fit perfectly for the training data and will not be able to generalize well on test set.

**MinMaxScaler:**


```python
# Check the numbers of row in train set:
X_train.shape[1]
```




    13




```python
range(1, int(np.sqrt(X_train.shape[0])))
```




    range(1, 40)



_The theoritical maximum depth a decision tree can achieve is one less than the number of training samples, but that will lead to severe overfitting, so I have decided to test out a range of 1 to 40._


```python
# A list of the maximum depths to try out and save to 'depths'
depths = range(1, int(np.sqrt(X_train.shape[0])))

# Empty lists to append to
train_acc_mm = []
test_acc_mm = []

# Loop through the depths
for max_depth in depths:
    
    # Instantiate the model 
    DT = DecisionTreeClassifier(max_depth=max_depth)
    
    # Fit the model 
    DT.fit(X_train_mm, y_train)
    
    # Score the model 
    train_acc_mm.append(DT.score(X_train_mm, y_train))
    test_acc_mm.append(DT.score(X_test_mm, y_test))
```


```python
# Plot the accuracies
plt.figure()
plt.plot(depths, train_acc_mm, c='red', label='train')
plt.plot(depths, test_acc_mm, c='blue', label='test')
plt.xlabel('max_depth')
plt.ylabel('accuracy score')
plt.legend()
plt.show()
```


    
![output_29_0](https://user-images.githubusercontent.com/70767722/124399244-3387e700-dce8-11eb-805c-2e85174f65be.png)    


We will select `max_depth=5`. 


```python
# The best DT

# Instantiate the model 
DT_mm = DecisionTreeClassifier(max_depth=5)

# Fit the model 
DT_mm.fit(X_train_mm, y_train)

# Score
print(DT_mm.score(X_train_mm, y_train))
print(DT_mm.score(X_test_mm, y_test))
```

    0.78735275883447
    0.7153465346534653


**StandardScaler:**


```python
# Empty lists to append to 
train_acc_ss = []
test_acc_ss = []

# Loop through the different depths
for max_depth in depths: 
    
    # Instantiate the model 
    DT = DecisionTreeClassifier(max_depth=max_depth)
    
    # Fit the model 
    DT.fit(X_train_ss, y_train)
    
    # Score the model 
    train_acc_ss.append(DT.score(X_train_ss, y_train))
    test_acc_ss.append(DT.score(X_test_ss, y_test))
```


```python
# Plot the accuracies
plt.figure()
plt.plot(depths, train_acc_ss, c='red', label='train')
plt.plot(depths, test_acc_ss, c='blue', label='test')
plt.xlabel('max_depth')
plt.ylabel('accuracy score')
plt.legend()
plt.show()
```


    
![output_34_0](https://user-images.githubusercontent.com/70767722/124399248-37b40480-dce8-11eb-8c03-398a9ff97cdd.png)    


`max_depth=5` again. 


```python
# The DT that performed best on the standardized data

# Instantiate the model 
DT_ss = DecisionTreeClassifier(max_depth=5)

# Fit the model 
DT_ss.fit(X_train_ss, y_train)

# Score
print(DT_ss.score(X_train_ss, y_train))
print(DT_ss.score(X_test_ss, y_test))
```

    0.78735275883447
    0.7153465346534653


Between the two sklearn scalers, `DecisionTreeClassifier()` performance on both for a given max_depth is almost identical. Even looking between the two figures we made, it's difficult to distinguish the differences. 

If we recall from our *Decision Trees* lecture, this is because every split or decision is made based on a single attribute at a time, so for Decision Trees it doesn't matter whether the different attributes are of different scales! 



```python
# Plot Tree

fig, ax = plt.subplots(figsize=(30,20))

out = plot_tree(DT_ss,
                feature_names=X.columns,
                filled=True,
                ax=ax,
                fontsize=12)

for o in out:
    arrow = o.arrow_patch
    if arrow is not None:
        arrow.set_edgecolor('red')
        arrow.set_linewidth(2)
```


    
![output_38_0](https://user-images.githubusercontent.com/70767722/124399251-3a165e80-dce8-11eb-9569-84fca2725724.png)    


This tree is straightforward and one of the advantages of the decision trees is they are interpretable. We can look at the process that the model has used to classify whether or not am user liked or disliked a particular song.
We can look at any nodes in this tree here and see explitcitly the decision that is being made to classify a song into one or the other value.

The other advantage of decision trees is the fact that we have to do very little preparation of our data. In fact, our data have a different sort of types. For example, most of our features have continous values ("Loudness", "Duration", and "Speechiness"); whereas, the other feature as "Key" is discreet in which we have 12 possible values of "Key" to predict.

You can also try KNN Classifier with this dataset and compare between 2 models.

END.
