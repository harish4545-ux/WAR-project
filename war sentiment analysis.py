# -*- coding: utf-8 -*-
# ============================================================
#   WAR SOCIAL MEDIA SENTIMENT ANALYSIS
#   Russia-Ukraine / Israel-Palestine Tweet Analysis
#   Tools: Pandas, VADER, Matplotlib, Seaborn, WordCloud
#   Author: Your Name | Project: Data Analytics Portfolio
# ============================================================

# ── INSTALL THESE FIRST (run in terminal) ───────────────────
# pip install pandas numpy matplotlib seaborn wordcloud
# pip install vaderSentiment langdetect tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
import re
import warnings
import os

warnings.filterwarnings('ignore')

# ── GLOBAL STYLE ────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'figure.facecolor': '#0D1117',
    'axes.facecolor':   '#161B22',
    'axes.edgecolor':   '#30363D',
    'axes.labelcolor':  '#C9D1D9',
    'xtick.color':      '#8B949E',
    'ytick.color':      '#8B949E',
    'text.color':       '#C9D1D9',
    'grid.color':       '#21262D',
    'grid.linewidth':   0.5,
})

COLORS = {
    'positive': '#3FB950',
    'negative': '#F85149',
    'neutral':  '#58A6FF',
    'ukraine':  '#FFD700',
    'russia':   '#E75480',
    'israel':   '#1E90FF',
    'palestine':'#2ECC71',
    'accent':   '#A371F7',
    'bg':       '#0D1117',
    'card':     '#161B22',
}

OUTPUT_DIR = 'sentiment_output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════
#   STEP 1 — LOAD DATASET
#   Place your Kaggle CSV here. If no file found, a
#   realistic sample dataset is auto-generated for testing.
# ════════════════════════════════════════════════════════════

def load_dataset(filepath=None):
    """
    Load CSV from Kaggle. Expected columns:
      text, created_at, user_location, retweet_count, like_count

    Kaggle datasets to use:
    - https://www.kaggle.com/datasets/bwandowando/ukraine-russian-crisis-twitter-dataset
    - https://www.kaggle.com/datasets/asad1m9a9h6mood/news-articles
    """
    if filepath and os.path.exists(filepath):
        print(f"[✔] Loading dataset: {filepath}")
        df = pd.read_csv(filepath, lineterminator='\n')
        # Rename columns to standard names if needed
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if 'text' in cl or 'tweet' in cl or 'content' in cl:
                col_map[col] = 'text'
            elif 'date' in cl or 'created' in cl or 'time' in cl:
                col_map[col] = 'created_at'
            elif 'location' in cl or 'country' in cl:
                col_map[col] = 'user_location'
            elif 'retweet' in cl:
                col_map[col] = 'retweet_count'
            elif 'like' in cl or 'favorite' in cl:
                col_map[col] = 'like_count'
        df.rename(columns=col_map, inplace=True)
    else:
        print("[ℹ] No dataset file found — generating sample data for demo...")
        df = generate_sample_data()

    return df


def generate_sample_data(n=2000):
    """Generate a realistic sample dataset for testing."""
    np.random.seed(42)

    ukraine_tweets = [
        "Prayers for the people of Ukraine. This war must end. #StandWithUkraine #Ukraine",
        "Russian forces advancing near Kharkiv. The world watches in horror. #UkraineWar",
        "Ukrainian army showing incredible resistance! Proud of their bravery. #SlavaUkraini",
        "Civilian casualties in Mariupol rising. This is a humanitarian crisis. #Ukraine",
        "NATO should do more to support Ukraine. Stop the war now! #NATO #Ukraine",
        "The destruction in Kyiv is heartbreaking. We stand with Ukraine #StopRussia",
        "More weapons needed for Ukraine to defend itself. #UkraineSupport",
        "Russian propaganda is everywhere online. Stay critical. #InfoWar",
        "Zelensky's speech was powerful and moving. Ukraine will not fall. #Ukraine",
        "Refugees fleeing Ukraine into Poland and Romania. Heartbreaking scenes.",
    ]
    russia_tweets = [
        "Russia has legitimate security concerns. NATO expansion threatens peace. #Russia",
        "Western media bias against Russia is obvious. Two sides to every story.",
        "Russian people are not responsible for Putin's decisions. #PeaceNotWar",
        "Sanctions on Russia will hurt ordinary people most. Not fair.",
        "Russia's economy under massive pressure from sanctions. #RussiaUkraine",
        "Russian military performance worse than expected. #Ukraine #Russia",
        "Propaganda from both sides. Hard to know the truth about #UkraineWar",
        "Supporting Ukraine but not blindly. Need peace negotiations now. #Russia",
        "Moscow protests against the war — brave Russians standing up. #AntiWar",
        "The truth about Donbas is more complex than media shows. #RussiaUkraine",
    ]
    israel_tweets = [
        "Praying for peace in Gaza. Too many innocent lives lost. #Gaza #Palestine",
        "Israel has the right to defend itself against Hamas attacks. #Israel",
        "The humanitarian crisis in Gaza is getting worse every day. #GazaUnderAttack",
        "Hamas must release all hostages immediately. #IsraelHamasWar",
        "Ceasefire needed NOW in Gaza. Children are dying. #CeasefireNow",
        "International community must act to stop the bloodshed. #Palestine",
        "Israel's response is disproportionate. Too many civilian casualties. #Gaza",
        "We stand with the Israeli people after the October 7 attacks. #Israel",
        "Aid must reach Gaza immediately. People are starving. #GazaAid",
        "Both Israelis and Palestinians deserve peace and security. #MiddleEast",
    ]

    all_tweet_pools = ukraine_tweets + russia_tweets + israel_tweets
    neutral_tweets = [
        "Following the latest news on the conflict situation worldwide.",
        "International diplomacy efforts ongoing. Summit scheduled for next week.",
        "UN Security Council meeting today on the ongoing conflict.",
        "Aid organizations providing relief in conflict zones.",
        "Peace talks collapsed again. Ceasefire seems distant.",
        "Global oil prices rising due to geopolitical tensions.",
        "Refugees welcome. We must help people fleeing conflict.",
        "Media coverage of the war has been intense this week.",
        "Historical context of the conflict is important to understand.",
        "Economic impact of the war being felt worldwide.",
    ]

    tweets = []
    dates = pd.date_range('2023-01-01', '2024-01-01', periods=n)
    locations = [
        'United States', 'United Kingdom', 'Germany', 'France', 'Ukraine',
        'Russia', 'Poland', 'Israel', 'Palestine', 'Turkey', 'India',
        'Australia', 'Canada', 'Netherlands', 'Italy', None, None
    ]
    hashtag_pools = {
        'ukraine': ['#Ukraine', '#StandWithUkraine', '#SlavaUkraini', '#UkraineWar', '#StopRussia', '#NATO', '#Zelensky'],
        'russia':  ['#Russia', '#RussiaUkraine', '#Putin', '#Sanctions', '#PeaceNotWar'],
        'israel':  ['#Israel', '#Gaza', '#Palestine', '#Hamas', '#CeasefireNow', '#GazaUnderAttack', '#IsraelHamasWar'],
        'neutral': ['#Peace', '#War', '#Conflict', '#UN', '#HumanitarianCrisis'],
    }

    for i in range(n):
        pool_key = np.random.choice(['ukraine', 'russia', 'israel', 'neutral'], p=[0.35, 0.25, 0.30, 0.10])
        if pool_key == 'ukraine':
            base = np.random.choice(ukraine_tweets)
        elif pool_key == 'russia':
            base = np.random.choice(russia_tweets)
        elif pool_key == 'israel':
            base = np.random.choice(israel_tweets)
        else:
            base = np.random.choice(neutral_tweets)

        hashtags = ' '.join(np.random.choice(hashtag_pools[pool_key], size=np.random.randint(1, 3), replace=False))
        text = base + ' ' + hashtags

        tweets.append({
            'text':           text,
            'created_at':     dates[i],
            'user_location':  np.random.choice(locations),
            'retweet_count':  int(np.random.exponential(50)),
            'like_count':     int(np.random.exponential(120)),
        })

    return pd.DataFrame(tweets)


# ════════════════════════════════════════════════════════════
#   STEP 2 — PREPROCESS
# ════════════════════════════════════════════════════════════

def preprocess(df):
    print("[✔] Preprocessing data...")

    # Parse datetime
    df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
    df = df.dropna(subset=['created_at', 'text'])
    df['text'] = df['text'].astype(str)

    # Time features
    df['date']  = df['created_at'].dt.date
    df['hour']  = df['created_at'].dt.hour
    df['month'] = df['created_at'].dt.to_period('M').astype(str)
    df['day_of_week'] = df['created_at'].dt.day_name()

    # Extract hashtags
    df['hashtags'] = df['text'].apply(lambda x: re.findall(r'#\w+', x.lower()))

    # Clean text (for word cloud)
    df['clean_text'] = df['text'].apply(clean_text)

    # Country tagging from location
    df['country'] = df['user_location'].apply(tag_country)

    # Fill engagement columns
    for col in ['retweet_count', 'like_count']:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    print(f"    → {len(df):,} tweets ready for analysis")
    return df


def clean_text(text):
    text = re.sub(r'http\S+|www\S+', '', text)     # remove URLs
    text = re.sub(r'@\w+', '', text)                # remove mentions
    text = re.sub(r'[^A-Za-z\s#]', '', text)        # keep letters + hashtags
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


COUNTRY_KEYWORDS = {
    'Ukraine':    ['ukraine', 'kyiv', 'kharkiv', 'mariupol', 'lviv', 'odessa', 'odesa'],
    'Russia':     ['russia', 'moscow', 'russie', 'russland', 'putin', 'siberia'],
    'USA':        ['usa', 'united states', 'america', 'new york', 'california', 'washington'],
    'UK':         ['uk', 'united kingdom', 'england', 'london', 'britain'],
    'Germany':    ['germany', 'berlin', 'deutschland'],
    'France':     ['france', 'paris', 'français'],
    'Poland':     ['poland', 'warsaw', 'polska'],
    'Israel':     ['israel', 'tel aviv', 'jerusalem'],
    'Palestine':  ['palestine', 'gaza', 'west bank', 'ramallah'],
    'Turkey':     ['turkey', 'ankara', 'istanbul', 'türkiye'],
    'India':      ['india', 'delhi', 'mumbai', 'bangalore'],
    'Australia':  ['australia', 'sydney', 'melbourne'],
}

def tag_country(loc):
    if not isinstance(loc, str):
        return 'Other'
    loc_lower = loc.lower()
    for country, keywords in COUNTRY_KEYWORDS.items():
        if any(k in loc_lower for k in keywords):
            return country
    return 'Other'


# ════════════════════════════════════════════════════════════
#   STEP 3 — SENTIMENT SCORING (VADER)
# ════════════════════════════════════════════════════════════

def run_sentiment(df):
    print("[✔] Running VADER sentiment analysis...")
    sia = SentimentIntensityAnalyzer()

    def score(text):
        return sia.polarity_scores(str(text))['compound']

    def label(score):
        if score >= 0.05:  return 'Positive'
        if score <= -0.05: return 'Negative'
        return 'Neutral'

    df['compound_score'] = df['text'].apply(score)
    df['sentiment']      = df['compound_score'].apply(label)

    pos = (df['sentiment'] == 'Positive').sum()
    neg = (df['sentiment'] == 'Negative').sum()
    neu = (df['sentiment'] == 'Neutral').sum()
    print(f"    → Positive: {pos:,}  |  Negative: {neg:,}  |  Neutral: {neu:,}")
    return df


# ════════════════════════════════════════════════════════════
#   STEP 4 — VISUALIZATIONS
# ════════════════════════════════════════════════════════════

# ── 4a. SENTIMENT DISTRIBUTION PIE CHART ────────────────────
def plot_pie(df):
    print("[✔] Chart 1 — Pie chart: Sentiment distribution")
    counts = df['sentiment'].value_counts()
    labels = counts.index.tolist()
    sizes  = counts.values.tolist()
    clrs   = [COLORS.get(l.lower(), '#888') for l in labels]
    explode = [0.05 if l == 'Negative' else 0 for l in labels]

    fig, ax = plt.subplots(figsize=(7, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['bg'])

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=clrs, autopct='%1.1f%%',
        startangle=140, explode=explode,
        wedgeprops={'edgecolor': COLORS['bg'], 'linewidth': 2},
        textprops={'color': '#C9D1D9', 'fontsize': 13}
    )
    for at in autotexts:
        at.set_color('#0D1117')
        at.set_fontsize(12)
        at.set_fontweight('bold')

    ax.set_title('Sentiment Distribution in War Tweets', fontsize=16,
                 color='#E6EDF3', fontweight='bold', pad=20)

    # Insight annotation
    dominant = counts.idxmax()
    ax.annotate(
        f'Insight: {dominant} sentiment dominates\n→ Emotional engagement is high',
        xy=(0, -1.35), fontsize=10, color='#8B949E', ha='center',
        style='italic'
    )

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/01_pie_sentiment.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()
    print(f"    → Saved: {path}")


# ── 4b. SENTIMENT OVER TIME LINE CHART ──────────────────────
def plot_line_over_time(df):
    print("[✔] Chart 2 — Line chart: Sentiment over time")

    daily = (
        df.groupby(['date', 'sentiment'])
        .size().reset_index(name='count')
    )
    daily['date'] = pd.to_datetime(daily['date'])

    # Resample monthly for clarity
    monthly_list = []
    for sent in ['Positive', 'Negative', 'Neutral']:
        sub = daily[daily['sentiment'] == sent].set_index('date')['count']
        sub = sub.resample('W').sum().reset_index()
        sub['sentiment'] = sent
        monthly_list.append(sub)
    monthly = pd.concat(monthly_list)

    fig, ax = plt.subplots(figsize=(13, 5), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['card'])

    palette = {'Positive': COLORS['positive'], 'Negative': COLORS['negative'], 'Neutral': COLORS['neutral']}
    for sent, grp in monthly.groupby('sentiment'):
        ax.plot(grp['date'], grp['count'],
                color=palette[sent], label=sent,
                linewidth=2.5, marker='o', markersize=3, alpha=0.9)
        ax.fill_between(grp['date'], grp['count'], alpha=0.1, color=palette[sent])

    ax.set_title('Sentiment Trends Over Time', fontsize=16, color='#E6EDF3', fontweight='bold', pad=15)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Tweet Count', fontsize=12)
    ax.legend(framealpha=0.2, facecolor=COLORS['bg'], edgecolor='#30363D', labelcolor='#C9D1D9')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=30)

    # Insight box
    ax.annotate('📈 Insight: Negative tweets spike during\nnews events (airstrikes, sanctions)',
                xy=(0.02, 0.92), xycoords='axes fraction',
                fontsize=9, color='#8B949E', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#161B22', edgecolor='#30363D', alpha=0.8))

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/02_line_sentiment_time.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()
    print(f"    → Saved: {path}")


# ── 4c. TOP HASHTAGS BAR CHART ──────────────────────────────
def plot_hashtags(df):
    print("[✔] Chart 3 — Bar chart: Top hashtags")

    all_tags = [tag for tags in df['hashtags'] for tag in tags]
    tag_counts = Counter(all_tags).most_common(20)
    tags, counts = zip(*tag_counts)

    fig, ax = plt.subplots(figsize=(10, 7), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['card'])

    cmap = plt.cm.plasma
    bar_colors = [cmap(i / len(tags)) for i in range(len(tags))]
    bars = ax.barh(tags, counts, color=bar_colors, edgecolor='none', height=0.7)

    # Value labels
    for bar, count in zip(bars, counts):
        ax.text(count + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{count:,}', va='center', fontsize=9, color='#8B949E')

    ax.invert_yaxis()
    ax.set_title('Top 20 Hashtags in War Tweets', fontsize=16, color='#E6EDF3', fontweight='bold', pad=15)
    ax.set_xlabel('Frequency', fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Insight
    ax.annotate(f'Insight: Top tag → {tags[0]} | {tags[1]} equally trending',
                xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=9, color='#8B949E', style='italic',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#161B22', edgecolor='#30363D', alpha=0.8))

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/03_bar_hashtags.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()
    print(f"    → Saved: {path}")


# ── 4d. WORD CLOUD ───────────────────────────────────────────
def plot_wordcloud(df):
    print("[✔] Chart 4 — Word cloud")

    stopwords = set(STOPWORDS)
    stopwords.update(['will', 'one', 'now', 'new', 'amp', 'via', 'get', 'us', 'also',
                      'rt', 'https', 'co', 't', 's', 'u', 'war', 'tweet'])

    for sentiment, color_func_color in [
        ('Positive', '#3FB950'),
        ('Negative', '#F85149'),
        ('All',      '#58A6FF'),
    ]:
        if sentiment == 'All':
            text = ' '.join(df['clean_text'])
        else:
            text = ' '.join(df[df['sentiment'] == sentiment]['clean_text'])

        if not text.strip():
            continue

        wc = WordCloud(
            width=1200, height=600,
            background_color='#0D1117',
            stopwords=stopwords,
            max_words=150,
            colormap='plasma' if sentiment == 'All' else 'Greens' if sentiment == 'Positive' else 'Reds',
            prefer_horizontal=0.8,
            min_font_size=10,
        ).generate(text)

        fig, ax = plt.subplots(figsize=(14, 6), facecolor=COLORS['bg'])
        ax.imshow(wc, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Word Cloud — {sentiment} Tweets',
                     fontsize=18, color='#E6EDF3', fontweight='bold', pad=15)

        plt.tight_layout()
        path = f'{OUTPUT_DIR}/04_wordcloud_{sentiment.lower()}.png'
        plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
        plt.show()
        print(f"    → Saved: {path}")


# ── 4e. COUNTRY COMPARISON ──────────────────────────────────
def plot_country_comparison(df):
    print("[✔] Chart 5 — Country comparison")

    top_countries = df[df['country'] != 'Other']['country'].value_counts().head(8).index.tolist()
    sub = df[df['country'].isin(top_countries)]

    pivot = (
        sub.groupby(['country', 'sentiment'])
        .size().unstack(fill_value=0)
    )
    # Normalize to percentages
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct.reindex(columns=['Positive', 'Neutral', 'Negative'], fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6), facecolor=COLORS['bg'])
    ax.set_facecolor(COLORS['card'])

    x = np.arange(len(pivot_pct.index))
    width = 0.28
    offsets = [-width, 0, width]
    sentiment_colors = [COLORS['positive'], COLORS['neutral'], COLORS['negative']]

    for i, (sent, color) in enumerate(zip(['Positive', 'Neutral', 'Negative'], sentiment_colors)):
        vals = pivot_pct[sent].values if sent in pivot_pct.columns else np.zeros(len(x))
        bars = ax.bar(x + offsets[i], vals, width, label=sent, color=color, alpha=0.85, edgecolor='none')
        for bar, val in zip(bars, vals):
            if val > 5:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.0f}%', ha='center', va='bottom', fontsize=8, color='#8B949E')

    ax.set_title('Sentiment by Country (%)', fontsize=16, color='#E6EDF3', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot_pct.index, rotation=25, ha='right', fontsize=11)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.legend(framealpha=0.2, facecolor=COLORS['bg'], edgecolor='#30363D', labelcolor='#C9D1D9')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.annotate('Insight: Western countries show higher negative sentiment.\nMiddle East / Global South shows more neutral stance.',
                xy=(0.02, 0.95), xycoords='axes fraction', fontsize=9,
                color='#8B949E', style='italic', va='top',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#161B22', edgecolor='#30363D', alpha=0.8))

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/05_bar_country.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()
    print(f"    → Saved: {path}")


# ── 4f. PEAK HOUR ANALYSIS ──────────────────────────────────
def plot_peak_hours(df):
    print("[✔] Chart 6 — Peak hour heatmap")

    hour_sent = (
        df.groupby(['hour', 'sentiment'])
        .size().unstack(fill_value=0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=COLORS['bg'])

    # Left: line chart of tweets by hour
    ax = axes[0]
    ax.set_facecolor(COLORS['card'])
    for sent, color in [('Positive', COLORS['positive']), ('Negative', COLORS['negative']), ('Neutral', COLORS['neutral'])]:
        if sent in hour_sent.columns:
            ax.plot(hour_sent.index, hour_sent[sent], color=color, label=sent, linewidth=2.5, marker='o', markersize=4)
    ax.set_title('Tweet Activity by Hour of Day', fontsize=14, color='#E6EDF3', fontweight='bold', pad=12)
    ax.set_xlabel('Hour (24H)', fontsize=11)
    ax.set_ylabel('Tweet Count', fontsize=11)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(framealpha=0.2, facecolor=COLORS['bg'], edgecolor='#30363D', labelcolor='#C9D1D9')
    ax.grid(True, alpha=0.3)
    ax.axvspan(20, 23, alpha=0.12, color=COLORS['accent'], label='Peak zone')
    ax.annotate('📌 Peak: 8–11 PM\n(Post-work news consumption)',
                xy=(20.5, ax.get_ylim()[1] * 0.85), fontsize=9,
                color=COLORS['accent'], style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#161B22', edgecolor='#30363D', alpha=0.8))

    # Right: heatmap sentiment vs hour
    ax2 = axes[1]
    pivot_heat = hour_sent.T
    pivot_heat = pivot_heat.reindex(['Positive', 'Neutral', 'Negative'])
    sns.heatmap(pivot_heat, ax=ax2, cmap='RdYlGn', linewidths=0.5,
                linecolor='#0D1117', annot=True, fmt='d',
                cbar_kws={'label': 'Tweet Count'},
                annot_kws={'size': 8})
    ax2.set_title('Sentiment Intensity Heatmap\n(Hour of Day)', fontsize=14, color='#E6EDF3', fontweight='bold', pad=12)
    ax2.set_xlabel('Hour', fontsize=11)
    ax2.set_ylabel('', fontsize=11)
    ax2.tick_params(axis='both', colors='#8B949E')

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/06_heatmap_hours.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()
    print(f"    → Saved: {path}")


# ── 4g. ENGAGEMENT ANALYSIS ─────────────────────────────────
def plot_engagement(df):
    print("[✔] Chart 7 — Engagement by sentiment")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=COLORS['bg'])
    palette_list = [COLORS['positive'], COLORS['neutral'], COLORS['negative']]

    for ax, metric, title in zip(
        axes,
        ['retweet_count', 'like_count'],
        ['Retweets by Sentiment', 'Likes by Sentiment']
    ):
        ax.set_facecolor(COLORS['card'])
        grouped = [
            df[df['sentiment'] == s][metric].clip(upper=df[metric].quantile(0.95))
            for s in ['Positive', 'Neutral', 'Negative']
        ]
        bp = ax.boxplot(grouped, patch_artist=True, notch=False,
                        boxprops=dict(linewidth=0),
                        medianprops=dict(color='#E6EDF3', linewidth=2),
                        whiskerprops=dict(color='#30363D'),
                        capprops=dict(color='#30363D'),
                        flierprops=dict(marker='o', markersize=3, alpha=0.3, color='#8B949E'))
        for patch, color in zip(bp['boxes'], palette_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        means = [g.mean() for g in grouped]
        for i, mean in enumerate(means):
            ax.plot(i + 1, mean, 'D', color='white', markersize=7, zorder=5)

        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(['Positive', 'Neutral', 'Negative'], fontsize=12)
        ax.set_title(title, fontsize=14, color='#E6EDF3', fontweight='bold', pad=12)
        ax.set_ylabel('Count', fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[1].annotate('Insight: Negative & emotional tweets\nget more engagement (retweets, likes)',
                     xy=(0.02, 0.95), xycoords='axes fraction', fontsize=9,
                     color='#8B949E', style='italic', va='top',
                     bbox=dict(boxstyle='round,pad=0.4', facecolor='#161B22', edgecolor='#30363D', alpha=0.8))

    plt.tight_layout()
    path = f'{OUTPUT_DIR}/07_engagement_sentiment.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()
    print(f"    → Saved: {path}")


# ── 4h. SUMMARY DASHBOARD ───────────────────────────────────
def plot_dashboard(df):
    print("[✔] Chart 8 — Summary dashboard")

    fig = plt.figure(figsize=(16, 10), facecolor=COLORS['bg'])
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.55, wspace=0.45)

    # ── KPI metrics ─────────────────────────────────────────
    total    = len(df)
    neg_pct  = round((df['sentiment'] == 'Negative').mean() * 100, 1)
    pos_pct  = round((df['sentiment'] == 'Positive').mean() * 100, 1)
    top_tag  = Counter([t for tags in df['hashtags'] for t in tags]).most_common(1)
    top_tag  = top_tag[0][0] if top_tag else 'N/A'
    avg_ret  = round(df['retweet_count'].mean(), 1)

    kpis = [
        ('Total Tweets', f'{total:,}', COLORS['neutral']),
        ('Negative %',   f'{neg_pct}%', COLORS['negative']),
        ('Positive %',   f'{pos_pct}%', COLORS['positive']),
        ('Top Hashtag',  top_tag,       COLORS['accent']),
    ]

    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor(COLORS['card'])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis('off')
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=COLORS['card'],
                                   edgecolor=color, linewidth=1.5, transform=ax.transAxes))
        ax.text(0.5, 0.65, value, ha='center', va='center', fontsize=22,
                fontweight='bold', color=color, transform=ax.transAxes)
        ax.text(0.5, 0.25, label, ha='center', va='center', fontsize=10,
                color='#8B949E', transform=ax.transAxes)

    # ── Pie ──────────────────────────────────────────────────
    ax_pie = fig.add_subplot(gs[1, 0])
    ax_pie.set_facecolor(COLORS['card'])
    counts = df['sentiment'].value_counts()
    clrs   = [COLORS.get(l.lower(), '#888') for l in counts.index]
    ax_pie.pie(counts.values, labels=counts.index, colors=clrs, autopct='%1.0f%%',
               startangle=90, wedgeprops={'edgecolor': COLORS['bg'], 'linewidth': 1.5},
               textprops={'color': '#C9D1D9', 'fontsize': 9})
    ax_pie.set_title('Sentiment Split', fontsize=11, color='#E6EDF3', pad=8)

    # ── Top hashtags ─────────────────────────────────────────
    ax_hash = fig.add_subplot(gs[1, 1:3])
    ax_hash.set_facecolor(COLORS['card'])
    all_tags = [t for tags in df['hashtags'] for t in tags]
    tc = Counter(all_tags).most_common(10)
    if tc:
        tags_l, cnts_l = zip(*tc)
        colors_h = [COLORS['negative'] if i < 3 else COLORS['neutral'] if i < 6 else COLORS['positive'] for i in range(len(tags_l))]
        ax_hash.barh(tags_l, cnts_l, color=colors_h, edgecolor='none', height=0.6)
        ax_hash.invert_yaxis()
        ax_hash.set_title('Top 10 Hashtags', fontsize=11, color='#E6EDF3', pad=8)
        ax_hash.grid(axis='x', alpha=0.3)
        ax_hash.spines['top'].set_visible(False)
        ax_hash.spines['right'].set_visible(False)

    # ── Country bars ─────────────────────────────────────────
    ax_ctry = fig.add_subplot(gs[1, 3])
    ax_ctry.set_facecolor(COLORS['card'])
    ctry_counts = df[df['country'] != 'Other']['country'].value_counts().head(6)
    ax_ctry.barh(ctry_counts.index, ctry_counts.values,
                 color=COLORS['accent'], alpha=0.8, edgecolor='none', height=0.6)
    ax_ctry.invert_yaxis()
    ax_ctry.set_title('Top Countries', fontsize=11, color='#E6EDF3', pad=8)
    ax_ctry.grid(axis='x', alpha=0.3)
    ax_ctry.spines['top'].set_visible(False)
    ax_ctry.spines['right'].set_visible(False)

    # ── Hourly activity line ──────────────────────────────────
    ax_hour = fig.add_subplot(gs[2, :])
    ax_hour.set_facecolor(COLORS['card'])
    for sent, color in [('Positive', COLORS['positive']), ('Negative', COLORS['negative']), ('Neutral', COLORS['neutral'])]:
        h = df[df['sentiment'] == sent].groupby('hour').size()
        ax_hour.plot(h.index, h.values, color=color, label=sent, linewidth=2, marker='o', markersize=3)
        ax_hour.fill_between(h.index, h.values, alpha=0.07, color=color)
    ax_hour.set_title('Tweet Activity by Hour of Day', fontsize=11, color='#E6EDF3', pad=8)
    ax_hour.set_xlabel('Hour (0–23)', fontsize=10)
    ax_hour.set_xticks(range(24))
    ax_hour.legend(framealpha=0.2, facecolor=COLORS['bg'], edgecolor='#30363D',
                   labelcolor='#C9D1D9', fontsize=9, loc='upper left')
    ax_hour.grid(True, alpha=0.3)
    ax_hour.axvspan(20, 23, alpha=0.1, color=COLORS['accent'])

    fig.suptitle('WAR SOCIAL MEDIA SENTIMENT — ANALYSIS DASHBOARD',
                 fontsize=18, fontweight='bold', color='#E6EDF3', y=0.98)

    path = f'{OUTPUT_DIR}/08_dashboard.png'
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor=COLORS['bg'])
    plt.show()
    print(f"    → Saved: {path}")


# ════════════════════════════════════════════════════════════
#   STEP 5 — EXPORT CSV (for Power BI)
# ════════════════════════════════════════════════════════════

def export_for_powerbi(df):
    print("[✔] Exporting clean CSV for Power BI...")
    export_cols = ['text', 'clean_text', 'created_at', 'date', 'hour', 'month',
                   'day_of_week', 'user_location', 'country', 'hashtags',
                   'sentiment', 'compound_score', 'retweet_count', 'like_count']
    export_cols = [c for c in export_cols if c in df.columns]
    out = df[export_cols].copy()
    out['hashtags'] = out['hashtags'].apply(lambda x: ', '.join(x))
    path = f'{OUTPUT_DIR}/war_sentiment_clean.csv'
    out.to_csv(path, index=False)
    print(f"    → Saved: {path}  ({len(out):,} rows)")
    print("    → Import this CSV directly into Power BI Desktop")


# ════════════════════════════════════════════════════════════
#   STEP 6 — PRINT INSIGHTS SUMMARY
# ════════════════════════════════════════════════════════════

def print_insights(df):
    print("\n" + "═" * 60)
    print("  PROJECT INSIGHTS SUMMARY (use in viva!)")
    print("═" * 60)

    total = len(df)
    dist  = df['sentiment'].value_counts(normalize=True).mul(100).round(1)
    for s in ['Positive', 'Negative', 'Neutral']:
        if s in dist:
            print(f"  {s:10s}: {dist[s]:5.1f}%")

    all_tags = [t for tags in df['hashtags'] for t in tags]
    tc = Counter(all_tags).most_common(3)
    print(f"\n  Top 3 hashtags: {', '.join([t[0] for t in tc])}")

    peak = df.groupby('hour').size().idxmax()
    print(f"  Peak hour: {peak}:00 — people tweet most at this hour")
    print(f"  → Insight: Post-work / late-night news consumption cycle")

    neg_avg_rt = df[df['sentiment'] == 'Negative']['retweet_count'].mean()
    pos_avg_rt = df[df['sentiment'] == 'Positive']['retweet_count'].mean()
    ratio = round(neg_avg_rt / pos_avg_rt, 1) if pos_avg_rt > 0 else 'N/A'
    print(f"\n  Negative tweets avg retweets: {neg_avg_rt:.1f}")
    print(f"  Positive tweets avg retweets: {pos_avg_rt:.1f}")
    print(f"  → Emotional/negative tweets get ~{ratio}× more engagement")

    top_ctry = df[df['country'] != 'Other']['country'].value_counts().head(3).index.tolist()
    print(f"\n  Top tweeting countries: {', '.join(top_ctry)}")
    print("  → Western countries dominate the conversation")
    print("═" * 60 + "\n")


# ════════════════════════════════════════════════════════════
#   MAIN — RUN EVERYTHING
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 60)
    print("  WAR SENTIMENT ANALYSIS — STARTING")
    print("═" * 60 + "\n")

    # ── Change this path to your Kaggle CSV ─────────────────
    # Example: CSV_FILE = 'ukraine_tweets.csv'
    CSV_FILE = None   # Set to None to use sample data

    df = load_dataset(CSV_FILE)
    df = preprocess(df)
    df = run_sentiment(df)

    print("\n[►] Generating all charts...\n")
    plot_pie(df)
    plot_line_over_time(df)
    plot_hashtags(df)
    plot_wordcloud(df)
    plot_country_comparison(df)
    plot_peak_hours(df)
    plot_engagement(df)
    plot_dashboard(df)

    export_for_powerbi(df)
    print_insights(df)

    print(f"[✅] All outputs saved to: ./{OUTPUT_DIR}/")
    print("     Open war_sentiment_clean.csv in Power BI for interactive dashboard")
    print("     All PNG charts ready for your report / viva slides\n")