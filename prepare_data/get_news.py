import pandas as pd



class NewsGetter:
    def __init__(self):
        self.categories = ["mental", "climate", "war", "poverty", "homeless", "wealth_inequality", "gender_inequality"]
        
        self.news_war = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/data/news_data/war.csv')
        self.news_mental = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/data/news_data/mental.csv')
        self.news_climate = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/data/news_data/climate.csv')
        self.news_poverty = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/data/news_data/poverty.csv')
        self.news_homeless = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/data/news_data/homeless.csv')
        self.news_wealth_inequality = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/data/news_data/wealth_inequality.csv')
        self.news_gender_inequality = pd.read_csv('/Users/weijiechua/Desktop/ImperialClasses/Courses/Term3/wj_SATbot2.0/data/news_data/gender_inequality.csv')

        self.categories_dict = {"mental": self.news_mental, "war": self.news_war, "climate": self.news_climate, "poverty": self.news_poverty,  \
                                "homeless": self.news_homeless, "wealth inequality": self.news_wealth_inequality, "gender inequality": self.news_gender_inequality}
        
        
    def get_categories(self):
        return self.categories

    def get_random_news(self, category):
        sample_news = self.categories_dict[category].sample()
        return [sample_news['titles'].to_string(index=False), \
                sample_news['content'].to_string(index=False), \
                sample_news['url'].to_string(index=False)]


newsgetter = NewsGetter()
print(newsgetter.get_random_news("mental"))

