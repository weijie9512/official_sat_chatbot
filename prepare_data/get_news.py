import pandas as pd



class NewsGetter:
    def __init__(self):
        self.categories = ["mental", "climate", ]
        
        
        self.mental_news = pd.read_csv("../data/mental.csv", index_col=0)


        self.categories_dict = {"mental": self.mental_news}
        
    def get_categories(self):
        return self.categories

    def get_random_news(self, category):
        sample_news = self.categories_dict[category].sample()
        return [sample_news['titles'].to_string(index=False), \
                sample_news['content'].to_string(index=False), \
                sample_news['url'].to_string(index=False)]


newsgetter = NewsGetter()
print(newsgetter.get_random_news("mental"))

