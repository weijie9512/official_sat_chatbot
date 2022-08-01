from newsapi import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

API_KEY = "188b1d9e2e794b0e98cb08db39f8d2e5"

class News:
    """
    News class
    """
    def __init__(self):
        self.newsapi = NewsApiClient(api_key=API_KEY)
    
    def get_top_news(self, query, category='general', country='us'):
        """
        Getting news 
        """
        news = self.newsapi.get_top_headlines(q=query,
                                        category=category,
                                        language='en',
                                        country=country)

        article_titles = []
        content = []
        url = []

        
        for i, new in enumerate(news['articles']):

            article_titles.append(new['title'])
            content.append(new['content'])
            url.append(new['url'])
        return article_titles, content, url
    
    def get_every_news(self, query, from_param='2022-06-01',
                                    to='2022-06-14',
                                    language='en',
                                    sort_by='relevancy',
                                    page=2):
        """
        Getting news 
        """
        news = self.newsapi.get_everything(q=query,
                                        from_param=from_param,
                                        to=to,
                                        language=language,
                                        sort_by=sort_by,
                                        page=page)

        article_titles = []
        content = []
        url = []
        print(news)
        for i, new in enumerate(news['articles']):
            print(new)
            article_titles.append(new['title'])
            content.append(new['content'])
            url.append(new['url'])
        return article_titles, content, url



if __name__ == "__main__":
    news = News()

    keyword = "war"

    from_date =   (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = datetime.today().strftime('%Y-%m-%d')
    print(f"Downloading news of title {keyword} from {from_date} to {to_date}.")
    #article_titles, content, url = news.get_top_news("ukraine")


    article_titles, content, url = news.get_every_news(keyword, from_param=from_date, to=to_date)
    """
    print(article_titles)
    print(len(article_titles))
    print(type(article_titles))
    """


    d = {'titles':article_titles, 'content':content, 'url': url}
    df = pd.DataFrame(d)

 
    df.to_csv(f"{keyword}.csv")
#
# print(content)