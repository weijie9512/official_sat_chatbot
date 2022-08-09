
try:
    from newsapi import NewsApiClient
except:
    from newsapi.newsapi_client import NewsApiClient
import pandas as pd
from datetime import datetime, timedelta

import os

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

        formatted_query = ' '.join(query.split("_"))
        news = self.newsapi.get_top_headlines(q=formatted_query,
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

        formatted_query = ' '.join(query.split("_"))
        print(formatted_query)
        if formatted_query in ["gender inequality", "wealth inequality", "homeless", "poverty"]:
            news = self.newsapi.get_everything(q=formatted_query,
                                            from_param=from_param,
                                            to=to,
                                            language=language,
                                            sort_by=sort_by,
                                           page=page)
        else:
            news = self.newsapi.get_everything(q=formatted_query,
                                            from_param=from_param,
                                            to=to,
                                            language=language,
                                            sort_by=sort_by,
                                            # source: news24: south africa
                                            # rt: russia
                                            # ary-news: palestine
                                            #sources = "abc-news, rt, al-jazeera-english, ary-news, bloomberg, cbc-news, google-news, google-news-ca, google-news-in, google-news-is, google-news-ru, google-news-sa, google-news-uk, news24, politico, rbc, the-hindu, the-times-of-india, time",
                                            #
                                            sources = "rt, reuters, al-jazeera-english, ary-news, bloomberg, news24, politico, abc-news",
                                            page=page)

        article_titles = []
        content = []
        url = []
        for i, new in enumerate(news['articles']):
            #article_titles.append(new['title'])
            #content.append(new['content'])
            url.append(new['url'])
        #return article_titles, content, url
        return url

    def save_and_update(self):
        categories = ["mental", "climate", "war", "poverty", "homeless", "wealth_inequality", "gender_inequality"]
        #categories = ["war"]
        for keyword in categories:
            from_date =   (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
            to_date = datetime.today().strftime('%Y-%m-%d')
            print(f"Downloading news of title {keyword} from {from_date} to {to_date}.")
            #article_titles, content, url = news.get_top_news("ukraine")

           
            url = news.get_every_news(keyword, from_param=from_date, to=to_date)
            d = {'url': url}
            df = pd.DataFrame(d)
            pwd = os.getcwd()
            df.to_csv(f"{pwd}/data/news_data/{keyword}.csv")

    def test_run(self):
        print("test")




if __name__ == "__main__":
    news = News()

    news.save_and_update()
    """
    
    keyword = "climate"

    from_date =   (datetime.today() - timedelta(days=30)).strftime('%Y-%m-%d')
    to_date = datetime.today().strftime('%Y-%m-%d')
    print(f"Downloading news of title {keyword} from {from_date} to {to_date}.")
    #article_titles, content, url = news.get_top_news("ukraine")

    try:
        article_titles, content, url = news.get_every_news(keyword, from_param=from_date, to=to_date)
        d = {'titles':article_titles, 'content':content, 'url': url}
        df = pd.DataFrame(d)
 
        df.to_csv(f"{keyword}.csv")
    except:
        article_titles, content, url = news.get_every_news(keyword, from_param=from_date, to=to_date)
        d = {'url': url}
        df = pd.DataFrame(d)
 
        df.to_csv(f"{keyword}.csv")

    
    #print(article_titles)
    #print(len(article_titles))
    #print(type(article_titles))
    


    #d = {'titles':article_titles, 'content':content, 'url': url}
    #df = pd.DataFrame(d)
    
    #d = {'url': url}
    #df = pd.DataFrame(d)
 
    #df.to_csv(f"{keyword}.csv")
    """
    
#
# print(content)