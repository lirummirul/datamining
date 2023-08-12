import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

keywords = {
    'наука': ['наука', 'исследование', 'технология', 'мозг', 'эксперимент', 'находка', 'наука о птицах', 'наука о животных', 'анализы', 'ассоциации', 'анализы', 'жизнь', 'биология', 'наука о растениях', 'врачебная наука', 'генетика'],
    'спорт': ['спорт', 'футбол', 'ходьба', 'тренировка', 'бег', 'испытания', 'тренер', 'триал', 'ставки', 'советский спорт', 'фаворит', 'ассоциации', 'арена', 'бизнес', 'бонусы', 'бокс', 'гетто', 'гига', 'спорт дома', 'спорт для детей', 'гейм спорт', 'хоккей', 'бокс', 'лига чемпионов', 'баскетбол', 'биатлон', 'олимпиада', 'спортсменка', 'ЦСКА'],
    'шоппинг': ['шоппинг', 'покупка', 'скидки', 'магазин', 'товар', 'продукт', 'каталог', 'сдэк', 'выгодный', 'блогер', 'мода', 'расспрадажа', 'адрес', 'магазины'],
    'новости': ['новости', 'политика', 'событие', 'расследование', 'происшествие', 'суд', 'очевидец', 'наказание', 'яндекс', 'РИА', 'РБК', 'Жириновский', 'последние новости', 'мира', 'Россия', 'Путин', 'США']
}

nayka = ['наука', 'исследование', 'технология', 'мозг', 'эксперимент', 'находка', 'наука о птицах', 'наука о животных', 'анализы', 'ассоциации', 'анализы', 'жизнь', 'биология', 'наука о растениях', 'врачебная наука', 'генетика']
sport = ['спорт', 'футбол', 'ходьба', 'тренировка', 'бег', 'испытания', 'тренер', 'триал', 'ставки', 'советский спорт', 'фаворит', 'ассоциации', 'арена', 'бизнес', 'бонусы', 'бокс', 'гетто', 'гига', 'спорт дома', 'спорт для детей', 'гейм спорт', 'хоккей', 'бокс', 'лига чемпионов', 'баскетбол', 'биатлон', 'олимпиада', 'спортсменка', 'ЦСКА']
shopping = ['шоппинг', 'покупка', 'скидки', 'магазин', 'товар', 'продукт', 'каталог', 'сдэк', 'выгодный', 'блогер', 'мода', 'расспрадажа', 'адрес', 'магазины']
news = ['новости', 'политика', 'событие', 'расследование', 'происшествие', 'суд', 'очевидец', 'наказание', 'яндекс', 'РИА', 'РБК', 'Жириновский', 'последние новости', 'мира', 'Россия', 'Путин', 'США']

def get_text_from_web_page(url):
    page = requests.get(url).content
    soup = BeautifulSoup(page, 'html.parser')
    text = ' '.join(soup.stripped_strings)
    return text

def calculate_jaccard_similarity(text, keywords_list):
    text_words = set(text.split())
    keywords_words = set(keywords_list)
    intersection = len(text_words.intersection(keywords_words))
    union = len(text_words.union(keywords_words))
    similarity = intersection / union
    return similarity

def calculate_cosine_similarity(text, keywords_list):
    vectorizer = TfidfVectorizer(stop_words='english', use_idf=True)
    vectorized_text = vectorizer.fit_transform([text])
    target_vec = vectorizer.transform(keywords_list)
    similarity = cosine_similarity(vectorized_text, target_vec).mean()
    return similarity

# наука
# url = 'https://www.nkj.ru/news/47921/'
# спорт
url = 'https://www.sports.ru/'
# шоппинг
# url = 'https://www.tourister.ru/world/europe/russia/city/kazan/publications/636'
# новости
# url = 'https://ria.ru/20230411/putin-1864492523.html'
text = get_text_from_web_page(url)

jaccard_results = {}
cosine_results = {}
for topic, topic_keywords in keywords.items():
    jaccard_similarity = calculate_jaccard_similarity(text, topic_keywords)
    similarity_value = calculate_cosine_similarity(text, topic_keywords)
    jaccard_results[topic] = jaccard_similarity
    cosine_results[topic] = similarity_value

# Вывод результатов
print('Метрика схожести Жаккарда:', jaccard_results)
print('Метрика схожести по Косинусу:', cosine_results)

jaccard_news = jaccard_results["новости"]
jaccard_shopping = jaccard_results["шоппинг"]
jaccard_sport = jaccard_results["спорт"]
jaccard_nayka = jaccard_results["наука"]

cosine_new = cosine_results["новости"]
cosine_shopping = cosine_results["шоппинг"]
cosine_sport = cosine_results["спорт"]
cosine_nayka = cosine_results["наука"]

categories_cosines = {"новости": cosine_new,
                      "спорт": cosine_sport,
                      "наука": cosine_nayka,
                      "шоппинг": cosine_shopping
                      }

categories_jaccard = {"новости": jaccard_news,
                      "спорт": jaccard_sport,
                      "наука": jaccard_nayka,
                      "шоппинг": jaccard_shopping
                      }

max_categories_cosine = max(categories_cosines.items(), key=lambda x: x[1])[0]
max_categories_jaccard = max(categories_jaccard.items(), key=lambda x: x[1])[0]

if max(categories_jaccard.values()) > categories_jaccard.get(max_categories_cosine):
    print("Категория: ", max_categories_jaccard)
else:
    print("Категория: ", max_categories_cosine)