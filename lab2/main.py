import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import networkx as nx

count = 0 # для подсчёта pagerank

# функция для извлечения ссылок из HTML-кода страницы
def get_links(url):
    links = []
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        for a in soup.find_all('a', href=True):
            link = a['href']
            if link.startswith('http') or link.startswith('https://'):
                links.append(link)
            # else:
            #     link = 'https://trello.com' + link
            #     links.append(link)
    except:
        pass
    return links

G = nx.DiGraph() # создание графа

def create_pagerank_graph(url, damping_factor=0.85, tol=1e-6):
    queue = [(url, 0)]
    visited = set()
    while queue:
        # if (level > depth):
        #     break
        link, level = queue.pop(0)
        links = get_links(link) # получение всех ссылок на странице
        G.add_node(link) # добавление вершины в граф
        for l in links:  # добавление ребер в граф
            if l not in G:
                G.add_node(l)
            G.add_edge(link, l)
        for l in links: # добавление ребер в граф
            G.add_node(l)
            G.add_edge(link, l)

        if level < 4: # добавление ссылок на другие сайты в очередь
            for l in links:
                if l not in visited:
                    if l.startswith(url):
                        queue.append((l, level + 1))
                    else:
                        if level < 2:
                            queue.append((l, level + 1))
                if l.startswith(url) and l not in visited:
                    queue.append((l, level + 1))
                elif not l.startswith(url) and l not in visited:
                    G.add_node(l)
                    G.add_edge(link, l)
        visited.add(link)

    pagerank_dict = nx.pagerank(G, alpha=damping_factor, tol=tol) # запуск алгоритма PageRank

    return pagerank_dict

# тестируем функцию
url = 'https://trello.com'
pr = create_pagerank_graph(url)

# # выводим результаты, но мне это не надо, поэтому я просто подсчитаю количество сайтов
print("PageRank:")
for node in pr:
    count += len(node)
    print(node, pr[node])

# ищем ссылки на другие сайты до 4 уровня включительно и добавляем их в граф
print("\nСсылки на другие сайты:")
for node in pr:
    if node not in G:
        G.add_node(node)
    paths = nx.single_source_shortest_path(G, node, cutoff=4)
    for path in paths:
        if path != node and (path.startswith('http') or path.startswith('https')):
            if all(n in G for n in paths[path]):
                print(node, "->", path)
                G.add_edge(node, path)

# пересчитываем PageRank для обновленного графа и выводим результаты
print("\nPageRank для обновленного графа:")
pr = nx.pagerank(G)
for node in pr:
    print(node, pr[node])



# # Сортировку я написала сама, оставляем, но нужно в порядке убывания сделать
print("\nSort PageRank:")
# # Сортируем словарь по значению PageRank
sorted_pagerank = sorted(pr.items(), key=lambda x: x[1], reverse=True)

# Выводим первые 10 элементов отсортированного списка
for link, pagerank in sorted_pagerank[:10]:
    print(f"{link} - {pagerank}")

for url, score in sorted(pr.items(), key=lambda x: x[1], reverse=True):
    # print(f'{url}, PageRank: {score}')
    count += score

# Разобраться в коде с показом фигуры
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111)
nx.draw_circular(G, ax=ax)
plt.show()
# print(count)
