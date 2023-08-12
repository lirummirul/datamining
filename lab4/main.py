import math
import mmh3
from bitarray import bitarray

class BloomFilter:
    def __init__(self, num_items, false_positive_rate):
        self.num_items = num_items
        self.false_positive_rate = false_positive_rate
        self.num_bits = self.calculate_num_bits()
        self.num_hashes = self.calculate_num_hashes()
        self.bit_array = bitarray(self.num_bits)
        self.bit_array.setall(0)

    def add(self, item):
        for i in range(self.num_hashes):
            index = mmh3.hash(item, i) % self.num_bits
            self.bit_array[index] = 1

    def __contains__(self, item):
        for i in range(self.num_hashes):
            index = mmh3.hash(item, i) % self.num_bits
            if not self.bit_array[index]:
                return False
        return True

    def calculate_num_bits(self):
        m = -1 * (self.num_items * math.log(self.false_positive_rate)) / math.pow(math.log(2), 2)
        return int(m)

    def calculate_num_hashes(self):
        k = (self.num_bits / self.num_items) * math.log(2)
        return int(k)

# Загрузила текстовый документ
with open("text_file.txt") as file:
        text = file.read()

words = text.split()

# Вычисляю 10-й процентиль частоты слов
word_frequency = {}
for word in words:
    if word in word_frequency:
        word_frequency[word] += 1
    else:
        word_frequency[word] = 1

word_frequencies = list(word_frequency.values())
percentile = sorted(word_frequencies)[int(len(word_frequencies)*0.1)]

# Разделяю текст на перекрывающиеся части на основе 10-го процентиля частоты слов
part_size = int(len(words) * 0.1 / 9) # каждая часть содержит 90% слов из предыдущей и следующей частей
overlap_size = int(part_size * 0.5) # размер перекрытия составляет 50% от размера детали

text_parts = []
for i in range(0, len(words), part_size - overlap_size):
    text_part = ' '.join(words[i:i+part_size])
    if i > 0:
        prev_part = text_parts[-1]
        text_part = ' '.join(prev_part.split()[:-overlap_size]) + ' ' + text_part
    if i + part_size < len(words):
        next_part = ' '.join(words[i+part_size:i+part_size+overlap_size])
        text_part += ' ' + next_part
    text_parts.append(text_part)

bloom_filters = []
for text_part in text_parts:
    num_items = len(text_part.split())
    false_positive_rate = 0.01
    bloom_filter = BloomFilter(num_items, false_positive_rate)
    for word in text_part.split():
        bloom_filter.add(word)
    bloom_filters.append(bloom_filter)

# Проверяю, есть ли слово в текстовом документе, используя все созданные фильтры Bloom
def is_word_in_document(word):
    for bloom_filter in bloom_filters:
        if word in bloom_filter:
            return True
    return False

# Если пользователь в консоли введёт '-1' - то программа завершиться
def main():
    while True:
        word = input("Введите слово для проверки: \n")
        if (word == '-1'):
            break
        print(is_word_in_document(word))

if __name__ == '__main__':
    main()