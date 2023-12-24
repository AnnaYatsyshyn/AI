import random
import matplotlib.pyplot as plt


# Клас Ant, який відображає кожну мураху в алгоритмі
class Ant:
    def __init__(self, start_position):
        self.current_position = start_position
        self.next_position = None
        self.tabu_list = [start_position]
        self.visited_count = 1
        self.travel_sequence = [start_position]
        self.total_distance = 0


# Функція для обчислення загальної відстань для заданого маршруту, використовуючи матрицю відстаней між містами
def yatsyshyn_calculate_distance(path, distance_matrix):
    total_distance = 0
    for i in range(len(path) - 1):
        total_distance += distance_matrix[path[i]][path[i + 1]]
    return total_distance


# Функція використовується для створення заданої кількості мурах, кожна з яких має початкову позицію
def yatsyshyn_initialize_ants(num_ants, start_position):
    ants = [Ant(start_position) for _ in range(num_ants)]
    return ants


# Функція для визначення наступної позиції для мурахи
def yatsyshyn_choose_next_position(ant, pheromone_matrix, distance_matrix, alpha, beta):
    current_city = ant.current_position
    unvisited_cities = [city for city in range(len(pheromone_matrix[current_city])) if city not in ant.tabu_list]

    if not unvisited_cities:
        # Якщо всі міста вже відвідані, повертаємось до початкового
        ant.next_position = ant.travel_sequence[0]
    else:
        probabilities = []
        for city in unvisited_cities:
            pheromone = pheromone_matrix[current_city][city]
            distance = distance_matrix[current_city][city]
            probabilities.append((city, (pheromone ** alpha) * ((1 / distance) ** beta)))

        total_prob = sum(prob[1] for prob in probabilities)
        probabilities = [(prob[0], prob[1] / total_prob) for prob in probabilities]

        chosen_city = random.choices(unvisited_cities, weights=[prob[1] for prob in probabilities])[0]
        ant.next_position = chosen_city


# Функція оновлює рівень феромонів на основі пройденого маршруту кожної мурахи
def yatsyshyn_update_pheromones(pheromone_matrix, ants, evaporation_rate):
    for i in range(len(pheromone_matrix)):
        for j in range(len(pheromone_matrix[i])):
            pheromone_matrix[i][j] *= (1 - evaporation_rate)

    for ant in ants:
        for i in range(len(ant.travel_sequence) - 1):
            current_city = ant.travel_sequence[i]
            next_city = ant.travel_sequence[i + 1]
            pheromone_matrix[current_city][next_city] += 1 / ant.total_distance


# Функція для відображення оптимального маршруту
def plot_best_path(distance_matrix, city_names, best_path):
    plt.figure(figsize=(10, 6))

    plt.plot(best_path, marker='o', linestyle='-', color='b')

    plt.title('Optimal Path')
    plt.xlabel('Step')
    plt.ylabel('City')
    plt.xticks(range(len(best_path)),
               labels=[str(i) for i in range(len(best_path))])  # Встановлюємо позначки X з номерами кроків
    plt.yticks(best_path, labels=[city_names[i] for i in best_path], rotation=45,
               ha="right")  # Встановлюємо позначки Y з назвами міст
    plt.show()


# Основна функція для вирішення задачуі комівояжера за допомогою методу мурашиних колоній
def yatsyshyn_ant_colony(distance_matrix, num_ants, max_iterations, alpha, beta, evaporation_rate, city_names):
    num_cities = len(distance_matrix)
    start_position = 14  # Рівне
    pheromone_matrix = [[1 for _ in range(num_cities)] for _ in range(num_cities)]

    ants = yatsyshyn_initialize_ants(num_ants, start_position)

    for iteration in range(max_iterations):
        for ant in ants:
            yatsyshyn_choose_next_position(ant, pheromone_matrix, distance_matrix, alpha, beta)
            ant.total_distance += distance_matrix[ant.current_position][ant.next_position]
            ant.tabu_list.append(ant.next_position)
            ant.visited_count += 1
            ant.travel_sequence.append(ant.next_position)
            ant.current_position = ant.next_position

        yatsyshyn_update_pheromones(pheromone_matrix, ants, evaporation_rate)

    best_ant = min(ants, key=lambda x: x.total_distance)
    plot_best_path(distance_matrix, city_names, best_ant.travel_sequence)

    return best_ant.travel_sequence, best_ant.total_distance


# Матриця відстаней між містами
D = [
    [0, 645, 868, 125, 748, 366, 256, 316, 1057, 382, 360, 471, 428, 593, 311, 844, 602, 232, 575, 734, 521, 120, 343,
     312, 396],
    [645, 0, 252, 664, 81, 901, 533, 294, 394, 805, 975, 343, 468, 196, 957, 446, 430, 877, 1130, 213, 376, 765, 324,
     891, 672],
    [868, 252, 0, 858, 217, 1171, 727, 520, 148, 1111, 1221, 611, 731, 390, 1045, 591, 706, 1100, 1391, 335, 560, 988,
     547, 1141, 867],
    [125, 664, 858, 0, 738, 431, 131, 407, 1182, 257, 423, 677, 557, 468, 187, 803, 477, 298, 671, 690, 624, 185, 321,
     389, 271],
    [748, 81, 217, 738, 0, 1119, 607, 303, 365, 681, 833, 377, 497, 270, 925, 365, 477, 977, 1488, 287, 297, 875, 405,
     957, 747],
    [366, 901, 1171, 431, 1119, 0, 561, 618, 1402, 328, 135, 747, 627, 898, 296, 1070, 908, 134, 280, 1040, 798, 246,
     709, 143, 701],
    [256, 533, 727, 131, 607, 561, 0, 298, 811, 388, 550, 490, 489, 337, 318, 972, 346, 427, 806, 478, 551, 315, 190,
     538, 149],
    [316, 294, 520, 407, 303, 618, 298, 0, 668, 664, 710, 174, 294, 246, 627, 570, 506, 547, 883, 387, 225, 435, 126,
     637, 363],
    [1057, 394, 148, 1182, 365, 1402, 811, 668, 0, 1199, 1379, 857, 977, 474, 1129, 739, 253, 1289, 1539, 333, 806,
     1177, 706, 1292, 951],
    [382, 805, 1111, 257, 681, 328, 388, 664, 1199, 0, 152, 780, 856, 725, 70, 1052, 734, 159, 413, 866, 869, 263, 578,
     336, 949],
    [360, 975, 1221, 423, 833, 135, 550, 710, 1379, 152, 0, 850, 970, 891, 232, 1173, 896, 128, 261, 1028, 1141, 240,
     740, 278, 690],
    [471, 343, 611, 677, 377, 747, 490, 174, 857, 780, 850, 0, 120, 420, 864, 282, 681, 754, 999, 556, 51, 590, 300,
     642, 640],
    [428, 468, 731, 557, 497, 627, 489, 294, 977, 856, 970, 120, 0, 540, 741, 392, 800, 660, 1009, 831, 171, 548, 420,
     515, 529],
    [593, 196, 390, 468, 270, 898, 337, 246, 474, 725, 891, 420, 540, 0, 665, 635, 261, 825, 1149, 141, 471, 653, 279,
     892, 477],
    [311, 957, 1045, 187, 925, 296, 318, 627, 1129, 70, 232, 864, 741, 665, 0, 1157, 664, 162, 484, 805, 834, 193, 508,
     331, 458],
    [844, 446, 591, 803, 365, 1070, 972, 570, 739, 1052, 1173, 282, 392, 635, 1157, 0, 896, 1097, 1363, 652, 221, 964,
     696, 981, 1112],
    [602, 430, 706, 477, 477, 908, 346, 506, 253, 734, 896, 681, 800, 261, 664, 896, 0, 774, 1138, 190, 732, 662, 540,
     883, 350],
    [232, 877, 1100, 298, 977, 134, 427, 547, 1289, 159, 128, 754, 660, 825, 162, 1097, 774, 0, 338, 987, 831, 112, 575,
     176, 568],
    [575, 1130, 1391, 671, 1488, 280, 806, 883, 1539, 413, 261, 999, 1009, 1149, 484, 1363, 1138, 338, 0, 1299, 1065,
     455, 984, 444, 951],
    [734, 213, 335, 690, 287, 1040, 478, 387, 333, 866, 1028, 556, 831, 141, 805, 652, 190, 987, 1299, 0, 576, 854, 420,
     1036, 608],
    [521, 376, 560, 624, 297, 798, 551, 225, 806, 869, 1141, 51, 171, 471, 834, 221, 732, 831, 1065, 576, 0, 641, 351,
     713, 691],
    [120, 765, 988, 185, 875, 246, 315, 435, 1177, 263, 240, 590, 548, 653, 193, 964, 662, 112, 455, 854, 641, 0, 463,
     190, 455],
    [343, 324, 547, 321, 405, 709, 190, 126, 706, 578, 740, 300, 420, 279, 508, 696, 540, 575, 176, 444, 1036, 713, 190,
     660, 330],
    [312, 891, 1141, 389, 957, 143, 538, 637, 1292, 336, 278, 642, 515, 892, 331, 981, 883, 176, 444, 1036, 713, 190,
     660, 0, 695],
    [396, 672, 867, 271, 747, 701, 149, 363, 951, 949, 690, 640, 529, 477, 458, 1112, 350, 568, 951, 608, 691, 455, 330,
     695, 0]
]

# Список імен міст
city_names = [
    "Вінниця", "Дніпро", "Донецьк", "Житомир", "Запоріжжя",
    "Ів-Франківськ", "Київ", "Кропивницький", "Луганськ", "Луцьк",
    "Львів", "Миколаїв", "Одеса", "Полтава", "Рівне", "Сімферополь",
    "Суми", "Тернопіль", "Ужгород", "Харків", "Херсон", "Хмельницький",
    "Черкаси", "Чернівці", "Чернігів"
]

# Виклик функції
result_path, result_distance = yatsyshyn_ant_colony(D, num_ants=25, max_iterations=25, alpha=1, beta=2, evaporation_rate=0.5,
                                                   city_names=city_names)

# Результат
print("Optimal Path:", result_path)
print("Optimal Distance:", result_distance)
