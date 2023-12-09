from utils import get_index_of_max_area

def test_import():
    # object_name, y_min, x_min, y_max, x_max

    list_2d = [('white', 0, 0, 10, 10), ('black', 0, 0, 5, 5), ('white', 0, 0, 20, 20)]

    print(get_index_of_max_area(list_2d))

def get_max():
    data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    index = 1  # column index

    max_value = max(row[index] for row in data)

    print(max_value)

if __name__ == '__main__':
    get_max()