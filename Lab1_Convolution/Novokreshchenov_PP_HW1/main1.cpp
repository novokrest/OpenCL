#define _CRT_SECURE_NO_WARNINGS

#include <fstream>
#include <iostream>
#include <vector>

typedef std::vector<int> vint;

void read_data(vint & input, size_t count)
{
	for (size_t i = 0; i < count; ++i) {
		std::cin >> input[i];
	}
}

void print_matrix(vint & matr, int size)
{
	std::cout << std::endl;

	for (size_t i = 0; i < size; ++i) {
		for (size_t j = 0; j < size; ++j) {
			std::cout << matr[i * size + j];
		}
		std::cout << std::endl;
	}
}

int main1()
{
	freopen("input.txt", "r", stdin);

	size_t test_array_size, mask_size;

	std::cin >> test_array_size;
	std::cin >> mask_size;

	size_t data_size = test_array_size * test_array_size;
	size_t mask_data_size = mask_size * mask_size;

	vint input(data_size);
	vint output(data_size);
	vint mask(mask_data_size);

	read_data(input, input.size());
	read_data(mask, mask.size());

	for (int id = 0; id < data_size; ++id) {
		int row = id / test_array_size;
		int column = id % test_array_size;
		int res = 0;
		for (int i = 0; i < mask_size; ++i) {
			for (int j = 0; j < mask_size; ++j) {
				int input_row = row - mask_size / 2 + i;
				int input_column = column - mask_size / 2 + j;
				if (input_row >= 0 && input_row < test_array_size && input_column >= 0 && input_column < test_array_size) {
					res += mask[i * mask_size + j] * input[input_row * test_array_size + input_column];
				}
			}
		}
		output[id] = res;
	}

	print_matrix(output, test_array_size);

	int i = 0;

	return 0;
}