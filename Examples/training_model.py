# Загружаем библиотеки
from SimpleNeuro import Neural
import numpy as np

# Создаём объект agent и передаём в него: количество входных нейронов, скрытых и выходных
agent = Neural(10, 256, 10)

# Цикл для обучения
epochs = 50
succes = 0

for epoch in range(epochs):
    # Цикл для вывода статистики об успехах нейросети, она будет выводиться каждые 10 000 шагов обучения
    for i in range(10000):
        # Создаём входные данные
        numbers = np.random.choice(np.arange(10), size=10, replace=False)
        input_data = np.reshape(numbers, (10, 1))
        input_data = input_data.astype(float)/10
        
        # Создаём выходные данные
        output_data = np.zeros((10, 1))
        max_index = np.argmax(input_data)
        output_data[max_index] = 1
        output_data = output_data.astype(float)
      
        # Запускаем функцию обучения, в ней нейросеть обучается с помощью метода Backpropagation, который корректирует веса
        result_agent = agent.training(learning_rate=0.0001, input_data=input_data, output_data=output_data)

        # Если нейросеть справилась с задачей, то добавляем её в счётчик 1 единицу
        succes += int(np.argmax(output_data) == result)

    # Выводим номер эпохи обучения и успешность нейросети
    print(f'Epoch {epoch}')
    print(f'Succes {float(succes)/100}%')

    # Обнуляем счётчик для новой эпохи
    succes = 0
# Сохраняем веса в файл 'model.npz'
agent.save_weights('model')
