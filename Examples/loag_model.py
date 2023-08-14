# Импортируем библиотеки
from SimpleNeuro import Neural
import numpy as np

# Передаём количество входных, скрытых и выходных нейронов
agent = Neural(10, 256, 10)

# Загружаем веса из файла 'model.npz'
agent.load_weights('model')

while True:
    # Создаём рандомные входные данные
    numbers = np.random.choice(np.arange(10), size=10, replace=False)
    data = np.reshape(numbers, (10, 1))
    data = data.astype(float)
    
    # Передаём в функцию usage() входные данные и записываем в переменную rez выходные
    rez = agent.usage(np.array(data))

    # Приводим данные в удабный для чтения вид и выводим их
    max_index = np.argmax(rez)
    rez = np.zeros((10, 1))
    rez[max_index] = 1
    data = data.astype(int)
    rez = rez.astype(int)
    print(data.T[0], end='\n')
    print(rez.T[0])
  
    input('\n')
