# Установка
<code>pip install SimpleNeuro</code>

# Условия для работы

<ul>
  <li>Передавать нейросети данные только в диапазоне от -1 до 1</li>
  <li>Размеры входной и выходной матрицы должны быть всегда равны количеству входных и выходных нейронов, количество скрытых нейронов можно указывать любое</li>
  <li>Нейросети очень сложно выдавать точные числа, ей намного легче выводить матрицу, в которой 1 число будет больше другово</li>
</ul>

# Работа с библиотекой

Для начала нужно импортировать библиотеки и указать количество нейронов
<code>
from SimpleNeuro import Neural
import numpy as np
agent = Neural(10, 256, 10)
</code>
