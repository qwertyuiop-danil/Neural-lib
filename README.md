# Установка
<code>pip install SimpleNeuro</code>

# Условия для работы

<ul>
  <li>Передавать нейросети данные только в диапазоне от -1 до 1</li>
  <li>Размеры входной и выходной матрицы должны быть всегда равны количеству входных и выходных нейронов, количество скрытых нейронов можно указывать любое</li>
  <li>Нейросети очень сложно выдавать точные числа, ей намного легче выводить матрицу, в которой 1 число будет больше другово</li>
</ul>

# Работа с библиотекой
<ul>
<li><code>agent = Neural(10, 256, 10)</code> - создание нейросети в параметрах указывается количество входных, скрытых и выходных нейронов</li>

<li><code>agent.training(0.0001, input_data, output_data)</code> - тренировка нейросети, в параметрах указывается скорость обучения, чем меньше скорость обучения тем точнее она обучиться. Скорость не может равняться 0</li>

<li><code>agent.save_weights('model')</code> - сохранение модели в файле model.npz</li>

<li><code>agent.load_weights('model')</code> - загрузка весов из файла model.npz. При загрузке весов количество всех нейронов должно быть указано такоеже, как и при обучении</li>

</ul>
