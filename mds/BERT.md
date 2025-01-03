- **Количество слоев (трансформеров):** 12
- **Размер скрытых слоев:** 768
- **Количество голов внимания в каждом слое:** 12
- **Общее количество параметров:** 110 миллионов
- **Vocab size**: 30 000 

Идея: использовать bidirectional encoder, который позволяет смотреть на слова как слева-направо так и справа-налево. И получать информативные эмбеддинги, которые после этого могут быть использованы в downstream tasks, не изменяя архитектуру модели, а изменив лишь финальный слой получить SOTA результаты во многих NLP задачах.

Обучался на задачах NSP(Next Sentence Prediction) и MLM(Masked Language Modeling)

## NSP
Взяли датасет, состоящий из пар предложений A и B, разделенные токеном \[SEP]. В 50% случаев предложение B действительно идет за предложение A и ставится label IsNext, в 50% ставится label NotNext.
![[Pasted image 20241008233019.png| 700]]
Каждый эмбеддинг является суммой токена ембеддинга, токена сегмента, эмбеддинга позиции.

## MLM
Задача предсказать индекс замаскированного токен. 

15% токенов случайным образом заменяются по следующей стратегии:

- в 80% случаях заменяются на токен \[MASK]
- в 10% процентах случаев на любой случайный токен
-  10% токен не изменяется

Для чего? 

Если бы все время использовался токен \[MASK], модель бы "привыкала" к такому поведению, которое не отражает реальных данных, т.е. модель лучше генерализуется 
![[Pasted image 20241012172748.png|500]]