## Мотивация
LLM становятся все больше, на момент статьи(2022 год) вышли модели PaLM(540B) и BLOOM(176B), чтобы заинференсить вторую необходимо 8 штучек A100, чтоб зафайнтюнить 72:) Поэтому нужна квантизация! 

**Квантизация** — это процесс преобразования данных из высокоточной формы (например, чисел с плавающей запятой) в более низкоточную форму (например, целые числа) для уменьшения объема памяти и ускорения вычислений. В контексте глубокого обучения это означает замену весов и входных данных нейронных сетей, представленных в формате с плавающей запятой (например, FP32 или FP16), на целочисленное представление (например, int8).


Авторы замечают что в LLM размером 6.7B возникает ~150000 экстремальных значений, то есть тех ктоорые выбиваются из распределения конкретного признака, также они отмечают, что если занулить эти экстремальные значения, то значение *softmax* у top-1 токена на ~20%, также значительно возрастает перплексия(600-1000%). В то время как если занулить такое же количество каких то других признаков, перплеския возрастает всего на 0.1%.
Из этого мы понимаем невероятную важность этих значений и что с ними надо работать иначе.

## Mixed precision quantization
Следовательно, если мы квантизуем эти значения в *int8*, мы сильно потеряем в качестве. Поэтому авторы предлагают следующий подход: оставлять такие экстремальные значения в FP16, а остальные при умножении матриц квантизовать в *int8*.
![[Pasted image 20241025224327.png|600]]

В матрице *X* выбросы выбираются по столбцам, в матрице весов *W* по строкам. 

## Что считают **выбросом**? 

Для трансформера с $L$ слоями, размерностью $hiddenstate$ $h$, размерностью последовательности $S$ выбросом считается признак, который удровлетворяет следующим критериям:
- величина не менее 6 (было выбрано эмперечиски)
- присутствует не менее чем в 25% слоев транформера($L$)
- затрагивает не менее 6% последовательности($S$)

![[Pasted image 20241026134037.png]]


Из этих двух графиков можно увидеть следующее: 
*кол-во выбросов растет, когда предсказательная способность модели улучшается, этим можно объяснить тот факт модели многое теряют при обычной квантизации, так как в этих экстремальных значениниях много информации, которую мы теряем*