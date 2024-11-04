## Архитектура
![[Pasted image 20241008205217.png|400]]
Архитектура модели трансформера. Как видно он состоит из двух компонентов: энкодер и декодер. В оригинальной статье используется 6 encoder layers и 6 decoder layers. 
Encoder layer состоит из двух sub-layers: Multi-Head Attention и Feed Forward. Также после каждого из них прокидываются inputs(residual-connection) и на выходы Feed Forward применяется LayerNorm.
Decoder же состоит из трех sub-layers, к Attention и LayerNorm добавляется Attention, применяемый к выходам Encoder'а. Также здесь максированный Attention, который маскирует все токены меньше i, гарантируя то, что при предскании следующего токена модель будет обладать информацией о токенах только с предыдущих позиций.

## Attention

В оригинальной статье используется Scaled-DotProduct Attention, который позднее изменится на Flash Attention и др. (так как они быстрее и занимают меньше памяти).
![[Pasted image 20241008211127.png | 350]]
$Attention(Q, K, V) = softmax(QK^T)/\sqrt(d^k))*V$

Интуиция - все что под softmax это вычисление весов, с которыми мы берем эмбеддинги из V.

Нормализация на $\sqrt(d^k)$ нужна так как при больших размерах векторов($d^k$), output softmax попадают в область с градиентами близкими к нулю. 
P.S To illustrate why the dot products get large, assume that the components of q and k are independent random variables with mean 0 and variance 1. Then their dot product, q · k = Pdk i=1 qiki, has mean 0 and variance dk.

Сложность Scaled-DotProduct Attn $O(n^2)$, что решается в...

## Multi-Head Attention

Работает хорошо - масштабируем. Разные головы выявляют разные паттерны. "Голова" -  одна тройка матриц Q, K, V. В оригинальной статье используется 8 голов.
Можно представить все в виде матричного умножения следующим образом

$MultiHead(Q, K, V) = Concat(head_1, ... head_h)W^o$ 
где $head_i=Attention(QW_i^Q, KW_i^K, VW_i^V)$, QKV - линейный обучаемый слои, а матрицы W имеют следующие размерности:

![[Pasted image 20241008222556.png]]

## Feed Forward layer
$FFN(x) = max(0, xW_1 + b_1)W_2 + b_2$

Идея feed-forward слоя в том что мы "разжимаем" входящий эмбеддинг в пространство большей размерности, после "сжимая" его обратно

![[Pasted image 20241008225009.png]]

В оригинальной статье $d_m=512$, а $d_{ff}=2048$  
