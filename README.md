# Fusion LSTM exercise
This is just a toy example in which we perform fusion of multiple LSTMs at each time step of the unfolding of an LSTM. This example is about French-to-English translation.

The dataset and the basic archetecture are adapted from: <br>
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

See model.py for details of this implementation. The architecture of the network looks like: <br>
<img src="img/fusionRNN.png" alt="architecture">



At the i-th time step (in this case, at the i-th word in a array of word, i. e. a sentence), the FusionLSTM takes in the hidden states of three different LSTM encoders, and outputs the fusion_state, i. e. fusion_state = FusionLSTM(hidden_1, hidden_2, hidden_3 ), and the encoders each takes in the input, its own hidden state, and the fusion_state, and outputs an encoding of the sentence, which is then decoded by a GRU decoder.

In this example, all three encoders have the same architecture except that they respectively have hidden state size 256, 64, 16. I was sort of expecting that having different hidden-sizes might make the encoders learn to encode features of a sentence at different length scales, but in practice I find it extremely hard and slow to train.

Some sample translation (> denotes the original French sentence, = denotes the target, < denotes the translated English). This is the output at epoch 1 (since it takes rather long to train):


```
> c est un homme aux multiples talents .
= he s a man of many talents .
< he s a to of . . <EOS>

> je ne suis pas vraiment occupee .
= i m not really busy .
< i m not sure i <EOS>

> je suis sur qu il reussira .
= i am sure that he will succeed .
< i m a of of of . . <EOS>

> je suis de ton cote .
= i m by your side .
< i m sorry for i <EOS>

> il travaille dur en vue de reussir son examen .
= he is working hard to pass the examination .
< he s a of of of . . <EOS>
```
