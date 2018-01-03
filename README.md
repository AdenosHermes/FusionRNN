# Fusion LSTM exercise (version 2, with Coefficient LSTM added)

This is just a toy example in which we perform fusion of multiple LSTMs at each time step of the unfolding of an LSTM. This example is about French-to-English translation.

The dataset and the basic archetecture are adapted from: <br>
http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

# General Architecture
See model.py for details of the network architecture. The architecture of the network looks like: <br>
<img src="img/fusionRNN.png" alt="architecture">

At the i-th time step (in this case, at the i-th word in a array of word, i. e. a sentence), the FusionLSTM takes in the hidden states of three different LSTM encoders, and outputs the fusion_state, i. e. fusion_state = FusionLSTM(hidden_1, hidden_2, hidden_3 ), and the encoders each takes in the input, its own hidden state, and the fusion_state, and outputs an encoding of the sentence, which is then decoded by a GRU decoder.

# Fusion with Coefficient LSTM
The Fusion LSTM is the core of this architecture, it involves the interaction of an LSTM fuser (see class fusionLSTM in model.py) and a coeffiecient LSTM (see WeightLSTM in model.py) that weights the fused states of the fuser, the fusion LSTM has the following complicated structure:
<img src="img/fusionLSTM.png" alt="fusion">

At each step, the coeffiencient takes in its hidden state and its last output (which is initialized to a vector with value 1 / Fusion_Size, i.e. a uniform weight), and the output is a weight vector computed by taking the softmax of its the hidden state (thus converting it to a distribution). We use the weight vector to weight the hidden states and use this as the input to the fusion LSTM, and cycle this for three steps (since more layers of LSTMs tend to give better result but are slow to train).

# Parameters and Results
In this example, all three encoders have the same architecture except that they respectively have hidden state size 256, 64, 16, and so fusion_size = 256 + 64 +16. The learnign rate is 0.01 with momentum 0.2 (I experimented with momentum 0.2, 0.5, and 0.9, and 0.2 seems to render the fastest and more stable convergence). I was sort of expecting that having different hidden-sizes might make the encoders learn to encode features of a sentence at different length scales, but in practice I find it extremely hard and slow to train.

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
# To Do
-This is a very tentative version, and the code is far from being polished. So I guess I need to polish the code <br>
-I realized that I forgot to pass on the cell states of the LSTMs, I will fix this slightly later <br>
-In the current implementation, the fusion_state is passed in alongside the input word (i.e. cat(input, fusion_state)) into a general LSTM, a better version should probably implement a customized LSTM, and do some separate operation on the fusion_state
