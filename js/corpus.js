// The tiny training corpus. Everything the app "knows" — the tokenizer's
// merges, the embedding geometry, and the bigram language prior — is derived
// from this text at load time. Deliberately small and readable so a curious
// user can open this file and see exactly where the behavior comes from.
//
// Vocabulary is controlled on purpose: recurring themes (cats and dogs, sun
// and moon, rivers, and the model talking about itself) give the bigram
// prior enough repetition to chain plausible sentences, and give the
// co-occurrence embeddings real structure (cat ~ dog, sun ~ moon).
// Morphological families (play / playing / played, light / sunlight /
// moonlight) are included so BPE learns reusable subword pieces.

const CORPUS_TEXT = `
the cat sat on the mat and the dog sat on the rug .
the cat sleeps in the warm sun and the dog sleeps in the shade .
a small cat chased the ball and a big dog chased the cat .
the black cat likes warm milk and the brown dog likes cold water .
the cat is playing with the ball in the garden .
the dog is playing with the stick in the park .
the cat played all day and slept all night .
the dog played in the river and the water was cold .
the bird sings in the morning and sleeps in the tree at night .
a small bird sat on the tree and looked at the garden .
the fish swims in the cold river near the old bridge .
the fish likes deep water and the bird likes the open sky .
the sun rises in the morning and the moon rises at night .
the sun is bright and warm and the moon is pale and cold .
sunlight fills the garden in the morning .
moonlight fills the river at night .
the light of the sun is warm and the light of the moon is soft .
the rain falls on the garden and the river grows big .
the rain fell all night and the morning was grey and cold .
the wind blows over the river and the trees move in the wind .
snow falls in winter and the garden sleeps under the snow .
in winter the night is long and the day is short .
in summer the day is long and the sun is high in the sky .
the sky is blue in summer and grey in winter .
the old man walks in the park every morning .
the old man likes to read a book in the warm sun .
a young child plays with the small cat in the garden .
the child likes the dog and the dog likes the child .
the child reads a small book about the moon and the stars .
the stars shine in the night sky over the quiet town .
the town is quiet at night and busy in the morning .
people walk over the old bridge to the market in the town .
the market sells warm bread and cold milk in the morning .
the bread is warm and the milk is cold .
the river runs through the town and under the old bridge .
the road runs through the town and over the hill .
a small house sits on the hill near the old tree .
the house is warm in winter and cool in summer .
the garden of the house is full of red and yellow flowers .
the flowers open in the morning light and close at night .
the model reads the input one token at a time .
the model turns each token into a vector of numbers .
each vector flows through the layers of the model .
attention lets each token look at the tokens before it .
each attention head learns to look at different tokens .
the layers of the model turn vectors into new vectors .
the neurons in each layer fire on different patterns .
the model gives each next token a probability .
the model picks the next token and starts again .
a token can be a word or a small piece of a word .
the tokens flow through the model and become a prediction .
a big model has many layers and many attention heads .
the network learns patterns from the text it reads .
words with similar meaning get similar vectors .
the vector for cat is near the vector for dog .
the vector for sun is near the vector for moon .
a probability is a number between zero and one .
high temperature makes the model pick rare tokens .
low temperature makes the model pick safe tokens .
the cat looked at the fish in the river and the fish swam away .
the dog looked at the bird in the tree and the bird flew away .
the cat and the dog sleep in the warm house in winter .
the child and the old man walk over the bridge in the rain .
the morning light is soft and the night wind is cold .
the quiet garden sleeps under the pale moonlight .
the busy market opens in the bright morning sunlight .
the small fish swims under the bridge in the deep water .
the young bird sings a soft song in the old tree .
the model is small but the idea is big .
this small model plays with words like a child plays with a ball .
the next token is a prediction and the prediction is a probability .
the input becomes tokens and the tokens become vectors .
the output of one layer is the input of the next layer .
the last layer turns the vectors into a prediction .
`;
