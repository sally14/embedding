# embedding


Embedding generation with text preprocessing. 


## Preprocessor

For Word2Vec, we want a soft yet important preprocessing. We want to denoise the text while keeping as much variety and information as possible. 

Preprocesses the text/set of text in the following way : 
 - 1. Detects and replaces numbers/float by a generic token 'FLOAT', 'INT'
 - 2. Add spaces in between punctuation so that tokenisation avoids adding 'word.' to the vocabulary instead of 'word', '.'. 
 - 3. Lowers words
 - 4. Recursive word phrases detection : with a simple probabilistic rule, gathers the tokens 'new', york' to a single token 'new_york'. 
 - 5. Frequency Subsampling : discards unfrequent words with a probability depending on their frequency. 

 