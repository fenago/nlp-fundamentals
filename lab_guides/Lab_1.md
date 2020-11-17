<img align="right" src="../logo.png">


Lab 1. Introduction to Natural Language Processing
==============================================

Learning Objectives
-------------------

By the end of this lab, you will be able to:

-   Describe what natural language processing (NLP) is all about
-   Describe the history of NLP
-   Differentiate between NLP and Text Analytics
-   Implement various preprocessing tasks
-   Describe the various phases of an NLP project

In this lab, you will learn about the basics of natural language
processing and various preprocessing steps that are required to clean
and analyze the data.


#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

All examples are present in `work/nlp-fundamentals/Lesson1` folder. 

You can access lab at `http://<host-ip>/lab/workspaces/lab1_Introduction`


Exercise 1: Basic Text Analytics
--------------------------------

In this exercise, we will perform some basic text analytics on the given
text data. Follow these steps to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell. Assign a `sentence` variable with
    \'`The quick brown fox jumps over the lazy dog`\'. Insert
    a new cell and add the following code to implement this:


    ```
    sentence = 'The quick brown fox jumps over the lazy dog'
    ```


3.  Check whether the word \'`quick`\' belongs to that text
    using the following code:

    ```
    'quick' in sentence
    ```


    The preceding code will return the output \'**True**\'.

4.  Find out the `index` value of the word \'`fox`\'
    using the following code:

    ```
    sentence.index('fox')
    ```


    The code will return the output **16**.

5.  To find out the rank of the word \'`lazy`\', use the
    following code:

    ```
    sentence.split().index('lazy')
    ```


    The code generates the output **7**.

6.  For printing the third word of the given text, use the following
    code:

    ```
    sentence.split()[2]
    ```


    This will return the output \'**brown**\'.

7.  To print the third word of the given sentence in reverse order, use
    the following code:

    ```
    sentence.split()[2][::-1]
    ```


    This will return the output \'**nworb**\'.

8.  To concatenate the first and last words of the given sentence, use
    the following code:

    ```
    words = sentence.split()first_word = words[0]last_word = words[len(words)-1]concat_word = first_word + last_word print(concat_word)
    ```


    The code will generate the output \'**Thedog**\'.

9.  For printing words at even positions, use the following code:

    ```
    [words[i] for i in range(len(words)) if i%2 == 0]
    ```


    The code generates the following output:

    
![](./images/C13142_01_03.jpg)


10. To print the last three letters of the text, use the following code:

    ```
    sentence[-3:]
    ```


    This will generate the output \'**dog**\'.

11. To print the text in reverse order, use the following code:

    ```
    sentence[::-1]
    ```


    The code generates the following output:

    
![](./images/C13142_01_04.jpg)


12. To print each word of the given text in reverse order, maintaining
    their sequence, use the following code:

    ```
    print(' '.join([word[::-1] for word in words]))
    ```


    The code generates the following output:

![](./images/C13142_01_05.jpg)


We are now well acquainted with NLP. In the next section, let\'s dive
deeper into the various steps involved in it.



Exercise 2: Tokenization of a Simple Sentence
---------------------------------------------

In this exercise, we will tokenize the words in a given sentence with
the help of the **NLTK** library. Follow these steps to implement this
exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add a following code to import the necessary
    libraries:


    ```
    import nltk
    from nltk import word_tokenize
    ```


3.  The `word_tokenize()` method is used to split the sentence
    into words/tokens. We need to add a sentence as input to the
    `word_tokenize()` method, so that it performs its job. The
    result obtained would be a list, which we will store in a
    `word` variable. To implement this, insert a new cell and
    add the following code:


    ```
    words = word_tokenize("I am reading NLP Fundamentals")
    ```


4.  In order to view the list of tokens generated, we need to view it
    using the `print()` function. Insert a new cell and add
    the following code to implement this:

    ```
    print(words)
    ```


    The code generates the following output:

    
![](./images/C13142_01_06.jpg)


Thus we can see the list of tokens generated with the help of the
`word_tokenize()` method. In the next section, we will see
another pre-processing step: **Parts-of-Speech (PoS) tagging**.

PoS Tagging
-----------

If we look at the sentence,
\"The sky is blue,\" we get four tokens -- \"The,\" \"sky,\" \"is,\" and
\"blue\" with the help of tokenization. Now, using **PoS tagger**, we
tag parts of speech to each word/token. This will look as follows:

*\[(\'The\', \'DT\'), (\'sky\', \'NN\'), (\'is\', \'VBZ\'), (\'blue\',
\'JJ\')\]*

*DT = determiner*

*NN = noun, common, singular or mass*

*VBZ = verb, present tense, 3rd* *person singular*

*JJ = Adjective*

An exercise in the next section will definitely give a better
understanding of this concept.

Exercise 3: PoS Tagging
-----------------------

In this exercise, we will find out the PoS for each word in the
sentence, \"`I am reading NLP Fundamentals`\". We first make
use of tokenization in order to get the tokens. Later, we use a PoS
tagger, which will help us find PoS for each word/token. Follow these
steps to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to import the necessary
    libraries:


    ```
    import nltk
    from nltk import word_tokenize
    ```


3.  For finding the tokens in the sentence, we make use of the
    `word_tokenize()` method. Insert a new cell and add the
    following code to implement this:


    ```
    words = word_tokenize("I am reading NLP Fundamentals")
    ```


4.  Print the tokens with the help of the `print()` function.
    To implement this, add a new cell and write the following code:

    ```
    print(words)
    ```


    The code generates the following output:

    
![](./images/C13142_01_07.jpg)


5.  In order to find the PoS for each word, we make use of the
    `pos_tag()` method of the `nltk` library. Insert
    a new cell and add the following code to implement this:

    ```
    nltk.pos_tag(words)
    ```


    The code generates the following output:

    
![](./images/C13142_01_08.jpg)


6.  In the preceding output, we can see that for each token, a PoS has
    been allotted. Here, **PRP** stands for **personal pronoun**,
    **VBP** stands for **verb present**, **VGB** stands for **verb
    gerund**, **NNP** stands for **proper noun singular**, and **NNS**
    stands for **noun plural**.

We have learned about labeling appropriate PoS to tokens in a sentence.
In the next section, we will learn about **stop words** in sentences and
ways to deal with them.



Exercise 4: Stop Word Removal
-----------------------------

In this exercise, we will check the list of stopwords provided by the
`nltk` library. Based on this list, we will filter out the
stopwords included in our text. Follow these steps to implement this
exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to import the necessary
    libraries:


    ```
    import nltk
    nltk.download('stopwords')
    from nltk import word_tokenize
    from nltk.corpus import stopwords
    ```


3.  In order to check the list of stopwords provided for the
    `English` language, we pass it as a parameter to the
    `words()` function. Insert a new cell and add the
    following code to implement this:


    ```
    stop_words = stopwords.words('english')
    ```


4.  In the code, the list of stopwords provided by the
    `English` language is stored in the `stop_words`
    variable. In order to view the list, we make use of the
    `print()` function. Insert a new cell and add the
    following code to view the list:

    ```
    print(stop_words)
    ```


    The code generates the following output:

    
![](./images/C13142_01_09.jpg)


5.  To remove the stop words from a sentence, we first assign a string
    to the `sentence` variable and tokenize it into words
    using the `word_tokenize()` method. Insert a new cell and
    add the following code to implement this:


    ```
    sentence = "I am learning Python. It is one of the most popular programming languages"
    sentence_words = word_tokenize(sentence)
    ```


6.  To print the list of tokens, insert a new cell and add the following
    code:

    ```
    print(sentence_words)
    ```


    The code generates the following output:

    
![](./images/C13142_01_10.jpg)


7.  To remove the stopwords, first we need to loop through each word in
    the sentence, check whether there are any stop words, and then
    finally combine them to form a complete sentence. To implement this,
    insert a new cell and add the following code:


    ```
    sentence_no_stops = ' '.join([word for word in sentence_words if word not in stop_words])
    ```


8.  To check whether the stopwords are filtered out from our sentence,
    we print the `sentence_no_stops` variable. Insert a new
    cell and add the following code to print:

    ```
    print(sentence_no_stops)
    ```


    The code generates the following output:

![](./images/C13142_01_11.jpg)


As you can see in the preceding figure, stopwords such as \"**am,**\"
\"**is,**\" \"**of,**\" \"**the,**\" and \"**most**\" are being filtered
out and text without stop words is produced as output.

We have learned how to remove stop words from given text. In the next
section, we will focus on normalizing text.


Exercise 5: Text Normalization
------------------------------

In this exercise, we will normalize a given text. Basically, we will be
trying to replace selected words with new words, using the
`replace()` function, and finally produce the normalized text.
Follow these steps to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to assign a string to
    the `sentence` variable:


    ```
    sentence = "I visited US from UK on 22-10-18"
    ```


3.  We want to replace \"`US`\" with
    \"`United States`\", \"`UK`\" with
    \"`United Kingdom`\", and \"`18`\" with
    \"`2018`\". To do so, we make use of the
    `replace()` function and store the updated output in the
    \"`normalized_sentence`\" variable. Insert a new cell and
    add the following code to implement this:


    ```
    normalized_sentence = sentence.replace("US", "United States").replace("UK", "United Kingdom").replace("-18", "-2018")
    ```


4.  Now, in order to check whether the text has been normalized, we
    insert a new cell and add the following code to print it:

    ```
    print(normalized_sentence)
    ```


    The code generates the following output:

![](./images/C13142_01_12.jpg)


In the preceding figure, we can see that our text has been normalized.

Now that we have learned the basics of text normalization, in the next
section, we explore various other ways that text can be normalized.

Spelling Correction
-------------------

Spelling correction is one of the most important tasks in any NLP
project. It can be time consuming, but without it there are high chances
of losing out on required information. We make use of the
\"**autocorrect**\" Python library to correct spellings. Let\'s look at
the following exercise to get a better understanding of this.

Exercise 6: Spelling Correction of a Word and a Sentence
--------------------------------------------------------

In this exercise, we will perform spelling correction on a word and a
sentence, with the help of the \'`autocorrect`\' library of
Python. Follow these steps in order to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to import the necessary
    libraries:


    ```
    import nltk
    from nltk import word_tokenize
    from autocorrect import spell
    ```


3.  In order to correct the spelling of a word, pass a wrongly spelled
    word as a parameter to the `spell()` function. Insert a
    new cell and add the following code to implement this:

    ```
    spell('Natureal')
    ```


    The code generates the following output:

    
![](./images/C13142_01_13.jpg)


4.  In order to correct the spelling of a sentence, we first need to
    tokenize it into words. After that, we loop through each word in
    `sentence`, autocorrect them, and finally combine them.
    Insert a new cell and add the following code to implement this:


    ```
    sentence = word_tokenize("Ntural Luanguage Processin deals with the art of extracting insightes from Natural Languaes")
    ```


5.  We make use of the `print()` function to print all tokens.
    Insert a new cell and add the following code to print tokens:

    ```
    print(sentence)
    ```


    The code generates the following output:

    
![](./images/C13142_01_14.jpg)


6.  Now that we have got the tokens, we loop through each token in
    `sentence`, correct them, and assign them to new variable.
    Insert a new cell and add the following code to implement this:


    ```
    sentence_corrected = ' '.join([spell(word) for word in sentence])
    ```


7.  To print the correct sentence, we insert a new cell and add the
    following code:

    ```
    print(sentence_corrected)
    ```


    The code generates the following output:

    
![](./images/C13142_01_15.jpg)


8.  In the preceding figure, we can see that most of the wrongly spelled
    words have been corrected. But the word \"**Processin**\" was
    wrongly converted into \"**Procession**\". It should have been
    \"**Processing**\". It happened because to change \"Processin\" to
    \"Procession\" or \"Processing,\" an equal number of edits is
    required. To rectify this, we need to use other kinds of spelling
    correctors that are aware of context.

    In the next section, we will look at **stemming**, which is another
    form of text normalization.

Stemming
--------

In languages such as English, words get transformed into various forms
when being used in a sentence. For example, the word \"product\" might
get transformed into \"production\" when referring to the process of
making something or transformed into \"products\" in plural form. It is
necessary to convert these words into their base forms, as they carry
the same meaning. Stemming is a process that helps us in doing so. If we
look at the following figure, we get a perfect idea about how words get
transformed into their base forms:

![](./images/C13142_01_16.jpg)


To get a better understanding about stemming, we shall look into an
exercise in the next section.

Exercise 7: Stemming
--------------------

In this exercise, we will pass a few words through the stemming process
such that they get converted into their base forms. Follow these steps
to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to import the necessary
    libraries:


    ```
    import nltk
    stemmer = nltk.stem.PorterStemmer()
    ```


3.  Now pass the following words as parameters to the `stem()`
    method. To implement this, insert a new cell and add the following
    code:

    ```
    stemmer.stem("production")
    ```


    When the input is \"`production`\", the following output
    is generated:

    
![](./images/C13142_01_17.jpg)


    ```
    stemmer.stem("coming")
    ```


    When the input is \"`coming`\", the following output is
    generated:

    
![](./images/C13142_01_18.jpg)


    ```
    stemmer.stem("firing")
    ```


    When the input is \"`firing`\", the following output is
    generated:

    
![](./images/C13142_01_19.jpg)


    ```
    stemmer.stem("battling")
    ```


    When the input is \"`battling`\", the following output is
    generated:

    
![](./images/C13142_01_20.jpg)


4.  From the preceding figures, we can see that the entered words are
    converted into their base forms.

In the next section, we will focus on **lemmatization**, which is
another form of text normalization.

Lemmatization
-------------

Sometimes, the stemming process leads to inappropriate results. For
example, in the last exercise, the word \"battling\" got transformed to
\"battl,\" which has no meaning. To overcome these problems with
stemming, we make use of lemmatization. In this process, an additional
check is being made, by looking through the dictionary to extract the
base form of a word. However, this additional check slows down the
process. To get a better understanding about lemmatization, we will look
at an exercise in the next section.

Exercise 8: Extracting the base word using Lemmatization
--------------------------------------------------------

In this exercise, we use the lemmatization process to produce the proper
form of a given word. Follow these steps to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to import the necessary
    libraries:


    ```
    import nltk
    nltk.download('wordnet')
    from nltk.stem.wordnet import WordNetLemmatizer
    ```

3.  Create object of the `WordNetLemmatizer` class. Insert a
    new cell and add the following code to implement this:


    ```
    lemmatizer = WordNetLemmatizer()
    ```


4.  Bring the word to its proper form by using the
    `lemmatize()` method of the `WordNetLemmatizer`
    class. Insert a new cell and add the following code to implement
    this:

    ```
    lemmatizer.lemmatize('products')
    ```


    With the input \"`products`\", the following output is
    generated:

![](./images/C13142_01_21.jpg)


```
lemmatizer.lemmatize('production')
```

With the input \"`production`\", the following output is
generated:

![](./images/C13142_01_22.jpg)


```
lemmatizer.lemmatize('coming')
```

With the input \"`coming`\", the following output is generated:

![](./images/C13142_01_23.jpg)


```
lemmatizer.lemmatize('battle')
```

With the input \"`battle`\", the following output is
generated:

![](./images/C13142_01_24.jpg)


We have learned how to use the lemmatization process to transform a
given word into its base form.



Exercise 9: Treating Named Entities
-----------------------------------

In this exercise, we will find out the named entities in a sentence.
Follow these steps to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to import the necessary
    libraries:


    ```
    import nltk
    from nltk import word_tokenize
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    ```


3.  Declare the `sentence` variable and assign it a string.
    Insert a new cell and add the following code to implement this:


    ```
    sentence = "We are reading a book published by Packt which is based out of Birmingham."
    ```


4.  To find the named entities from the preceding text, insert a new
    cell and the following code:

    ```
    i = nltk.ne_chunk(nltk.pos_tag(word_tokenize(sentence)), binary=True)
    [a for a in i if len(a)==1]
    ```


Word Sense Disambiguation
-------------------------

Word sense disambiguation is the process of mapping a word to
the correct sense it carries. We need to disambiguate words based on the
sense they carry so that they can be treated as different entities when
being analyzed. The following figure displays a perfect example of how
ambiguity is caused due to the usage of the same word in different
sentences:

![](./images/C13142_01_26.jpg)


Exercise 10: Word Sense Disambiguation
--------------------------------------

In this exercise, we will find the sense of the word \"bank\" in two
different sentences. Follow these steps to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to import the necessary
    libraries:


    ```
    import nltk
    from nltk.wsd import lesk
    from nltk import word_tokenize
    ```


3.  Declare two variables, `sentence1` and
    `sentence2`, and assign them with appropriate strings.
    Insert a new cell and the following code to implement this:


    ```
    sentence1 = "Keep your savings in the bank"
    sentence2 = "It's so risky to drive over the banks of the road"
    ```


4.  In order to find the sense of the word \"bank\" in the preceding two
    sentences, we make use of the `lesk` algorithm provided by
    the `nltk.wsd` library. Insert a new cell and add the
    following code to implement this:

    ```
    print(lesk(word_tokenize(sentence1), 'bank'))
    ```


    The code generates the following output:

![](./images/C13142_01_27.jpg)


Here, `savings_bank.n.02` refers to a container for keeping
money safely at home. To check the other sense of the word bank, write
the following code:

```
print(lesk(word_tokenize(sentence2), 'bank')) 
```

The code generates the following output:

![](./images/C13142_01_28.jpg)


Here, `bank.v.07` refers to a slope in the turn of a road.


#### Sentence Boundary Detection
---------------------------

Sentence boundary detection is the method of detecting where one
sentence ends and where another sentence begins. Various analyses need to be performed at a sentence level,
so detecting boundaries of sentences is essential. An exercise in the
next section will provide a better understanding of this process.

Exercise 11: Sentence Boundary Detection
----------------------------------------

In this exercise, we will extract sentences from a paragraph. Follow
these steps to implement this exercise:

1.  Open a Jupyter notebook.

2.  Insert a new cell and add the following code to import the necessary
    libraries:


    ```
    import nltk
    from nltk.tokenize import sent_tokenize
    ```


3.  We make use of the `sent_tokenize()` method to detect
    sentences in a given text. Insert a new cell and add the following
    code to implement this:

    ```
    sent_tokenize("We are reading a book. Do you know who is the publisher? It is Packt. Packt is based out of Birmingham.")
    ```


    The code generates the following output:

![](./images/C13142_01_29.jpg)



Activity 1: Preprocessing of Raw Text
-------------------------------------

We have a text corpus that is in an improper format. In this activity,
we will perform all the pre-processing steps that were discussed earlier
to get some meaning out of the text.

**Note:** The `file.txt` file can be found at this location:

`/home/jovyan/work/nlp-fundamentals/Lesson1/data_ch1`

Follow these steps to implement this activity:

1.  Import the necessary libraries.

2.  Load the text corpus to a variable.

3.  Apply the tokenization process to the text corpus and print the
    first 20 tokens.

4.  Apply spelling correction on each token and print the initial 20
    corrected tokens as well as the corrected text corpus.

5.  Apply PoS tags to each corrected token and print them.

6.  Remove stop words from the corrected token list and print the
    initial 20 tokens.

7.  Apply stemming and lemmatization to the corrected token list and the
    print initial 20 tokens.

8.  Detect the sentence boundaries in the given text corpus and print
    the total number of sentences.

    ##### Note

    The solution for this activity can be found in the current directory.

By now, you should be familiar with what NLP is and what basic
pre-processing steps are needed to carry out any NLP project. In the
next section, we will focus on different phases that are included in an
NLP project.


#### Kick Starting an NLP Project

We can divide an NLP project into several sub-projects or phases. These
phases are followed sequentially. This tends to increase the overall
efficiency of the process as each phase is generally carried out by
specialized resources. An NLP project has to go through six major
phases, which are outlined in the following figure:

![](./images/C13142_01_30.jpg)


#### Summary

In this lab, we learned how NLP is different from text analytics. We
also covered the various pre-processing steps that are included in NLP.
We looked at the different phases an NLP project has to pass through. In
the next lab, you will learn about the different methods required
for extracting features from unstructured texts, such as TF-IDF and bag
of words. You will also learn about NLP tasks such as tokenization,
lemmatization, and stemming in more detail. Furthermore, text
visualization techniques such as word clouds will be introduced.
