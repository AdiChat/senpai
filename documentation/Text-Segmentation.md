##Feature: Text Segmentation

Concerned File: [**TextTiling.py**](https://github.com/AdiChat/Read-Less/blob/master/readless/Segmentation/texttiling.py)

#####* `tokenize_string(input_string, w)`: 

   Tokenize a string using the following four steps:
            1) Turn all text into lowercase and split into tokens by
               removing all punctuation except for apostrophes and internal
               hyphens
            2) Remove stop words 
            3) Perform lemmatization 
            4) Group the tokens into groups of size w, which represents the 
               pseudo-sentence size.
     
        Arguments :
            input_string : A string to tokenize
            w: pseudo-sentence size
        Returns:
            A tuple (token_sequences, unique_tokens, paragraph_breaks), where:
                token_sequences: A list of token sequences, each w tokens long.
                unique_tokens: A set of all unique words used in the text.
                paragraph_breaks: A list of indices such that paragraph breaks
                                  occur immediately after each index.
#####* `vocabulary_introduction(token_sequences, w)`:

   Computes lexical score for the gap between pairs of text sequences.
      It starts assigning scores after the first sequence.
    
        Arguments :
            w: size of a sequence
        Returns:
            list of scores where scores[i] corresponds to the score at gap position i that is the score after sequence i.
        Raises:
            None
#####* `block_score(k, token_sequence, unique_tokens)`:
    Computes the similarity scores for adjacent blocks of token sequences.
    
        Arguments:
            k: the block size
            token_seq_ls: list of token sequences, each of the same length
            unique_tokens: A set of all unique words used in the text.
        Returns:
            list of block scores from gap k through gap (len(token_sequence)-k-2) both inclusive.
        Raises:
            None.

#####* `getDepthCutoff(lexScores, liberal=True)`:
        Compute the cutoff for depth scores above which gaps are considered boundaries.
        Args:
            lexScores: list of lexical scores for each token-sequence gap
            liberal: True IFF liberal criterion will be used for determining cutoff
        Returns:
            A float representing the depth cutoff score
        Raises:
            None

#####* `getDepthSideScore( lexScores, currentGap, left)`:
        Computes the depth score for the specified side of the specified gap
        Args:
            lexScores: list of lexical scores for each token-sequence gap
            currentGap: index of gap for which to get depth side score
            left: True IFF the depth score for left side is desired
        Returns:
            A float representing the depth score for the specified side and gap,
            calculated by finding the "peak" on the side of the gap and returning
            the difference between the lexical scores of the peak and gap.
        Raises:
            None

#####* `getGapBoundaries(lexScores)`:
        Get the gaps to be considered as boundaries based on gap lexical scores
        Args:
            lexScores: list of lexical scores for each token-sequence gap
        Returns:
            A list of gaps (identified by index) that are considered boundaries.
        Raises:
            None

#####* `getBoundaries(lexScores, pLocs, w)`:
      Get locations of paragraphs where subtopic boundaries occur
        Args:
            lexScores: list of lexical scores for each token-sequence gap
            pLocs: list of token indices such that paragraph breaks occur after them
            w: number of tokens to be grouped into each token-sequence
        Returns:
            A sorted list of unique paragraph locations (measured in terms of token
            indices) after which a subtopic boundary occurs.
        Raises:
            None

#####* `segmentText(boundaries, pLocs, inputText)`:
     Get TextTiles in the input text based on paragraph locations and boundaries.
        Args:
            boundaries: list of paragraph locations where subtopic boundaries occur
            pLocs: list of token indices such that paragraph breaks occur after them
            inputText: a string of the initial (unsanitized) text
        Returns:
            output: A list of segmented text
        Raises:
            None

#####* `run(data, w=4, k=10, select_segment=2)`:
        Helper function that runs the TextTiling on the entire corpus
        for a given set of parameters w and k, and writes the average 
        statistics to outfile. 
        Args:
            w: pseudo-sentence size; default value taken as 4
            k: length of window; default value taken as 10
            data: input string
            select_segment: (0,1) indicates whether to use block comparison or vocabulary introduction
        Returns:
            segmented text
        Raises:
            None.
#####* `segmentFile(pathToFile)`:
        Helper function to segment a textual data
        Arguments:
            pathToFile: path to the file containing the textual data to be segmented
        Returns:
            segmented text
        Raises:
            None