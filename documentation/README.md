#Documentation

##Major Features:
* Conversation Summarization
* Text Summarization
* Text Segmentation

##Feature: Conversation Summarization

Concerned File: [**ClusterRank.py**](https://github.com/AdiChat/Read-Less/blob/master/readless/Summarization/clusterrank.py)

Related file: [**TextTiling.py**](https://github.com/AdiChat/Read-Less/blob/master/readless/Segmentation/texttiling.py)

Functionalities of **ClusterRank summarization**:

* `lDistance(firstString, secondString)`:
* `buildGraph(nodes)`:
* `extractSentences(text)`:
* `summarize(data)`:
* `summarizeFile(pathToFile)`:

##Feature: Text Summarization

Concerned Files: [**TextRank.py**](https://github.com/AdiChat/Read-Less/blob/master/readless/Summarization/textrank.py), [**RandomSum.py**](https://github.com/AdiChat/Read-Less/blob/master/readless/Summarization/randomSum.py)

Functionalities of **TextRank summarization**:

* `lDistance(firstString, secondString)`:
* `buildGraph(nodes)`:
* `extractSentences(text)`:
* `summarize(data)`:
* `summarizeFile(pathToFile)`:

Functionalities of **Random Summarization**:

* `random(firstString, secondString)`:
* `buildGraph(nodes)`:
* `extractSentences(text)`:
* `summarize(data)`:
* `summarizeFile(pathToFile)`:

##Feature: Text Segmentation

Concerned File: [**TextTiling.py**](https://github.com/AdiChat/Read-Less/blob/master/readless/Segmentation/texttiling.py)

* `tokenize_string(input_string, w)`: 
* `vocabulary_introduction(token_sequences, w)`:
* `block_score(k, token_sequence, unique_tokens)`:
* `getDepthCutoff(lexScores, liberal=True)`:
* `getDepthSideScore( lexScores, currentGap, left)`:
* `getGapBoundaries(lexScores)`:
* `getBoundaries(lexScores, pLocs, w)`:
* `segmentText(boundaries, pLocs, inputText)`:
* `run(data, w=4, k=10, select_segment=2)`:
* `segmentFile(pathToFile)`:

