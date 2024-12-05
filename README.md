# MindMiner

MindMiner is a class designed to build a graph or map of references per initial reference. It receives a list of references as input and constructs a network of interconnected papers. The primary goal of MindMiner is to create a comprehensive text corpus that can be used as training data for a text-to-activation map in the brain. This corpus is intended to facilitate the analysis of how different texts or concepts activate specific regions of the brain.

By traversing the graph of references, MindMiner aims to aggregate a vast amount of text data related to a particular topic or set of topics. This aggregated data can then be used to train machine learning models that predict brain activity patterns in response to different texts or concepts. The ultimate objective is to gain a deeper understanding of how the brain processes and responds to various types of information.

MindMiner's functionality is divided into several key steps:

1. **Fetching papers**: Given a list of paper titles, MindMiner attempts to fetch the papers' metadata, including their abstracts and references.
2. **Building the graph**: MindMiner constructs a graph where each node represents a paper, and the edges represent the references between papers.
3. **Expanding the corpus**: By recursively fetching papers referenced by the initial papers, MindMiner expands the corpus to include a broader range of related texts.
4. **Failed paper tracking**: MindMiner keeps track of papers that fail to be fetched or parsed, allowing for the identification of potential issues or gaps in the corpus.

The resulting corpus can be used for various applications, including but not limited to:

* Training machine learning models to predict brain activity patterns
* Analyzing the relationships between different research topics or concepts
* Identifying influential papers or authors within a particular field
* Facilitating the discovery of new research areas or topics

MindMiner is designed to be a flexible and extensible tool, allowing researchers to customize its behavior and adapt it to their specific needs.