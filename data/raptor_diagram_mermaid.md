flowchart TD
 subgraph subGraph0["Tree Construction"]
        TreeBuilder["TreeBuilder"]
        TreeConstruction["Tree Construction"]
        Summarization["Summarization"]
        Embedding["Embedding"]
        Clustering["Clustering"]
        SummarizationModels["Summarization Models"]
        EmbeddingModels["Embedding Models"]
        ClusteringAlgorithms["Clustering Algorithms"]
  end
 subgraph subGraph1["Information Retrieval"]
        TreeRetriever["TreeRetriever"]
        InformationRetrieval("Information Retrieval")
        FaissRetriever["FaissRetriever"]
        InformationRetriever["InformationRetriever"]
        SimilaritySearch["Similarity Search"]
        FaissIndex["Faiss Index"]
  end
 subgraph subGraph2["Question Answering"]
        QAModels["Question Answering Models"]
        QuestionAnswering["Question Answering"]
        LanguageModels["Language Models"]
  end
 subgraph subGraph3["raptor Project"]
        RetrievalAugmentation["RetrievalAugmentation"]
        subGraph0
        subGraph1
        subGraph2
        GPT3TurboSummarizationModel["GPT3TurboSummarizationModel"]
        GPT3SummarizationModel["GPT3SummarizationModel"]
        OpenAIEmbeddingModel["OpenAIEmbeddingModel"]
        SBertEmbeddingModel["SBertEmbeddingModel"]
        RAPTOR_Clustering["RAPTOR_Clustering"]
        GPT3QAModel["GPT3QAModel"]
        GPT3TurboQAModel["GPT3TurboQAModel"]
        GPT4QAModel["GPT4QAModel"]
        UnifiedQAModel["UnifiedQAModel"]
        OpenAIModels["OpenAIModels"]
        HuggingFaceModels["HuggingFaceModels"]
        CosineDistance["CosineDistance"]
        TopKSelection["TopKSelection"]
        IndexFlatIP["IndexFlatIP"]
  end
    RetrievalAugmentation --> TreeConstruction & InformationRetrieval & QuestionAnswering
    TreeConstruction --> TreeBuilder
    TreeBuilder --> Summarization & Embedding & Clustering
    Summarization --> SummarizationModels
    Embedding --> EmbeddingModels
    Clustering --> ClusteringAlgorithms
    InformationRetrieval --> TreeRetriever
    InformationRetriever --> FaissRetriever
    TreeRetriever --> SimilaritySearch
    FaissRetriever --> FaissIndex
    QuestionAnswering --> QAModels
    QAModels --> LanguageModels & GPT3QAModel & GPT3TurboQAModel & GPT4QAModel & UnifiedQAModel
    SummarizationModels --> GPT3TurboSummarizationModel & GPT3SummarizationModel
    EmbeddingModels --> OpenAIEmbeddingModel & SBertEmbeddingModel
    ClusteringAlgorithms --> RAPTOR_Clustering
    LanguageModels --> OpenAIModels & HuggingFaceModels
    SimilaritySearch --> CosineDistance & TopKSelection
    FaissIndex --> IndexFlatIP
