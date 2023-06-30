# Detection-of-Change-Points
Identification of Temporal Transition of Functional States Using Recurrent Neural Networks from Functional MRI

Split the entire functional MRI scan into segments by detecting change points of functional signals to facilitate better characterization of temporally dynamic functional connectivity patterns

Abstract:
Dynamic functional connectivity analysis provides valuable information for understanding brain functional activity underlying different cognitive processes. Besides sliding window based approaches, a variety of methods have been developed to automatically split the entire functional MRI scan into segments by detecting change points of functional signals to facilitate better characterization of temporally dynamic functional connectivity patterns. However, these methods are based on certain assumptions for the functional signals, such as Gaussian distribution, which are not necessarily suitable for the fMRI data. In this study, we develop a deep learning based framework for adaptively detecting temporally dynamic functional state transitions in a data-driven way without any explicit modeling assumptions, by leveraging recent advances in recurrent neural networks (RNNs) for sequence modeling. Particularly, we solve this problem in an anomaly detection framework with an assumption that the functional profile of one single time point could be reliably predicted based on its preceding profiles within a stable functional state, while large prediction errors would occur around change points of functional states. We evaluate the proposed method using both task and resting-state fMRI data obtained from the human connectome project and experimental results have demonstrated that the proposed change point detection method could effectively identify change points between different task events and split the resting-state fMRI into segments with distinct functional connectivity patterns.

Keywords: Brain fMRI; Change point detection; Functional dynamics; Recurrent neural networks.

Reference:
Li H, Fan Y. Identification of Temporal Transition of Functional States Using Recurrent Neural Networks from Functional MRI. Med Image Comput Comput Assist Interv. 2018 Sep;11072:232-239. doi: 10.1007/978-3-030-00931-1_27. Epub 2018 Sep 13. PMID: 30320310; PMCID: PMC6180329. https://pubmed.ncbi.nlm.nih.gov/30320310/
