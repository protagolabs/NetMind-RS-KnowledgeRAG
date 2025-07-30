# ERNIE: Enhanced Language Representation with Informative Entities  

hengyan Zhang1,2,3∗, $\mathbf{X}\mathbf{u}\ \mathbf{Han}^{1,2,3*}$ , Zhiyuan $\mathbf{Liu^{1,2,3\dag}}$ , Xin Jiang4, Maosong $\mathbf{Sun^{1,2,3}}$ , Qun Liu4 1Department of Computer Science and Technology, Tsinghua University, Beijing, China 2Institute for Artificial Intelligence, Tsinghua University, Beijing, China 3State Key Lab on Intelligent Technology and Systems, Tsinghua University, Beijing, China 4Huawei Noah’s Ark Lab {zhangzhengyan14,hanxu17}@mails.tsinghua.edu.cn  

# Abstract  

Neural language representation models such as BERT pre-trained on large-scale corpora can well capture rich semantic patterns from plain text, and be fine-tuned to consistently improve the performance of various NLP tasks. However, the existing pre-trained language models rarely consider incorporating knowledge graphs (KGs), which can provide rich structured knowledge facts for better language understanding. We argue that informative entities in KGs can enhance language representation with external knowledge. In this paper, we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously. The experimental results have demonstrated that ERNIE achieves significant improvements on various knowledge-driven tasks, and meanwhile is comparable with the state-of-the-art model BERT on other common NLP tasks. The source code and experiment details of this paper can be obtained from https:// github.com/thunlp/ERNIE.  

# 1 Introduction  

Pre-trained language representation models, including feature-based (Mikolov et al., 2013; Pennington et al., 2014; Peters et al., 2017, 2018) and fine-tuning (Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018; Devlin et al., 2019) approaches, can capture rich language information from text and then benefit many NLP applications. BERT (Devlin et al., 2019), as one of the most recently proposed models, obtains the stateof-the-art results on various NLP applications by simple fine-tuning, including named entity recognition (Sang and De Meulder, 2003), question answering (Rajpurkar et al., 2016; Zellers et al., 2018), natural language inference (Bowman et al., 2015), and text classification (Wang et al., 2018).  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/690b9bf1dee9456b866120e1da675407/a93dec15447c254171c4ddbf17bcf96fb43db62756d16b82196a86a8e69fef44.jpg)  
Figure 1: An example of incorporating extra knowledge information for language understanding. The solid lines present the existing knowledge facts. The red dotted lines present the facts extracted from the sentence in red. The green dotdash lines present the facts extracted from the sentence in green.   
Bob Dylan wrote Blowin’ in the Wind in 1962, and wrote Chronicles: Volume One in 2004.  

Although pre-trained language representation models have achieved promising results and worked as a routine component in many NLP tasks, they neglect to incorporate knowledge information for language understanding. As shown in Figure 1, without knowing Blowin’ in the Wind and Chronicles: Volume One are song and book respectively, it is difficult to recognize the two occupations of Bob Dylan, i.e., songwriter and writer, on the entity typing task. Furthermore, it is nearly impossible to extract the fine-grained relations, such as composer and author on the relation classification task. For the existing pre-trained language representation models, these two sentences are syntactically ambiguous, like “UNK wrote UNK in UNK”. Hence, considering rich knowledge information can lead to better language understanding and accordingly benefits various knowledge-driven applications, e.g. entity typing and relation classification.  

For incorporating external knowledge into language representation models, there are two main challenges. (1) Structured Knowledge Encoding: regarding to the given text, how to effectively extract and encode its related informative facts in KGs for language representation models is an important problem; (2) Heterogeneous Information Fusion: the pre-training procedure for language representation is quite different from the knowledge representation procedure, leading to two individual vector spaces. How to design a special pre-training objective to fuse lexical, syntactic, and knowledge information is another challenge.  

To overcome the challenges mentioned above, we propose Enhanced Language RepresentatioN with Informative Entities (ERNIE), which pretrains a language representation model on both large-scale textual corpora and KGs:  

(1) For extracting and encoding knowledge information, we firstly recognize named entity mentions in text and then align these mentions to their corresponding entities in KGs. Instead of directly using the graph-based facts in KGs, we encode the graph structure of KGs with knowledge embedding algorithms like TransE (Bordes et al., 2013), and then take the informative entity embeddings as input for ERNIE. Based on the alignments between text and KGs, ERNIE integrates entity representations in the knowledge module into the underlying layers of the semantic module.  

(2) Similar to BERT, we adopt the masked language model and the next sentence prediction as the pre-training objectives. Besides, for the better fusion of textual and knowledge features, we design a new pre-training objective by randomly masking some of the named entity alignments in the input text and asking the model to select appropriate entities from KGs to complete the alignments. Unlike the existing pre-trained language representation models only utilizing local context to predict tokens, our objectives require models to aggregate both context and knowledge facts for predicting both tokens and entities, and lead to a knowledgeable language representation model.  

We conduct experiments on two knowledgedriven NLP tasks, i.e., entity typing and relation classification. The experimental results show that ERNIE significantly outperforms the state-of-theart model BERT on these knowledge-driven tasks, by taking full advantage of lexical, syntactic, and knowledge information. We also evaluate ERNIE on other common NLP tasks, and ERNIE still achieves comparable results.  

# 2 Related Work  

Many efforts are devoted to pre-training language representation models for capturing language information from text and then utilizing the information for specific NLP tasks. These pre-training approaches can be divided into two classes, i.e., feature-based approaches and finetuning approaches.  

The early work (Collobert and Weston, 2008; Mikolov et al., 2013; Pennington et al., 2014) focuses on adopting feature-based approaches to transform words into distributed representations. As these pre-trained word representations capture syntactic and semantic information in textual corpora, they are often used as input embeddings and initialization parameters for various NLP models, and offer significant improvements over random initialization parameters (Turian et al., 2010). Since these word-level models often suffer from the word polysemy, Peters et al. (2018) further adopt the sequence-level model (ELMo) to capture complex word features across different linguistic contexts and use ELMo to generate context-aware word embeddings.  

Different from the above-mentioned featurebased language approaches only using the pretrained language representations as input features, Dai and Le (2015) train auto-encoders on unlabeled text, and then use the pre-trained model architecture and parameters as a starting point for other specific NLP models. Inspired by Dai and Le (2015), more pre-trained language representation models for fine-tuning have been proposed. Howard and Ruder (2018) present AWDLSTM (Merity et al., 2018) to build a universal language model (ULMFiT). Radford et al. (2018) propose a generative pre-trained Transformer (Vaswani et al., 2017) (GPT) to learn language representations. Devlin et al. (2019) propose a deep bidirectional model with multiplelayer Transformers (BERT), which achieves the state-of-the-art results for various NLP tasks.  

Though both feature-based and fine-tuning language representation models have achieved great success, they ignore the incorporation of knowledge information. As demonstrated in recent work, injecting extra knowledge information can significantly enhance original models, such as reading comprehension (Mihaylov and Frank, 2018; Zhong et al., 2018), machine translation (Zaremoodi et al., 2018), natural language inference (Chen et al., 2018), knowledge acquisition (Han et al., 2018a), and dialog systems (Madotto et al., 2018). Hence, we argue that extra knowledge information can effectively benefit existing pre-training models. In fact, some work has attempted to joint representation learning of words and entities for effectively leveraging external KGs and achieved promising results (Wang et al., 2014; Toutanova et al., 2015; Han et al., 2016; Yamada et al., 2016; Cao et al., 2017, 2018). Sun et al. (2019) propose the knowledge masking strategy for masked language model to enhance language representation by knowledge 1. In this paper, we further utilize both corpora and KGs to train an enhanced language representation model based on BERT.  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/690b9bf1dee9456b866120e1da675407/fa045cc47c378907f1816a6229bb386bdb450662367b85ae8d30cfebaba596ca.jpg)  
Figure 2: The left part is the architecture of ERNIE. The right part is the aggregator for the mutual integration of the input of tokens and entities. Information fusion layer takes two kinds of input: one is the token embedding, and the other one is the concatenation of the token embedding and entity embedding. After information fusion, it outputs new token embeddings and entity embeddings for the next layer.  

# 3 Methodology  

In this section, we present the overall framework of ERNIE and its detailed implementation, including the model architecture in Section 3.2, the novel pre-training task designed for encoding informative entities and fusing heterogeneous information in Section 3.4, and the details of the fine-tuning procedure in Section 3.5.  

# 3.1 Notations  

We denote a token sequence as $\{w_{1},\ldots,w_{n}\}$ 2, where $n$ is the length of the token sequence. Meanwhile, we denote the entity sequence aligning to the given tokens as $\{e_{1},\ldots,e_{m}\}$ , where $m$ is the length of the entity sequence. Note that $m$ is not equal to $n$ in most cases, as not every token can be aligned to an entity in KGs. Furthermore, we denote the whole vocabulary containing all tokens as $\nu$ , and the entity list containing all entities in KGs as $\mathcal{E}$ . If a token $w\in\mathcal{V}$ has a corresponding entity $e\in\mathcal{E}$ , their alignment is defined as $f(w)=e$ . In this paper, we align an entity to the first token in its named entity phrase, as shown in Figure 2.  

# 3.2 Model Architecture  

As shown in Figure 2, the whole model architecture of ERNIE consists of two stacked modules: (1) the underlying textual encoder (T-Encoder) responsible to capture basic lexical and syntactic information from the input tokens, and (2) the upper knowledgeable encoder (K-Encoder) responsible to integrate extra token-oriented knowledge information into textual information from the underlying layer, so that we can represent heterogeneous information of tokens and entities into a united feature space. Besides, we denote the number of T-Encoder layers as $N$ , and the number of K-Encoder layers as $M$ .  

To be specific, given a token sequence $\{w_{1},\ldots,w_{n}\}$ and its corresponding entity sequence $\{e_{1},\ldots,e_{m}\}$ , the textual encoder firstly sums the token embedding, segment embedding, positional embedding for each token to compute its input embedding, and then computes lexical and syntactic features $\{\pmb{w}_{1},\dots,\pmb{w}_{n}\}$ as follows,  

$$
\{{\pmb w}_{1},\ldots,{\pmb w}_{n}\}=\mathrm{T}\mathrm{-}\mathrm{Encoder}(\{{\pmb w}_{1},\ldots,{\pmb w}_{n}\}),
$$  

where $\mathrm{T}{\mathrm{-Encoder}}(\cdot)$ is a multi-layer bidirectional Transformer encoder. As T-Encoder( ) is identical to its implementation in BERT and BERT is prevalent, we exclude a comprehensive description of this module and refer readers to Devlin et al. (2019) and Vaswani et al. (2017).  

After computing $\{\pmb{w}_{1},\dots,\pmb{w}_{n}\}$ , ERNIE adopts a knowledgeable encoder K-Encoder to inject the knowledge information into language representation. To be specific, we represent $\{e_{1},\ldots,e_{m}\}$ with their entity embeddings $\{e_{1},\ldots,e_{m}\}$ , which are pre-trained by the effective knowledge embedding model TransE (Bordes et al., 2013). Then, both $\{\pmb{w}_{1},\dots,\pmb{w}_{n}\}$ and $\{e_{1},\ldots,e_{m}\}$ are fed into K-Encoder for fusing heterogeneous information and computing final output embeddings,  

$$
\begin{array}{c}{{\{w_{1}^{o},\ldots,w_{n}^{o}\},\{e_{1}^{o},\ldots,e_{n}^{o}\}=\mathrm{K-Encoder}(}}\\ {{\{w_{1},\ldots,w_{n}\},\{e_{1},\ldots,e_{m}\}).}}\end{array}
$$  

$\{\pmb{w}_{1}^{o},\dots,\pmb{w}_{n}^{o}\}$ and $\{e_{1}^{o},\ldots,e_{n}^{o}\}$ will be used as features for specific tasks. More details of the knowledgeable encoder K-Encoder will be introduced in Section 3.3.  

# 3.3 Knowledgeable Encoder  

As shown in Figure 2, the knowledgeable encoder K-Encoder consists of stacked aggregators, which are designed for encoding both tokens and entities as well as fusing their heterogeneous features. In the $i$ -th aggregator, the input token embeddings {w(1i−1), . $\{\boldsymbol{w}_{1}^{(i-1)},\cdot\cdot\cdot,\boldsymbol{w}_{n}^{(i-1)}\}$ (i−1) and entity embeddings $\{e_{1}^{(i-1)},\ldots,e_{m}^{(i-1)}\}$ (mi−1) from the preceding aggregator are fed into two multi-head self-attentions (MH-ATTs) (Vaswani et al., 2017) respectively,  

$$
\begin{array}{r l r}&{}&{\{\tilde{{\pmb w}}_{1}^{(i)},\ldots,\tilde{{\pmb w}}_{n}^{(i)}\}=\mathtt{M H-A T T}\big(\{{\pmb w}_{1}^{(i-1)},\ldots,{\pmb w}_{n}^{(i-1)}\}\big),}\\ &{}&{\{\tilde{e}_{1}^{(i)},\ldots,\tilde{e}_{m}^{(i)}\}=\mathtt{M H-A T T}\big(\{{\pmb e}_{1}^{(i-1)},\ldots,{\pmb e}_{m}^{(i-1)}\}\big).}\end{array}
$$  

Then, the $i$ -th aggregator adopts an information fusion layer for the mutual integration of the token and entity sequence, and computes the output embedding for each token and entity. For a token $w_{j}$ and its aligned entity $e_{k}=f(w_{j})$ , the information fusion process is as follows,  

$$
\begin{array}{r l}&{\quad\pmb{h}_{j}=\sigma(\tilde{\pmb{W}}_{t}^{(i)}\tilde{\pmb{w}}_{j}^{(i)}+\tilde{\pmb{W}}_{e}^{(i)}\tilde{\pmb{e}}_{k}^{(i)}+\tilde{\pmb{b}}^{(i)}),}\\ &{\pmb{w}_{j}^{(i)}=\sigma(\pmb{W}_{t}^{(i)}\pmb{h}_{j}+\pmb{b}_{t}^{(i)}),}\\ &{\quad\pmb{e}_{k}^{(i)}=\sigma(\pmb{W}_{e}^{(i)}\pmb{h}_{j}+\pmb{b}_{e}^{(i)}).}\end{array}
$$  

where $h_{j}$ is the inner hidden state integrating the information of both the token and the entity. $\sigma(\cdot)$ is the non-linear activation function, which usually is the GELU function (Hendrycks and Gimpel, 2016). For the tokens without corresponding entities, the information fusion layer computes the output embeddings without integration as follows,  

$$
\begin{array}{r}{\pmb{h}_{j}=\sigma(\tilde{\pmb{W}}_{t}^{(i)}\tilde{\pmb{w}}_{j}^{(i)}+\tilde{\pmb{b}}^{(i)}),}\\ {\pmb{w}_{j}^{(i)}=\sigma(\pmb{W}_{t}^{(i)}\pmb{h}_{j}+\pmb{b}_{t}^{(i)}).\qquad}\end{array}
$$  

For simplicity, the $i$ -th aggregator operation is denoted as follows,  

$$
\begin{array}{r l r}&{}&{\{\pmb{w}_{1}^{(i)},\ldots,\pmb{\omega}_{n}^{(i)}\},\{\pmb{e}_{1}^{(i)},\ldots,\pmb{e}_{m}^{(i)}\}=\mathbb{A}\mathbb{g}\mathbb{g}\mathbb{r}\mathrm{e}\mathbb{g}\mathrm{a}\mathrm{t}\mathrm{or}(}\\ &{}&{\{\pmb{w}_{1}^{(i-1)},\ldots,\pmb{w}_{n}^{(i-1)}\},\{\pmb{e}_{1}^{(i-1)},\ldots,\pmb{e}_{m}^{(i-1)}\}).}\end{array}
$$  

The output embeddings of both tokens and entities computed by the top aggregator will be used as the final output embeddings of the knowledgeable encoder K-Encoder.  

# 3.4 Pre-training for Injecting Knowledge  

In order to inject knowledge into language representation by informative entities, we propose a new pre-training task for ERNIE, which randomly masks some token-entity alignments and then requires the system to predict all corresponding entities based on aligned tokens. As our task is similar to training a denoising auto-encoder (Vincent et al., 2008), we refer to this procedure as a denoising entity auto-encoder (dEA). Considering that the size of $\mathcal{E}$ is quite large for the softmax layer, we thus only require the system to predict entities based on the given entity sequence instead of all entities in KGs. Given the token sequence $\{w_{1},\ldots,w_{n}\}$ and its corresponding entity sequence $\{e_{1},\ldots,e_{m}\}$ , we define the aligned entity distribution for the token $w_{i}$ as follows,  

$$
p(e_{j}|w_{i})=\frac{\exp(\mathtt{l i n e a r}({\pmb w}_{i}^{o})\cdot{\pmb e}_{j})}{\sum_{k=1}^{m}\exp(\mathtt{l i n e a r}({\pmb w}_{i}^{o})\cdot{\pmb e}_{k})},
$$  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/690b9bf1dee9456b866120e1da675407/32459b490a615fac89df659078ef85c6a178e98e4ca41784c5caa47037ed7206.jpg)  
Figure 3: Modifying the input sequence for the specific tasks. To align tokens among different types of input, we use dotted rectangles as placeholder. The colorful rectangles present the specific mark tokens.  

where $\mathtt{l i n e a r(\cdot)}$ is a linear layer. Eq. 7 will be used to compute the cross-entropy loss function for dEA.  

Considering that there are some errors in tokenentity alignments, we perform the following operations for dEA: (1) In $5\%$ of the time, for a given token-entity alignment, we replace the entity with another random entity, which aims to train our model to correct the errors that the token is aligned with a wrong entity; (2) In $15\%$ of the time, we mask token-entity alignments, which aims to train our model to correct the errors that the entity alignment system does not extract all existing alignments; (3) In the rest of the time, we keep tokenentity alignments unchanged, which aims to encourage our model to integrate the entity information into token representations for better language understanding.  

Similar to BERT, ERNIE also adopts the masked language model (MLM) and the next sentence prediction (NSP) as pre-training tasks to enable ERNIE to capture lexical and syntactic information from tokens in text. More details of these pre-training tasks can be found from Devlin et al. (2019). The overall pre-training loss is the sum of the dEA, MLM and NSP loss.  

# 3.5 Fine-tuning for Specific Tasks  

As shown in Figure 3, for various common NLP tasks, ERNIE can adopt the fine-tuning procedure similar to BERT. We can take the final output embedding of the first token, which corresponds to the special [CLS] token, as the representation of the input sequence for specific tasks. For some knowledge-driven tasks (e.g., relation classification and entity typing), we design special finetuning procedure:  

For relation classification, the task requires systems to classify relation labels of given entity pairs based on context. The most straightforward way to fine-tune ERNIE for relation classification is to apply the pooling layer to the final output embeddings of the given entity mentions, and represent the given entity pair with the concatenation of their mention embeddings for classification. In this paper, we design another method, which modifies the input token sequence by adding two mark tokens to highlight entity mentions. These extra mark tokens play a similar role like position embeddings in the conventional relation classification models (Zeng et al., 2015). Then, we also take the [CLS] token embedding for classification. Note that we design different tokens [HD] and [TL] for head entities and tail entities respectively.  

The specific fine-tuning procedure for entity typing is a simplified version of relation classification. As previous typing models make full use of both context embeddings and entity mention embeddings (Shimaoka et al., 2016; Yaghoobzadeh and Schu¨tze, 2017; Xin et al., 2018), we argue that the modified input sequence with the mention mark token [ENT] can guide ERNIE to combine both context information and entity mention information attentively.  

# 4 Experiments  

In this section, we present the details of pretraining ERNIE and the fine-tuning results on five NLP datasets, which contain both knowledgedriven tasks and the common NLP tasks.  

# 4.1 Pre-training Dataset  

The pre-training procedure primarily acts in accordance with the existing literature on pre-training language models. For the large cost of training ERNIE from scratch, we adopt the parameters of BERT released by Google3 to initialize the Transformer blocks for encoding tokens. Since pretraining is a multi-task procedure consisting of NSP, MLM, and dEA, we use English Wikipedia as our pre-training corpus and align text to Wikidata. After converting the corpus into the formatted data for pre-training, the annotated input has nearly 4, 500M subwords and 140M entities, and discards the sentences having less than 3 entities.  

Before pre-training ERNIE, we adopt the knowledge embeddings trained on Wikidata4 by TransE as the input embeddings for entities. To be specific, we sample part of Wikidata which contains 5, 040, 986 entities and 24, 267, 796 fact triples. The entity embeddings are fixed during training and the parameters of the entity encoding modules are all initialized randomly.  

# 4.2 Parameter Settings and Training Details  

In this work, we denote the hidden dimension of token embeddings and entity embeddings as $H_{w}$ , $H_{e}$ respectively, and the number of self-attention heads as $A_{w}$ , $A_{e}$ respectively. In detail, we have the following model size: $N=6,M=6,H_{w}=$ $768,H_{e}=100,A_{w}=12,A_{e}=4$ . The total parameters are about 114M.  

The total amount of parameters of BERTBASE is about 110M, which means the knowledgeable module of ERNIE is much smaller than the language module and has little impact on the run-time performance. And, we only pre-train ERNIE on the annotated corpus for one epoch. To accelerate the training process, we reduce the max sequence length from 512 to 256 as the computation of selfattention is a quadratic function of the length. To keep the number of tokens in a batch as same as BERT, we double the batch size to 512. Except for setting the learning rate as $5e^{-5}$ , we largely follow the pre-training hyper-parameters used in BERT. For fine-tuning, most hyper-parameters are the same as pre-training, except batch size, learning rate, and number of training epochs. We find the following ranges of possible values work well on the training datasets with gold annotations, i.e., batch size: 32, learning rate (Adam): $5e^{-5}$ , $3e^{-5}$ , $2e^{-5}$ , number of epochs ranging from 3 to 10.  

We also evaluate ERNIE on the distantly supervised dataset, i.e., FIGER (Ling et al., 2015). As the powerful expression ability of deeply stacked Transformer blocks, we found small batch size would lead the model to overfit the training data. Hence, we use a larger batch size and less training epochs to avoid overfitting, and keep the range of learning rate unchanged, i.e., batch size: 2048, number of epochs: 2, 3.  

Table 1: The statistics of the entity typing datasets FIGER and Open Entity.   


| | | | | |
| :--- | :--- | :--- | :--- | :--- |
| Dataset | Train | Develop | Test | Type |
| FIGER | 2,000,000 | 10.000 | 563 | 113 |
| Open Entity | 2.000 | 2.000 | 2.000 | 6 |
  

Table 2: Results of various models on FIGER $(\%)$ .   


| | | | |
| :--- | :--- | :--- | :--- |
| Model | Acc. | Macro | Micro |
| NFGEC (Attentive) | 54.53 | 74.76 | 71.58 |
| NFGEC (LSTM) | 55.60 | 75.15 | 71.73 |
| BERT | 52.04 | 75.16 | 71.63 |
| ERNIE | 57.19 | 76.51 | 73.39 |
  

As most datasets do not have entity annotations, we use TAGME (Ferragina and Scaiella, 2010) to extract the entity mentions in the sentences and link them to their corresponding entities in KGs.  

# 4.3 Entity Typing  

Given an entity mention and its context, entity typing requires systems to label the entity mention with its respective semantic types. To evaluate performance on this task, we fine-tune ERNIE on two well-established datasets FIGER (Ling et al., 2015) and Open Entity (Choi et al., 2018). The training set of FIGER is labeled with distant supervision, and its test set is annotated by human. Open Entity is a completely manually-annotated dataset. The statistics of these two datasets are shown in Table 1. We compare our model with the following baseline models for entity typing:  

NFGEC. NFGEC is a hybrid model proposed by Shimaoka et al. (2016). NFGEC combines the representations of entity mention, context and extra hand-craft features as input, and is the stateof-the-art model on FIGER. As this paper focuses on comparing the general language representation abilities of various neural models, we thus do not use the hand-craft features in this work.  

UFET. For Open Entity, we add a new hybrid model UFET (Choi et al., 2018) for comparison. UFET is proposed with the Open Entity dataset, which uses a Bi-LSTM for context representation instead of two Bi-LSTMs separated by entity mentions in NFGEC.  

Besides NFGEC and UFET, we also report the result of fine-tuning BERT with the same input format introduced in Section 3.5 for fair comparison. Following the same evaluation criteria used in the previous work, we compare NFGEC, BERT, ERNIE on FIGER, and adopt strict accuracy, loose macro, loose micro scores for evaluation. We compare NFGEC, BERT, UFET, ERNIE on Open Entity, and adopt precision, recall, microF1 scores for evaluation.  

Table 3: Results of various models on Open Entity $(\%)$ .   


| | | | |
| :--- | :--- | :--- | :--- |
| Model | P | R | F1 |
| NFGEC (LSTM) | 68.80 | 53.30 | 60.10 |
| UFET | 77.40 | 60.60 | 68.00 |
| BERT | 76.37 | 70.96 | 73.56 |
| ERNIE | 78.42 | 72.90 | 75.56 |
  

Table 4: The statistics of the relation classification datasets FewRel and TACRED.   


| | | | | |
| :--- | :--- | :--- | :--- | :--- |
| Dataset | Train | Develop | Test | Relation |
| FewRel | 8.000 | 16.000 | 16.000 | 80 |
| TACRED | 68,124 | 22.631 | 15,509 | 42 |
  

The results on FIGER are shown in Table 2. From the results, we observe that: (1) BERT achieves comparable results with NFGEC on the macro and micro metrics. However, BERT has lower accuracy than the best NFGEC model. As strict accuracy is the ratio of instances whose predictions are identical to human annotations, it illustrates some wrong labels from distant supervision are learned by BERT due to its powerful fitting ability. (2) Compared with BERT, ERNIE significantly improves the strict accuracy, indicating the external knowledge regularizes ERNIE to avoid fitting the noisy labels and accordingly benefits entity typing.  

The results on Open Entity are shown in Table 3. From the table, we observe that: (1) BERT and ERNIE achieve much higher recall scores than the previous entity typing models, which means pre-training language models make full use of both the unsupervised pre-training and manuallyannotated training data for better entity typing. (2) Compared to BERT, ERNIE improves the precision by $2\%$ and the recall by $2\%$ , which means the informative entities help ERNIE predict the labels more precisely.  

In summary, ERNIE effectively reduces the noisy label challenge in FIGER, which is a widely-used distantly supervised entity typing dataset, by injecting the information from KGs. Besides, ERNIE also outperforms the baselines on Open Entity which has gold annotations.  

Table 5: Results of various models on FewRel and TACRED $(\%)$ .   


| | | |
| :--- | :--- | :--- |
|  | FewRel | TACRED |
| Model | P | R | F1 | P | R | F1 |
| CNN | 69.51 | 69.64 | 69.35 | 70.30 | 54.20 | 61.20 |
| PA-LSTM |  |  |  | 65.70 | 64.50 | 65.10 |
| C-GCN |  |  |  | 69.90 | 63.30 | 66.40 |
| BERT | 85.05 | 85.11 | 84.89 | 67.23 | 64.81 | 66.00 |
| ERNIE | 88.49 | 88.44 | 88.32 | 69.97 | 66.08 | 67.97 |
  

# 4.4 Relation Classification  

Relation classification aims to determine the correct relation between two entities in a given sentence, which is an important knowledge-driven NLP task. To evaluate performance on this task, we fine-tune ERNIE on two well-established datasets FewRel (Han et al., 2018c) and TACRED (Zhang et al., 2017). The statistics of these two datasets are shown in Table 4. As the original experimental setting of FewRel is few-shot learning, we rearrange the FewRel dataset for the common relation classification setting. Specifically, we sample 100 instances from each class for the training set, and sample 200 instances for the development and test respectively. There are 80 classes in FewRel, and there are 42 classes (including a special relation “no relation”) in TACRED. We compare our model with the following baseline models for relation classification:  

CNN. With a convolution layer, a max-pooling layer, and a non-linear activation layer, CNN gets the output sentence embedding, and then feeds it into a relation classifier. To better capture the position of head and tail entities, position embeddings are introduced into CNN (Zeng et al., 2015; Lin et al., 2016; Wu et al., 2017; Han et al., 2018b).  

PA-LSTM. Zhang et al. (2017) propose PALSTM introducing a position-aware attention mechanism over an LSTM network, which evaluates the relative contribution of each word in the sequence for the final sentence representation.  

C-GCN. Zhang et al. (2018) adopt the graph convolution operations to model dependency trees for relation classification. To encode the word order and reduce the side effect of errors in dependency parsing, Contextualized GCN (C-GCN) firstly uses Bi-LSTM to generate contextualized representations as input for GCN models.  

In addition to these three baselines, we also finetune BERT with the same input format introduced in Section 3.5 for fair comparison.  

Table 6: Results of BERT and ERNIE on different tasks of GLUE $(\%)$ .   


| | | | | |
| :--- | :--- | :--- | :--- | :--- |
| Model | MNLI-(m/mm) 392k | QQP 363k | QNLI 104k | SST-2 67k |
| BERTBASE | 84.6/83.4 | 71.2 |  | 93.5 |
| ERNIE | 84.0/83.2 | 71.2 | 91.3 | 93.5 |
| Model | CoLA 8.5k | STS-B 5.7k | MRPC 3.5k | RTE 2.5k |
| BERTBASE | 52.1 | 85.8 | 88.9 | 66.4 |
| ERNIE | 52.3 | 83.2 | 88.2 | 68.8 |
  

As FewRel does not have any null instance where there is not any relation between entities, we adopt macro averaged metrics to present the model performances. Since FewRel is built by checking whether the sentences contain facts in Wikidata, we drop the related facts in KGs before pre-training for fair comparison. From Table 5, we have two observations: (1) As the training data does not have enough instances to train the CNN encoder from scratch, CNN just achieves an F1 score of $69.35\%$ . However, the pre-training models including BERT and ERNIE increase the F1 score by at least $15\%$ . (2) ERNIE achieves an absolute F1 increase of $3.4\%$ over BERT, which means fusing external knowledge is very effective.  

In TACRED, there are nearly $80\%$ null instances so that we follow the previous work (Zhang et al., 2017) to adopt micro averaged metrics to represent the model performances instead of the macro. The results of CNN, PA-LSTM, and C-GCN come from the paper by Zhang et al. (2018), which are the best results of CNN, RNN, and GCN respectively. From Table 5, we observe that: (1) The C-GCN model outperforms the strong BERT model by an F1 increase of $0.4\%$ , as C-GCN utilizes the dependency trees and the entity mask strategy. The entity mask strategy refers to replacing each subject (and object similarly) entity with a special NER token, which is similar to our proposed pre-training task dEA. (2) ERNIE achieves the best recall and F1 scores, and increases the F1 of BERT by nearly $2.0\%$ , which proves the effectiveness of the knowledgeable module for relation classification.  

In conclusion, we find that the pre-trained language models can provide more information for relation classification than the vanilla encoder CNN and RNN. And ERNIE outperforms BERT on both of the relation classification datasets, especially on the FewRel which has a much smaller training set. It demonstrates extra knowledge helps the model make full use of small training data, which is important for most NLP tasks as large-scale annotated data is unavailable.  

Table 7: Ablation study on FewRel $(\%)$ .   


| | | | |
| :--- | :--- | :--- | :--- |
| Model 一 | P | R | F1 |
| BERT | 一 85.05 | 85.11 | 84.89 |
| ERNIE | 88.49 | 88.44 | 88.32 |
| w/o entities | 85.89 | 85.89 | 85.79 |
| w/o dEA | 85.85 | 85.75 | 85.62 |
  

# 4.5 GLUE  

The General Language Understanding Evaluation (GLUE) benchmark (Wang et al., 2018) is a collection of diverse natural language understanding tasks (Warstadt et al., 2018; Socher et al., 2013; Dolan and Brockett, 2005; Agirre et al., 2007; Williams et al., 2018; Rajpurkar et al., 2016; Dagan et al., 2006; Levesque et al., 2011), which is the main benchmark used in Devlin et al. (2019). To explore whether our knowledgeable module degenerates the performance on common NLP tasks, we evaluate ERNIE on 8 datasets of GLUE and compare it with BERT.  

In Table 6, we report the results of our evaluation submissions and those of BERT from the leaderboard. We notice that ERNIE is consistent with BERTBASE on big datasets like MNLI, QQP, QNLI, and SST-2. The results become more unstable on small datasets, that is, ERNIE is better on CoLA and RTE, but worse on STS-B and MRPC.  

In short, ERNIE achieves comparable results with BERTBASE on GLUE. On the one hand, it means GLUE does not require external knowledge for language representation. On the other hand, it illustrates ERNIE does not lose the textual information after heterogeneous information fusion.  

# 4.6 Ablation Study  

In this subsection, we explore the effects of the informative entities and the knowledgeable pretraining task (dEA) for ERNIE using FewRel dataset. w/o entities and w/o dEA refer to finetuning ERNIE without entity sequence input and the pre-training task dEA respectively. As shown in Table 7, we have the following observations: (1) Without entity sequence input, dEA still injects knowledge information into language representation during pre-training, which increases the F1 score of BERT by $0.9\%$ . (2) Although the informative entities bring much knowledge information which intuitively benefits relation classification, ERNIE without dEA takes little advantage of this, leading to the F1 increase of $0.7\%$ .  

# 5 Conclusion  

In this paper, we propose ERNIE to incorporate knowledge information into language representation models. Accordingly, we propose the knowledgeable aggregator and the pre-training task dEA for better fusion of heterogeneous information from both text and KGs. The experimental results demonstrate that ERNIE has better abilities of both denoising distantly supervised data and fine-tuning on limited data than BERT. There are three important directions remain for future research: (1) inject knowledge into feature-based pre-training models such as ELMo (Peters et al., 2018); (2) introduce diverse structured knowledge into language representation models such as ConceptNet (Speer and Havasi, 2012) which is different from the world knowledge database Wikidata; (3) annotate more real-world corpora heuristically for building larger pre-training data. These directions may lead to more general and effective language understanding.  

# Acknowledgement  

This work is funded by the Natural Science Foundation of China (NSFC) and the German Research Foundation (DFG) in Project Crossmodal Learning, NSFC 61621136008 / DFG TRR-169, the National Natural Science Foundation of China (NSFC No. 61572273) and China Association for Science and Technology (2016QNRC001).  

# References  

Eneko Agirre, Llu’is M‘arquez, and Richard Wicentowski. 2007. Proceedings of the fourth international workshop on semantic evaluations (semeval-2007). In Proceedings of SemEval-2007.   
Antoine Bordes, Nicolas Usunier, Alberto Garcia-Duran, Jason Weston, and Oksana Yakhnenko. 2013. Translating embeddings for modeling multi-relational data. In Proceedings of NIPS, pages 2787–2795.   
Samuel R Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning. 2015. A large annotated corpus for learning natural language inference. In Proceedings of EMNLP, pages 632–642.   
Yixin Cao, Lei Hou, Juanzi Li, Zhiyuan Liu, Chengjiang Li, Xu Chen, and Tiansi Dong. 2018. Joint representation learning of cross-lingual words and entities via attentive distant supervision. In Proceedings of EMNLP, pages 227–237.   
Yixin Cao, Lifu Huang, Heng Ji, Xu Chen, and Juanzi Li. 2017. Bridge text and knowledge by learning multiprototype entity mention embedding. In Proceedings of ACL, pages 1623–1633.   
Qian Chen, Xiaodan Zhu, Zhen-Hua Ling, Diana Inkpen, and Si Wei. 2018. Neural natural language inference models enhanced with external knowledge. In Proceedings of ACL, pages 2406–2417.   
Eunsol Choi, Omer Levy, Yejin Choi, and Luke Zettlemoyer. 2018. Ultra-fine entity typing. In Proceedings of ACL, pages 87–96.   
Ronan Collobert and Jason Weston. 2008. A unified architecture for natural language processing: Deep neural networks with multitask learning. In Proceedings of ICML, pages 160–167.   
Ido Dagan, Oren Glickman, and Bernardo Magnini. 2006. The PASCAL recognising textual entailment challenge. In Proceedings of MLCW, pages 177–190.   
Andrew M Dai and Quoc V Le. 2015. Semi-supervised sequence learning. In Proceedings of NIPS, pages 3079– 3087.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT.   
William B Dolan and Chris Brockett. 2005. Automatically constructing a corpus of sentential paraphrases. In Proceedings of IWP.   
Paolo Ferragina and Ugo Scaiella. 2010. Tagme: on-the-fly annotation of short text fragments (by wikipedia entities). In Proceedings of CIKM, pages 1625–1628.   
Xu Han, Zhiyuan Liu, and Maosong Sun. 2016. Joint representation learning of text and knowledge for knowledge graph completion. arXiv preprint arXiv:1611.04125.   
Xu Han, Zhiyuan Liu, and Maosong Sun. 2018a. Neural knowledge acquisition via mutual attention between knowledge graph and text. In Proceedings of AAAI.   
Xu Han, Pengfei Yu, Zhiyuan Liu, Maosong Sun, and Peng Li. 2018b. Hierarchical relation extraction with coarse-tofine grained attention. In Proceedings of EMNLP, pages 2236–2245.   
Xu Han, Hao Zhu, Pengfei Yu, Ziyun Wang, Yuan Yao, Zhiyuan Liu, and Maosong Sun. 2018c. Fewrel: A largescale supervised few-shot relation classification dataset with state-of-the-art evaluation. In Proceedings of EMNLP, pages 4803–4809.   
Dan Hendrycks and Kevin Gimpel. 2016. Gaussian error linear units (gelus). arXiv preprint arXiv:1606.08415.   
Jeremy Howard and Sebastian Ruder. 2018. Universal language model fine-tuning for text classification. In Proceedings of ACL, pages 328–339.   
Hector J Levesque, Ernest Davis, and Leora Morgenstern. 2011. The Winograd schema challenge. In AAAI Spring Symposium: Logical Formalizations of Commonsense Reasoning, volume 46, page 47.   
Yankai Lin, Shiqi Shen, Zhiyuan Liu, Huanbo Luan, and Maosong Sun. 2016. Neural relation extraction with selective attention over instances. In Proceedings of ACL, volume 1, pages 2124–2133.   
Xiao Ling, Sameer Singh, and Daniel S Weld. 2015. Design challenges for entity linking. TACL, 3:315–328.   
Andrea Madotto, Chien-Sheng Wu, and Pascale Fung. 2018. Mem2seq: Effectively incorporating knowledge bases into end-to-end task-oriented dialog systems. In Proceedings of ACL, pages 1468–1478.   
Stephen Merity, Nitish Shirish Keskar, and Richard Socher. 2018. Regularizing and optimizing lstm language models. In Proceedings of ICLR.   
Todor Mihaylov and Anette Frank. 2018. Knowledgeable reader: Enhancing cloze-style reading comprehension with external commonsense knowledge. In Proceedings of ACL, pages 821–832.   
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013. Distributed representations of words and phrases and their compositionality. In Proceedings of NIPS, pages 3111–3119.   
Jeffrey Pennington, Richard Socher, and Christopher Manning. 2014. Glove: Global vectors for word representation. In Proceedings of EMNLP, pages 1532–1543.   
Matthew Peters, Waleed Ammar, Chandra Bhagavatula, and Russell Power. 2017. Semi-supervised sequence tagging with bidirectional language models. In Proceedings of ACL, pages 1756–1765.   
Matthew Peters, Mark Neumann, Mohit Iyyer, Matt Gardner, Christopher Clark, Kenton Lee, and Luke Zettlemoyer. 2018. Deep contextualized word representations. In Proceedings of NAACL-HLT, pages 2227–2237.   
Alec Radford, Karthik Narasimhan, Tim Salimans, and Ilya Sutskever. 2018. Improving language understanding by generative pre-training. In Proceedings of Technical report, OpenAI.   
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: $100{,}000{+}$ questions for machine comprehension of text. In Proceedings of EMNLP, pages 2383–2392.   
Erik F Tjong Kim Sang and Fien De Meulder. 2003. Introduction to the conll-2003 shared task: Languageindependent named entity recognition. In Proceedings of NAACL-HLT, pages 142–147.   
Sonse Shimaoka, Pontus Stenetorp, Kentaro Inui, and Sebastian Riedel. 2016. An attentive neural architecture for fine-grained entity type classification. In Proceedings of AKBC, pages 69–74.   
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of EMNLP, pages 1631–1642.   
Robert Speer and Catherine Havasi. 2012. Representing general relational knowledge in conceptnet 5. In Proceedings of LREC, pages 3679–3686.   
Yu Sun, Shuohuan Wang, Yukun Li, Shikun Feng, Xuyi Chen, Han Zhang, Xin Tian, Danxiang Zhu, Hao Tian, and Hua Wu. 2019. Ernie: Enhanced representation through knowledge integration.   
Kristina Toutanova, Danqi Chen, Patrick Pantel, Hoifung Poon, Pallavi Choudhury, and Michael Gamon. 2015. Representing text for joint embedding of text and knowledge bases. In Proceedings of EMNLP, pages 1499–1509.   
Joseph Turian, Lev Ratinov, and Yoshua Bengio. 2010. Word representations: a simple and general method for semisupervised learning. In Proceedings of ACL, pages 384– 394.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Proceedings of NIPS, pages 5998–6008.   
Pascal Vincent, Hugo Larochelle, Yoshua Bengio, and PierreAntoine Manzagol. 2008. Extracting and composing robust features with denoising autoencoders. In Proceedings of ICML, pages 1096–1103.   
Alex Wang, Amapreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman. 2018. Glue: A multitask benchmark and analysis platform for natural language understanding. In Proceedings of EMNLP, pages 353– 355.   
Zhen Wang, Jianwen Zhang, Jianlin Feng, and Zheng Chen. 2014. Knowledge graph and text jointly embedding. In Proceedings of EMNLP, pages 1591–1601.   
Alex Warstadt, Amanpreet Singh, and Samuel R. Bowman. 2018. Neural network acceptability judgments. arXiv preprint 1805.12471.   
Adina Williams, Nikita Nangia, and Samuel R. Bowman. 2018. A broad-coverage challenge corpus for sentence understanding through inference. In Proceedings of NAACLHLT, pages 1112–1122.   
Yi Wu, David Bamman, and Stuart Russell. 2017. Adversarial training for relation extraction. In Proceedings of EMNLP, pages 1778–1783.   
Ji Xin, Hao Zhu, Xu Han, Zhiyuan Liu, and Maosong Sun. 2018. Put it back: Entity typing with language model enhancement. In Proceedings of EMNLPs, pages 993–998.   
Yadollah Yaghoobzadeh and Hinrich Schu¨tze. 2017. Multilevel representations for fine-grained typing of knowledge base entities. In Proceedings of EACL, pages 578–589.   
Ikuya Yamada, Hiroyuki Shindo, Hideaki Takeda, and Yoshiyasu Takefuji. 2016. Joint learning of the embedding of words and entities for named entity disambiguation. In Proceedings of CoNLL, pages 250–259.   
Poorya Zaremoodi, Wray Buntine, and Gholamreza Haffari. 2018. Adaptive knowledge sharing in multi-task learning: Improving low-resource neural machine translation. In Proceedings of ACL, pages 656–661.   
Rowan Zellers, Yonatan Bisk, Roy Schwartz, and Yejin Choi. 2018. Swag: A large-scale adversarial dataset for grounded commonsense inference. In Proceedings of EMNLP, pages 93–104.   
Daojian Zeng, Kang Liu, Yubo Chen, and Jun Zhao. 2015. Distant supervision for relation extraction via piecewise convolutional neural networks. In Proceedings of EMNLP, pages 1753–1762.   
Yuhao Zhang, Peng Qi, and Christopher D Manning. 2018. Graph convolution over pruned dependency trees improves relation extraction. In Proceedings of EMNLP, pages 2205–2215.   
Yuhao Zhang, Victor Zhong, Danqi Chen, Gabor Angeli, and Christopher D Manning. 2017. Position-aware attention and supervised data improve slot filling. In Proceedings of EMNLP, pages 35–45.   
Wanjun Zhong, Duyu Tang, Nan Duan, Ming Zhou, Jiahai Wang, and Jian Yin. 2018. Improving question answering by commonsense-based pre-training. arXiv preprint arXiv:1809.03568.  