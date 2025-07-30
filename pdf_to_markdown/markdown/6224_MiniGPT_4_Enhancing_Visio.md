# MINIGPT-4: ENHANCING VISION-LANGUAGE UNDERSTANDING WITH ADVANCED LARGE LANGUAGE MODELS  

Deyao $\mathbf{Z}\mathbf{h}\mathbf{u}^{*}$ , Jun Chen∗, Xiaoqian Shen, Xiang Li, Mohamed Elhoseiny King Abdullah University of Science and Technology {deyao.zhu,jun.chen,xiaoqian.shen, xiang.li.1,mohamed.elhoseiny @kaust.edu.sa  

# ABSTRACT  

The recent GPT-4 has demonstrated extraordinary multi-modal abilities, such as directly generating websites from handwritten text and identifying humorous elements within images. These features are rarely observed in previous visionlanguage models. However, the technical details behind GPT-4 continue to remain undisclosed. We believe that the enhanced multi-modal generation capabilities of GPT-4 stem from the utilization of sophisticated large language models (LLM). To examine this phenomenon, we present MiniGPT-4, which aligns a frozen visual encoder with a frozen advanced LLM, Vicuna, using one projection layer. Our work, for the first time, uncovers that properly aligning the visual features with an advanced large language model can possess numerous advanced multi-modal abilities demonstrated by GPT-4, such as detailed image description generation and website creation from hand-drawn drafts. Furthermore, we also observe other emerging capabilities in MiniGPT-4, including writing stories and poems inspired by given images, teaching users how to cook based on food photos, and so on. In our experiment, we found that the model trained on short image caption pairs could produce unnatural language outputs (e.g., repetition and fragmentation). To address this problem, we curate a detailed image description dataset in the second stage to finetune the model, which consequently improves the model’s generation reliability and overall usability. Our code, pre-trained model, and collected dataset are available at https://minigpt-4.github.io/.  

# 1 INTRODUCTION  

In recent years, large language models (LLMs) have experienced rapid advancements (Ouyang et al., 2022; OpenAI, 2022; Brown et al., 2020; Scao et al., 2022a; Touvron et al., 2023; Chowdhery et al., 2022; Hoffmann et al., 2022). With exceptional language understanding capabilities, these models can perform a variety of intricate linguistic tasks in a zero-shot manner. Notably, GPT-4, a large-scale multimodal model, has been recently introduced and demonstrated several impressive capabilities of vision-language understanding and generation (OpenAI, 2023). For example, GPT-4 can produce detailed and accurate image descriptions, explain unusual visual phenomena, and even construct websites based on handwritten text instructions.  

Although GPT-4 has exhibited remarkable vision language capabilities, the methods behind its exceptional abilities are still a mystery (OpenAI, 2023). We believe that these impressive skills may stem from the utilization of a more advanced large language model (LLM). LLMs have demonstrated various emergent abilities, as evidenced in GPT-3’s few-shot prompting setup (Brown et al., 2020) and the findings of Wei et al. (2022) (Wei et al., 2022). Such emergent properties are hard to find in smaller-scale models. It is conjectured that these emergent abilities are also applicable to multi-modal models, which could be the foundation of GPT-4’s impressive visual description capabilities.  

To substantiate our hypothesis, we present a novel vision-language model named MiniGPT-4. It utilizes an advanced large language model (LLM), Vicuna (Chiang et al., 2023), which is built upon  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/3e3f23265491388537ccd5c890e7ea973f8021605af3c3386c50e6611c537460.jpg)  
Figure 1: The architecture of MiniGPT-4. It consists of a vision encoder with a pretrained ViT and Q-Former, a single linear projection layer, and an advanced Vicuna large language model. MiniGPT-4 only requires training the linear projection layer to align the visual features with the Vicuna.  

LLaMA (Touvron et al., 2023) and reported to achieve $90\%$ of ChatGPT’s quality as per GPT-4’s evaluation, as the language decoder. In terms of visual perception, we employ the same pretrained vision components of BLIP-2 (Li et al., 2023c) that consists of a ViT-G/14 from EVA-CLIP (Fang et al., 2022) and a Q-Former network. MiniGPT-4 adds a single projection layer to align the encoded visual features with the Vicuna language model and freezes all the other vision and language components. MiniGPT-4 is initially trained for $20\mathrm{k}$ steps using a batch size of 256 on 4 A100 GPUs, leveraging a combined image captioning dataset that includes images from LAION (Schuhmann et al., 2021), Conceptual Captions (Changpinyo et al., 2021; Sharma et al., 2018), and SBU (Ordonez et al., 2011) to align visual features with the Vicuna language model. Nevertheless, merely aligning visual features with the language model (LLM) is inadequate to ensure robust visual conversation capabilities, resembling that of a chatbot. The presence of underlying noise in raw image-text pairs can lead to subpar language outputs. Therefore, we collect another 3,500 detailed image description pairs to further fine-tune the model with a designed conversational template in order to improve the naturalness of the generated language and its usability.  

In our experiments, we discovered that MiniGPT-4 possesses numerous capabilities similar to those demonstrated by GPT-4. For instance, MiniGPT-4 can generate intricate image descriptions, create websites based on handwritten text instructions, and explain unusual visual phenomena. Furthermore, our findings revealed that MiniGPT-4 also has a variety of other intriguing abilities not showcased in the GPT-4 demonstrations. For example, MiniGPT-4 can directly generate detailed cooking recipes from food photos, write stories or poems inspired by images, write advertisements for products in images, identify problems shown in photos and provide corresponding solutions, and retrieve rich facts about people, movies, or art directly from images, among other capabilities. These abilities are absent in previous vision-language models like Kosmos-1 (Huang et al., 2023) and BLIP-2 (Li et al., 2023c) that use less powerful language models. This further validates that integrating visual features with an advanced language model is one of the keys to enhancing vision-language models. We present a summary of our key findings:  

• Our research reveals with compelling evidence that by aligning visual features with advanced large language models like Vicuna, MiniGPT-4 can achieve advanced vision-language capabilities comparable to those exhibited in the GPT-4 demonstrations. • Our findings suggest that training merely one projection layer can effectively align a pretrained vision encoder with the large language model. Our MiniGPT-4 only requires training approximately 10 hours on 4 A100 GPUs. • We discovered that simply aligning visual features with large language models using short image caption pairs is not sufficient for developing a well-performing model and leads to unnatural language generation. Further finetuning with a small but detailed image description pairs can address this limitation and significantly improves its usability.  

# 2 RELATED WORKS  

Large language models have experienced tremendous success in recent years due to the scaling up of training data and an increase in the number of parameters. Early models, such as BERT (Devlin et al., 2018), GPT-2 (Radford et al., 2019), and T5 (Raffel et al., 2020), laid the foundation for this progress. Subsequently, GPT-3 (Brown et al., 2020), with a massive scale of 175 billion parameters, was introduced, demonstrating significant breakthroughs across numerous language benchmarks. This development inspired the creation of various other large language models, including MegatronTuring NLG (Smith et al., 2022), Chinchilla (Hoffmann et al., 2022), PaLM (Chowdhery et al., 2022), OPT (Zhang et al., 2022), BLOOM (Scao et al., 2022b), and LLaMA (Touvron et al., 2023), among others. Wei et al. (Wei et al., 2022) further discovered several emergent abilities, which appear exclusively in large models. The emergence of these abilities underscores the importance of scaling up in the development of large language models. Moreover, by aligning the pre-trained large language model GPT-3 with human intent, instructions and human feedback, InstructGPT (Ouyang et al., 2022) and ChatGPT (OpenAI, 2022) enable conversational interactions with humans and can answer a wide range of diverse and complex questions. More recently, several open-sourced models, such as Alpaca (Taori et al., 2023) and Vicuna (Chiang et al., 2023), have been developed based on LLaMA (Touvron et al., 2023) and also exhibit similar performance.  

Leveraging Pre-trained LLMs in Vision-Language Tasks. The use of autoregressive language models as decoders in vision-language tasks has become increasingly popular (Chen et al., 2022; Huang et al., 2023; Yang et al., 2022; Tiong et al., 2022; Alayrac et al., 2022; Li et al., 2023c; 2022; Driess et al., 2023), facilitating cross-modal knowledge transfer. Notable examples include VisualGPT (Chen et al., 2022) and Frozen (Tsimpoukelli et al., 2021), which integrate pre-trained language models for decoding. Flamingo (Alayrac et al., 2022) aligns a vision encoder and language model, excelling in few-shot learning. BLIP-2 (Li et al., 2023c) combines a Flan-T5 (Chung et al., 2022) with Q-Former for efficient alignment. PaLM-E (Driess et al., 2023), with its 562 billion parameters, merges real-world sensor data into an LLM, linking perceptions and languages. GPT4 (OpenAI, 2023) further advances visual understanding and reasoning after extensive image-text data pre-training. Contemporary works such as LLaVa (Liu et al., 2023a), InstructBLIP (Dai et al., 2023), mPLUG-Owl (Ye et al., 2023), Multimodal-GPT (Gong et al., 2023), and Otter (Li et al., 2023b) align language models with visual encoders using multimodal instruction following datasets. Compared to these methods, MiniGPT-4 demonstrates both data efficiency and parameter efficiency, where only a single linear layer is learnable and the training time is just 10 hours with 4 A100 GPUs. In addition, LLaVa (Liu et al., 2023a), MIMIC-IT (Li et al., 2023a), and M3IT (Li et al., 2023e) collect visual instruction datasets by either generating from ChatGPT or from the human annotators. Such methods require access to image datasets with ground truth image information in text format. Compared to these methods, the visual instruction dataset used in MiniGPT-4 is generated by MiniGPT-4 itself, making data collection model-informed.  

LLMs like ChatGPT can enhance vision-language tasks by collaborating with specialized models. Visual ChatGPT (Wu et al., 2023) and MM-REACT (Yang\* et al., 2023) show ChatGPT integrating various visual models for complex challenges. ChatCaptioner (Zhu et al., 2023) uses ChatGPT to generate questions for BLIP-2, summarizing image content through dialogue. Video ChatCaptioner (Chen et al., 2023) extends this to video understanding. ViperGPT (Surı´s et al., 2023) combines an LLM with vision models for visual queries. MiniGPT-4 aligns visual information with the language model directly, avoiding external models.  

# 3 METHOD  

MiniGPT-4 aims to align visual information from a pretrained vision encoder with an advanced large language model (LLM). Specifically, we utilize the Vicuna (Chiang et al., 2023) as our language decoder, which is constructed upon LLaMA (Touvron et al., 2023) and can perform a wide range of complex linguistic tasks. For visual perception, we employ the same visual encoder as used in BLIP-2 (Li et al., 2023c), a ViT backbone (Fang et al., 2022) coupled with their pre-trained Q-Former. Both language and vision models are open-sourced. We target to bridge the gap between the visual encoder and LLM using a linear projection layer, with an overview of our model displayed in Fig.1.  

We use a two-stage training method. First, we pretrain it on a vast set of image-text pairs to learn vision-language skills. Then, we finetune the model using a smaller, high-quality image-text dataset and a conversational template, improving generation reliability and usability.  

# 3.1 FIRST PRETRAINING STAGE  

In the initial pretraining stage, our model uses a large collection of aligned image-text pairs to gain vision-language knowledge. The output from the projection layer serves as a soft prompt for the LLM, leading it to generate corresponding ground-truth texts. Throughout pretraining, the pretrained vision encoder and LLM remain frozen, with only the linear projection layer undergoing training. We utilize datasets from Conceptual Caption (Changpinyo et al., 2021; Sharma et al., 2018), SBU (Ordonez et al., 2011), and LAION (Schuhmann et al., 2021) for this process. The model undergoes 20,000 training steps with a batch size of 256, covering about 5 million image-text pairs, and completes in around 10 hours on 4 A100 (80GB) GPUs.  

Issues of the first pretraining stage After its initial pretraining, MiniGPT-4 shows the ability to hold a wealth of knowledge and respond reasonably to human queries. Yet, it sometimes generates incoherent outputs like repetitive words or sentences, fragmented phrases, or irrelevant content, which impairs its capacity for fluent visual conversation with humans.  

GPT-3, despite its extensive language dataset pretraining, faced challenges in aligning outputs with user intentions. Instruction finetuning and reinforcement learning from human feedback transformed it into GPT-3.5 (Ouyang et al., 2022; OpenAI, 2022), enhancing its ability to produce human-friendly outputs. This mirrors MiniGPT-4’s state after pretraining, explaining its current difficulties in generating fluent, natural human language outputs.  

# 3.2 CURATING A HIGH-QUALITY ALIGNMENT DATASET FOR VISION-LANGUAGE DOMAIN.  

To achieve greater naturalness in the generated language and enhance the model’s usability, a secondstage alignment process is essential. While in the realm of NLP, instruction fine-tuning datasets (Taori et al., 2023) and conversations (sha, 2023) are easily accessible, no equivalent datasets exist for the vision-language domain at the time of this project. To address this deficiency, we curated a detailed image description dataset, specifically tailored for vision-language alignment purposes. This dataset is subsequently utilized to fine-tune our MiniGPT-4 during the second-stage alignment process.  

Initial aligned image-text generation In the initial phase, we employ the model derived from the first pretraining stage to generate comprehensive descriptions of input images. To enable our model to produce more detailed image descriptions, we designed a prompt that adheres to the conversational format of the Vicuna (Chiang et al., 2023) language model, as shown below. In this prompt, $<$ <ImageFeature $>$ represents the visual features produced by the linear projection layer.  

###Human: $<I m g><$ <ImageFeature></Img $>$ Describe this image in detail. Give as many details as possible. Say everything you see. ###Assistant:  

To identify incomplete sentences, we examine whether the generated sentence exceeds 80 tokens. If it does not, we incorporate an additional prompt, ###Human: Continue ###Assistant: , prompting our MiniGPT-4 to extend the generation process. By concatenating the outputs from both steps, we can create a more comprehensive image description. This approach enables us to generate image-text pairs with detailed and informative image descriptions. We randomly select 5,000 images from the Conceptual Caption dataset (Changpinyo et al., 2021; Sharma et al., 2018) and use the pretrained model to generate corresponding language descriptions for each image.  

Data post-processing The generated image descriptions are marred by issues like repetitive words or sentences, fragmented sentences, and irrelevant content. To rectify these, we use ChatGPT with a specific prompt to improve the descriptions.  

Fix the error in the given paragraph. Remove any repeating sentences, meaningless characters, not English sentences, and so on. Remove unnecessary repetition. Rewrite any incomplete sentences. Return directly the results without explanation. Return directly the input paragraph if it is already correct without explanation.  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/c41f46a2c234ef8ccc740d8b757fa3ff20ea5d4ecf419f9964cac4eaf8a15533.jpg)  
Figure 2: Detailed description  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/551d9f80512935dc76d20c073f45e7f460073804ab421ab69e491df97acf5644.jpg)  
Figure 3: Advertisement promotion  

Upon completing the post-processing stage, we manually verify the correctness of each image description to guarantee its high quality. Specifically, we first identified several frequently shown errors (“I’m sorry I made a mistake...”, or “I apologize for that ...”) and then hard-coded rules to automatically filter them out. We also manually refine the generated captions by eliminating redundant words or sentences that ChatGPT fails to detect. Finally, only approximately 3,500 out of 5,000 image-text pairs satisfy our requirement, and these pairs are subsequently utilized for the second-stage alignment process.  

# 3.3 SECOND-STAGE FINETUNING  

During the second stage, we finetune our pretrained model with the curated high-quality image-text pairs. During the finetuning, we use the predefined prompts in the following template:  

###Human: $<I m g><$ ImageFeature></Img><Instruction $>$ ###Assistant:  

In this prompt, <Instruction $>$ represents a randomly sampled instruction from our predefined instruction set containing variant forms of instructions such as “Describe this image in detail” or “Could you describe the contents of this image for me”. It is important to note that we do not calculate the regression loss for this specific text-image prompt.  

As a result, MiniGPT-4 is now capable of producing more natural and reliable language outputs. Furthermore, we observed that this fine-tuning process is remarkably efficient, only requiring a mere 400 training steps with a batch size of 12, which takes around 7 minutes with a single A100 GPU.  

# 4 EXPERIMENTS  

In the experiment, we aim to showcase the diverse and emergent capabilities of our MiniGPT-4 model through various qualitative examples. These abilities include generating detailed image descriptions, identifying amusing aspects within memes, providing food recipes from photos, writing poems for images, etc. Additionally, we present quantitative results on the task of image captioning.  

# 4.1 UNCOVERING EMERGENT ABILITIES WITH MINIGPT-4 THROUGH QUALITATIVE EXAMPLES  

MiniGPT-4 demonstrates many advanced abilities compared to traditional vision-language models. For example, it can describe images in detail and interpret the humorous aspects of a given meme. Here, we qualitatively compared our model to one of the leading vision-language models, BLIP-2 (Li et al., 2023c), with eight distinct examples, each highlighting a different ability.  

Fig.2 shows MiniGPT-4’s ability to identify multiple elements in an image, like busy streets, clock towers, shops, streetlights, and restaurants, whereas BLIP-2 only notes streets, people, and motorcycles. In another instance, Fig.4a, MiniGPT-4 aptly explains the humor in a meme by relating the dog’s expression to common Monday blues, a concept BLIP-2 misses, merely describing the image without grasping its humorous aspect.  

MiniGPT-4 has many other capabilities, including creating ads from images (Fig.3), extracting facts from movie photos (Fig.8), generating recipes from food images (Fig.11), diagnosing and suggesting treatments for plant diseases (Fig.12), designing websites from hand-written drafts (Fig.4b), and writing poems inspired by images (Fig.10). These abilities surpass those of traditional models like BLIP-2, which uses Flan-T5 XXL (Chung et al., 2022) as a language model. This difference highlights the importance of aligning visual features with an advanced LLM like Vicuna (Chiang et al., 2023) to unlock advanced vision-language capabilities.  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/1181561484ec7e7f87c94b7c5649d083c9b6e36a022a6541958ce03107769b51.jpg)  
Figure 4: Model generations from BLIP-2, BLIP-2 finetuned our second stage data (BLIP-2 FT), MiniGPT-4 finetuned with Local Narrative data in the second stage (MiniGPT-4 LocNa), MiniGPT-4 model without Q-Former (MiniGPT-4 No Q-Former), and MQinuiaGlPitTa-t4i.  

Table 1: Quantitative results on advanced vision-language tasks. MiniGPT-4 shows strong performance and successfully responses to $65\%$ of the requests.   


| | | | | | |
| :--- | :--- | :--- | :--- | :--- | :--- |
|  | Meme | Recipes | Ads | Poem | Avg. |
| BLIP-2 | 0/25 | 4/25 | 1/25 | 0/25 | 5/100 |
| MiniGPT-4 | 8/25 | 18/25 | 19/25 | 20/25 | 65/100 |
  

# 4.2 QUANTITATIVE ANALYSIS  

Advanced Abilities Our evaluation dataset for vision-language tasks included 100 images divided across four tasks: meme interpretation, recipe generation, advertisement creation, and poem composition, each with 25 images. Human evaluators assessed the model’s responses. We compared MiniGPT-4 with BLIP-2, as detailed in Tab.1. MiniGPT-4 outperformed BLIP-2 (Li et al., 2023c), especially in recipe, advertisement, and poem tasks, successfully handling $80\%$ of these. It also interpreted humor in memes correctly in 8 out of 25 cases, a challenging aspect for BLIP-2.  

Image Captioning We evaluate the performance of MiniGPT-4 on the COCO caption benchmark and compare it with BLIP-2 (Li et al., 2023c). Our model’s generated captions typically contain rich visual details. As such, conventional similarity-based image-caption evaluation metrics struggle to provide an accurate evaluation. To evaluate, we check how many of COCO’s 5 ground truth captions per image are covered by MiniGPT-4’s captions, using GPT-4 turbo. Evaluation details can be found in Appx.A.3. Results in Tab.2 show MiniGPT-4 averaged 2.22 ground truth captions, better than BLIP-2’s 1.96, proving its captions to be more informative. Additional evaluations on traditional VQA tasks are detailed in Appx.A.2.  

Video Understanding Here, we evaluate MiniGPT-4 for video understanding. We finetuned MiniGPT4 on $1.2\mathrm{k}$ videos from the VideoInstruct100K (Maaz et al., 2023), using 50 frames and subtitles per video. Experimental results on the video-based generative performance benchmark (Maaz et al., 2023) in Tab. 4 show that MiniGPT-4 outperformed the strongest baseline Video-ChatGPT (Maaz et al., 2023) in correctness, detail, context, and time comprehension, while also showing strong consistency, demonstrating MiniGPT-4’s potential in processing videos.  

Other Benchmarks MinGPT-4 has been densely evaluated and compared with contemporary baselines like LLaVa (Liu et al., 2023a) and mPlug-Owl (Ye et al., 2023) by many popular benchmarks like MMBench (Liu et al., 2023b) quantitatively. A detailed discussion of MiniGPT-4’s performance on these benchmarks can be found in Appx.A.5.  

# 4.3 ANALYSIS ON THE SECOND-STAGE FINETUNING  

Effectiveness of the second-stage finetuning Utilizing MiniGPT-4 solely after the first pretraining stage leads to issues like repetitive or fragmented sentences. These are largely resolved after the second-stage finetuning, as shown in Fig.5, where MiniGPT-4 evolves from generating incomplete to fluent captions. This section assesses the second-stage finetuning’s importance and effectiveness.  

To measure its impact, we sampled 100 images from the COCO test set for the detailed description and poem writing tasks, using the prompts “Describe the image in detail.” and “Can you write a beautiful poem about this image?”. Both pre- and post-second-stage finetuned models attempted these tasks. Results in Tab.3 show a significant drop in failures post-finetuning, with less than two failures in 100 images for each task, indicating a notable improvement in output quality. Fig.5 provides qualitative examples of this enhancement.  

Table 2: COCO caption evaluation. We use GPT-4 turbo to count the number of ground truth captions the model output can cover. MiniGPT-4(GPT-4v) denotes a variant trained using GPT-4V generated data in the second stage.   


| | | | |
| :--- | :--- | :--- | :--- |
|  | BLIP-2 | MiniGPT-4 | MiniGPT-4 (GPT-4v) |
| #GT Cover | 1.96 | 2.22 | 2.26 |
  

Table 3: Failure rates of detailed caption and poem generation tasks before and after second-stage finetuning. The finetuning stage significantly reduces generation failures.   


| | | |
| :--- | :--- | :--- |
| Failure rate | Detailed caption | Poem |
| Before stage-2 | 35% | 32% |
| After stage-2 | 2% | 1% |
  

Table 4: Video understanding on the video-based generative performance benchmark.   


| | | | | | |
| :--- | :--- | :--- | :--- | :--- | :--- |
|  | Correctness | Detail | Contextual | Temporal | Consistency |
| Video Chat (Li et al., 2023d) | 2.23 | 2.50 | 2.53 | 1.94 | 2.24 |
| Llama Adapter (Zhang et al., 2023b) | 2.03 | 2.32 | 2.30 | 1.98 | 2.15 |
| Video LLama (Zhang et al., 2023a) | 1.96 | 2.18 | 2.16 | 1.82 | 1.79 |
| Video-ChatGPT (Maaz et al., 2023) | 2.40 | 2.52 | 2.62 | 1.98 | 2.37 |
| MiniGPT-4 | 2.68 | 2.76 | 3.20 | 2.26 | 2.18 |
  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/83e343ff63e0b19011b9fe9f41b13b6db0f9e780170f758d5d976054359b96f6.jpg)  

Figure 5: MiniGPT-4 before second-stage fine- Figure 6: An example of MiniGPT-4’s limitations. tuning fails to output completed texts. The gener- MiniGPT-4 hallucinates unexisting tablecloths ation is improved after the finetuning. and can’t locate the windows correctly.  

Can the original BLIP-2 benefit from the second-stage data? In this study, we finetune BLIP-2 (Li et al., 2023c) with our second-stage data in the same way as MiniGPT-4, and check if it can obtain similar advanced abilities as MiniGPT-4. The finetuned BLIP-2 is denoted as BLIP-2 FT. Note that MiniGPT-4 uses the same visual module as BLIP-2; while BLIP-2 uses FlanT5 XXL (Chung et al., 2022) as the language model, which is not as strong as the Vicuna (Chiang et al., 2023) model used in our MiniGPT-4 model. We rely on the same prompts to assess the advanced capabilities of our model. Qualitative results are shown in Fig.4, 13, and 14. We discover that BLIP-2 FT still generates short responses and fails to generalize to advanced tasks like meme explaining and website coding (Fig.4). Our finding suggests that BLIP-2’s relatively weaker language model FlanT5 XXL benefits less from such a small dataset, and highlights the effectiveness of a more advanced LLM in a VLM system.  

Second stage with Localized Narratives We tested MiniGPT-4’s performance by substituting our self-collected dataset with the Localized Narratives dataset (Pont-Tuset et al., 2020) in the second training stage. We name this variant MiniGPT-4 LocNa. The Localized Narratives dataset features detailed image descriptions with corresponding regional localizations. Qualitative results shown in Fig.4, 13, and 14 reveal that MiniGPT-4 LocNa can produce lengthy image descriptions (as seen in Fig.14). However, these outputs are of lower quality, often with monotonous expressions. MiniGPT-4 LocNa also shows weaker generalization in complex tasks, like explaining meme humor (Fig.4a), compared to the original MiniGPT-4. This performance difference may stem from the repetitive and monotonous nature of the Localized Narratives dataset.  

Second stage with GPT-4V generated data. We conduct further ablation experiments using 2,000 GPT-4V generated image-text pairs collected by LAION (LAION, 2023) in the second stage. Results in Tab.2 shows performance improvements from this fine-tuning.  

Table 5: Ablation on architecture designs   


| | | |
| :--- | :--- | :--- |
| Model | AOK-VQA | GQA |
| MiniGPT-4 | 58.2 | 32.2 |
| (a) MiniGPT-4 w/o Q-Former | 56.9 | 33.4 |
| (b) MiniGPT-4 + 3 Layers | 49.7 | 31.0 |
| (c) MiniGPT-4 + Finetune Q-Former | 52.1 | 28.0 |
  

Table 6: Hallucination Evaluation   


| | | |
| :--- | :--- | :--- |
|  | CHAIR | Avg. Length |
| Blip-2 | 1.3 | 6.5 |
| mPLUG-Owl | 30.2 | 98.5 |
| LLaVa | 18.8 | 90.7 |
| MiniGPT-4 (short) | 7.2 | 28.8 |
| MiniGPT-4 (long) | 9.6 | 175 |
  

Amount of traing data in the first stage This ablation study can be found in Appx.A.4.  

# 4.4 ABLATION ON THE ARCHITECTURE DESIGNS  

To further demonstrate the effectiveness of using one single linear layer to align visual features with LLM, we conduct experiments with different architecture designs, including (a) removing the QFormer and directly mapping the VIT’s output to Vicuna’s embedding space (i.e., without Q-former), (b) using three linear layers instead of one layer, and (c) additionally finetuning the Q-Former in the vision module. All the variants are trained in the same way as the original design. Results on AOK-VQA (Schwenk et al., 2022) and GQA (Hudson & Manning, 2019) datasets in Tab.5 show that the variant (a) MiniGPT-4 w/o Q-Former has a similar performance to the original design. Qualitative results of this variant in Fig.4, 13, and 14 also show similar advanced skills. This reveals that the Q-Former from BLIP-2 doesn’t plays a critical roles for advanced skills. Besides, both variants (b) MiniGPT- $\mathbf{4}\substack{+3}$ Layers and (c) MiniGPT- $^{4+}$ finetuning Q-Former, perform slightly worse than the original MiniGPT-4. This indicates a single projection layer is sufficient to align the vision encoder and the large language model in our limited training data setting.  

# 4.5 LIMITATION ANALYSIS  

Hallucination As MiniGPT-4 is built upon LLMs, it inherits LLM’s limitations like hallucinating nonexistent knowledge. An example in Fig. 6 shows that MiniGPT-4 incorrectly identifies the presence of white tablecloths in the image, despite their absence. Here, we use the metric $\mathrm{CHAIR}_{i}$ (Rohrbach et al., 2018) to gauge the hallucination rate of the generation, with the two distinct prompts to control the model generation length: MiniGPT-4 (long): Please describe this image as detailed as possible. MiniGPT-4 (short): Please describe the image shortly and precisely, in less than 20 words.  

Results in Tab.6 show that longer captions tend to have higher hallucination rates. For example, MiniGPT-4 (long) generates captions averaging 175 words with a higher hallucination rate, while MiniGPT-4 (short) averages 28.8 words with a lower rate. BLIP-2, averaging 6.5 words, hallucinates less but covers fewer objects as seen in Tab.2. Compared to contemporary methods like LLaVa or mPlug-Owl, MiniGPT-4 generates longer descriptions with fewer hallucination. Hallucination in detailed image descriptions is still an unresolved issue. Using Reinforcement Learning with AI feadback with hallucination detection modules may be a potential solution.  

Spatial Information Understanding MiniGPT-4’s visual perception remains limited. It may struggle to differentiate spatial localization. For example, MiniGPT-4 in Fig. 6 fails to identify the location of the windows. This limitation may stem from a lack of aligned image-text data designed for spatial information understanding. Training on such datasets like RefCOCO (Kazemzadeh et al., 2014) or Visual Genome (Krishna et al., 2017) could potentially alleviate this issue.  

# 5 DISCUSSION  

How does MiniGPT-4 obtain these advanced abilities? Many of the advanced vision-language capabilities demonstrated by GPT-4 can be understood as compositional skills rooted in two foundational skills: image understanding and language generation. Take the task of image-based poem writing as an example. Advanced LLMs like ChatGPT and Vicuna can already craft poems based on users’ instructions. If they acquire the ability to understand images, compositionally generalizing to the task of image-based poem writing even without having image-poem pairs in their training data is possible.  

In its first pretraining stage, MiniGPT-4 learns image understanding by correlating images with short descriptions from caption datasets. However, the language style in these datasets differs from that of modern LLMs, leading to distorted language generation and impeding compositional generalization. To address this, a second-stage finetuning is introduced to improve language generation. Post two-stage training, MiniGPT-4 successfully demonstrates advanced compositional vision-language abilities, such as draft-to-website or interpreting memes, confirming our approach. Future research could explore the mechanisms of compositional generalization further. Our work, as a preliminary exploration of vision-based LLM capabilities, aims to encourage more studies in this area.  

REFERENCES   
Sharegpt. https://github.com/domeccleston/sharegpt, 2023.   
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. Flamingo: a visual language model for few-shot learning. In Advances in Neural Information Processing Systems, 2022.   
Anas Awadalla, Irena Gao, Josh Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Shiori Sagawa, et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.   
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.   
Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual $12\mathrm{m}$ : Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3558–3568, 2021.   
Jun Chen, Han Guo, Kai Yi, Boyang Li, and Mohamed Elhoseiny. Visualgpt: Data-efficient adaptation of pretrained language models for image captioning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18030–18040, 2022.   
Jun Chen, Deyao Zhu, Kilichbek Haydarov, Xiang Li, and Mohamed Elhoseiny. Video chatcaptioner: Towards the enriched spatiotemporal descriptions. arXiv preprint arXiv:2304.04227, 2023.   
Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. Vicuna: An open-source chatbot impressing gpt-4 with $90\%*$ chatgpt quality, March 2023. URL https: //vicuna.lmsys.org.   
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311, 2022.   
Hyung Won Chung, Le Hou, Shayne Longpre, Barret Zoph, Yi Tay, William Fedus, Eric Li, Xuezhi Wang, Mostafa Dehghani, Siddhartha Brahma, et al. Scaling instruction-finetuned language models. arXiv preprint arXiv:2210.11416, 2022.   
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. Instructblip: Towards general-purpose vision-language models with instruction tuning, 2023.   
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.   
Danny Driess, Fei Xia, Mehdi SM Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, et al. Palm-e: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.   
Zhengxiao Du, Yujie Qian, Xiao Liu, Ming Ding, Jiezhong Qiu, Zhilin Yang, and Jie Tang. Glm: General language model pretraining with autoregressive blank infilling. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 320–335, 2022.   
Yuxin Fang, Wen Wang, Binhui Xie, Quan Sun, Ledell Wu, Xinggang Wang, Tiejun Huang, Xinlong Wang, and Yue Cao. Eva: Exploring the limits of masked visual representation learning at scale. arXiv preprint arXiv:2211.07636, 2022.   
Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen. Multimodal-gpt: A vision and language model for dialogue with humans. arXiv preprint arXiv:2305.04790, 2023.  

ordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 2022.  

Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021.  

Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al. Language is not all you need: Aligning perception with language models. arXiv preprint arXiv:2302.14045, 2023.  

Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700–6709, 2019.  

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 787–798, 2014.  

Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.  

LAION. Laion gpt4v dataset. https://huggingface.co/datasets/laion/ gpt4v-dataset, 2023.  

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Fanyi Pu, Jingkang Yang, Chunyuan Li, and Ziwei Liu. Mimic-it: Multi-modal in-context instruction tuning. arXiv preprint arXiv:2306.05425, 2023a.  

Bo Li, Yuanhan Zhang, Liangyu Chen, Jinghao Wang, Jingkang Yang, and Ziwei Liu. Otter: A multi-modal model with in-context instruction tuning. arXiv preprint arXiv:2305.03726, 2023b.  

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pp. 12888–12900. PMLR, 2022.  

Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. arXiv preprint arXiv:2301.12597, 2023c.  

KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. Videochat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355, 2023d.  

Lei Li, Yuwei Yin, Shicheng Li, Liang Chen, Peiyi Wang, Shuhuai Ren, Mukai Li, Yazheng Yang, Jingjing Xu, Xu Sun, et al. M ˆ3 it: A large-scale dataset towards multi-modal multilingual instruction tuning. arXiv preprint arXiv:2306.04387, 2023e.  

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. arXiv preprint arXiv:2304.08485, 2023a.  

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? arXiv preprint arXiv:2307.06281, 2023b.  

Muhammad Maaz, Hanoona Rasheed, Salman Khan, and Fahad Shahbaz Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. arXiv preprint arXiv:2306.05424, 2023.  

OpenAI. Introducing chatgpt. https://openai.com/blog/chatgpt, 2022.  

OpenAI. Gpt-4 technical report, 2023.  

Vicente Ordonez, Girish Kulkarni, and Tamara Berg. Im2text: Describing images using 1 million captioned photographs. Advances in neural information processing systems, 24, 2011.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730–27744, 2022.   
Jordi Pont-Tuset, Jasper Uijlings, Soravit Changpinyo, Radu Soricut, and Vittorio Ferrari. Connecting vision and language with localized narratives. In Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part V 16, pp. 647–664. Springer, 2020.   
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485–5551, 2020.   
Anna Rohrbach, Lisa Anne Hendricks, Kaylee Burns, Trevor Darrell, and Kate Saenko. Object hallucination in image captioning. arXiv preprint arXiv:1809.02156, 2018.   
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagn´e, Alexandra Sasha Luccioni, Franc¸ois Yvon, Matthias Gall´e, et al. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022a.   
Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ili´c, Daniel Hesslow, Roman Castagn´e, Alexandra Sasha Luccioni, Franc¸ois Yvon, Matthias Gall´e, et al. Bloom: A 176bparameter open-access multilingual language model. arXiv preprint arXiv:2211.05100, 2022b.   
Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021.   
Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, Kenneth Marino, and Roozbeh Mottaghi. A-okvqa: A benchmark for visual question answering using world knowledge. In European Conference on Computer Vision, pp. 146–162. Springer, 2022.   
Piyush Sharma, Nan Ding, Sebastian Goodman, and Radu Soricut. Conceptual captions: A cleaned, hypernymed, image alt-text dataset for automatic image captioning. In Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 2556–2565, 2018.   
Shaden Smith, Mostofa Patwary, Brandon Norick, Patrick LeGresley, Samyam Rajbhandari, Jared Casper, Zhun Liu, Shrimai Prabhumoye, George Zerveas, Vijay Korthikanti, et al. Using deepspeed and megatron to train megatron-turing nlg 530b, a large-scale generative language model. arXiv preprint arXiv:2201.11990, 2022.   
Dı´dac Surı´s, Sachit Menon, and Carl Vondrick. Vipergpt: Visual inference via python execution for reasoning. arXiv preprint arXiv:2303.08128, 2023.   
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.   
Anthony Meng Huat Tiong, Junnan Li, Boyang Li, Silvio Savarese, and Steven CH Hoi. Plug-andplay vqa: Zero-shot vqa by conjoining large pretrained models with zero training. arXiv preprint arXiv:2210.08773, 2022.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timoth´ee Lacroix, Baptiste Rozi\`ere, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.   
Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill. Multimodal few-shot learning with frozen language models. Advances in Neural Information Processing Systems, 34:200–212, 2021.   
Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, Ed H. Chi, Tatsunori Hashimoto, Oriol Vinyals, Percy Liang, Jeff Dean, and William Fedus. Emergent abilities of large language models. Transactions on Machine Learning Research, 2022. ISSN 2835-8856. URL https://openreview.net/forum?id=yzkSU5zdwD. Survey Certification.   
Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan. Visual chatgpt: Talking, drawing and editing with visual foundation models. arXiv preprint arXiv:2303.04671, 2023.   
Antoine Yang, Antoine Miech, Josef Sivic, Ivan Laptev, and Cordelia Schmid. Zero-shot video question answering via frozen bidirectional language models. arXiv preprint arXiv:2206.08155, 2022.   
Zhengyuan Yang\*, Linjie Li\*, Jianfeng Wang\*, Kevin Lin\*, Ehsan Azarnasab\*, Faisal Ahmed\*, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang. Mm-react: Prompting chatgpt for multimodal reasoning and action. 2023.   
Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al. mplug-owl: Modularization empowers large language models with multimodality. arXiv preprint arXiv:2304.14178, 2023.   
Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023a.   
Renrui Zhang, Jiaming Han, Aojun Zhou, Xiangfei Hu, Shilin Yan, Pan Lu, Hongsheng Li, Peng Gao, and Yu Qiao. Llama-adapter: Efficient fine-tuning of language models with zero-init attention. arXiv preprint arXiv:2303.16199, 2023b.   
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.   
Deyao Zhu, Jun Chen, Kilichbek Haydarov, Xiaoqian Shen, Wenxuan Zhang, and Mohamed Elhoseiny. Chatgpt asks, blip-2 answers: Automatic questioning towards enriched visual descriptions. arXiv preprint arXiv:2303.06594, 2023.  

A APPENDIX  

# A.1 MORE QUALITATIVE RESULTS  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/6cacf733790e3153c7661d61cdf7ab2be8f23dd02d7e8a8ce6751304ddc176cb.jpg)  

Human Explain why this meme is funny.  

BLIP-2 A dog laying down on the floor with the caption monday just Monday.  

MiniGPT-4 This meme is funny because it shows a dog laying on the floor with its head resting on its paws, as if it is tired or sleepy. The caption, "monday just monday," adds to the humor by suggesting that the dog is feeling the same way as many people do on Mondays, which are often considered to be the most dreaded day of the week. The meme is relatable and humorous, making it a popular one among internet users.  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/82dc69632d9aa97eaa5226a035f66c39498cc36d6a204c15a6b072d65fc435b5.jpg)  
Human Can you give me a short introduction to this movie?  

BLIP-2 The godfather is a 1969 american gangster film directed by martin scorses.  

MiniGPT-4 The Godfather is a classic American crime drama film directed by Francis Ford Coppola and released in 1972. It is based on the novel of the same name by Mario Puzo and follows the story of the Corleone family, a powerful mafia clan in New York City. The film stars Marlon Brando as the patriarch of the family, Vito Corleone, and Al Pacino as his son, Michael Corleone. The Godfather is known for its iconic performances, memorable dialogue, and its portrayal of the American mafia lifestyle. It is considered one of the greatest films of all time and has had a lasting impact on popular culture.  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/a5d40229421a39a5a28d19609f7ae3c8c25a9f1844c726260802afeece54aa1c.jpg)  
Figure 7: Advertisement promotion   
Figure 9: Website creating  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/2164fea3212ecd0a467cf972411697456f1cc55655e5c0456c8c2d3cc78f207d.jpg)  
Figure 8: Factual retrieval   
Figure 10: Poem writing  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/8f95d7b505607d6540b84b77e1f07955cdddce53ed71478caf99b63b3ffcdb67.jpg)  
Figure 11: Food recipe generation   
Figure 12: Plant cultivating  

# A.2 EVALUATION IN TRADITIONAL VQA BENCHMARKS  

The aim of this study is to replicate the remarkable multi-modal capabilities demonstrated in GPT-4, such as generating detailed image descriptions and creating websites from hand-drawn drafts. To emphasize the most crucial component of advanced vision-language skills, the methodology of MiniGPT-4 is intentionally kept minimal. For instance, the learnable model capacity is limited (only one linear layer), and MiniGPT-4 is trained with just 5 million pairs, in contrast to BLIP-2 with 129 million image-text pairs. Such a pared-down approach is anticipated to yield suboptimal results on traditional benchmarks. While this isn’t our primary goal, we offer a quantitative analysis of the VQA datasets A-OKVQA (multi-choice) (Schwenk et al., 2022) and GQA (Hudson & Manning, 2019). Additionally, to showcase the potential of MiniGPT-4 with traditional benchmarks, we conduct a straightforward ablation study. Here, we simply unfreeze the LLM using LoRA (Hu et al., 2021) and incorporate more training data from the VQAv2, OKVQA, and A-OKVQA datasets during the second finetuning stage. Results in Tab. 7 indicate that the original MiniGPT-4 lags behind BLIP-2 by a reasonable margin, and merely augmenting the learning capacity and the training data results in a substantial performance improvement, which confirms our expectations. We believe our model’s performance on conventional vision benchmarks can be enhanced with a carefully designed training strategy (e.g., dataset sample ratios, learning rate schedule, etc.), more training data/datasets, and additional learnable parameters. Since enhancing performance on traditional vision benchmarks isn’t this project’s objective, we reserve this aspect for future research.  

Table 7: Performance Comparison between BLIP-2 and MiniGPT-4   


| | | | |
| :--- | :--- | :--- | :--- |
| Model | Training data | AOK-VQA | GQA |
| Blip-2 | 129M image-text pairs | 80.2 | 42.4 |
| MiniGPT-4 | 5M image-text pairs | 58.2 | 32.2 |
| MiniGPT-4 (Finetune Vicuna) | 5M image-text pairs | 67.2 | 43.5 |
  

# A.3 DETAILS OF CAPTION EVALUATION  

We utilize GPT-4 turbo (gpt-4-1106-preview) to assess whether the generated descriptions capture the content of each ground truth caption individually. In the COCO dataset, each image is accompanied by 5 ground truth captions. For every image, we calculate the number of captions covered by the generated descriptions and then average this count across 5000 random sampled images from the validation set to derive the final score.  

Here is the prompt we use in GPT-4 turbo   
Given a test image description and a list of gt image caption,   
verify whether the information in gt caption is included in the test description. The answer should be yes or no.   
Input is in this format:   
Test: (test sentence)   
1: (gt1)   
2: (gt2)   
3: (gt3)   
you need to answer yes or no for each gt in the following format:   
1: (yes/no)   
2: (yes/no)   
3: (yes/no)  

# A.4 AMOUNT OF TRAINING DATA IN THE FIRST STAGE.  

We evaluate the impact of training data volume in the first stage by using checkpoints at $10\%$ , $30\%$ , and $50\%$ of stage 1 duration, subsequently finetuned in stage 2. Results in Tab. 8 show a significant performance drop with only $10\%$ of stage 1 data. However, utilizing $30\%$ of stage 1 data, equivalent to $1.5\mathbf{M}$ image-text pairs can achieve similar performance with the original MiniGPT-4. No gains were seen beyond $50\%$ of stage 1 data, indicating potential saturation of the model’s learnable capacity at this juncture.  

Table 8: Captioning performance with different amount of training data in stage-1.   


| | | | | |
| :--- | :--- | :--- | :--- | :--- |
| Metric | 10% | 30% | 50% | 100% |
| #GT Cover | 1.62 | 2.15 | 2.26 | 2.22 |
  

# A.5 MINIGPT-4 ON MMBENCH  

MMBench (Liu et al., 2023b) is a new multi-modality benchmark with diverse evaluation questions to evaluate different abilities of vision language model. MMBench evaluated MiniGPT-4 together with other contemporary vision language models like OpenFlamingo (Awadalla et al., 2023), VisualGLM (Du et al., 2022), LLaVa (Liu et al., 2023a), and InstructBlip Dai et al. (2023). Here, we show the performance of MiniGPT-4 and other baseline models in Tab. 9. Results show that MiniGPT-4 demonstrates competitive performance compared to contemporary methods, e.g., InstructBlip. It surpasses InstructBlip in several key areas: logical reasoning (LR), fine-grained perception for single instance (FP-S), and fine-grained perception across instances (FP-C). Additionally, MiniGPT-4 achieves competitive results in relation reasoning (RR), attribute reasoning (AR), and coarse perception (CP).  

Table 9: Perforance on MMBench benchmark. Numbers are from Liu et al. (2023b).   


| | | | | | | | |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Model | Overall | LR | AR | RR | FP-S | FP-C | CP |
| OpenFlamingo | 4.6 | 6.7 | 8.0 | 0.0 | 6.7 | 2.8 | 2.0 |
| VisualGLM | 38.1 | 10.8 | 44.3 | 35.7 | 43.8 | 23.4 | 47.3 |
| LLaVa | 38.7 | 16.7 | 48.3 | 30.4 | 45.5 | 32.4 | 40.6 |
| InstructBlip | 44.0 | 19.1 | 54.2 | 34.8 | 47.8 | 24.8 | 56.4 |
| MiniGPT-4 | 42.3 | 20.8 | 50.7 | 30.4 | 49.5 | 26.2 | 50.7 |
  

# A.6 MORE QUALITATIVE ABLATION RESULTS  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/67ffbcaf29352cc6773d99bc5024ede7b993666ee02d1c02266893592920dc63.jpg)  
iFGigPuTr-e413N:oAQbl-aFtiornmSetru)d,ythone RMeicnipieGPGTe-n4eratiitohn  

![](https://netmind-public-files.s3.us-west-2.amazonaws.com/3deaeab8e5794f52b2050346fcf7be3f/a1a2c193784b411b8b331cc111dfa72e471049d940c526541bb292654416ac36.jpg)  
Figure 14: Ablation Study on Detailed Description  