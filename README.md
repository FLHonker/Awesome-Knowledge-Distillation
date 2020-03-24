# Awesome Knowledge-Distillation

![](https://img.shields.io/badge/Number-257-green)

- [Awesome Knowledge-Distillation](#awesome-knowledge-distillation)
  - [Different forms of knowledge](#different-forms-of-knowledge)
    - [Knowledge from logits](#knowledge-from-logits)
    - [Knowledge from intermediate layers](#knowledge-from-intermediate-layers)
    - [Graph-based](#graph-based)
    - [Mutual Information](#mutual-information)
    - [Self-KD](#self-kd)
    - [Structured Knowledge](#structured-knowledge)
    - [Privileged Information](#privileged-information)
  - [KD + GAN](#kd--gan)
  - [KD + Meta-learning](#kd--meta-learning)
  - [Data-free KD](#data-free-kd)
  - [KD + AutoML](#kd--automl)
  - [KD + RL](#kd--rl)
  - [Multi-teacher KD](#multi-teacher-kd)
    - [Knowledge Amalgamation（KA) - zju-VIPA](#knowledge-amalgamationka---zju-vipa)
  - [Cross-modal KD & DA](#cross-modal-kd--da)
  - [Application of KD](#application-of-kd)
    - [for NLP](#for-nlp)
  - [Model Pruning or Quantization](#model-pruning-or-quantization)
  - [Beyond](#beyond)

## Different forms of knowledge

### Knowledge from logits

1. Distilling the knowledge in a neural network. Hinton et al. arXiv:1503.02531
2. Learning from Noisy Labels with Distillation. Li, Yuncheng et al. ICCV 2017
3. Training Deep Neural Networks in Generations:A More Tolerant Teacher Educates Better Students. arXiv:1805.05551
4. Knowledge distillation by on-the-fly native ensemble. Lan, Xu et al. NIPS 2018
5. Learning Metrics from Teachers: Compact Networks for Image Embedding. Yu, Lu et al. CVPR 2019
6. Relational Knowledge Distillation.  Park, Wonpyo et al, CVPR 2019
7. Like What You Like: Knowledge Distill via Neuron Selectivity Transfer. Huang, Zehao and Wang, Naiyan. 2017
8. On Knowledge Distillation from Complex Networks for Response Prediction. Arora, Siddhartha et al. NAACL 2019
9. On the Efficacy of Knowledge Distillation. Cho, Jang Hyun and Hariharan, Bharath. arXiv:1910.01348. ICCV 2019
10. [noval]Revisit Knowledge Distillation: a Teacher-free Framework. Yuan, Li et al. arXiv:1909.11723
11. Improved Knowledge Distillation via Teacher Assistant: Bridging the Gap Between Student and Teacher. Mirzadeh et al. arXiv:1902.03393
12. Ensemble Distribution Distillation. ICLR 2020
13. Noisy Collaboration in Knowledge Distillation. ICLR 2020
14. On Compressing U-net Using Knowledge Distillation. arXiv:1812.00249
15. Distillation-Based Training for Multi-Exit Architectures. Phuong, Mary and Lampert, Christoph H. ICCV 2019
16. Self-training with Noisy Student improves ImageNet classification. Xie, Qizhe et al.(Google) CVPR 2020
17. Variational Student: Learning Compact and Sparser Networks in Knowledge Distillation Framework. arXiv:1910.12061
18. Preparing Lessons: Improve Knowledge Distillation with Better Supervision. arXiv:1911.07471
19. Adaptive Regularization of Labels. arXiv:1908.05474
20. Positive-Unlabeled Compression on the Cloud. Xu, Yixing(HUAWEI) et al. NIPS 2019
21. Snapshot Distillation: Teacher-Student Optimization in One Generation. Yang, Chenglin et al. CVPR 2019
22. QUEST: Quantized embedding space for transferring knowledge. Jain, Himalaya et al. CVPR 2020(pre)
23. Conditional teacher-student learning. Z. Meng et al. ICASSP 2019
24. Subclass Distillation. Müller, Rafael et al. arXiv:2002.03936
25. MarginDistillation: distillation for margin-based softmax. Svitov, David & Alyamkin, Sergey. arXiv:2003.02586
26. An Embarrassingly Simple Approach for Knowledge Distillation. Gao, Mengya et al. MLR 2018
27. Sequence-Level Knowledge Distillation. Kim, Yoon & Rush, Alexander M. arXiv:1606.07947
28. Boosting **Self-Supervised** Learning via Knowledge Transfer. Noroozi, Mehdi et al. CVPR 2018

### Knowledge from intermediate layers

1. Fitnets: Hints for thin deep nets. Romero, Adriana et al. arXiv:1412.6550
2. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. Zagoruyko et al. ICLR 2017
3. Knowledge Projection for Effective Design of Thinner and Faster Deep Neural Networks. Zhang, Zhi et al. arXiv:1710.09505
4. A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning. Yim, Junho et al. CVPR 2017
5. Paraphrasing complex network: Network compression via factor transfer. Kim, Jangho et al. NIPS 2018
6. Knowledge transfer with jacobian matching. ICML 2018
7. Self-supervised knowledge distillation using singular value decomposition. Lee, Seung Hyun et al. ECCV 2018
8. Variational Information Distillation for Knowledge Transfer. Ahn, Sungsoo et al. CVPR 2019
9  <!-- * 通过互信息导出关于student中间层表示和teacher中间表示的关系。 -->
10. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019
11. Knowledge Distillation via Route Constrained Optimization. Jin, Xiao et al. ICCV 2019
12. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
13. MEAL: Multi-Model Ensemble via Adversarial Learning. Shen,Zhiqiang, He,Zhankui, and Xue Xiangyang. AAAI 2019
14. A Comprehensive Overhaul of Feature Distillation. Heo, Byeongho et al. ICCV 2019
15. Feature-map-level Online Adversarial Knowledge Distillation. ICLR 2020
16. Distilling Object Detectors with Fine-grained Feature Imitation. ICLR 2020
17. Knowledge Squeezed Adversarial Network Compression. Changyong, Shu et al. AAAI 2020
18. Stagewise Knowledge Distillation. Kulkarni, Akshay et al. arXiv: 1911.06786
19. Knowledge Distillation from Internal Representations. AAAI 2020
20. Knowledge Flow:Improve Upon Your Teachers. ICLR 2019
21. LIT: Learned Intermediate Representation Training for Model Compression. ICML 2019
22. Learning Deep Representations with Probabilistic Knowledge Transfer. Passalis et al. ECCV 2018
23. Improving the Adversarial Robustness of Transfer Learning via Noisy Feature Distillation. Chin, Ting-wu et al. arXiv:2002.02998
24. Knapsack Pruning with Inner Distillation. Aflalo, Yonathan et al. arXiv:2002.08258
25. Residual Knowledge Distillation. Gao, Mengya et al. arXiv:2002.09168
26. Knowledge distillation via adaptive instance normalization. Yang, Jing et al. arXiv:2003.04289
27. Bert-of-Theseus: Compressing bert by progressive module replacing. Xu, Canwen et al. arXiv:2002.02925 [[code][2.27]]

### Graph-based
1. Graph-based Knowledge Distillation by Multi-head Attention Network. Lee, Seunghyun and Song, Byung. Cheol arXiv:1907.02226
2. Graph Representation Learning via Multi-task Knowledge Distillation. arXiv:1911.05700
3. Deep geometric knowledge distillation with graphs. arXiv:1911.03080
4. Better and faster: Knowledge transfer from multiple self-supervised learning tasks via graph distillation for video classification. IJCAI 2018

### Mutual Information

1. Correlation Congruence for Knowledge Distillation. Peng, Baoyun et al. ICCV 2019
2. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
3. Variational Information Distillation for Knowledge Transfer. Ahn, Sungsoo et al. CVPR 2019
4. Contrastive Representation Distillation. Tian, Yonglong et al. ICLR 2020

### Self-KD

1. Moonshine:Distilling with Cheap Convolutions. Crowley, Elliot J. et al. NIPS 2018 
2. Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. Zhang, Linfeng et al. ICCV 2019
3. Learning Lightweight Lane Detection CNNs by Self Attention Distillation. Hou, Yuenan et al. ICCV 2019
4. BAM! Born-Again Multi-Task Networks for Natural Language Understanding. Clark, Kevin et al. ACL 2019,short
5. Self-Knowledge Distillation in Natural Language Processing. Hahn, Sangchul and Choi, Heeyoul. arXiv:1908.01851
6. Rethinking Data Augmentation: Self-Supervision and Self-Distillation. Lee, Hankook et al. ICLR 2020
7. Regularizing Predictions via Class wise Self knowledge Distillation. ICLR 2020
8. MSD: Multi-Self-Distillation Learning via Multi-classifiers within Deep Neural Networks. arXiv:1911.09418
9. Self-Distillation Amplifies Regularization in Hilbert Space. Mobahi, Hossein et al. arXiv:2002.05715
10. MINILM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. Wang, Wenhui et al. arXiv:2002.10957

### Structured Knowledge

1. Paraphrasing Complex Network:Network Compression via Factor Transfer. Kim, Jangho et al. NIPS 2018
2. Relational Knowledge Distillation.  Park, Wonpyo et al. CVPR 2019
   <!-- * 通过对输出embedding表示构建了instance之间的二阶距离关系和三届角度关系，作为一种知识引导student学习 -->
3. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019
   <!-- * 通过instance中间层表示构建了instance之间的图，并将图作为一种知识进行传递。 -->
4. Contrastive Representation Distillation. Tian, Yonglong et al. arXiv: 1910.10699
5. Teaching To Teach By Structured Dark Knowledge. ICLR 2020

### Privileged Information

1. Learning using privileged information: similarity control and knowledge transfer. Vapnik, Vladimir and Rauf, Izmailov. MLR 2015  
2. Unifying distillation and privileged information. Lopez-Paz, David et al. ICLR 2016
3. Model compression via distillation and quantization. Polino, Antonio et al. ICLR 2018
4. KDGAN:Knowledge Distillation with Generative Adversarial Networks. Wang, Xiaojie. NIPS 2018
5. [noval]Efficient Video Classification Using Fewer Frames. Bhardwaj, Shweta et al. CVPR 2019
6. Retaining privileged information for multi-task learning. Tang, Fengyi et al. KDD 2019
7. A Generalized Meta-loss function for regression and classification using privileged information. Asif, Amina et al. arXiv:1811.06885

## KD + GAN

1. Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks. Xu, Zheng et al. arXiv:1709.00513
2. KTAN: Knowledge Transfer Adversarial Network. Liu, Peiye et al. arXiv:1810.08126
3. KDGAN:Knowledge Distillation with Generative Adversarial Networks. Wang, Xiaojie. NIPS 2018
4. Adversarial Learning of Portable Student Networks. Wang, Yunhe et al. AAAI 2018
5. Adversarial Network Compression. Belagiannis, Vasileios et al. ECCV 2018
6. Cross-Modality Distillation: A case for Conditional Generative Adversarial Networks. ICASSP 2018
7. Adversarial Distillation for Efficient Recommendation with External Knowledge. TOIS 2018
8. Training student networks for acceleration with conditional adversarial networks. Xu, Zheng et al. BMVC 2018
9. [noval]DAFL:Data-Free Learning of Student Networks. Chen, Hanting et al. ICCV 2019
10. MEAL: Multi-Model Ensemble via Adversarial Learning. Shen,Zhiqiang, He,Zhankui, and Xue Xiangyang. AAAI 2019
11. Knowledge Distillation with Adversarial Samples Supporting Decision Boundary. Heo, Byeongho et al. AAAI 2019
12. Exploiting the Ground-Truth: An Adversarial Imitation Based Knowledge Distillation Approach for Event Detection. Liu, Jian et al. AAAI 2019
13. Adversarially Robust Distillation. Goldblum, Micah et al. AAAI 2020
14. GAN-Knowledge Distillation for one-stage Object Detection. Hong, Wei et al. arXiv:1906.08467
15. Lifelong GAN: Continual Learning for Conditional Image Generation. Kundu et al. arXiv:1908.03884
16. Compressing GANs using Knowledge Distillation. Aguinaldo, Angeline et al. arXiv:1902.00159
17. Feature-map-level Online Adversarial Knowledge Distillation. ICLR 2020
18. MineGAN: effective knowledge transfer from GANs to target domains with few images. Wang, Yaxing et al. arXiv:1912.05270
19. Distilling portable Generative Adversarial Networks for Image Translation. Chen, Hanting et al. AAAI 2020
20. GAN Compression: Efficient Architectures for Interactive Conditional GANs. Junyan Zhu et al. CVPR 2020 [[code][8.20]]

## KD + Meta-learning

1. Few Sample Knowledge Distillation for Efficient Network Compression. Li, Tianhong et al. ICLR 2020
2. Learning What and Where to Transfer. Jang, Yunhun et al, ICML 2019
3. Transferring Knowledge across Learning Processes. Moreno, Pablo G et al. ICLR 2019
4. Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval. Liu, Qing et al. ICCV 2019
5. Diversity with Cooperation: Ensemble Methods for Few-Shot Classification. Dvornik, Nikita et al. ICCV 2019
6. Knowledge Representing: Efficient, Sparse Representation of Prior Knowledge for Knowledge Distillation. arXiv:1911.05329v1
7. Progressive Knowledge Distillation For Generative Modeling. ICLR 2020
8. Few Shot Network Compression via Cross Distillation. AAAI 2020

## Data-free KD
1. Data-Free Knowledge Distillation for Deep Neural Networks. NIPS 2017
2. Zero-Shot Knowledge Distillation in Deep Networks. ICML 2019
3. DAFL:Data-Free Learning of Student Networks. ICCV 2019
4. Zero-shot Knowledge Transfer via Adversarial Belief Matching. Micaelli, Paul and Storkey, Amos. NIPS 2019
5. Dream Distillation: A Data-Independent Model Compression Framework. Kartikeya et al. ICML 2019
6. Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion. Yin, Hongxu et al. CVPR 2020
7. Data-Free Adversarial Distillation. Fang, Gongfan et al. CVPR 2020
8. The Knowledge Within: Methods for Data-Free Model Compression. Haroush, Matan et al. arXiv:1912.01274
9. Knowledge Extraction with No Observable Data. Yoo, Jaemin et al. NIPS 2019 [[code][10.9]]
10. Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN. CVPR 2020

- other data-free model compression:

11. Data-free Parameter Pruning for Deep Neural Networks. Srinivas, Suraj et al. arXiv:1507.06149
12. Data-Free Quantization Through Weight Equalization and Bias Correction. Nagel, Markus et al. ICCV 2019
13. ZeroQ: A Novel Zero Shot Quantization Framework. Cai, Yaohui et al. arxiv:2001.00281

## KD + AutoML

1. Improving Neural Architecture Search Image Classifiers via Ensemble Learning. Macko, Vladimir et al. 2019
2. Blockwisely Supervised Neural Architecture Search with Knowledge Distillation. Li, Changlin et al. arXiv:1911.13053v1
3. Towards Oracle Knowledge Distillation with Neural Architecture Search. Kang, Minsoo et al. AAAI 2020
4. Search for Better Students to Learn Distilled Knowledge. Gu, Jindong & Tresp, Volker arXiv:2001.11612

## KD + RL

1. N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning. Ashok, Anubhav et al. ICLR 2018
2. Knowledge Flow:Improve Upon Your Teachers. Liu, Iou-jen et al. ICLR 2019
3. Transferring Knowledge across Learning Processes. Moreno, Pablo G et al. ICLR 2019
4. Exploration by random network distillation. Burda, Yuri et al. ICLR 2019
5. Periodic Intra-Ensemble Knowledge Distillation for Reinforcement Learning. Hong, Zhang-Wei et al. arXiv:2002.00149
6. Transfer Heterogeneous Knowledge Among Peer-to-Peer Teammates: A Model Distillation Approach. Xue, Zeyue et al. arXiv:2002.02202

## Multi-teacher KD 

1. Learning from Multiple Teacher Networks. You, Shan et al. KDD 2017
2. Semi-Supervised Knowledge Transfer for Deep Learning from Private Training Data. ICLR 2017
    <!-- * 也是多teacher，但是从隐私保护的角度来融合teacher输出的结果 -->
3. Knowledge Adaptation: Teaching to Adapt. Arxiv:1702.02052
    <!-- * 迁移学习，每个source domain对应一个teacher。KD中的温度值设定为5。 -->
4. Deep Model Compression: Distilling Knowledge from Noisy Teachers.  Sau, Bharat Bhusan et al. arXiv:1610.09650v2 
5. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Tarvainen, Antti and Valpola, Harri. NIPS 2017
6. Born-Again Neural Networks. Furlanello, Tommaso et al. ICML 2018
   <!-- * 教师网络和学生网络具有同样结果，多个网络交替依次训练，最终结果进行平均融合 -->
7. Deep Mutual Learning. Zhang, Ying et al. CVPR 2018
   <!-- * 多个学生模型之间同时互相学习 -->
8. Knowledge distillation by on-the-fly native ensemble. Lan, Xu et al. NIPS 2018
9. Collaborative learning for deep neural networks. Song, Guocong and Chai, Wei. NIPS 2018
10. Data Distillation: Towards Omni-Supervised Learning. Radosavovic, Ilija et al. CVPR 2018
11. Multilingual Neural Machine Translation with Knowledge Distillation. ICLR 2019
    <!-- * 多个teacher（一个teacher一个语言对），简单融合。 -->
12. Unifying Heterogeneous Classifiers with Distillation. Vongkulbhisal et al. CVPR 2019
    <!-- * 有多个不完全一样场景下的分类器（分类目标不尽相同），如何将它们统一起来，构造一个总的分类器：认为每一个classifier同等重要，都要去拟合它们。 -->
13. Distilled Person Re-Identification: Towards a More Scalable System. Wu, Ancong et al. CVPR 2019
    <!-- * 知识不再是soft-label，而是similarity matrix。权重引入是teacher-level的，而不是instance-level的。 -->
14. Diversity with Cooperation: Ensemble Methods for Few-Shot Classification. Dvornik, Nikita et al. ICCV 2019
15. Model Compression with Two-stage Multi-teacher Knowledge Distillation for Web Question Answering System. Yang, Ze et al. WSDM 2020 
16. FEED: Feature-level Ensemble for Knowledge Distillation. Park, SeongUk and Kwak, Nojun. arXiv:1909.10754(AAAI20 pre)
17. Stochasticity and Skip Connection Improve Knowledge Transfer. Lee, Kwangjin et al. ICLR 2020
18. Online Knowledge Distillation with Diverse Peers. Chen, Defang et al. AAAI 2020
19. Hydra: Preserving Ensemble Diversity for Model Distillation. Tran, Linh et al. arXiv:2001.04694
20. Distilled Hierarchical Neural Ensembles with Adaptive Inference Cost. Ruiz, Adria et al. arXv:2003.01474

### Knowledge Amalgamation（KA) - zju-VIPA

[VIPA - KA][13.1]

1. Amalgamating Knowledge towards Comprehensive Classification. Shen, Chengchao et al. AAAI 2019
2. Amalgamating Filtered Knowledge : Learning Task-customized Student from Multi-task Teachers. Ye, Jingwen et al. IJCAI 2019
3. Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning. Luo, Sihui et al. IJCAI 2019
4. Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More. Ye, Jingwen et al. CVPR 2019
5. Customizing Student Networks From Heterogeneous Teachers via Adaptive Knowledge Amalgamation. ICCV 2019
6. Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN. CVPR 2020

## Cross-modal KD & DA

1. SoundNet: Learning Sound Representations from Unlabeled Video SoundNet Architecture. Aytar, Yusuf et al. ECCV 2016
2. Cross Modal Distillation for Supervision Transfer. Gupta, Saurabh et al. CVPR 2016
3. Emotion recognition in speech using cross-modal transfer in the wild. Albanie, Samuel et al. ACM MM 2018
4. Through-Wall Human Pose Estimation Using Radio Signals. Zhao, Mingmin et al. CVPR 2018
5. Compact Trilinear Interaction for Visual Question Answering. Do, Tuong et al. ICCV 2019
6. Cross-Modal Knowledge Distillation for Action Recognition. Thoker, Fida Mohammad and Gall, Juerge. ICIP 2019
7. Learning to Map Nearly Anything. Salem, Tawfiq et al. arXiv:1909.06928
8. Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval. Liu, Qing et al. ICCV 2019
9. UM-Adapt: Unsupervised Multi-Task Adaptation Using Adversarial Cross-Task Distillation. Kundu et al. ICCV 2019
10. CrDoCo: Pixel-level Domain Transfer with Cross-Domain Consistency. Chen, Yun-Chun et al. CVPR 2019
11. XD:Cross lingual Knowledge Distillation for Polyglot Sentence Embeddings. ICLR 2020
12. Effective Domain Knowledge Transfer with Soft Fine-tuning. Zhao, Zhichen et al. arXiv:1909.02236
13. ASR is all you need: cross-modal distillation for lip reading. Afouras et al. arXiv:1911.12747v1
14. Knowledge distillation for semi-supervised _domain adaptation_. arXiv:1908.07355
15. _Domain Adaptation_ via Teacher-Student Learning for End-to-End _Speech Recognition_. Meng, Zhong et al. arXiv:2001.01798
16. Cluster Alignment with a Teacher for Unsupervised _Domain Adaptation_. ICCV 2019
17. Attention Bridging Network for Knowledge Transfer. Li, Kunpeng et al. ICCV 2019
18. Unpaired Multi-modal Segmentation via Knowledge Distillation. Dou, Qi et al. arXiv:2001.03111
19. Multi-source Distilling Domain Adaptation. Zhao, Sicheng et al. arXiv:1911.11554

## Application of KD

1. Face model compression by distilling knowledge from neurons. Luo, Ping et al. AAAI 2016
2. Learning efficient object detection models with knowledge distillation. Chen, Guobin et al. NIPS 2017
3. Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy. Mishra, Asit et al. NIPS 2018
4. Distilled Person _Re-identification_: Towars a More Scalable System. Wu, Ancong et al. CVPR 2019
5. [noval]Efficient _Video Classification_ Using Fewer Frames. Bhardwaj, Shweta et al. CVPR 2019
6. Fast Human _Pose Estimation_. Zhang, Feng et al. CVPR 2019
7. Distilling knowledge from a deep _pose_ regressor network. Saputra et al. arXiv:1908.00858 (2019)
8. Learning Lightweight _Lane Detection_ CNNs by Self Attention Distillation. Hou, Yuenan et al. ICCV 2019
9. Structured Knowledge Distillation for _Semantic Segmentation_. Liu, Yifan et al. CVPR 2019
10. Relation Distillation Networks for _Video Object Detection_. Deng, Jiajun et al. ICCV 2019
11. Teacher Supervises Students How to Learn From Partially Labeled Images for _Facial Landmark Detection_. Dong, Xuanyi and Yang, Yi. ICCV 2019
12. Progressive Teacher-student Learning for Early _Action Prediction_. Wang, Xionghui et al. CVPR2019
13. Lightweight Image _Super-Resolution_ with Information Multi-distillation Network. Hui, Zheng et al. ICCVW 2019
14. AWSD:Adaptive Weighted Spatiotemporal Distillation for _Video Representation_. Tavakolian, Mohammad et al. ICCV 2019
15. Dynamic Kernel Distillation for Efficient _Pose Estimation_ in Videos. Nie, Xuecheng et al. ICCV 2019
16. Teacher Guided _Architecture Search_. Bashivan, Pouya and Tensen, Mark. ICCV 2019
17. Online Model Distillation for Efficient _Video Inference_. Mullapudi et al. ICCV 2019
18. Distilling _Object Detectors_ with Fine-grained Feature Imitation. Wang, Tao et al. CVPR2019
19. Relation Distillation Networks for _Video Object Detection_. Deng, Jiajun et al. ICCV 2019
20. Knowledge Distillation for Incremental Learning in _Semantic Segmentation_. arXiv:1911.03462
21. MOD: A Deep Mixture Model with Online Knowledge Distillation for Large Scale Video Temporal Concept Localization. arXiv:1910.12295
22. Teacher-Students Knowledge Distillation for _Siamese Trackers_. arXiv:1907.10586
23. LaTeS: Latent Space Distillation for Teacher-Student _Driving_ Policy Learning. Zhao, Albert et al. CVPR 2020(pre)
24. Knowledge Distillation for _Brain Tumor Segmentation_. arXiv:2002.03688
25. ROAD: Reality Oriented Adaptation for _Semantic Segmentation_ of Urban Scenes. Chen, Yuhua et al. CVPR 2018
26. Next Point-of-Interest _Recommendation_ on Resource-Constrained Mobile Devices. WWW 2020
27. Multi-Representation Knowledge Distillation For Audio Classification. Gao, Liang et al. arXiv:2002.09607
28. Collaborative Distillation for Ultra-Resolution Universal Style Transfer. Wang, Huan et al. CVPR 2020

### for NLP
1. Patient Knowledge Distillation for BERT Model Compression. Sun, Siqi et al. arXiv:1908.09355
2. TinyBERT: Distilling BERT for Natural Language Understanding. Jiao, Xiaoqi et al. arXiv:1909.10351
3. Learning to Specialize with Knowledge Distillation for Visual Question Answering. NIPS 2018
4. Knowledge Distillation for Bilingual Dictionary Induction. EMNLP 2017
5. A Teacher-Student Framework for Maintainable Dialog Manager. EMNLP 2018
6. Understanding Knowledge Distillation in Non-Autoregressive Machine Translation. arxiv 2019
7. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. Sanh, Victor et al. arXiv:1910.01108
8. Well-Read Students Learn Better: On the Importance of Pre-training Compact Models. Turc, Iulia et al. arXiv:1908.08962
9. On Knowledge distillation from complex networks for response prediction. Arora, Siddhartha et al. NAACL 2019
10. Distilling the Knowledge of BERT for Text Generation. arXiv:1911.03829v1
11. Understanding Knowledge Distillation in Non-autoregressive Machine Translation. arXiv:1911.02727
12. MobileBERT: Task-Agnostic Compression of BERT by Progressive Knowledge Transfer. ICLR 2020
13. Acquiring Knowledge from Pre-trained Model to Neural Machine Translation. Weng, Rongxiang et al. AAAI 2020
14. TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval. Lu, Wenhao et al. KDD 2020
15. Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation. Xu, Yige et al. arXiv:2002.10345

## Model Pruning or Quantization
1. Accelerating Convolutional Neural Networks with Dominant Convolutional Kernel and Knowledge Pre-regression. ECCV 2016
2. N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning. Ashok, Anubhav et al. ICLR 2018
3. Slimmable Neural Networks. Yu, Jiahui et al. ICLR 2018
4. Co-Evolutionary Compression for Unpaired Image Translation. Shu, Han et al. ICCV 2019
5. MetaPruning: Meta Learning for Automatic Neural Network Channel Pruning. Liu, Zechun et al. ICCV 2019
6. LightPAFF: A Two-Stage Distillation Framework for Pre-training and Fine-tuning. ICLR 2020
7. Pruning with hints: an efficient framework for model acceleration. ICLR 2020
8. Training convolutional neural networks with cheap convolutions and online distillation. arXiv:1909.13063
9. Cooperative Pruning in Cross-Domain Deep Neural Network Compression. [Chen, Shangyu][17.9] et al. IJCAI 2019
10. QKD: Quantization-aware Knowledge Distillation. Kim, Jangho et al. arXiv:1911.12491v1

## Beyond

1. Do deep nets really need to be deep?. Ba,Jimmy, and Rich Caruana. NIPS 2014
2. When Does Label Smoothing Help? Müller, Rafael, Kornblith, and Hinton. NIPS 2019
3. Towards Understanding Knowledge Distillation. Phuong, Mary and Lampert, Christoph. AAAI 2019
4. Harnessing deep neural networks with logucal rules. ACL 2016
   <!-- * 融合先验知识 -->
5. Adaptive Regularization of Labels. Ding, Qianggang et al. arXiv:1908.05474
6. Knowledge Isomorphism between Neural Networks. Liang, Ruofan et al. arXiv:1908.01581
7. Role-Wise Data Augmentation for Knowledge Distillation. ICLR 2020
8. [Neural Network Distiller][18.8]: A Python Package For DNN Compression Research. arXiv:1910.12232
9. (survey)Modeling Teacher-Student Techniques in Deep Neural Networks for Knowledge Distillation. arXiv:1912.13179
10. Understanding and Improving Knowledge Distillation. Tang, Jiaxi et al. arXiv:2002.03532
11. The State of Knowledge Distillation for Classification. Ruffy, Fabian and Chahal, Karanbir. arXiv:1912.10850 [[code]][18.11]
12. [TextBrewer][18.12]: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing. HIT and iFLYTEK. arXiv:2002.12620
13. Explaining Knowledge Distillation by Quantifying the Knowledge. [Zhang, Quanshi][18.13] et al. aiXiv:2003.03622
14. DeepVID: deep visual interpretation and diagnosis for image classifiers via knowledge distillation. IEEE Trans, 2019.



---
Note: All papers pdf can be found and downloaded on [Bing](https://www.bing.com) or [Google](https://www.google.com).

Source: <https://github.com/FLHonker/Awesome-Knowledge-Distillation>
<!--
Thanks for all contributors:

[![shivmgg](https://avatars0.githubusercontent.com/u/21128481?s=28&v=4)](https://github.com/shivmgg)
-->

Contact: Yuang Liu(<frankliu624@outlook.com>), AIDA, [ECNU](https://www.ecnu.edu.cn/).


[2.27]: https://github.com/JetRunner/BERT-of-Theseus
[8.20]: https://github.com/mit-han-lab/gan-compression
[10.9]: https://github.com/snudatalab/KegNet
[13.1]: https://github.com/zju-vipa/KamalEngine
[17.9]: https://csyhhu.github.io/
[18.8]: https://github.com/NervanaSystems/distiller
[18.11]: https://github.com/karanchahal/distiller
[18.12]: https://github.com/airaria/TextBrewer
[18.13]: http://qszhang.com/
