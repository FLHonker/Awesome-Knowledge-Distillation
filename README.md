# Awesome Knowledge-Distillation

![](https://img.shields.io/badge/Number-389-green)

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
10. Revisit Knowledge Distillation: a Teacher-free Framework(Revisiting Knowledge Distillation via Label Smoothing Regularization). Yuan, Li et al. CVPR 2020 [[code]][1.10]
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
28. Boosting Self-Supervised Learning via Knowledge Transfer. Noroozi, Mehdi et al. CVPR 2018
29. Meta Pseudo Labels. Pham, Hieu et al. ICML 2020
30. Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model. CVPR 2020 [[code]][1.30]
31. Distilled Binary Neural Network for Monaural Speech Separation. Chen Xiuyi et al. IJCNN 2018
32. Teacher-Class Network: A Neural Network Compression Mechanism. Malik et al. arXiv:2004.03281
33. Deeply-supervised knowledge synergy. Sun, Dawei et al. CVPR 2019
34. What it Thinks is Important is Important: Robustness Transfers through Input Gradients. Chan, Alvin et al. CVPR 2020
35. Triplet Loss for Knowledge Distillation. Oki, Hideki et al. IJCNN 2020
36. Role-Wise Data Augmentation for Knowledge Distillation. ICLR 2020 [[code]][1.36]
37. Distilling Spikes: Knowledge Distillation in Spiking Neural Networks. arXiv:2005.00288
38. Improved Noisy Student Training for Automatic Speech Recognition. Park et al.arXiv:2005.09629
39. Learning from a Lightweight Teacher for Efficient Knowledge Distillation. Yuang Liu et al. arXiv:2005.09163
40. ResKD: Residual-Guided Knowledge Distillation. Li, Xuewei et al. arXiv:2006.04719
41. Distilling Effective Supervision from Severe Label Noise. Zhang, Zizhao. et al. CVPR 2020 [[code]][1.41]
42. Knowledge Distillation Meets Self-Supervision. Xu, Guodong et al. ECCV 2020 [[code]][1.42]
43. Self-supervised Knowledge Distillation for Few-shot Learning. arXiv:2006.09785 [[code]][1.43]
44. Learning with Noisy Class Labels for Instance Segmentation. ECCV 2020
45. Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation. Wang, Liwei et al. arXiv:2007.01951

### Knowledge from intermediate layers

1. Fitnets: Hints for thin deep nets. Romero, Adriana et al. arXiv:1412.6550
2. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. Zagoruyko et al. ICLR 2017
3. Knowledge Projection for Effective Design of Thinner and Faster Deep Neural Networks. Zhang, Zhi et al. arXiv:1710.09505
4. A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning. Yim, Junho et al. CVPR 2017
5. Paraphrasing complex network: Network compression via factor transfer. Kim, Jangho et al. NIPS 2018
6. Knowledge transfer with jacobian matching. ICML 2018
7. Self-supervised knowledge distillation using singular value decomposition. Lee, Seung Hyun et al. ECCV 2018
8. Learning Deep Representations with Probabilistic Knowledge Transfer. Passalis et al. ECCV 2018
9. Variational Information Distillation for Knowledge Transfer. Ahn, Sungsoo et al. CVPR 2019
10. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019
11. Knowledge Distillation via Route Constrained Optimization. Jin, Xiao et al. ICCV 2019
12. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
13. MEAL: Multi-Model Ensemble via Adversarial Learning. Shen,Zhiqiang, He,Zhankui, and Xue Xiangyang. AAAI 2019
14. A Comprehensive Overhaul of Feature Distillation. Heo, Byeongho et al. ICCV 2019
15. Feature-map-level Online Adversarial Knowledge Distillation. ICML 2020
16. Distilling Object Detectors with Fine-grained Feature Imitation. ICLR 2020
17. Knowledge Squeezed Adversarial Network Compression. Changyong, Shu et al. AAAI 2020
18. Stagewise Knowledge Distillation. Kulkarni, Akshay et al. arXiv: 1911.06786
19. Knowledge Distillation from Internal Representations. AAAI 2020
20. Knowledge Flow:Improve Upon Your Teachers. ICLR 2019
21. LIT: Learned Intermediate Representation Training for Model Compression. ICML 2019
22. Improving the Adversarial Robustness of Transfer Learning via Noisy Feature Distillation. Chin, Ting-wu et al. arXiv:2002.02998
23. Knapsack Pruning with Inner Distillation. Aflalo, Yonathan et al. arXiv:2002.08258
24. Residual Knowledge Distillation. Gao, Mengya et al. arXiv:2002.09168
25. Knowledge distillation via adaptive instance normalization. Yang, Jing et al. arXiv:2003.04289
26. Bert-of-Theseus: Compressing bert by progressive module replacing. Xu, Canwen et al. arXiv:2002.02925 [[code]][2.27]
27. Distilling Spikes: Knowledge Distillation in Spiking Neural Networks. arXiv:2005.00727
28. Generalized Bayesian Posterior Expectation Distillation for Deep Neural Networks. Meet et al. arXiv:2005.08110
29. Feature-map-level Online Adversarial Knowledge Distillation. Chung, Inseop et al. ICML 2020
30. Channel Distillation: Channel-Wise Attention for Knowledge Distillation. Zhou, Zaida et al. arXiv:2006.01683 [[code]][2.30]
31. Matching Guided Distillation. ECCV 2020
32. Differentiable Feature Aggregation Search for Knowledge Distillation. ECCV 2020
33. Interactive Knowledge Distillation. Fu, Shipeng et al. arXiv:2007.01476

### Graph-based

1. Graph-based Knowledge Distillation by Multi-head Attention Network. Lee, Seunghyun and Song, Byung. Cheol arXiv:1907.02226
2. Graph Representation Learning via Multi-task Knowledge Distillation. arXiv:1911.05700
3. Deep geometric knowledge distillation with graphs. arXiv:1911.03080
4. Better and faster: Knowledge transfer from multiple self-supervised learning tasks via graph distillation for video classification. IJCAI 2018
5. Distillating Knowledge from Graph Convolutional Networks. Yang, Yiding et al. CVPR 2020

### Mutual Information

1. Correlation Congruence for Knowledge Distillation. Peng, Baoyun et al. ICCV 2019
2. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
3. Variational Information Distillation for Knowledge Transfer. Ahn, Sungsoo et al. CVPR 2019
4. Contrastive Representation Distillation. Tian, Yonglong et al. ICLR 2020 [[RepDistill]][4.4]
5. Online Knowledge Distillation via Collaborative Learning. Guo, Qiushan et al. CVPR 2020
6. Peer Collaborative Learning for Online Knowledge Distillation. Wu, Guile & Gong, Shaogang. arXiv:2006.04147
7. Online Knowledge Distillation via Collaborative Learning. Guo, Qiushan et al. CVPR 2020
8. Knowledge Transfer via Dense Cross-layer Mutual-distillation. ECCV 2020
9. MutualNet: Adaptive ConvNet via Mutual Learning from Network Width and Resolution. Yang, Taojiannan et al. ECCV 2020 [[code]][4.9]

### Self-KD

1. Moonshine:Distilling with Cheap Convolutions. Crowley, Elliot J. et al. NIPS 2018 
2. Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. Zhang, Linfeng et al. ICCV 2019
3. Learning Lightweight Lane Detection CNNs by Self Attention Distillation. Hou, Yuenan et al. ICCV 2019
4. BAM! Born-Again Multi-Task Networks for Natural Language Understanding. Clark, Kevin et al. ACL 2019,short
5. Self-Knowledge Distillation in Natural Language Processing. Hahn, Sangchul and Choi, Heeyoul. arXiv:1908.01851
6. Rethinking Data Augmentation: Self-Supervision and Self-Distillation. Lee, Hankook et al. ICLR 2020
7. MSD: Multi-Self-Distillation Learning via Multi-classifiers within Deep Neural Networks. arXiv:1911.09418
8. Self-Distillation Amplifies Regularization in Hilbert Space. Mobahi, Hossein et al. arXiv:2002.05715
9. MINILM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. Wang, Wenhui et al. arXiv:2002.10957
10. Regularizing Class-wise Predictions via Self-knowledge Distillation. CVPR 2020 [[code]][5.11]
11. Self-Distillation as Instance-Specific Label Smoothing. Zhang, Zhilu & Sabuncu, Mert R. arXiv:2006.05065

### Structured Knowledge

1. Paraphrasing Complex Network:Network Compression via Factor Transfer. Kim, Jangho et al. NIPS 2018
2. Relational Knowledge Distillation.  Park, Wonpyo et al. CVPR 2019
3. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019
4. Contrastive Representation Distillation. Tian, Yonglong et al. ICLR 2020
5. Teaching To Teach By Structured Dark Knowledge. ICLR 2020
6. Inter-Region Affinity Distillation for Road Marking Segmentation. Hou, Yuenan et al. CVPR 2020 [[code]][6.6]
7. Heterogeneous Knowledge Distillation using Information Flow Modeling. Passalis et al. CVPR 2020 [[code]][6.7]
8. Asymmetric metric learning for knowledge transfer. Budnik, Mateusz & Avrithis, Yannis. arXiv:2006.16331
9. Local Correlation Consistency for Knowledge Distillation. 2020
10. Few-Shot Class-Incremental Learning. Tao, Xiaoyu et al. CVPR 2020

### Privileged Information

1. Learning using privileged information: similarity control and knowledge transfer. Vapnik, Vladimir and Rauf, Izmailov. MLR 2015  
2. Unifying distillation and privileged information. Lopez-Paz, David et al. ICLR 2016
3. Model compression via distillation and quantization. Polino, Antonio et al. ICLR 2018
4. KDGAN:Knowledge Distillation with Generative Adversarial Networks. Wang, Xiaojie. NIPS 2018
5. Efficient Video Classification Using Fewer Frames. Bhardwaj, Shweta et al. CVPR 2019
6. Retaining privileged information for multi-task learning. Tang, Fengyi et al. KDD 2019
7. A Generalized Meta-loss function for regression and classification using privileged information. Asif, Amina et al. arXiv:1811.06885
8. Private Knowledge Transfer via Model Distillation with Generative Adversarial Networks. Gao, Di & Zhuo, Cheng. AAAI 2020

## KD + GAN

1. Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks. Xu, Zheng et al. arXiv:1709.00513
2. KTAN: Knowledge Transfer Adversarial Network. Liu, Peiye et al. arXiv:1810.08126
3. KDGAN:Knowledge Distillation with Generative Adversarial Networks. Wang, Xiaojie. NIPS 2018
4. Adversarial Learning of Portable Student Networks. Wang, Yunhe et al. AAAI 2018
5. Adversarial Network Compression. Belagiannis, Vasileios et al. ECCV 2018
6. Cross-Modality Distillation: A case for Conditional Generative Adversarial Networks. ICASSP 2018
7. Adversarial Distillation for Efficient Recommendation with External Knowledge. TOIS 2018
8. Training student networks for acceleration with conditional adversarial networks. Xu, Zheng et al. BMVC 2018
9. DAFL:Data-Free Learning of Student Networks. Chen, Hanting et al. ICCV 2019
10. MEAL: Multi-Model Ensemble via Adversarial Learning. Shen,Zhiqiang, He,Zhankui, and Xue Xiangyang. AAAI 2019
11. Knowledge Distillation with Adversarial Samples Supporting Decision Boundary. Heo, Byeongho et al. AAAI 2019
12. Exploiting the Ground-Truth: An Adversarial Imitation Based Knowledge Distillation Approach for Event Detection. Liu, Jian et al. AAAI 2019
13. Adversarially Robust Distillation. Goldblum, Micah et al. AAAI 2020
14. GAN-Knowledge Distillation for one-stage Object Detection. Hong, Wei et al. arXiv:1906.08467
15. Lifelong GAN: Continual Learning for Conditional Image Generation. Kundu et al. arXiv:1908.03884
16. Compressing GANs using Knowledge Distillation. Aguinaldo, Angeline et al. arXiv:1902.00159
17. Feature-map-level Online Adversarial Knowledge Distillation. ICML 2020
18. MineGAN: effective knowledge transfer from GANs to target domains with few images. Wang, Yaxing et al. CVPR 2020
19. Distilling portable Generative Adversarial Networks for Image Translation. Chen, Hanting et al. AAAI 2020
20. GAN Compression: Efficient Architectures for Interactive Conditional GANs. Junyan Zhu et al. CVPR 2020 [[code]][8.20]
21. Adversarial network compression. Belagiannis et al. ECCV 2018

## KD + Meta-learning

1. Few Sample Knowledge Distillation for Efficient Network Compression. Li, Tianhong et al. CVPR 2020
2. Learning What and Where to Transfer. Jang, Yunhun et al, ICML 2019
3. Transferring Knowledge across Learning Processes. Moreno, Pablo G et al. ICLR 2019
4. Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval. Liu, Qing et al. ICCV 2019
5. Diversity with Cooperation: Ensemble Methods for Few-Shot Classification. Dvornik, Nikita et al. ICCV 2019
6. Knowledge Representing: Efficient, Sparse Representation of Prior Knowledge for Knowledge Distillation. arXiv:1911.05329v1
7. Progressive Knowledge Distillation For Generative Modeling. ICLR 2020
8. Few Shot Network Compression via Cross Distillation. AAAI 2020
9. MetaDistiller: Network Self-boosting via Meta-learned Top-down Distillation. ECCV 2020

## Data-free KD

1. Data-Free Knowledge Distillation for Deep Neural Networks. NIPS 2017
2. Zero-Shot Knowledge Distillation in Deep Networks. ICML 2019
3. DAFL:Data-Free Learning of Student Networks. ICCV 2019
4. Zero-shot Knowledge Transfer via Adversarial Belief Matching. Micaelli, Paul and Storkey, Amos. NIPS 2019
5. Dream Distillation: A Data-Independent Model Compression Framework. Kartikeya et al. ICML 2019
6. Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion. Yin, Hongxu et al. CVPR 2020
7. Data-Free Adversarial Distillation. Fang, Gongfan et al. CVPR 2020
8. The Knowledge Within: Methods for Data-Free Model Compression. Haroush, Matan et al. CVPR 2020
9. Knowledge Extraction with No Observable Data. Yoo, Jaemin et al. NIPS 2019 [[code]][10.9]
10. Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN. CVPR 2020
11. DeGAN : Data-Enriching GAN for Retrieving Representative Samples from a Trained Classifier. Addepalli, Sravanti et al. arXiv:1912.11960
12. Generative Low-bitwidth Data Free Quantization. Xu, Shoukai et al. arXiv:2003.03603
13. This dataset does not exist: training models from generated images. arXiv:1911.02888
14. MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation. Sanjay et al. arXiv:2005.03161
15. Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data. Such et al. ECCV 2020
16. Billion-scale semi-supervised learning for image classification. FAIR. arXiv:1905.00546 [[code]][10.16]

- other data-free model compression:

17. Data-free Parameter Pruning for Deep Neural Networks. Srinivas, Suraj et al. arXiv:1507.06149
18. Data-Free Quantization Through Weight Equalization and Bias Correction. Nagel, Markus et al. ICCV 2019
19. DAC: Data-free Automatic Acceleration of Convolutional Networks. Li, Xin et al. WACV 2019

## KD + AutoML

1. Improving Neural Architecture Search Image Classifiers via Ensemble Learning. Macko, Vladimir et al. arXiv:1903.06236
2. Blockwisely Supervised Neural Architecture Search with Knowledge Distillation. Li, Changlin et al. arXiv:1911.13053v1
3. Towards Oracle Knowledge Distillation with Neural Architecture Search. Kang, Minsoo et al. AAAI 2020
4. Search for Better Students to Learn Distilled Knowledge. Gu, Jindong & Tresp, Volker arXiv:2001.11612
5. Circumventing Outliers of AutoAugment with Knowledge Distillation. Wei, Longhui et al. arXiv:2003.11342
6. Network Pruning via Transformable Architecture Search. Dong, Xuanyi & Yang, Yi. NIPS 2019
7. Search to Distill: Pearls are Everywhere but not the Eyes. Liu Yu et al. CVPR 2020
8. AutoGAN-Distiller: Searching to Compress Generative Adversarial Networks. Fu, Yonggan et al. ICML 2020 [[code]][11.8]

## KD + RL

1. N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning. Ashok, Anubhav et al. ICLR 2018
2. Knowledge Flow:Improve Upon Your Teachers. Liu, Iou-jen et al. ICLR 2019
3. Transferring Knowledge across Learning Processes. Moreno, Pablo G et al. ICLR 2019
4. Exploration by random network distillation. Burda, Yuri et al. ICLR 2019
5. Periodic Intra-Ensemble Knowledge Distillation for Reinforcement Learning. Hong, Zhang-Wei et al. arXiv:2002.00149
6. Transfer Heterogeneous Knowledge Among Peer-to-Peer Teammates: A Model Distillation Approach. Xue, Zeyue et al. arXiv:2002.02202
7. Proxy Experience Replay: Federated Distillation for Distributed Reinforcement Learning. Cha, han et al. arXiv:2005.06105
8. Dual Policy Distillation. Lai, Kwei-Herng et al. arXiv:2006.04061

## Multi-teacher KD 

1. Learning from Multiple Teacher Networks. You, Shan et al. KDD 2017
2. Semi-Supervised Knowledge Transfer for Deep Learning from Private Training Data. ICLR 2017
3. Knowledge Adaptation: Teaching to Adapt. Arxiv:1702.02052
4. Deep Model Compression: Distilling Knowledge from Noisy Teachers.  Sau, Bharat Bhusan et al. arXiv:1610.09650v2 
5. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Tarvainen, Antti and Valpola, Harri. NIPS 2017
6. Born-Again Neural Networks. Furlanello, Tommaso et al. ICML 2018
7. Deep Mutual Learning. Zhang, Ying et al. CVPR 2018
8. Knowledge distillation by on-the-fly native ensemble. Lan, Xu et al. NIPS 2018
9. Collaborative learning for deep neural networks. Song, Guocong and Chai, Wei. NIPS 2018
10. Data Distillation: Towards Omni-Supervised Learning. Radosavovic, Ilija et al. CVPR 2018
11. Multilingual Neural Machine Translation with Knowledge Distillation. ICLR 2019
12. Unifying Heterogeneous Classifiers with Distillation. Vongkulbhisal et al. CVPR 2019
13. Distilled Person Re-Identification: Towards a More Scalable System. Wu, Ancong et al. CVPR 2019
14. Diversity with Cooperation: Ensemble Methods for Few-Shot Classification. Dvornik, Nikita et al. ICCV 2019
15. Model Compression with Two-stage Multi-teacher Knowledge Distillation for Web Question Answering System. Yang, Ze et al. WSDM 2020 
16. FEED: Feature-level Ensemble for Knowledge Distillation. Park, SeongUk and Kwak, Nojun. AAAI 2020
17. Stochasticity and Skip Connection Improve Knowledge Transfer. Lee, Kwangjin et al. ICLR 2020
18. Online Knowledge Distillation with Diverse Peers. Chen, Defang et al. AAAI 2020
19. Hydra: Preserving Ensemble Diversity for Model Distillation. Tran, Linh et al. arXiv:2001.04694
20. Distilled Hierarchical Neural Ensembles with Adaptive Inference Cost. Ruiz, Adria et al. arXv:2003.01474
21. Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition. Gao, Yan et al. arXiv:2005.09310
22. Adaptive multi-teacher multi-level knowledge distillation. Yuang Liu et al. Neurocomputing, 2020
23. Large-Scale Few-Shot Learning via Multi-Modal Knowledge Discovery. ECCV 2020
24. Collaborative Learning for Faster StyleGAN Embedding. Guan, Shanyan et al. arXiv:2007.01758
25. Temporal Self-Ensembling Teacher for Semi-Supervised Object Detection. Chen, Cong et al. IEEE 2020 [[code]][12.25]
26. Dual-Teacher: Integrating Intra-domain and Inter-domain Teachers for Annotation-efficient Cardiac Segmentation. MICCAI 2020

### Knowledge Amalgamation（KA) - zju-VIPA

[VIPA - KA][13.24]

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
20. Creating Something from Nothing: Unsupervised Knowledge Distillation for Cross-Modal Hashing. Hu, Hengtong et al. CVPR 2020
21. Improving Semantic Segmentation via Self-Training. Zhu, Yi et al. arXiv:2004.14960
22. Speech to Text Adaptation: Towards an Efficient Cross-Modal Distillation. arXiv:2005.08213
23. Joint Progressive Knowledge Distillation and Unsupervised Domain Adaptation. arXiv:2005.07839
24. Knowledge as Priors: Cross-Modal Knowledge Generalization for Datasets without Superior Knowledge. Zhao, Long et al. CVPR 2020
25. Large-Scale Domain Adaptation via Teacher-Student Learning. Li, Jinyu et al. arXiv:1708.05466
26. Large Scale Audiovisual Learning of Sounds with Weakly Labeled Data. Fayek, Haytham M. & Kumar, Anurag. IJCAI 2020
27. Distilling Cross-Task Knowledge via Relationship Matching. Ye, Han-Jia. et al. CVPR 2020 [[code]][14.27]
28. Modality distillation with multiple stream networks for action recognition. Garcia, Nuno C. et al. ECCV 2018
29. Domain Adaptation through Task Distillation. ECCV 2020

## Application of KD

1. Face model compression by distilling knowledge from neurons. Luo, Ping et al. AAAI 2016
2. Learning efficient object detection models with knowledge distillation. Chen, Guobin et al. NIPS 2017
3. Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy. Mishra, Asit et al. NIPS 2018
4. Distilled Person _Re-identification_: Towars a More Scalable System. Wu, Ancong et al. CVPR 2019
5. Efficient _Video Classification_ Using Fewer Frames. Bhardwaj, Shweta et al. CVPR 2019
6. Fast Human _Pose Estimation_. Zhang, Feng et al. CVPR 2019
7. Distilling knowledge from a deep _pose_ regressor network. Saputra et al. arXiv:1908.00858 (2019)
8. Learning Lightweight _Lane Detection_ CNNs by Self Attention Distillation. Hou, Yuenan et al. ICCV 2019
9. Structured Knowledge Distillation for _Semantic Segmentation_. Liu, Yifan et al. CVPR 2019
10. Relation Distillation Networks for _Video Object Detection_. Deng, Jiajun et al. ICCV 2019
11. Teacher Supervises Students How to Learn From Partially Labeled Images for _Facial Landmark Detection_. Dong, Xuanyi and Yang, Yi. ICCV 2019
12. Progressive Teacher-student Learning for Early _Action Prediction_. Wang, Xionghui et al. CVPR 2019
13. Lightweight Image _Super-Resolution_ with Information Multi-distillation Network. Hui, Zheng et al. ICCVW 2019
14. AWSD:Adaptive Weighted Spatiotemporal Distillation for _Video Representation_. Tavakolian, Mohammad et al. ICCV 2019
15. Dynamic Kernel Distillation for Efficient _Pose Estimation_ in Videos. Nie, Xuecheng et al. ICCV 2019
16. Teacher Guided _Architecture Search_. Bashivan, Pouya and Tensen, Mark. ICCV 2019
17. Online Model Distillation for Efficient _Video Inference_. Mullapudi et al. ICCV 2019
18. Distilling _Object Detectors_ with Fine-grained Feature Imitation. Wang, Tao et al. CVPR 2019
19. Relation Distillation Networks for _Video Object Detection_. Deng, Jiajun et al. ICCV 2019
20. Knowledge Distillation for Incremental Learning in _Semantic Segmentation_. arXiv:1911.03462
21. MOD: A Deep Mixture Model with Online Knowledge Distillation for Large Scale Video Temporal Concept Localization. arXiv:1910.12295
22. Teacher-Students Knowledge Distillation for _Siamese Trackers_. arXiv:1907.10586
23. LaTeS: Latent Space Distillation for Teacher-Student _Driving_ Policy Learning. Zhao, Albert et al. CVPR 2020(pre)
24. Knowledge Distillation for _Brain Tumor Segmentation_. arXiv:2002.03688
25. ROAD: Reality Oriented Adaptation for _Semantic Segmentation_ of Urban Scenes. Chen, Yuhua et al. CVPR 2018
26. Next Point-of-Interest _Recommendation_ on Resource-Constrained Mobile Devices. WWW 2020
27. Multi-Representation Knowledge Distillation For Audio Classification. Gao, Liang et al. arXiv:2002.09607
28. Collaborative Distillation for Ultra-Resolution Universal _Style Transfer_. Wang, Huan et al. CVPR 2020 [[code]][15.28]
29. ShadowTutor: Distributed Partial Distillation for Mobile _Video_ DNN Inference. Chung, Jae-Won et al. ICPP 2020 [[code]][15.29]
30. Object Relational Graph with Teacher-Recommended Learning for _Video Captioning_. Zhang, Ziqi et al. CVPR 2020
31. Spatio-Temporal Graph for _Video Captioning_ with Knowledge distillation. CVPR 2020 [[code]][15.31]
32. Squeezed Deep _6DoF Object Detection_ Using Knowledge Distillation. Felix, Heitor et al. arXiv:2003.13586
33. Distilled Semantics for Comprehensive _Scene Understanding_ from Videos. Tosi, Fabio et al. arXiv:2003.14030
34. Parallel WaveNet: Fast high-fidelity _speech synthesis_. Van et al. ICML 2018
35. Distill Knowledge From NRSfM for Weakly Supervised _3D Pose_ Learning. Wang Chaoyang et al. ICCV 2019
36. KD-MRI: A knowledge distillation framework for _image reconstruction_ and image restoration in MRI workflow. Murugesan et al. MIDL 2020
37. Geometry-Aware Distillation for Indoor _Semantic Segmentation_. Jiao, Jianbo et al. CVPR 2019
38. Teacher Supervises Students How to Learn From Partially Labeled Images for _Facial Landmark Detection_. ICCV 2019
39. Distill Image _Dehazing_ with Heterogeneous Task Imitation. Hong, Ming et al. CVPR 2020
40. Knowledge Distillation for _Action Anticipation_ via Label Smoothing. Camporese et al. arXiv:2004.07711
41. More Grounded _Image Captioning_ by Distilling Image-Text Matching Model. Zhou, Yuanen et al. CVPR 2020
42. Distilling Knowledge from Refinement in Multiple _Instance Detection_ Networks. Zeni, Luis Felipe & Jung, Claudio. arXiv:2004.10943
43. A General Knowledge Distillation Framework for Counterfactual _Recommendation_ via Uniform Data. SIGIR 2020
44. Enabling Incremental Knowledge Transfer for _Object Detection_ at the Edge. arXiv:2004.05746
45. Uninformed Students: Student-Teacher _Anomaly Detection_ with Discriminative Latent Embeddings. Bergmann, Paul et al. CVPR 2020
46. TA-Student _VQA_: Multi-Agents Training by Self-Questioning. Xiong, Peixi & Wu Ying. CVPR 2020
47. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. Jiang, Lu et al. ICML 2018
48. A Multi-Task Mean Teacher for Semi-Supervised _Shadow Detection_. Chen, Zhihao et al. CVPR 2020 [[code]][15.48]
49. Learning Lightweight _Face Detector_ with Knowledge Distillation. Zhang Shifeng et al. IEEE 2019
50. Learning Lightweight _Pedestrian Detector_ with Hierarchical Knowledge Distillation. ICIP 2019
51. Distilling _Object Detectors_ with Task Adaptive Regularization. Sun, Ruoyu et al. arXiv:2006.13108
52. Privileged Features Distillation at Taobao _Recommendations_. Xu, Chen et al. KDD 2020
53. Intra-class Compactness Distillation for _Semantic Segmentation_. ECCV 2020
54. Boosting Weakly Supervised _Object Detection_ with Progressive Knowledge Transfer. ECCV 2020
55. DOPE: Distillation Of Part Experts for whole-body _3D pose estimation_ in the wild. ECCV 2020
56. Self-similarity Student for Partial Label Histopathology Image _Segmentation_. ECCV 2020
57. Self-Supervised _GAN Compression_. Yu, Chong & Pool, Jeff. arXiv:2007.01491
58. Robust _Re-Identification_ by Multiple Views Knowledge Distillation. Porrello et al. ECCV 2020 [[code]][15.58]
59. LabelEnc: A New Intermediate Supervision Method for _Object Detection_. Hao, Miao et al. arXiv:2007.03282
60. Optical Flow Distillation: Towards Efficient and Stable _Video Style Transfer_. Chen, Xinghao et al. ECCV 2020
61. Adversarial Self-Supervised Learning for Semi-Supervised _3D Action Recognition_. Si, Chenyang et al. ECCV 2020

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
12. MobileBERT: a Compact Task-Agnostic BERT for Resource-Limited Devices. Sun, Zhiqing et al. ACL 2020
13. Acquiring Knowledge from Pre-trained Model to Neural Machine Translation. Weng, Rongxiang et al. AAAI 2020
14. TwinBERT: Distilling Knowledge to Twin-Structured BERT Models for Efficient Retrieval. Lu, Wenhao et al. KDD 2020
15. Improving BERT Fine-Tuning via Self-Ensemble and Self-Distillation. Xu, Yige et al. arXiv:2002.10345
16. FastBERT: a Self-distilling BERT with Adaptive Inference Time. Liu, Weijie et al. ACL 2020
17. LightRec: a Memory and Search-Efficient Recommender System. Lian Defu et al. WWW 2020
18. LadaBERT: Lightweight Adaptation of BERT through Hybrid Model Compression. Mao, Yihuan et al. arXiv:2004.04124
19. DynaBERT: Dynamic BERT with Adaptive Width and Depth. Hou, Lu et al. arXiv:2004.04037
20. Structure-Level Knowledge Distillation For Multilingual Sequence Labeling. Wang, Xinyu et al. ACL 2020
21. Distilled embedding: non-linear embedding factorization using knowledge distillation. Lioutas, Vasileios et al. arXiv:1910.06720
22. TinyMBERT: Multi-Stage Distillation Framework for Massive Multi-lingual NER. Mukherjee & Awadallah. ACL 2020
23. Knowledge Distillation for Multilingual Unsupervised Neural Machine Translation. Sun, Haipeng et al. arXiv:2004.10171
24. Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. Reimers, Nils & Gurevych, Iryna arXiv:2004.09813
25. Distilling Knowledge for Fast Retrieval-based Chat-bots. Tahami et al. arXiv:2004.11045
26. Single-/Multi-Source Cross-Lingual NER via Teacher-Student Learning on Unlabeled Data in Target Language. ACL 2020
27. Local Clustering with Mean Teacher for Semi-supervised Learning. arXiv:2004.09665
28. Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher. arXiv:2004.08780 
29. Syntactic Structure Distillation Pretraining For Bidirectional Encoders. arXiv: 2005.13482
30. Distill, Adapt, Distill: Training Small, In-Domain Models for Neural Machine Translation. arXiv:2003.02877
31. Distilling Neural Networks for Faster and Greener Dependency Parsing. arXiv:2006.00844
32. Distilling Knowledge from Well-informed Soft Labels for Neural Relation Extraction. AAAI 2020 [[code]][16.32]
33. More Grounded Image Captioning by Distilling Image-Text Matching Model. Zhou, Yuanen et al. CVPR 2020
34. Multimodal Learning with Incomplete Modalities by Knowledge Distillation. Wang, Qi et al. KDD 2020

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
11. Neural Network Pruning with Residual-Connections and Limited-Data. Luo, Jian-Hao & Wu, Jianxin. CVPR 2020
12. Training Quantized Neural Networks with a Full-precision Auxiliary Module. Zhuang, Bohan et al. CVPR 2020
13. Towards Effective Low-bitwidth Convolutional Neural Networks. Zhuang, Bohan et al. CVPR 2018
14. Effective Training of Convolutional Neural Networks with Low-bitwidth Weights and Activations. Zhuang, Bohan et al. arXiv:1908.04680
15. Paying more attention to snapshots of Iterative Pruning: Improving Model Compression via Ensemble Distillation. Le et al. arXiv:2006.11487 [[code]][17.15]
16. Knowledge Distillation Beyond Model Compression. Choi, Arthur et al. arxiv:2007.01493
17. Distillation Guided Residual Learning for Binary Convolutional Neural Networks. Ye, Jianming et al. ECCV 20

## Beyond

1. Do deep nets really need to be deep?. Ba,Jimmy, and Rich Caruana. NIPS 2014
2. When Does Label Smoothing Help? Müller, Rafael, Kornblith, and Hinton. NIPS 2019
3. Towards Understanding Knowledge Distillation. Phuong, Mary and Lampert, Christoph. AAAI 2019
4. Harnessing deep neural networks with logical rules. ACL 2016
5. Adaptive Regularization of Labels. Ding, Qianggang et al. arXiv:1908.05474
6. Knowledge Isomorphism between Neural Networks. Liang, Ruofan et al. arXiv:1908.01581
7. [Neural Network Distiller][18.8]: A Python Package For DNN Compression Research. arXiv:1910.12232
8. (survey)Modeling Teacher-Student Techniques in Deep Neural Networks for Knowledge Distillation. arXiv:1912.13179
9. Understanding and Improving Knowledge Distillation. Tang, Jiaxi et al. arXiv:2002.03532
10. The State of Knowledge Distillation for Classification. Ruffy, Fabian and Chahal, Karanbir. arXiv:1912.10850 [[code]][18.11]
11. [TextBrewer][18.12]: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing. HIT and iFLYTEK. arXiv:2002.12620
12. Explaining Knowledge Distillation by Quantifying the Knowledge. [Zhang, Quanshi][18.13] et al. CVPR 2020
13. DeepVID: deep visual interpretation and diagnosis for image classifiers via knowledge distillation. IEEE Trans, 2019.
14. On the Unreasonable Effectiveness of Knowledge Distillation: Analysis in the Kernel Regime. Rahbar, Arman et al. arXiv:2003.13438
15. (survey)Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks. Wang, Lin & Yoon, Kuk-Jin. arXiv:2004.05937
16. Why distillation helps: a statistical perspective. arXiv:2005.10419
17. Transferring Inductive Biases through Knowledge Distillation. Abnar, Samira et al. arXiv:2006.00555
18. Does label smoothing mitigate label noise? Lukasik, Michal et al. ICML 2020
19. An Empirical Analysis of the Impact of Data Augmentation on Knowledge Distillation. Das, Deepan et al. arXiv:2006.03810
20. Knowledge Distillation: A Survey. Gou, Jianping et al. arXiv:2006.05525
21. Does Adversarial Transferability Indicate Knowledge Transferability? Liang, Kaizhao et al. arXiv:2006.14512
22. On the Demystification of Knowledge Distillation: A Residual Network Perspective. Jha et al. arXiv:2006.16589

---
Note: All papers pdf can be found and downloaded on [Bing](https://www.bing.com) or [Google](https://www.google.com).

Source: <https://github.com/FLHonker/Awesome-Knowledge-Distillation>
<!--
Thanks for all contributors:

[![shivmgg](https://avatars0.githubusercontent.com/u/21128481?s=28&v=4)](https://github.com/shivmgg)
-->

Contact:[Yuang Liu](https://flhonker.github.io)(<frankliu624@outlook.com>), [ECNU](https://www.ecnu.edu.cn/).


[1.10]: https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation
[1.30]: https://github.com/dwang181/active-mixup
[1.36]: https://github.com/bigaidream-projects/role-kd
[1.41]: https://github.com/google-research/google-research/tree/master/ieg
[1.42]: https://github.com/xuguodong03/SSKD
[1.43]: https://github.com/brjathu/SKD
[2.27]: https://github.com/JetRunner/BERT-of-Theseus
[2.30]: https://github.com/zhouzaida/channel-distillation
[4.4]: https://github.com/HobbitLong/RepDistiller
[4.9]: https://github.com/taoyang1122/MutualNet
[5.11]: https://github.com/alinlab/cs-kd
[6.6]: https://github.com/cardwing/Codes-for-IntRA-KD
[6.7]: https://github.com/passalis/pkth
[8.20]: https://github.com/mit-han-lab/gan-compression
[10.9]: https://github.com/snudatalab/KegNet
[10.16]: https://github.com/leaderj1001/Billion-scale-semi-supervised-learning
[11.8]: https://github.com/TAMU-VITA/AGD
[12.25]: http://github.com/SYangDong/tse-t
[13.24]: https://github.com/zju-vipa/KamalEngine
[14.27]:  https://github.com/njulus/ReFilled
[15.28]: https://github.com/mingsun-tse/collaborative-distillation
[15.29]: https://github.com/jaywonchung/ShadowTutor
[15.31]: https://github.com/StanfordVL/STGraph
[15.48]: https://github.com/eraserNut/MTMT
[15.58]: https://github.com/aimagelab/VKD
[16.32]: https://github.com/zzysay/KD4NRE
[17.9]: https://csyhhu.github.io/
[17.15]: https://github.com/lehduong/ginp
[18.8]: https://github.com/NervanaSystems/distiller
[18.11]: https://github.com/karanchahal/distiller
[18.12]: https://github.com/airaria/TextBrewer
[18.13]: http://qszhang.com/
