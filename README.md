# Awesome Knowledge-Distillation


![counter](https://img.shields.io/badge/Paper-565-green) 
[![star](https://img.shields.io/github/stars/FLHonker/Awesome-Knowledge-Distillation?label=star&style=social)](https://github.com/FLHonker/Awesome-Knowledge-Distillation)

- [Awesome Knowledge-Distillation](#awesome-knowledge-distillation)
  - [Different forms of knowledge](#different-forms-of-knowledge)
    - [Knowledge from logits](#knowledge-from-logits)
    - [Knowledge from intermediate layers](#knowledge-from-intermediate-layers)
    - [Graph-based](#graph-based)
    - [Mutual Information & Online Learning](#mutual-information--online-learning)
    - [Self-KD](#self-kd)
    - [Structural Knowledge](#structural-knowledge)
    - [Privileged Information](#privileged-information)
  - [KD + GAN](#kd--gan)
  - [KD + Meta-learning](#kd--meta-learning)
  - [Data-free KD](#data-free-kd)
  - [KD + AutoML](#kd--automl)
  - [KD + RL](#kd--rl)
  - [KD + Self-supervised](#kd--self-supervised)
  - [Multi-teacher and Ensemble KD](#multi-teacher-and-ensemble-kd)
    - [Knowledge Amalgamation（KA) - zju-VIPA](#knowledge-amalgamationka---zju-vipa)
  - [Cross-modal KD & DA](#cross-modal-kd--da)
  - [Application of KD](#application-of-kd)
    - [for NLP & Data-Mining](#for-nlp--data-mining)
    - [for RecSys](#for-recsys)
  - [Model Pruning or Quantization](#model-pruning-or-quantization)
  - [Beyond](#beyond)
  - [Distiller Tools](#distiller-tools)

## Different forms of knowledge

### Knowledge from logits

1. Distilling the knowledge in a neural network. Hinton et al. arXiv:1503.02531
2. Learning from Noisy Labels with Distillation. Li, Yuncheng et al. ICCV 2017
3. Training Deep Neural Networks in Generations:A More Tolerant Teacher Educates Better Students. arXiv:1805.05551
4. Learning Metrics from Teachers: Compact Networks for Image Embedding. Yu, Lu et al. CVPR 2019
5. Relational Knowledge Distillation. Park, Wonpyo et al. CVPR 2019
6. On Knowledge Distillation from Complex Networks for Response Prediction. Arora, Siddhartha et al. NAACL 2019
7. On the Efficacy of Knowledge Distillation. Cho, Jang Hyun & Hariharan, Bharath. arXiv:1910.01348. ICCV 2019
8. Revisit Knowledge Distillation: a Teacher-free Framework (Revisiting Knowledge Distillation via Label Smoothing Regularization). Yuan, Li et al. CVPR 2020 [[code]][1.10]
9. Improved Knowledge Distillation via Teacher Assistant: Bridging the Gap Between Student and Teacher. Mirzadeh et al. arXiv:1902.03393
10. Ensemble Distribution Distillation. ICLR 2020
11. Noisy Collaboration in Knowledge Distillation. ICLR 2020
12. On Compressing U-net Using Knowledge Distillation. arXiv:1812.00249
13. Self-training with Noisy Student improves ImageNet classification. Xie, Qizhe et al.(Google) CVPR 2020
14. Variational Student: Learning Compact and Sparser Networks in Knowledge Distillation Framework. AAAI 2020
15. Preparing Lessons: Improve Knowledge Distillation with Better Supervision. arXiv:1911.07471
16. Adaptive Regularization of Labels. arXiv:1908.05474
17. Positive-Unlabeled Compression on the Cloud. Xu, Yixing et al. (HUAWEI) NeurIPS 2019
18. Snapshot Distillation: Teacher-Student Optimization in One Generation. Yang, Chenglin et al. CVPR 2019
19. QUEST: Quantized embedding space for transferring knowledge. Jain, Himalaya et al. arXiv:2020
20. Conditional teacher-student learning. Z. Meng et al. ICASSP 2019
21. Subclass Distillation. Müller, Rafael et al. arXiv:2002.03936
22. MarginDistillation: distillation for margin-based softmax. Svitov, David & Alyamkin, Sergey. arXiv:2003.02586
23. An Embarrassingly Simple Approach for Knowledge Distillation. Gao, Mengya et al. MLR 2018
24. Sequence-Level Knowledge Distillation. Kim, Yoon & Rush, Alexander M. arXiv:1606.07947
25. Boosting Self-Supervised Learning via Knowledge Transfer. Noroozi, Mehdi et al. CVPR 2018
26. Meta Pseudo Labels. Pham, Hieu et al. ICML 2020 [[code]][1.26]
27. Neural Networks Are More Productive Teachers Than Human Raters: Active Mixup for Data-Efficient Knowledge Distillation from a Blackbox Model. CVPR 2020 [[code]][1.30]
28. Distilled Binary Neural Network for Monaural Speech Separation. Chen Xiuyi et al. IJCNN 2018
29. Teacher-Class Network: A Neural Network Compression Mechanism. Malik et al. arXiv:2004.03281
30. Deeply-supervised knowledge synergy. Sun, Dawei et al. CVPR 2019
31. What it Thinks is Important is Important: Robustness Transfers through Input Gradients. Chan, Alvin et al. CVPR 2020
32. Triplet Loss for Knowledge Distillation. Oki, Hideki et al. IJCNN 2020
33. Role-Wise Data Augmentation for Knowledge Distillation. ICLR 2020 [[code]][1.36]
34. Distilling Spikes: Knowledge Distillation in Spiking Neural Networks. arXiv:2005.00288
35. Improved Noisy Student Training for Automatic Speech Recognition. Park et al. arXiv:2005.09629
36. Learning from a Lightweight Teacher for Efficient Knowledge Distillation. Yuang Liu et al. arXiv:2005.09163
37. ResKD: Residual-Guided Knowledge Distillation. Li, Xuewei et al. arXiv:2006.04719
38. Distilling Effective Supervision from Severe Label Noise. Zhang, Zizhao. et al. CVPR 2020 [[code]][1.41]
39. Knowledge Distillation Meets Self-Supervision. Xu, Guodong et al. ECCV 2020 [[code]][1.42]
40. Self-supervised Knowledge Distillation for Few-shot Learning. arXiv:2006.09785 [[code]][1.43]
41. Learning with Noisy Class Labels for Instance Segmentation. ECCV 2020
42. Improving Weakly Supervised Visual Grounding by Contrastive Knowledge Distillation. Wang, Liwei et al. arXiv:2007.01951
43. Deep Streaming Label Learning. Wang, Zhen et al. ICML 2020 [[code]][1.46]
44. Teaching with Limited Information on the Learner's Behaviour. Zhang, Yonggang et al. ICML 2020
45. Discriminability Distillation in Group Representation Learning. Zhang, Manyuan et al. ECCV 2020
46. Local Correlation Consistency for Knowledge Distillation. ECCV 2020
47. Prime-Aware Adaptive Distillation. Zhang, Youcai et al. ECCV 2020
48. One Size Doesn't Fit All: Adaptive Label Smoothing. Krothapalli et al. arXiv:2009.06432
49. Learning to learn from noisy labeled data. Li, Junnan et al. CVPR 2019
50. Combating Noisy Labels by Agreement: A Joint Training Method with Co-Regularization. Wei, Hongxin et al. CVPR 2020
51. Online Knowledge Distillation via Multi-branch Diversity Enhancement. Li, Zheng et al. ACCV 2020
52. Pea-KD: Parameter-efficient and Accurate Knowledge Distillation. arXiv:2009.14822
53. Extending Label Smoothing Regularization with Self-Knowledge Distillation. Wang, Jiyue et al. arXiv:2009.05226
54. Spherical Knowledge Distillation. Guo, Jia et al. arXiv:2010.07485
55. Soft-Label Dataset Distillation and Text Dataset Distillation. arXiv:1910.02551
56. Wasserstein Contrastive Representation Distillation. Chen, Liqun et al. cvpr 2021
57. Computation-Efficient Knowledge Distillation via Uncertainty-Aware Mixup. Xu, Guodong et al. cvpr 2021 [[code]][1.59]
58. Knowledge Refinery: Learning from Decoupled Label. Ding, Qianggang et al. AAAI 2021
59. Rocket Launching: A Universal and Efficient Framework for Training Well-performing Light Net. Zhou, Guorui et al. AAAI 2018

### Knowledge from intermediate layers

1. Fitnets: Hints for thin deep nets. Romero, Adriana et al. arXiv:1412.6550
2. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. Zagoruyko et al. ICLR 2017
3. Knowledge Projection for Effective Design of Thinner and Faster Deep Neural Networks. Zhang, Zhi et al. arXiv:1710.09505
4. A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning. Yim, Junho et al. CVPR 2017
5. Like What You Like: Knowledge Distill via Neuron Selectivity Transfer. Huang, Zehao & Wang, Naiyan. 2017
6. Paraphrasing complex network: Network compression via factor transfer. Kim, Jangho et al. NeurIPS 2018
7. Knowledge transfer with jacobian matching. ICML 2018
8. Self-supervised knowledge distillation using singular value decomposition. Lee, Seung Hyun et al. ECCV 2018
9. Learning Deep Representations with Probabilistic Knowledge Transfer. Passalis et al. ECCV 2018
10. Variational Information Distillation for Knowledge Transfer. Ahn, Sungsoo et al. CVPR 2019
11. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019
12. Knowledge Distillation via Route Constrained Optimization. Jin, Xiao et al. ICCV 2019
13. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
14. MEAL: Multi-Model Ensemble via Adversarial Learning. Shen,Zhiqiang, He,Zhankui, and Xue Xiangyang. AAAI 2019
15. A Comprehensive Overhaul of Feature Distillation. Heo, Byeongho et al. ICCV 2019
16. Feature-map-level Online Adversarial Knowledge Distillation. ICML 2020
17. Distilling Object Detectors with Fine-grained Feature Imitation. ICLR 2020
18. Knowledge Squeezed Adversarial Network Compression. Changyong, Shu et al. AAAI 2020
19. Stagewise Knowledge Distillation. Kulkarni, Akshay et al. arXiv: 1911.06786
20. Knowledge Distillation from Internal Representations. AAAI 2020
21. Knowledge Flow:Improve Upon Your Teachers. ICLR 2019
22. LIT: Learned Intermediate Representation Training for Model Compression. ICML 2019
23. Improving the Adversarial Robustness of Transfer Learning via Noisy Feature Distillation. Chin, Ting-wu et al. arXiv:2002.02998
24. Knapsack Pruning with Inner Distillation. Aflalo, Yonathan et al. arXiv:2002.08258
25. Residual Knowledge Distillation. Gao, Mengya et al. arXiv:2002.09168
26. Knowledge distillation via adaptive instance normalization. Yang, Jing et al. arXiv:2003.04289
27. Bert-of-Theseus: Compressing bert by progressive module replacing. Xu, Canwen et al. arXiv:2002.02925 [[code]][2.27]
28. Distilling Spikes: Knowledge Distillation in Spiking Neural Networks. arXiv:2005.00727
29. Generalized Bayesian Posterior Expectation Distillation for Deep Neural Networks. Meet et al. arXiv:2005.08110
30. Feature-map-level Online Adversarial Knowledge Distillation. Chung, Inseop et al. ICML 2020
31. Channel Distillation: Channel-Wise Attention for Knowledge Distillation. Zhou, Zaida et al. arXiv:2006.01683 [[code]][2.30]
32. Matching Guided Distillation. ECCV 2020 [[code]][2.31]
33. Differentiable Feature Aggregation Search for Knowledge Distillation. ECCV 2020
34. Interactive Knowledge Distillation. Fu, Shipeng et al. arXiv:2007.01476
35. Feature Normalized Knowledge Distillation for Image Classification. ECCV 2020 [[code]][2.34]
36. Layer-Level Knowledge Distillation for Deep Neural Networks. Li, Hao Ting et al. Applied Sciences, 2019
37. Knowledge Distillation with Feature Maps for Image Classification. Chen, Weichun et al. ACCV 2018
38. Efficient Kernel Transfer in Knowledge Distillation. Qian, Qi et al. arXiv:2009.14416
39. Collaborative Distillation in the Parameter and Spectrum Domains for Video Action Recognition. arXiv:2009.06902
40. Kernel Based Progressive Distillation for Adder Neural Networks. Xu, Yixing et al. NeurIPS 2020
41. Feature Distillation With Guided Adversarial Contrastive Learning. Bai, Tao et al. arXiv:2009.09922
42. Pay Attention to Features, Transfer Learn Faster CNNs. Wang, Kafeng et al. ICLR 2019
43. Multi-level Knowledge Distillation. Ding, Fei et al. arXiv:2012.00573
44. Cross-Layer Distillation with Semantic Calibration. Chen, Defang et al. AAAI 2021 [[code]][2.44]
45. Harmonized Dense Knowledge Distillation Training for Multi-­Exit Architectures. Wang, Xinglu & Li, Yingming. AAAI 2021
46. Robust Knowledge Transfer via Hybrid Forward on the Teacher-Student Model. Song, Liangchen et al. AAAI 2021
47. Show, Attend and Distill: Knowledge Distillation via Attention-­Based Feature Matching. Ji, Mingi et al. AAAI 2021
48. MINILMv2: Multi-Head Self-Attention Relation Distillation for Compressing Pretrained Transformers. Wang, Wenhui et al. arXiv:2012.15828
49. ALP-KD: Attention-Based Layer Projection for Knowledge Distillation. Peyman et al. AAAI 2021

### Graph-based

1. Graph-based Knowledge Distillation by Multi-head Attention Network. Lee, Seunghyun and Song, Byung. Cheol arXiv:1907.02226
2. Graph Representation Learning via Multi-task Knowledge Distillation. arXiv:1911.05700
3. Deep geometric knowledge distillation with graphs. arXiv:1911.03080
4. Better and faster: Knowledge transfer from multiple self-supervised learning tasks via graph distillation for video classification. IJCAI 2018
5. Distillating Knowledge from Graph Convolutional Networks. Yang, Yiding et al. CVPR 2020
6. Saliency Prediction with External Knowledge. Zhang, Yifeng et al. arXiv:2007.13839
7. Multi-label Zero-shot Classification by Learning to Transfer from External Knowledge. Huang, He et al. arXiv:2007.15610
8. Reliable Data Distillation on Graph Convolutional Network. Zhang, Wentao et al. ACM SIGMOD 2020
9. Mutual Teaching for Graph Convolutional Networks. Zhan, Kun et al. Future Generation Computer Systems, 2021
10. DistilE: Distiling Knowledge Graph Embeddings for Faster and Cheaper Reasoning. Zhu, Yushan et al. arXiv:2009.05912
11. Distill2Vec: Dynamic Graph Representation Learning with Knowledge Distillation. Antaris, Stefanos & Rafailidis, Dimitrios. arXiv:2011.05664
12. On Self-Distilling Graph Neural Network. Chen, Yuzhao et al. arXiv:2011.02255
13. Iterative Graph Self Distillation. iclr 2021

### Mutual Information & Online Learning

1. Correlation Congruence for Knowledge Distillation. Peng, Baoyun et al. ICCV 2019
2. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
3. Variational Information Distillation for Knowledge Transfer. Ahn, Sungsoo et al. CVPR 2019
4. Contrastive Representation Distillation. Tian, Yonglong et al. ICLR 2020 [[RepDistill]][4.4]
5. Online Knowledge Distillation via Collaborative Learning. Guo, Qiushan et al. CVPR 2020
6. Peer Collaborative Learning for Online Knowledge Distillation. Wu, Guile & Gong, Shaogang. AAAI 2021
7. Knowledge Transfer via Dense Cross-layer Mutual-distillation. ECCV 2020
8. MutualNet: Adaptive ConvNet via Mutual Learning from Network Width and Resolution. Yang, Taojiannan et al. ECCV 2020 [[code]][4.9]
9. AMLN: Adversarial-based Mutual Learning Network for Online Knowledge Distillation. ECCV 2020
10. Towards Cross-modality Medical Image Segmentation with Online Mutual Knowledge. Li, Kang et al. AAAI 2021
11. *Federated Knowledge Distillation. Seo, Hyowoon et al. arXiv:2011.02367
12. Unsupervised Image Segmentation using Mutual Mean-Teaching. Wu, Zhichao et al.arXiv:2012.08922
13. Exponential Moving Average Normalization for Self-supervised and Semi-supervised Learning. Cai, Zhaowei et al. arXiv:2101.08482

### Self-KD

1. Moonshine:Distilling with Cheap Convolutions. Crowley, Elliot J. et al. NeurIPS 2018 
2. Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. Zhang, Linfeng et al. ICCV 2019
3. Learning Lightweight Lane Detection CNNs by Self Attention Distillation. Hou, Yuenan et al. ICCV 2019
4. BAM! Born-Again Multi-Task Networks for Natural Language Understanding. Clark, Kevin et al. ACL 2019,short
5. Self-Knowledge Distillation in Natural Language Processing. Hahn, Sangchul and Choi, Heeyoul. arXiv:1908.01851
6. Rethinking Data Augmentation: Self-Supervision and Self-Distillation. Lee, Hankook et al. ICLR 2020
7. MSD: Multi-Self-Distillation Learning via Multi-classifiers within Deep Neural Networks. arXiv:1911.09418
8. Self-Distillation Amplifies Regularization in Hilbert Space. Mobahi, Hossein et al. NeurIPS 2020
9. MINILM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers. Wang, Wenhui et al. arXiv:2002.10957
10. Regularizing Class-wise Predictions via Self-knowledge Distillation. CVPR 2020 [[code]][5.11]
11. Self-Distillation as Instance-Specific Label Smoothing. Zhang, Zhilu & Sabuncu, Mert R. NeurIPS 2020
12. Self-PU: Self Boosted and Calibrated Positive-Unlabeled Training. Chen, Xuxi et al. ICML 2020 [[code]][5.13]
13. S2SD: Simultaneous Similarity-based Self-Distillation for Deep Metric Learning. arXiv:2009.08348
14. Comprehensive Attention Self-Distillation for Weakly-Supervised Object Detection. Huang, Zeyi et al. NeurIPS 2020
15. Distillation-Based Training for Multi-Exit Architectures. Phuong, Mary and Lampert, Christoph H. ICCV 2019
16. Pair-based self-distillation for semi-supervised domain adaptation. iclr 2021
17. SEED: SElf-SupErvised Distillation. ICLR 2021

### Structural Knowledge

1. Paraphrasing Complex Network:Network Compression via Factor Transfer. Kim, Jangho et al. NeurIPS 2018
2. Relational Knowledge Distillation.  Park, Wonpyo et al. CVPR 2019
3. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019
4. Contrastive Representation Distillation. Tian, Yonglong et al. ICLR 2020
5. Teaching To Teach By Structured Dark Knowledge. ICLR 2020
6. Inter-Region Affinity Distillation for Road Marking Segmentation. Hou, Yuenan et al. CVPR 2020 [[code]][6.6]
7. Heterogeneous Knowledge Distillation using Information Flow Modeling. Passalis et al. CVPR 2020 [[code]][6.7]
8. Asymmetric metric learning for knowledge transfer. Budnik, Mateusz & Avrithis, Yannis. arXiv:2006.16331
9. Local Correlation Consistency for Knowledge Distillation. ECCV 2020
10. Few-Shot Class-Incremental Learning. Tao, Xiaoyu et al. CVPR 2020
11. Semantic Relation Preserving Knowledge Distillation for Image-to-Image Translation. ECCV 2020
12. Interpretable Foreground Object Search As Knowledge Distillation. ECCV 2020
13. Improving Knowledge Distillation via Category Structure. ECCV 2020
14. Few-Shot Class-Incremental Learning via Relation Knowledge Distillation. Dong, Songlin et al. AAAI 2021

### Privileged Information

1. Learning using privileged information: similarity control and knowledge transfer. Vapnik, Vladimir and Rauf, Izmailov. MLR 2015  
2. Unifying distillation and privileged information. Lopez-Paz, David et al. ICLR 2016
3. Model compression via distillation and quantization. Polino, Antonio et al. ICLR 2018
4. KDGAN:Knowledge Distillation with Generative Adversarial Networks. Wang, Xiaojie. NeurIPS 2018
5. Efficient Video Classification Using Fewer Frames. Bhardwaj, Shweta et al. CVPR 2019
6. Retaining privileged information for multi-task learning. Tang, Fengyi et al. KDD 2019
7. A Generalized Meta-loss function for regression and classification using privileged information. Asif, Amina et al. arXiv:1811.06885
8. Private Knowledge Transfer via Model Distillation with Generative Adversarial Networks. Gao, Di & Zhuo, Cheng. AAAI 2020
9. Privileged Knowledge Distillation for Online Action Detection. Zhao, Peisen et al. cvpr 2021

## KD + GAN

1. Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks. Xu, Zheng et al. arXiv:1709.00513
2. KTAN: Knowledge Transfer Adversarial Network. Liu, Peiye et al. arXiv:1810.08126
3. KDGAN:Knowledge Distillation with Generative Adversarial Networks. Wang, Xiaojie. NeurIPS 2018
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
22. P-KDGAN: Progressive Knowledge Distillation with GANs for One-class Novelty Detection. Zhang, Zhiwei et al. IJCAI 2020
23. StyleGAN2 Distillation for Feed-forward Image Manipulation. Viazovetskyi et al. ECCV 2020 [[code]][8.23]
24. HardGAN: A Haze-Aware Representation Distillation GAN for Single Image Dehazing. ECCV 2020
25. TinyGAN: Distilling BigGAN for Conditional Image Generation. ACCV 2020 [[code]][8.25]
26. Learning Efficient GANs via Differentiable Masks and co-Attention Distillation. Li, Shaojie et al. aaai 2021 [[code]][8.26]
27. Self-Supervised GAN Compression. Yu, Chong & Pool, Jeff. arXiv:2007.01491
28. Efficient Conditional GAN Transfer with Knowledge Propagation across Classes. Shahbaziet al. arXiv:2102.06696 [[code]][8.28]

## KD + Meta-learning

1. Few Sample Knowledge Distillation for Efficient Network Compression. Li, Tianhong et al. CVPR 2020
2. Learning What and Where to Transfer. Jang, Yunhun et al, ICML 2019
3. Transferring Knowledge across Learning Processes. Moreno, Pablo G et al. ICLR 2019
4. Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval. Liu, Qing et al. ICCV 2019
5. Diversity with Cooperation: Ensemble Methods for Few-Shot Classification. Dvornik, Nikita et al. ICCV 2019
6. Knowledge Representing: Efficient, Sparse Representation of Prior Knowledge for Knowledge Distillation. arXiv:1911.05329v1
7. Progressive Knowledge Distillation For Generative Modeling. ICLR 2020
8. Few Shot Network Compression via Cross Distillation. AAAI 2020
9. MetaDistiller: Network Self-boosting via Meta-learned Top-down Distillation. Liu, Benlin et al. ECCV 2020
10. Few-Shot Learning with Intra-Class Knowledge Transfer. arXiv:2008.09892
11. Few-Shot Object Detection via Knowledge Transfer. Kim, Geonuk et al. arXiv:2008.12496
12. Distilled One-Shot Federated Learning. arXiv:2009.07999
13. Meta-KD: A Meta Knowledge Distillation Framework for Language Model Compression across Domains. Pan, Haojie et al. arXiv:2012.01266
14. Progressive Network Grafting for Few-Shot Knowledge Distillation. Shen, Chengchao et al. AAAI 2021

## Data-free KD

1. Data-Free Knowledge Distillation for Deep Neural Networks. NeurIPS 2017
2. Zero-Shot Knowledge Distillation in Deep Networks. ICML 2019
3. DAFL:Data-Free Learning of Student Networks. ICCV 2019
4. Zero-shot Knowledge Transfer via Adversarial Belief Matching. Micaelli, Paul and Storkey, Amos. NeurIPS 2019
5. Dream Distillation: A Data-Independent Model Compression Framework. Kartikeya et al. ICML 2019
6. Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion. Yin, Hongxu et al. CVPR 2020 [[code]][10.6]
7. Data-Free Adversarial Distillation. Fang, Gongfan et al. CVPR 2020
8. The Knowledge Within: Methods for Data-Free Model Compression. Haroush, Matan et al. CVPR 2020
9. Knowledge Extraction with No Observable Data. Yoo, Jaemin et al. NeurIPS 2019 [[code]][10.9]
10. Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN. CVPR 2020
11. DeGAN: Data-Enriching GAN for Retrieving Representative Samples from a Trained Classifier. Addepalli, Sravanti et al. arXiv:1912.11960
12. Generative Low-bitwidth Data Free Quantization. Xu, Shoukai et al. ECCV 2020 [[code]][10.12]
13. This dataset does not exist: training models from generated images. arXiv:1911.02888
14. MAZE: Data-Free Model Stealing Attack Using Zeroth-Order Gradient Estimation. Sanjay et al. arXiv:2005.03161
15. Generative Teaching Networks: Accelerating Neural Architecture Search by Learning to Generate Synthetic Training Data. Such et al. ECCV 2020
16. Billion-scale semi-supervised learning for image classification. FAIR. arXiv:1905.00546 [[code]][10.16]
17. Data-Free Network Quantization With Adversarial Knowledge Distillation. Choi, Yoojin et al. CVPRW 2020
18. Adversarial Self-Supervised Data-Free Distillation for Text Classification. EMNLP 2020
19. Towards Accurate Quantization and Pruning via Data-free Knowledge Transfer. arXiv:2010.07334
20. Data-free Knowledge Distillation for Segmentation using Data-Enriching GAN. Bhogale et al. arXiv:2011.00809
21. Layer-Wise Data-Free CNN Compression. Horton, Maxwell et al (Apple Inc.). cvpr 2021
22. Effectiveness of Arbitrary Transfer Sets for Data-free Knowledge Distillation. Nayak et al. WACV 2021
23. Learning in School: Multi-teacher Knowledge Inversion for Data-Free Quantization. Li, Yuhang et al. cvpr 2021
24. Large-Scale Generative Data-Free Distillation. Luo, Liangchen et al. cvpr 2021
25. Domain Impression: A Source Data Free Domain Adaptation Method. Kurmi et al. WACV 2021

- other data-free model compression:

26. Data-free Parameter Pruning for Deep Neural Networks. Srinivas, Suraj et al. arXiv:1507.06149
27. Data-Free Quantization Through Weight Equalization and Bias Correction. Nagel, Markus et al. ICCV 2019
28. DAC: Data-free Automatic Acceleration of Convolutional Networks. Li, Xin et al. WACV 2019
29. A Privacy-Preserving DNN Pruning and Mobile Acceleration Framework. Zhan, Zheng et al. arXiv:2003.06513
30. ZeroQ: A Novel Zero Shot Quantization Framework. Cai et al. CVPR 2020 [[code]][10.29]

## KD + AutoML

1. Improving Neural Architecture Search Image Classifiers via Ensemble Learning. Macko, Vladimir et al. arXiv:1903.06236
2. Blockwisely Supervised Neural Architecture Search with Knowledge Distillation. Li, Changlin et al. CVPR 2020
3. Towards Oracle Knowledge Distillation with Neural Architecture Search. Kang, Minsoo et al. AAAI 2020
4. Search for Better Students to Learn Distilled Knowledge. Gu, Jindong & Tresp, Volker arXiv:2001.11612
5. Circumventing Outliers of AutoAugment with Knowledge Distillation. Wei, Longhui et al. arXiv:2003.11342
6. Network Pruning via Transformable Architecture Search. Dong, Xuanyi & Yang, Yi. NeurIPS 2019
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
8. Dual Policy Distillation. Lai, Kwei-Herng et al. IJCAI 2020
9. Student-Teacher Curriculum Learning via Reinforcement Learning: Predicting Hospital Inpatient Admission Location. El-Bouri, Rasheed et al. ICML 2020
10. Reinforced Multi-Teacher Selection for Knowledge Distillation. Yuan, Fei et al. AAAI 2021
11. Universal Trading for Order Execution with Oracle Policy Distillation. Fang, Yuchen et al. AAAI 2021

## KD + Self-supervised

1. Reversing the cycle: self-supervised deep stereo through enhanced monocular distillation. ECCV 2020
2. Self-supervised Label Augmentation via Input Transformations. Lee, Hankook et al. ICML 2020 [[code]][12.2]
3. Improving Object Detection with Selective Self-supervised Self-training. Li, Yandong et al. ECCV 2020
4. Distilling Visual Priors from Self-Supervised Learning. Zhao, Bingchen & Wen, Xin. ECCVW 2020
5. Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning. Grill et al. arXiv:2006.07733 [[code]][12.5]
6. Unpaired Learning of Deep Image Denoising. Wu, Xiaohe et al. arXiv:2008.13711 [[code]][12.6]
7. SSKD: Self-Supervised Knowledge Distillation for Cross Domain Adaptive Person Re-Identification. Yin, Junhui et al. arXiv:2009.05972
8. Introspective Learning by Distilling Knowledge from Online Self-explanation. Gu, Jindong et al. ACCV 2020
9. Robust Pre-Training by Adversarial Contrastive Learning. Jiang, Ziyu et al. NeurIPS 2020 [[code]][12.9]
10. CompRess: Self-Supervised Learning by Compressing Representations. Koohpayegani et al. NeurIPS 2020 [[code]][12.10]
11. Big Self-Supervised Models are Strong Semi-Supervised Learners. Che, Ting et al. NeurIPS 2020 [[code]][12.11]
12. Rethinking Pre-training and Self-training. Zoph, Barret et al. NeurIPS 2020 [[code]][12.12]
13. ISD: Self-Supervised Learning by Iterative Similarity Distillation. Tejankar et al. cvpr 2021 [[code]][12.13]
14. Momentum^2 Teacher: Momentum Teacher with Momentum Statistics for Self-Supervised Learning. Li, Zeming et al. arXiv:2101.07525

## Multi-teacher and Ensemble KD 

1. Learning from Multiple Teacher Networks. You, Shan et al. KDD 2017
2. Learning with single-teacher multi-student. You, Shan et al. AAAI 2018
3. Knowledge distillation by on-the-fly native ensemble. Lan, Xu et al. NeurIPS 2018
4. Semi-Supervised Knowledge Transfer for Deep Learning from Private Training Data. ICLR 2017
5. Knowledge Adaptation: Teaching to Adapt. Arxiv:1702.02052
6. Deep Model Compression: Distilling Knowledge from Noisy Teachers.  Sau, Bharat Bhusan et al. arXiv:1610.09650
7. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Tarvainen, Antti and Valpola, Harri. NeurIPS 2017
8. Born-Again Neural Networks. Furlanello, Tommaso et al. ICML 2018
9. Deep Mutual Learning. Zhang, Ying et al. CVPR 2018
10. Collaborative learning for deep neural networks. Song, Guocong and Chai, Wei. NeurIPS 2018
11. Data Distillation: Towards Omni-Supervised Learning. Radosavovic, Ilija et al. CVPR 2018
12. Multilingual Neural Machine Translation with Knowledge Distillation. ICLR 2019
13. Unifying Heterogeneous Classifiers with Distillation. Vongkulbhisal et al. CVPR 2019
14. Distilled Person Re-Identification: Towards a More Scalable System. Wu, Ancong et al. CVPR 2019
15. Diversity with Cooperation: Ensemble Methods for Few-Shot Classification. Dvornik, Nikita et al. ICCV 2019
16. Model Compression with Two-stage Multi-teacher Knowledge Distillation for Web Question Answering System. Yang, Ze et al. WSDM 2020 
17. FEED: Feature-level Ensemble for Knowledge Distillation. Park, SeongUk and Kwak, Nojun. AAAI 2020
18. Stochasticity and Skip Connection Improve Knowledge Transfer. Lee, Kwangjin et al. ICLR 2020
19. Online Knowledge Distillation with Diverse Peers. Chen, Defang et al. AAAI 2020
20. Hydra: Preserving Ensemble Diversity for Model Distillation. Tran, Linh et al. arXiv:2001.04694
21. Distilled Hierarchical Neural Ensembles with Adaptive Inference Cost. Ruiz, Adria et al. arXv:2003.01474
22. Distilling Knowledge from Ensembles of Acoustic Models for Joint CTC-Attention End-to-End Speech Recognition. Gao, Yan et al. arXiv:2005.09310
23. Large-Scale Few-Shot Learning via Multi-Modal Knowledge Discovery. ECCV 2020
24. Collaborative Learning for Faster StyleGAN Embedding. Guan, Shanyan et al. arXiv:2007.01758
25. Temporal Self-Ensembling Teacher for Semi-Supervised Object Detection. Chen, Cong et al. IEEE 2020 [[code]][12.25]
26. Dual-Teacher: Integrating Intra-domain and Inter-domain Teachers for Annotation-efficient Cardiac Segmentation. MICCAI 2020
27. Joint Progressive Knowledge Distillation and Unsupervised Domain Adaptation. Nguyen-Meidine et al. WACV 2020
28. Semi-supervised Learning with Teacher-student Network for Generalized Attribute Prediction. Shin, Minchul et al. ECCV 2020
29. Knowledge Distillation for Multi-task Learning. Li, WeiHong & Bilen, Hakan. arXiv:2007.06889 [[project]][12.29]
30. Adaptive Multi-Teacher Multi-level Knowledge Distillation. Liu, Yuang et al. Neurocomputing 2020 [[code]][12.30]
31. Online Ensemble Model Compression using Knowledge Distillation. ECCV 2020
32. Learning From Multiple Experts: Self-paced Knowledge Distillation for Long-tailed Classification. ECCV 2020
33. Group Knowledge Transfer: Collaborative Training of Large CNNs on the Edge. He, Chaoyang et al. arXiv:2007.14513
34. Densely Guided Knowledge Distillation using Multiple Teacher Assistants. Son, Wonchul et l. arXiv:2009.08825
35. ProxylessKD: Direct Knowledge Distillation with Inherited Classifier for Face Recognition. Shi, Weidong et al. arXiv:2011.00265
36. Agree to Disagree: Adaptive Ensemble Knowledge Distillation in Gradient Space. Du, Shangchen et al. NeurIPS 2020 [[code]][12.37]
37. Reinforced Multi‐Teacher Selection for Knowledge Distillation. Yuan, Fei et al. AAAI 2021
38. Class-­Incremental Instance Segmentation via Multi­‐Teacher Networks. Gu, Yanan et al. AAAI 2021

### Knowledge Amalgamation（KA) - zju-VIPA

[VIPA - KA][13.24]

1. Amalgamating Knowledge towards Comprehensive Classification. Shen, Chengchao et al. AAAI 2019
2. Amalgamating Filtered Knowledge : Learning Task-customized Student from Multi-task Teachers. Ye, Jingwen et al. IJCAI 2019
3. Knowledge Amalgamation from Heterogeneous Networks by Common Feature Learning. Luo, Sihui et al. IJCAI 2019
4. Student Becoming the Master: Knowledge Amalgamation for Joint Scene Parsing, Depth Estimation, and More. Ye, Jingwen et al. CVPR 2019
5. Customizing Student Networks From Heterogeneous Teachers via Adaptive Knowledge Amalgamation. ICCV 2019
6. Data-Free Knowledge Amalgamation via Group-Stack Dual-GAN. CVPR 2020

## Cross-modal KD & DA

1. SoundNet: Learning Sound Representations from Unlabeled Video SoundNet Architecture. Aytar, Yusuf et al. NeurIPS 2016
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
14. Knowledge distillation for semi-supervised domain adaptation. arXiv:1908.07355
15. Domain Adaptation via Teacher-Student Learning for End-to-End Speech Recognition. Meng, Zhong et al. arXiv:2001.01798
16. Cluster Alignment with a Teacher for Unsupervised Domain Adaptation. ICCV 2019
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
29. Domain Adaptation through Task Distillation. Zhou, Brady et al. ECCV 2020 [[code]][14.29]
30. Dual Super-Resolution Learning for Semantic Segmentation. Wang, Li et al. CVPR 2020 [[code]][14.30]
31. Adaptively-Accumulated Knowledge Transfer for Partial Domain Adaptation. Jing, Taotao et al. ACM MM 2020
32. Domain2Vec: Domain Embedding for Unsupervised Domain Adaptation. Peng, Xingchao et al. ECCV 2020 [[code]][14.32]
33. Unsupervised Domain Adaptive Knowledge Distillation for Semantic Segmentation. Kothandaraman et al. arXiv:2011.08007
34. A Student‐Teacher Architecture for Dialog Domain Adaptation under the Meta‐Learning Setting. Qian, Kun et al. AAAI 2021
35. Multimodal Fusion via Teacher-­‐Student Network for Indoor Action Recognition. Bruce et al. AAAI 2021
36. Dual-Teacher++: Exploiting Intra-domain and Inter-domain Knowledge with Reliable Transfer for Cardiac Segmentation. Li, Kang et al. TMI 2021
37. Knowledge Distillation Methods for Efficient Unsupervised Adaptation Across Multiple Domains. Nguyen et al. IVC 2021
38. Feature-Supervised Action Modality Transfer. Thoker, Fida Mohammad and Snoek, Cees. ICPR 2020.

## Application of KD

1. Face model compression by distilling knowledge from neurons. Luo, Ping et al. AAAI 2016
2. Learning efficient object detection models with knowledge distillation. Chen, Guobin et al. NeurIPS 2017
3. Apprentice: Using Knowledge Distillation Techniques To Improve Low-Precision Network Accuracy. Mishra, Asit et al. NeurIPS 2018
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
26. Multi-Representation Knowledge Distillation For Audio Classification. Gao, Liang et al. arXiv:2002.09607
27. Collaborative Distillation for Ultra-Resolution Universal _Style Transfer_. Wang, Huan et al. CVPR 2020 [[code]][15.28]
28. ShadowTutor: Distributed Partial Distillation for Mobile _Video_ DNN Inference. Chung, Jae-Won et al. ICPP 2020 [[code]][15.29]
29. Object Relational Graph with Teacher-Recommended Learning for _Video Captioning_. Zhang, Ziqi et al. CVPR 2020
30. Spatio-Temporal Graph for _Video Captioning_ with Knowledge distillation. CVPR 2020 [[code]][15.31]
31. Squeezed Deep _6DoF Object Detection_ Using Knowledge Distillation. Felix, Heitor et al. arXiv:2003.13586
32. Distilled Semantics for Comprehensive _Scene Understanding_ from Videos. Tosi, Fabio et al. arXiv:2003.14030
33. Parallel WaveNet: Fast high-fidelity _speech synthesis_. Van et al. ICML 2018
34. Distill Knowledge From NRSfM for Weakly Supervised _3D Pose_ Learning. Wang Chaoyang et al. ICCV 2019
35. KD-MRI: A knowledge distillation framework for _image reconstruction_ and image restoration in MRI workflow. Murugesan et al. MIDL 2020
36. Geometry-Aware Distillation for Indoor _Semantic Segmentation_. Jiao, Jianbo et al. CVPR 2019
37. Teacher Supervises Students How to Learn From Partially Labeled Images for _Facial Landmark Detection_. ICCV 2019
38. Distill Image _Dehazing_ with Heterogeneous Task Imitation. Hong, Ming et al. CVPR 2020
39. Knowledge Distillation for _Action Anticipation_ via Label Smoothing. Camporese et al. arXiv:2004.07711
40. More Grounded _Image Captioning_ by Distilling Image-Text Matching Model. Zhou, Yuanen et al. CVPR 2020
41. Distilling Knowledge from Refinement in Multiple _Instance Detection_ Networks. Zeni, Luis Felipe & Jung, Claudio. arXiv:2004.10943
42. Enabling Incremental Knowledge Transfer for _Object Detection_ at the Edge. arXiv:2004.05746
43. Uninformed Students: Student-Teacher _Anomaly Detection_ with Discriminative Latent Embeddings. Bergmann, Paul et al. CVPR 2020
44. TA-Student _VQA_: Multi-Agents Training by Self-Questioning. Xiong, Peixi & Wu Ying. CVPR 2020
45. Mentornet: Learning data-driven curriculum for very deep neural networks on corrupted labels. Jiang, Lu et al. ICML 2018
46. A Multi-Task Mean Teacher for Semi-Supervised _Shadow Detection_. Chen, Zhihao et al. CVPR 2020 [[code]][15.48]
47. Learning Lightweight _Face Detector_ with Knowledge Distillation. Zhang Shifeng et al. IEEE 2019
48. Learning Lightweight _Pedestrian Detector_ with Hierarchical Knowledge Distillation. ICIP 2019
49. Distilling _Object Detectors_ with Task Adaptive Regularization. Sun, Ruoyu et al. arXiv:2006.13108
50. Intra-class Compactness Distillation for _Semantic Segmentation_. ECCV 2020
51. DOPE: Distillation Of Part Experts for whole-body _3D pose estimation_ in the wild. ECCV 2020
52. Self-similarity Student for Partial Label Histopathology Image _Segmentation_. ECCV 2020
53. Robust _Re-Identification_ by Multiple Views Knowledge Distillation. Porrello et al. ECCV 2020 [[code]][15.58]
54. LabelEnc: A New Intermediate Supervision Method for _Object Detection_. Hao, Miao et al. arXiv:2007.03282
55. Optical Flow Distillation: Towards Efficient and Stable _Video Style Transfer_. Chen, Xinghao et al. ECCV 2020
56. Adversarial Self-Supervised Learning for Semi-Supervised _3D Action Recognition_. Si, Chenyang et al. ECCV 2020
57. Dual-Path Distillation: A Unified Framework to Improve Black-Box Attacks. Zhang, Yonggang et al. ICML 2020
58. RGB-IR Cross-modality Person _ReID_ based on Teacher-Student GAN Mode. Zhang, Ziyue et al. arXiv:2007.07452
59. _Defocus Blur Detection_ via Depth Distillation. Cun, Xiaodong & Pun, Chi-Man. ECCV 2020 [[code]][15.64]
60. Boosting Weakly Supervised _Object Detection_ with Progressive Knowledge Transfer. Zhong, Yuanyi et al. ECCV 2020 [[code]][15.64]
61. Weight Decay Scheduling and Knowledge Distillation for _Active Learning_. ECCV 2020
62. Circumventing Outliers of AutoAugment with Knowledge Distillation. ECCV 2020
63. Improving _Face Recognition_ from Hard Samples via Distribution Distillation Loss. ECCV 2020
64. Exclusivity-Consistency Regularized Knowledge Distillation for _Face Recognition_. ECCV 2020
65. Self-similarity Student for Partial Label Histopathology Image _Segmentation_. Cheng, Hsien-Tzu et al. ECCV 2020
66. Deep Semi-supervised Knowledge Distillation for Overlapping Cervical Cell _Instance Segmentation_. Zhou, Yanning et al. arXiv:2007.10787 [[code]][15.70]
67. Two-Level Residual Distillation based Triple Network for Incremental _Object Detection_. Yang, Dongbao et al. arXiv:2007.13428
68. Towards Unsupervised _Crowd Counting_ via Regression-Detection Bi-knowledge Transfer. Liu, Yuting et al. ACM MM 2020
69. Teacher-Critical Training Strategies for _Image Captioning_. Huang, Yiqing & Chen, Jiansheng. arXiv:2009.14405
70. Object Relational Graph with Teacher-Recommended Learning for _Video Captioning_. Zhang, Ziqi et al. CVPR 2020
71. Multi-Frame to Single-Frame: Knowledge Distillation for _3D Object Detection_. Wang Yue et al. ECCV 2020
72. Residual Feature Distillation Network for Lightweight Image _Super-Resolution_. Liu, Jie et al. ECCV 2020
73. Intra-Utterance Similarity Preserving Knowledge Distillation for Audio Tagging. Interspeech 2020
74. Federated Model Distillation with Noise-Free Differential Privacy. arXiv:2009.05537
75. _Long-tailed Recognition_ by Routing Diverse Distribution-Aware Experts. Wang, Xudong et al. arXiv:2010.01809
76. Fast _Video Salient Object Detection_ via Spatiotemporal Knowledge Distillation. Yi, Tang & Yuan, Li. arXiv:2010.10027
77. Multiresolution Knowledge Distillation for _Anomaly Detection_. Salehi et al. cvpr 2021
78. Channel-wise Distillation for _Semantic Segmentation_. Shu, Changyong et al. arXiv: 2011.13256
79. Teach me to segment with mixed supervision: Confident students become masters. Dolz, Jose et al. arXiv:2012.08051
80. Invariant Teacher and Equivariant Student for Unsupervised _3D Human Pose Estimation_. Xu, Chenxin et al. AAAI 2021 [[code]][15.80]
81. Training data-efficient _image transformers_ & distillation through attention. Touvron, Hugo et al. arXiv:2012.12877 [[code]][15.81]
82. SID: Incremental Learning for Anchor-Free Object Detection via Selective and Inter-Related Distillation. Peng, Can et al. arXiv:2012.15439
83. PSSM-Distil: Protein Secondary Structure Prediction (PSSP) on Low-Quality PSSM by Knowledge Distillation with Contrastive Learning. Wang, Qin et al. AAAI 2021
84. Diverse Knowledge Distillation for End-­to‐End Person Search. Zhang, Xinyu et al. AAAI 2021
85. Enhanced Audio Tagging via Multi­‐ to Single­‐Modal Teacher­‐Student Mutual Learning. Yin, Yifang et al. AAAI 2021
86. Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks. Li, Yige et al. ICLR 2021 [[code]][15.86]
87. Unbiased Teacher for Semi-Supervised Object Detection. Liu, Yen-Cheng et al. ICLR 2021

### for NLP & Data-Mining

1. Patient Knowledge Distillation for BERT Model Compression. Sun, Siqi et al. arXiv:1908.09355
2. TinyBERT: Distilling BERT for Natural Language Understanding. Jiao, Xiaoqi et al. arXiv:1909.10351
3. Learning to Specialize with Knowledge Distillation for Visual Question Answering. NeurIPS 2018
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
17. LadaBERT: Lightweight Adaptation of BERT through Hybrid Model Compression. Mao, Yihuan et al. arXiv:2004.04124
18. DynaBERT: Dynamic BERT with Adaptive Width and Depth. Hou, Lu et al. NeurIPS 2020
19. Structure-Level Knowledge Distillation For Multilingual Sequence Labeling. Wang, Xinyu et al. ACL 2020
20. Distilled embedding: non-linear embedding factorization using knowledge distillation. Lioutas, Vasileios et al. arXiv:1910.06720
21. TinyMBERT: Multi-Stage Distillation Framework for Massive Multi-lingual NER. Mukherjee & Awadallah. ACL 2020
22. Knowledge Distillation for Multilingual Unsupervised Neural Machine Translation. Sun, Haipeng et al. arXiv:2004.10171
23. Making Monolingual Sentence Embeddings Multilingual using Knowledge Distillation. Reimers, Nils & Gurevych, Iryna arXiv:2004.09813
24. Distilling Knowledge for Fast Retrieval-based Chat-bots. Tahami et al. arXiv:2004.11045
25. Single-/Multi-Source Cross-Lingual NER via Teacher-Student Learning on Unlabeled Data in Target Language. ACL 2020
26. Local Clustering with Mean Teacher for Semi-supervised Learning. arXiv:2004.09665
27. Time Series Data Augmentation for Neural Networks by Time Warping with a Discriminative Teacher. arXiv:2004.08780 
28. Syntactic Structure Distillation Pretraining For Bidirectional Encoders. arXiv: 2005.13482
29. Distill, Adapt, Distill: Training Small, In-Domain Models for Neural Machine Translation. arXiv:2003.02877
30. Distilling Neural Networks for Faster and Greener Dependency Parsing. arXiv:2006.00844
31. Distilling Knowledge from Well-informed Soft Labels for Neural Relation Extraction. AAAI 2020 [[code]][16.32]
32. More Grounded Image Captioning by Distilling Image-Text Matching Model. Zhou, Yuanen et al. CVPR 2020
33. Multimodal Learning with Incomplete Modalities by Knowledge Distillation. Wang, Qi et al. KDD 2020
34. Distilling the Knowledge of BERT for Sequence-to-Sequence ASR. Futami, Hayato et al. arXiv:2008.03822
35. Contrastive Distillation on Intermediate Representations for Language Model Compression. Sun, Siqi et al. EMNLP 2020 [[code]][16.37]
36. Noisy Self-Knowledge Distillation for Text Summarization. arXiv:2009.07032
37. Simplified TinyBERT: Knowledge Distillation for Document Retrieval. arXiv:2009.07531
38. Autoregressive Knowledge Distillation through Imitation Learning. arXiv:2009.07253
39. BERT-EMD: Many-to-Many Layer Mapping for BERT Compression with Earth Mover’s Distance. EMNLP 2020 [[code]][16.392]
40. Interpretable Embedding Procedure Knowledge Transfer. Seunghyun Lee et al. AAAI 2021 [[code]][16.40]
41. LRC-BERT: Latent-representation Contrastive Knowledge Distillation for Natural Language Understanding. Fu, Hao et al. AAAI 2021
42. Towards Zero-Shot Knowledge Distillation for Natural Language Processing. Ahmad et al. arXiv:2012.15495
43. Meta-KD: A Meta Knowledge Distillation Framework for Language Model Compression across Domains. Pan, Haojie et al. AAAI 2021
44. Learning to Augment for Data-Scarce Domain BERT Knowledge Distillation. Feng, Lingyun et al. AAAI 2021
45. Label Confusion Learning to Enhance Text Classification Models. Guo, Biyang et al. AAAI 2021

### for RecSys

1. Improving session recommendation with recurrent neural networks by exploiting dwell time. Dallmann et al. arXiv:1706.10231
2. Developing Multi-Task Recommendations with Long-Term Rewards via Policy Distilled Reinforcement Learning. Liu, Xi et al. arXiv:2001.09595
3. A General Knowledge Distillation Framework for Counterfactual Recommendation via Uniform Data. Liu, Dugang et al. SIGIR 2020 [[Sildes]][16.35] [[code]][16.352]
4. LightRec: a Memory and Search-Efficient Recommender System. Lian, Defu et al. WWW 2020
5. Privileged Features Distillation at Taobao Recommendations. Xu, Chen et al. KDD 2020
6. Next Point-of-Interest Recommendation on Resource-Constrained Mobile Devices. WWW 2020
7. Adversarial Distillation for Efficient Recommendation with External Knowledge. Chen, Xu et al. ACM Trans, 2018
8. Ranking Distillation: Learning Compact Ranking Models With High Performance for Recommender System. Tang, Jiaxi et al. SIGKDD 2018
9. A novel Enhanced Collaborative Autoencoder with knowledge distillation for top-N recommender systems. Pan, Yiteng et al. Neurocomputing 2019 [[code]][16.38]
10. ADER: Adaptively Distilled Exemplar Replay Towards Continual Learning for Session-based Recommendation. Mi, Fei et al. ACM RecSys 2020
11. Ensembled CTR Prediction via Knowledge Distillation. Zhu, Jieming et al.(Huawei) CIKM 2020
12. DE-RRD: A Knowledge Distillation Framework for Recommender System. Kang, Seongku et al. CIKM 2020 [[code]][16.39]
13. Neural Compatibility Modeling with Attentive Knowledge Distillation. Song, Xuemeng et al. SIGIR 2018
14. Binarized Collaborative Filtering with Distilling Graph Convolutional Networks. Wang, Haoyu et al. IJCAI 2019
15. Collaborative Distillation for Top-N Recommendation. Jae-woong Lee, et al. CIKM 2019
16. Distilling Structured Knowledge into Embeddings for Explainable and Accurate Recommendation. Zhang Yuan et al. WSDM 2020
17. UMEC:Unified Model and Embedding Compression for Efficient Recommendation Systems. ICLR 2021

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
17. Distillation Guided Residual Learning for Binary Convolutional Neural Networks. Ye, Jianming et al. ECCV 2020
18. Cascaded channel pruning using hierarchical self-distillation. Miles & Mikolajczyk. BMVC 2020
19. TernaryBERT: Distillation-aware Ultra-low Bit BERT. Zhang, Wei et al. EMNLP 2020
20. Weight Distillation: Transferring the Knowledge in Neural Network Parameters. arXiv:2009.09152
21. Stochastic Precision Ensemble: Self-­‐Knowledge Distillation for Quantized Deep Neural Networks. Boo, Yoonho et al. AAAI 2021

## Beyond

1. Do deep nets really need to be deep?. Ba,Jimmy, and Rich Caruana. NeurIPS 2014
2. When Does Label Smoothing Help? Müller, Rafael, Kornblith, and Hinton. NeurIPS 2019
3. Towards Understanding Knowledge Distillation. Phuong, Mary and Lampert, Christoph. ICML 2019
4. Harnessing deep neural networks with logical rules. ACL 2016
5. Adaptive Regularization of Labels. Ding, Qianggang et al. arXiv:1908.05474
6. Knowledge Isomorphism between Neural Networks. Liang, Ruofan et al. arXiv:1908.01581
7. (survey)Modeling Teacher-Student Techniques in Deep Neural Networks for Knowledge Distillation. arXiv:1912.13179
8. Understanding and Improving Knowledge Distillation. Tang, Jiaxi et al. arXiv:2002.03532
9. The State of Knowledge Distillation for Classification. Ruffy, Fabian and Chahal, Karanbir. arXiv:1912.10850 [[code]][18.11]
10. Explaining Knowledge Distillation by Quantifying the Knowledge. [Zhang, Quanshi][18.13] et al. CVPR 2020
11. DeepVID: deep visual interpretation and diagnosis for image classifiers via knowledge distillation. IEEE Trans, 2019.
12. On the Unreasonable Effectiveness of Knowledge Distillation: Analysis in the Kernel Regime. Rahbar, Arman et al. arXiv:2003.13438
13. (survey)Knowledge Distillation and Student-Teacher Learning for Visual Intelligence: A Review and New Outlooks. Wang, Lin & Yoon, Kuk-Jin. arXiv:2004.05937
14. Why distillation helps: a statistical perspective. arXiv:2005.10419
15. Transferring Inductive Biases through Knowledge Distillation. Abnar, Samira et al. arXiv:2006.00555
16. Does label smoothing mitigate label noise? Lukasik, Michal et al. ICML 2020
17. An Empirical Analysis of the Impact of Data Augmentation on Knowledge Distillation. Das, Deepan et al. arXiv:2006.03810
18. Knowledge Distillation: A Survey. Gou, Jianping et al. arXiv:2006.05525
19. Does Adversarial Transferability Indicate Knowledge Transferability? Liang, Kaizhao et al. arXiv:2006.14512
20. On the Demystification of Knowledge Distillation: A Residual Network Perspective. Jha et al. arXiv:2006.16589
21. Enhancing Simple Models by Exploiting What They Already Know. Dhurandhar et al. ICML 2020
22. Feature-Extracting Functions for Neural Logic Rule Learning. Gupta & Robles-Kelly.arXiv:2008.06326
23. On the Orthogonality of Knowledge Distillation with Other Techniques: From an Ensemble Perspective. SeongUk et al. arXiv:2009.04120
24. Knowledge Distillation in Wide Neural Networks: Risk Bound, Data Efficiency and Imperfect Teacher. Ji, Guangda & Zhu, Zhanxing. NeurIPS 2020
25. In Defense of Feature Mimicking for Knowledge Distillation. Wang, Guo-Hua et al. arXiv:2011.0142
26. Solvable Model for Inheriting the Regularization through Knowledge Distillation. Luca Saglietti & Lenka Zdeborova. arXiv:2012.00194
27. Undistillable: Making A Nasty Teacher That CANNOT Teach Students. ICLR 2021
28. Towards Understanding Ensemble, Knowledge Distillation and Self-Distillation in Deep Learning. Allen-Zhu, Zeyuan & Li, Yuanzhi.(Microsoft) arXiv:2012.09816

## Distiller Tools

1. [Neural Network Distiller][18.8]: A Python Package For DNN Compression Research. arXiv:1910.12232
2. [TextBrewer][18.12]: An Open-Source Knowledge Distillation Toolkit for Natural Language Processing. HIT and iFLYTEK. arXiv:2002.12620
3. [torchdistill][18.28]: A Modular, Configuration-Driven Framework for Knowledge Distillation. 
4. [KD-Lib][18.29]: A PyTorch library for Knowledge Distillation, Pruning and Quantization. Shen, Het et al. arXiv:2011.14691
5. [Knowledge-Distillation-Zoo][18.30]
6. [RepDistiller][18.31]
7. [classification distiller][18.11]

---
Note: All papers' pdf can be found and downloaded on [arXiv](https://arxiv.org/search/), [Bing](https://www.bing.com) or [Google](https://www.google.com).

Source: <https://github.com/FLHonker/Awesome-Knowledge-Distillation>

Thanks for all contributors:

[![yuang](https://avatars.githubusercontent.com/u/20468157?s=28&v=4)](https://github.com/FLHonker)  [![lioutasb](https://avatars.githubusercontent.com/u/9558061?s=28&v=4)](https://github.com/lioutasb)  [![KaiyuYue](https://avatars.githubusercontent.com/u/19852297?s=28&v=4)](https://github.com/KaiyuYue)  [<img src="https://avatars.githubusercontent.com/u/21128481?s=28&v=4" width = "28" height = "28" alt="avatar" />](https://github.com/shivmgg)  [![cardwing](https://avatars.githubusercontent.com/u/23656119?s=28&v=4)](https://github.com/cardwing)  [![jaywonchung](https://avatars1.githubusercontent.com/u/29395896?s=28&v=4)](https://github.com/jaywonchung)  [![ZainZhao](https://avatars.githubusercontent.com/u/28838928?s=28&v=4)](https://github.com/ZainZhao)

Contact: [Yuang Liu](https://flhonker.github.io/)(frankliu624![](https://res.cloudinary.com/flhonker/image/upload/v1605363963/frankio/at1.png)outlook.com), [ECNU](https://www.ecnu.edu.cn/). Supervisor: [Wei Zhang](https://weizhangltt.github.io), Jun Wang. 


[1.10]: https://github.com/yuanli2333/Teacher-free-Knowledge-Distillation
[1.26]: https://github.com/google-research/google-research/tree/master/meta_pseudo_labels
[1.30]: https://github.com/dwang181/active-mixup
[1.36]: https://github.com/bigaidream-projects/role-kd
[1.41]: https://github.com/google-research/google-research/tree/master/ieg
[1.42]: https://github.com/xuguodong03/SSKD
[1.43]: https://github.com/brjathu/SKD
[1.46]: https://github.com/DSLLcode/DSLL
[1.59]: https://github.com/xuguodong03/UNIXKD
[2.27]: https://github.com/JetRunner/BERT-of-Theseus
[2.30]: https://github.com/zhouzaida/channel-distillation
[2.31]: https://github.com/KaiyuYue/mgd
[2.34]: https://github.com/aztc/FNKD
[2.44]: https://github.com/DefangChen/SemCKD
[4.4]: https://github.com/HobbitLong/RepDistiller
[4.9]: https://github.com/taoyang1122/MutualNet
[5.11]: https://github.com/alinlab/cs-kd
[5.13]: https://github.com/TAMU-VITA/Self-PU
[6.6]: https://github.com/cardwing/Codes-for-IntRA-KD
[6.7]: https://github.com/passalis/pkth
[8.20]: https://github.com/mit-han-lab/gan-compression
[8.23]: https://github.com/EvgenyKashin/stylegan2-distillation
[8.25]: https://github.com/terarachang/ACCV_TinyGAN
[8.26]: https://github.com/SJLeo/DMAD
[8.28]: https://github.com/mshahbazi72/cGANTransfer
[10.6]: https://github.com/NVlabs/DeepInversion
[10.9]: https://github.com/snudatalab/KegNet
[10.12]: https://github.com/xushoukai/GDFQ
[10.16]: https://github.com/leaderj1001/Billion-scale-semi-supervised-learning
[10.12]: https://github.com/amirgholami/ZeroQ
[11.8]: https://github.com/TAMU-VITA/AGD
[12.2]: https://github.com/hankook/SLA
[12.5]: https://github.com/sthalles/PyTorch-BYOL
[12.6]: https://github.com/XHWXD/DBSN
[12.9]: https://github.com/VITA-Group/Adversarial-Contrastive-Learning
[12.10]: https://github.com/UMBCvision/CompRess
[12.11]: https://github.com/google-research/simclr
[12.12]: https://github.com/tensorflow/tpu/tree/master/models/official/detection/projects/self_training
[12.13]: https://github.com/UMBCvision/ISD
[12.25]: http://github.com/SYangDong/tse-t
[12.29]: https://weihonglee.github.io/Projects/KD-MTL/KD-MTL.htm
[12.30]: https://github.com/FLHonker/AMTML-KD-code
[12.37]: https://github.com/AnTuo1998/AE-KD
[13.24]: https://github.com/zju-vipa/KamalEngine
[14.27]: https://github.com/njulus/ReFilled
[14.29]: https://github.com/bradyz/task-distillation
[14.30]: https://github.com/wanglixilinx/DSRL
[14.32]: https://github.com/VisionLearningGroup/Domain2Vec
[15.5]: https://github.com/lucidrains/byol-pytorch
[15.28]: https://github.com/mingsun-tse/collaborative-distillation
[15.29]: https://github.com/jaywonchung/ShadowTutor
[15.31]: https://github.com/StanfordVL/STGraph
[15.48]: https://github.com/eraserNut/MTMT
[15.58]: https://github.com/aimagelab/VKD
[15.64]: https://github.com/vinthony/depth-distillation
[15.64]: https://github.com/mikuhatsune/wsod_transfer
[15.70]: https://github.com/SIAAAAAA/MMT-PSM
[15.80]: https://github.com/sjtuxcx/ITES
[15.81]: https://github.com/facebookresearch/deit
[15.86]: https://github.com/bboylyg/NAD
[16.32]: https://github.com/zzysay/KD4NRE
[16.35]: http://csse.szu.edu.cn/staff/panwk/publications/Conference-SIGIR-20-KDCRec-Slides.pdf
[16.352]:https://github.com/dgliu/SIGIR20_KDCRec
[16.37]: https://github.com/intersun/CoDIR
[16.38]: https://github.com/graytowne/rank_distill
[16.39]: https://github.com/SeongKu-Kang/DE-RRD_CIKM20
[16.392]:https://github.com/lxk00/BERT-EMD
[16.40]: https://github.com/sseung0703/IEPKT
[17.9]: https://csyhhu.github.io/
[17.15]: https://github.com/lehduong/ginp
[18.8]: https://github.com/IntelLabs/distiller
[18.11]: https://github.com/karanchahal/distiller
[18.12]: https://github.com/airaria/TextBrewer
[18.13]: http://qszhang.com/
[18.28]: https://github.com/yoshitomo-matsubara/torchdistill
[18.29]: https://github.com/SforAiDl/KD_Lib
[18.30]: https://github.com/AberHu/Knowledge-Distillation-Zoo
[18.31]: https://github.com/HobbitLong/RepDistiller
