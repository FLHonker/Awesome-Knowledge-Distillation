# Awesome Knowledge-Distillation

- [Awesome Knowledge-Distillation](#awesome-knowledge-distillation)
  - [Different form of knowledge](#different-form-of-knowledge)
    - [Knowledge from logits](#knowledge-from-logits)
    - [Knowledge from intermediate layers](#knowledge-from-intermediate-layers)
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
  - [Cross-modal KD](#cross-modal-kd)
  - [Application of KD](#application-of-kd)
    - [for NLP](#for-nlp)
  - [Beyond](#beyond)

## Different form of knowledge

### Knowledge from logits

1. Distilling the knowledge in a neural network. Hinton et al. arXiv:1503.02531
2. A Gift from Knowledge Distillation: Fast Optimization, Network Minimization and Transfer Learning. Yim, Junho et al. CVPR 2017
3. Knowledge distillation by on-the-fly native ensemble. Lan, Xu et al. NIPS 2018
4. Learning Metrics from Teachers: Compact Networks for Image Embedding. Yu, Lu et al. CVPR 2019
5. Relational Knowledge Distillation.  Park, Wonpyo et al, CVPR 2019
6. Like What You Like: Knowledge Distill via Neuron Selectivity Transfer. Huang, Zehao and Wang, Naiyan. 2017
7. On Knowledge Distillation from Complex Networks for Response Prediction. Arora, Siddhartha et al. NAACL 2019
8. On the Efficacy of Knowledge Distillation. Cho, Jang Hyun and Hariharan, Bharath. arXiv:1910.01348
9.[noval]Revisit Knowledge Distillation: a Teacher-free Framework. Yuan, Li et al. arXiv:1909.11723

### Knowledge from intermediate layers

1. Fitnets: Hints for thin deep nets. Romero, Adriana et al. arXiv:1412.6550
2. Paying more attention to attention: Improving the performance of convolutional neural networks via attention transfer. Zagoruyko et al. ICLR 2017
3. Knowledge Projection for Effective Design of Thinner and Faster Deep Neural Networks. Zhang, Zhi et al. arXiv:1710.09505
4. Paraphrasing complex network: Network compression via factor transfer. Kim, Jangho et al. NIPS 2018
5. Variational Information Distillation for Knowledge Transfer. Ahn, Sungsoo et al. CVPR 2019
   <!-- * 通过互信息导出关于student中间层表示和teacher中间表示的关系。 -->
6. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019
7. Knowledge Distillation via Route Constrained Optimization. Jin, Xiao et al. ICCV 2019
8. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
9. MEAL: Multi-Model Ensemble via Adversarial Learning. Shen,Zhiqiang, He,Zhankui, and Xue Xiangyang. AAAI 2019
10. A Comprehensive Overhaul of Feature Distillation. Heo, Byeongho et al. ICCV 2019

### Mutual Information

1. Correlation Congruence for Knowledge Distillation. Peng, Baoyun et al. ICCV 2019
2. Similarity-Preserving Knowledge Distillation. Tung, Frederick, and Mori Greg. ICCV 2019
3. Variational Information Distillation for Knowledge Transfer. Ahn, Sungsoo et al. CVPR 2019
4. Contrastive Representation Distillation. Tian, Yonglong et al. arXiv: 1910.10699

### Self-KD

1.  Moonshine:Distilling with Cheap Convolutions. Crowley, Elliot J. et al. NIPS 2018 
2. Be Your Own Teacher: Improve the Performance of Convolutional Neural Networks via Self Distillation. Zhang, Linfeng et al. ICCV 2019
3. Learning Lightweight Lane Detection CNNs by Self Attention Distillation. Hou, Yuenan et al. ICCV 2019
4. BAM! Born-Again Multi-Task Networks for Natural Language Understanding. Clark, Kevin et al. ACL 2019,short
5. Self-Knowledge Distillation in Natural Language Processing. Hahn, Sangchul and Choi, Heeyoul. arXiv:1908.01851

### Structured Knowledge

1. Paraphrasing Complex Network:Network Compression via Factor Transfer. Kim, Jangho et al. NIPS 2018
2. Relational Knowledge Distillation.  Park, Wonpyo et al. CVPR 2019
   <!-- * 通过对输出embedding表示构建了instance之间的二阶距离关系和三届角度关系，作为一种知识引导student学习 -->
3. Knowledge Distillation via Instance Relationship Graph. Liu, Yufan et al. CVPR 2019
   <!-- * 通过instance中间层表示构建了instance之间的图，并将图作为一种知识进行传递。 -->
4. Contrastive Representation Distillation. Tian, Yonglong et al. arXiv: 1910.10699

### Privileged Information

1. Learning using privileged information: similarity control and knowledge transfer. Vapnik, Vladimir and Rauf, Izmailov. MLR 2015  
2. Unifying distillation and privileged information. Lopez-Paz, David et al. ICLR 2016
3. Model compression via distillation and quantization. Polino, Antonio et al. ICLR 2018
4. KDGAN:Knowledge Distillation with Generative Adversarial Networks. Wang, Xiaojie. NIPS 2018
5. [noval]Efficient Video Classification Using Fewer Frames. Bhardwaj, Shweta et al. CVPR 2019

## KD + GAN

1. Training Shallow and Thin Networks for Acceleration via Knowledge Distillation with Conditional Adversarial Networks. Xu, Zheng et al. ArXiv:1709.00513
2. KDGAN:Knowledge Distillation with Generative Adversarial Networks. Wang, Xiaojie. NIPS 2018
3. Adversarial Learning of Portable Student Networks. Wang, Yunhe et al. AAAI 2018
4. Adversarial Network Compression. Belagiannis, Vasileios et al. ECCV 2018
5. Cross-Modality Distillation: A case for Conditional Generative Adversarial Networks. ICASSP 2018
6. Adversarial Distillation for Efficient Recommendation with External Knowledge. TOIS 2018
7. [noval]DAFL:Data-Free Learning of Student Networks. Chen, Hanting et al. ICCV 2019
8. MEAL: Multi-Model Ensemble via Adversarial Learning. Shen,Zhiqiang, He,Zhankui, and Xue Xiangyang. AAAI 2019
9. Knowledge Distillation with Adversarial Samples Supporting Decision Boundary. Heo, Byeongho et al. AAAI 2019
10. Exploiting the Ground-Truth: An Adversarial Imitation Based Knowledge Distillation Approach for Event Detection. Liu, Jian et al. AAAI 2019
11. Adversarially Robust Distillation. Goldblum, Micah et al. arXiv:1905.09747
12. GAN-Knowledge Distillation for one-stage Object Detection. Hong, Wei et al. arXiv:1906.08467
13. Lifelong GAN: Continual Learning for Conditional Image Generation. Kundu et al. arXiv:1908.03884

## KD + Meta-learning

1. Sample Knowledge Distillation for Efficient Network Compression. Li, Tianhong et al. Arxiv:1812.01839
2. Learning What and Where to Transfer. Jang, Yunhun et al, ICML 2019
3. Transferring Knowledge across Learning Processes. Moreno, Pablo G et al. ICLR 2019

## Data-free KD
1. Data-Free Knowledge Distillation for Deep Neural Networks. NIPS 2017
2. Zero-Shot Knowledge Distillation in Deep Networks. ICML 2019
3. DAFL:Data-Free Learning of Student Networks. ICCV 2019

## KD + AutoML

1. Improving Neural Architecture Search Image Classifiers via Ensemble Learning. Macko, Vladimir et al. 2019

## KD + RL

1. N2N Learning: Network to Network Compression via Policy Gradient Reinforcement Learning. Ashok, Anubhav et al. ICLR 2018
2. Knowledge Flow:Improve Upon Your Teachers. Liu, Iou-jen et al. ICLR 2019
3. Transferring Knowledge across Learning Processes. Moreno, Pablo G et al. ICLR 2019

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
9. Multilingual Neural Machine Translation with Knowledge Distillation. ICLR 2019
    <!-- * 多个teacher（一个teacher一个语言对），简单融合。 -->
10. Unifying Heterogeneous Classifiers with Distillation. Vongkulbhisal et al. CVPR 2019
    <!-- * 有多个不完全一样场景下的分类器（分类目标不尽相同），如何将它们统一起来，构造一个总的分类器：认为每一个classifier同等重要，都要去拟合它们。 -->
11. Distilled Person Re-Identification: Towards a More Scalable System. Wu, Ancong et al. CVPR 2019
    <!-- * 知识不再是soft-label，而是similarity matrix。权重引入是teacher-level的，而不是instance-level的。 -->
12. Model Compression with Two-stage Multi-teacher Knowledge Distillation for Web Question Answering System。 Yang, Ze et al. WSDM 2020 
13. FEED: Feature-level Ensemble for Knowledge Distillation. Park, SeongUk and Kwak, Nojun. arXiv:1909.10754(AAAI20 pre)

## Cross-modal KD

1. Cross Modal Distillation for Supervision Transfer. Gupta, Saurabh et al. CVPR 2016
2. Emotion recognition in speech using cross-modal transfer in the wild. Albanie, Samuel et al. ACM MM 2018
3. Through-Wall Human Pose Estimation Using Radio Signals. Zhao, Mingmin et al. CVPR 2018
4. Compact Trilinear Interaction for Visual Question Answering. Do, Tuong et al. ICCV 2019
5. Cross-Modal Knowledge Distillation for Action Recognition. Thoker, Fida Mohammad and Gall, Juerge. ICIP 2019
6. Learning to Map Nearly Anything. Salem, Tawfiq et al. arXiv:1909.06928
7. Semantic-Aware Knowledge Preservation for Zero-Shot Sketch-Based Image Retrieval. Liu, Qing et al. ICCV 2019
8. UM-Adapt: Unsupervised Multi-Task Adaptation Using Adversarial Cross-Task Distillation. Kundu et al. ICCV 2019

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
12. Progressive Teacher-student Learning for Early Action Prediction. Wang, Xionghui et al. CVPR2019
13. Lightweight Image _Super-Resolution_ with Information Multi-distillation Network. Hui, Zheng et al. ICCVW 2019
14. AWSD:Adaptive Weighted Spatiotemporal Distillation for _Video Representation_. Tavakolian, Mohammad et al. ICCV 2019
15. Dynamic Kernel Distillation for Efficient _Pose Estimation_ in Videos. Nie, Xuecheng et al. ICCV 2019
16. Teacher Guided _Architecture Search_. Bashivan, Pouya and Tensen, Mark. ICCV 2019

### for NLP
1. Patient Knowledge Distillation for BERT Model Compression. Sun, Siqi et al. arXiv:1908.09355
2. TinyBERT: Distilling BERT for Natural Language Understanding. Jiao, Xiaoqi et al. arXiv:1909.10351
3. Learning to Specialize with Knowledge Distillation for Visual Question Answering. NIPS 2018
4. Knowledge Distillation for Bilingual Dictionary Induction. EMNLP 2017
5. A Teacher-Student Framework for Maintainable Dialog Manager. EMNLP 2018
6. Understanding Knowledge Distillation in Non-Autoregressive Machine Translation. arxiv 2019
7. DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter. Sanh, Victor et al. arXiv:1910.01108

## Beyond

1. Do deep nets really need to be deep?. Ba,Jimmy, and Rich Caruana. NIPS 2014
2. When Does Label Smoothing Help? Müller, Rafael, Kornblith, and Hinton. NIPS 2019
3. Towards Understanding Knowledge Distillation. Phuong, Mary and Lampert, Christoph. AAAI 2019
4. Harnessing deep neural networks with logucal rules. ACL 2016
   <!-- * 融合先验知识 -->
5. Adaptive Regularization of Labels. Ding, Qianggang et al. arXiv:1908.05474

---
Note: All papers pdf can be found and downloaded on bing or Google.

Source: <https://github.com/FLHonker/Awesome-Knowledge-Distillation>

Contact: Yuang Liu(<frankliu624@gmail.com>), AIDA, [ECNU](https://www.ecnu.edu.cn/).
