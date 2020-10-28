# 1.中文分词的发展历程

对380篇英文文献进行分析，大多是会议论文，来源包括ACL、EMNLP、COLING、IJCNLP等，收录最多的是ACL。SIGHAN是国际计算语言学协会中文处理特别兴趣组。SIGHAN采用多家机构的评测数据组织多次评测(即BakeOff)，评测使用封闭测试和开放测试两种方法。封闭测试只允许使用固定训练语料学习相应的模型，而开放测试可以使用任意资源。测试使用的评价标准包括准确率、召回率和F值。其中对比的是人工标注的数据集。CIPS-SIGHAN为中文处理资源与评测国际会议。

以SIGHAN和CIPS-SIGHAN的评测为主线，展示历届评测的重点内容和相关联的国际会议、时间，如下图所示。图中左侧使用不同颜色矩形框区分各个会议，圆形中的数字表示举办到第几届，评测与会议联合举办则增加了连线。

![image-20201022211145828](https://gitee.com/chengbo123/images/raw/master/image-20201022211145828.png)



SIGHAN2005提供的数据集包括训练集、测试集以及测试集黄金分割标准，除此之外还提供一个用于评分的脚本。比赛数据包括简体中文的北京大学PKU数据集和微软研究院MSR数据集；繁体中文的CityU数据集和AS数据集。

![image-20201022212202458](https://gitee.com/chengbo123/images/raw/master/image-20201022212202458.png)

![image-20201022212608319](https://gitee.com/chengbo123/images/raw/master/image-20201022212608319.png)

![image-20201022212655894](https://gitee.com/chengbo123/images/raw/master/image-20201022212655894.png)

![image-20201022212745046](https://gitee.com/chengbo123/images/raw/master/image-20201022212745046.png)

![image-20201022212921282](https://gitee.com/chengbo123/images/raw/master/image-20201022212921282.png)

# 2.中文分词的关键问题及模型算法

![中文分词](https://gitee.com/chengbo123/images/raw/master/%E4%B8%AD%E6%96%87%E5%88%86%E8%AF%8D.png)

![image-20201022225047590](https://gitee.com/chengbo123/images/raw/master/image-20201022225047590.png)

![image-20201022225117453](https://gitee.com/chengbo123/images/raw/master/image-20201022225117453.png)

![image-20201022225227050](https://gitee.com/chengbo123/images/raw/master/image-20201022225227050.png)



**参考文献：**
[1] 唐琳，郭崇慧，陈静锋 . 中文分词技术研究综述［J］. 数据分析与知识发现，2020，4（2/3）：1-17.